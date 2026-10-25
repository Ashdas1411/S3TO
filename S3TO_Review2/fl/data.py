from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _synthetic_dataset(n=8000, seed=42):
    """Offline fallback only. Creates label-specific simple 28x28 patterns."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n, dtype=torch.long) % 10
    x = torch.randn((n, 1, 28, 28), generator=g) * 0.10
    for i in range(n):
        label = int(y[i])
        row = 2 + label * 2
        col = 2 + (label * 2) % 20
        x[i, 0, row:row + 5, :] += 0.45
        x[i, 0, :, col:col + 3] += 0.45
    x = x.clamp(0, 1)
    return TensorDataset(x, y)


def load_mnist(data_dir="./data", download=True, allow_fallback=True):
    data_dir = Path(data_dir)
    transform = transforms.ToTensor()
    try:
        train = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        test = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
        return train, test, False
    except Exception as exc:
        if not allow_fallback:
            raise
        print(f"[WARN] MNIST could not be loaded: {exc}")
        print("[WARN] Using offline synthetic digit-like data so the prototype can still run.")
        train = _synthetic_dataset(8000, seed=42)
        test = _synthetic_dataset(2000, seed=123)
        return train, test, True


def _labels(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if isinstance(dataset, TensorDataset):
        return dataset.y.numpy()
    raise TypeError("Dataset must expose labels via .targets or be TensorDataset")



def partition_noniid(
    train_dataset,
    num_clients=6,
    samples_per_client=500,
    seed=42,
    mode="fixed",
    dirichlet_alpha=0.5,
):
    """
    Partition a dataset without reusing sample indices.

    Modes:
      fixed: Preserve the original hard-coded client label preferences.
      dirichlet: Draw a label distribution independently for each client.

    Lower Dirichlet alpha generally produces more concentrated
    label distributions. Higher alpha generally produces more
    balanced distributions.

    Exactly samples_per_client indices are returned for each client.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be positive.")

    if samples_per_client < 1:
        raise ValueError("samples_per_client must be positive.")

    if num_clients * samples_per_client > len(train_dataset):
        raise ValueError(
            "Requested samples exceed the dataset size; "
            "unique partitions cannot be created."
        )

    mode = mode.lower()

    if mode not in {"fixed", "dirichlet"}:
        raise ValueError(
            "mode must be either 'fixed' or 'dirichlet'."
        )

    if mode == "dirichlet" and (
        not np.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0
    ):
        raise ValueError(
            "dirichlet_alpha must be a finite positive number."
        )

    rng = np.random.default_rng(seed)
    labels = _labels(train_dataset)

    label_to_indices = {
        label: np.where(labels == label)[0].tolist()
        for label in np.unique(labels)
    }

    for indices in label_to_indices.values():
        rng.shuffle(indices)

    # Original behaviour is retained for backwards compatibility.
    if mode == "fixed":
        preferences = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7],
            4: [8, 9],
            5: [0, 3, 6, 9],
        }

        partitions = {}

        for client_id in range(num_clients):
            labels_for_client = preferences.get(
                client_id, [client_id % 10]
            )

            missing_labels = [
                label for label in labels_for_client
                if label not in label_to_indices
            ]
            if missing_labels:
                raise ValueError(
                    f"Dataset is missing labels: {missing_labels}"
                )

            per_label = (
                samples_per_client // len(labels_for_client)
            )
            remainder = (
                samples_per_client
                - per_label * len(labels_for_client)
            )

            selected = []

            for position, label in enumerate(labels_for_client):
                take = per_label + (
                    1 if position < remainder else 0
                )
                bucket = label_to_indices[label]

                if len(bucket) < take:
                    raise RuntimeError(
                        f"Not enough examples of label {label} "
                        f"for client {client_id}."
                    )

                selected.extend(bucket[:take])
                del bucket[:take]

            rng.shuffle(selected)
            partitions[client_id] = selected

        return partitions

    # Dirichlet mode: draw each client's class proportions, then
    # select actual dataset indices without replacement.
    class_labels = sorted(label_to_indices)
    num_classes = len(class_labels)

    partitions = {}

    for client_id in range(num_clients):
        probabilities = rng.dirichlet(
            np.full(num_classes, dirichlet_alpha)
        )

        quotas = rng.multinomial(
            samples_per_client,
            probabilities,
        )

        selected = []
        missing = 0

        # Take each requested class quota from its remaining pool.
        for class_position, label in enumerate(class_labels):
            requested = int(quotas[class_position])
            bucket = label_to_indices[label]
            take = min(requested, len(bucket))

            if take:
                selected.extend(bucket[:take])
                del bucket[:take]

            missing += requested - take

        # If any class pool was exhausted, fill the deficit from
        # remaining classes so each client still gets the requested
        # number of unique samples.
        if missing:
            available_labels = [
                label for label, bucket in label_to_indices.items()
                if bucket
            ]
            rng.shuffle(available_labels)

            for label in available_labels:
                bucket = label_to_indices[label]
                take = min(missing, len(bucket))

                if take:
                    selected.extend(bucket[:take])
                    del bucket[:take]
                    missing -= take

                if missing == 0:
                    break

        if missing:
            raise RuntimeError(
                f"Could not fill client {client_id}: "
                f"{missing} samples remain unavailable."
            )

        rng.shuffle(selected)
        partitions[client_id] = selected

    return partitions


def label_histogram(dataset, indices, num_classes=10):
    labels = _labels(dataset)[indices]
    counts = Counter(int(v) for v in labels)
    return np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)


def make_loader(dataset, indices, batch_size=64, shuffle=True):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def make_test_loader(test_dataset, batch_size=256):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
