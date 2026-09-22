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


def partition_noniid(train_dataset, num_clients=6, samples_per_client=500, seed=42):
    """Create deterministic label-skewed partitions for six clients.

    Client label preferences are intentionally different so the label histograms
    are useful to the S3TO controller.
    """
    rng = np.random.default_rng(seed)
    labels = _labels(train_dataset)
    label_to_indices = {label: np.where(labels == label)[0].tolist() for label in range(10)}
    for inds in label_to_indices.values():
        rng.shuffle(inds)

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
        labels_for_client = preferences.get(client_id, [client_id % 10])
        per_label = samples_per_client // len(labels_for_client)
        remainder = samples_per_client - per_label * len(labels_for_client)
        selected = []
        for pos, label in enumerate(labels_for_client):
            take = per_label + (1 if pos < remainder else 0)
            if len(label_to_indices[label]) < take:
                raise RuntimeError(f"Not enough examples of label {label} for client {client_id}")
            selected.extend(label_to_indices[label][:take])
            label_to_indices[label] = label_to_indices[label][take:]
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
