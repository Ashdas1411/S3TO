"""Reproducible Non-IID MNIST partitioning for S3TO."""

from collections import Counter
from typing import Dict, List, Tuple

import torch
from torchvision import datasets, transforms


NUM_CLIENTS = 6
NUM_CLASSES = 10
SEED = 42

# Explicit class allocation: every client receives exactly two digits.
CLIENT_DIGITS = {
    0: (0, 1),
    1: (1, 2),
    2: (3, 4),
    3: (4, 5),
    4: (6, 7),
    5: (8, 9),
}


def load_mnist(root: str = "data", download: bool = True):
    """Load the MNIST training dataset."""
    return datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=transforms.ToTensor(),
    )


def create_noniid_partitions(
    dataset,
    num_clients: int = NUM_CLIENTS,
    seed: int = SEED,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, int]]]:
    """Create deterministic, disjoint client partitions with two digits each."""
    if num_clients != NUM_CLIENTS:
        raise ValueError(f"This setup requires exactly {NUM_CLIENTS} clients.")

    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    # Collect and deterministically shuffle the indices for each digit.
    class_indices = {}
    for digit in range(NUM_CLASSES):
        indices = torch.where(targets == digit)[0]
        order = torch.randperm(len(indices), generator=generator)
        class_indices[digit] = indices[order].tolist()

    # Determine which clients are assigned each digit.
    digit_owners = {digit: [] for digit in range(NUM_CLASSES)}
    for client_id, digits in CLIENT_DIGITS.items():
        for digit in digits:
            digit_owners[digit].append(client_id)

    partitions = {client_id: [] for client_id in range(NUM_CLIENTS)}

    # Divide each digit's examples among its assigned clients.
    for digit, owners in digit_owners.items():
        indices = class_indices[digit]
        chunk_size = len(indices) // len(owners)

        for position, client_id in enumerate(owners):
            start = position * chunk_size
            end = (
                (position + 1) * chunk_size
                if position < len(owners) - 1
                else len(indices)
            )
            partitions[client_id].extend(indices[start:end])

    # Shuffle each client's index order reproducibly.
    for client_id, indices in partitions.items():
        client_generator = torch.Generator().manual_seed(seed + client_id + 1)
        order = torch.randperm(len(indices), generator=client_generator).tolist()
        partitions[client_id] = [indices[i] for i in order]

    histograms = {}
    for client_id, indices in partitions.items():
        labels = targets[indices].tolist()
        histograms[client_id] = dict(sorted(Counter(labels).items()))

    return partitions, histograms


def make_client_subsets(dataset, partitions):
    """Create a PyTorch Subset for each client."""
    from torch.utils.data import Subset

    return {
        client_id: Subset(dataset, indices)
        for client_id, indices in partitions.items()
    }


def main():
    dataset = load_mnist()
    partitions, histograms = create_noniid_partitions(dataset)

    all_indices = [index for indices in partitions.values() for index in indices]

    assert len(partitions) == NUM_CLIENTS
    assert len(all_indices) == len(dataset)
    assert len(set(all_indices)) == len(dataset)
    assert set(all_indices) == set(range(len(dataset)))

    for client_id, digits in CLIENT_DIGITS.items():
        assert set(histograms[client_id]) == set(digits), (
            f"Client {client_id} has unexpected digits: {histograms[client_id]}"
        )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of clients: {len(partitions)}")
    print(f"Total partitioned samples: {len(all_indices)}")
    print("Every example assigned exactly once: YES")

    print("\nLabel histograms:")
    for client_id in sorted(histograms):
        print(
            f"Client {client_id}: {histograms[client_id]} "
            f"(samples={len(partitions[client_id])})"
        )

    print("\nNon-IID partition validation: PASSED")


if __name__ == "__main__":
    main()
