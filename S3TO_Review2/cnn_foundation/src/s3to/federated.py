"""Local client training and sample-weighted FedAvg for S3TO."""

import copy
import random
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.s3to.model import MNISTCNN


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_local_model(
    global_state: Dict[str, torch.Tensor],
    dataset,
    indices: List[int],
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.01,
) -> Tuple[Dict[str, torch.Tensor], int, float]:
    """Train a local model and return its weights, sample count, and loss."""
    if not indices:
        raise ValueError("A client must have at least one training sample.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    model = MNISTCNN().to(device)
    model.load_state_dict(copy.deepcopy(global_state))
    model.train()

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_loss = 0.0
    total_examples = 0

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            total_examples += batch_count

    average_loss = total_loss / total_examples

    # Move returned tensors to CPU so aggregation does not retain GPU memory.
    local_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    return local_state, len(indices), average_loss


def fedavg(
    client_states: List[Dict[str, torch.Tensor]],
    client_sizes: List[int],
) -> Dict[str, torch.Tensor]:
    """Aggregate client models, weighting each by its sample count."""
    if not client_states:
        raise ValueError("At least one client model is required.")
    if len(client_states) != len(client_sizes):
        raise ValueError("Each client model must have a matching sample count.")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("Every client sample count must be positive.")

    reference_keys = set(client_states[0].keys())

    for state in client_states[1:]:
        if set(state.keys()) != reference_keys:
            raise ValueError("Client model states have different keys.")

    total_samples = sum(client_sizes)
    aggregated = {}

    for name in client_states[0]:
        reference_tensor = client_states[0][name]

        if not torch.is_floating_point(reference_tensor):
            # This CNN has no integer-valued model buffers. Keep this explicit
            # so future model changes do not silently average integer buffers.
            if any(
                not torch.equal(state[name], reference_tensor)
                for state in client_states[1:]
            ):
                raise ValueError(
                    f"Non-floating state '{name}' differs across clients."
                )
            aggregated[name] = reference_tensor.clone()
            continue

        result = torch.zeros_like(reference_tensor)

        for state, sample_count in zip(client_states, client_sizes):
            if state[name].shape != reference_tensor.shape:
                raise ValueError(f"Shape mismatch for model tensor '{name}'.")
            result.add_(
                state[name].to(dtype=result.dtype),
                alpha=sample_count / total_samples,
            )

        aggregated[name] = result

    return aggregated


def evaluate_model(
    model_state: Dict[str, torch.Tensor],
    dataset,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[float, float]:
    """Return test loss and accuracy for a supplied model state."""
    model = MNISTCNN().to(device)
    model.load_state_dict(model_state)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    """Run a small one-client smoke test, not a full FL experiment."""
    from src.s3to.data_partition import (
        create_noniid_partitions,
        load_mnist,
    )

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_mnist(download=False)
    partitions, _ = create_noniid_partitions(dataset)

    initial_model = MNISTCNN()
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in initial_model.state_dict().items()
    }

    # Use only a small subset to verify the training implementation first.
    client_id = 0
    smoke_indices = partitions[client_id][:256]

    updated_state, sample_count, loss = train_local_model(
        global_state=initial_state,
        dataset=dataset,
        indices=smoke_indices,
        device=device,
        epochs=1,
        batch_size=64,
        learning_rate=0.01,
    )

    assert sample_count == len(smoke_indices)
    assert loss >= 0.0
    assert set(updated_state) == set(initial_state)

    changed_tensors = sum(
        not torch.equal(initial_state[name], updated_state[name])
        for name in initial_state
    )
    assert changed_tensors > 0, "Training did not change any model tensors."

    print(f"Device: {device}")
    print(f"Smoke-test client: {client_id}")
    print(f"Training samples: {sample_count}")
    print(f"Average local training loss: {loss:.4f}")
    print(f"Model tensors changed: {changed_tensors}")
    print("Local-training smoke test: PASSED")


if __name__ == "__main__":
    main()
