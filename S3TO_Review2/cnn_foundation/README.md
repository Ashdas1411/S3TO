# S3TO CNN Foundation

This directory contains a standalone development module for the S3TO
project. It preserves the newer CNN, Non-IID MNIST partitioning, local
client training, FedAvg aggregation, and model evaluation code.

It complements the existing `S3TO_Review2` prototype; it does not replace
or claim to be integrated with its networking and Ryu/Mininet pipeline.

## Contents

- `src/s3to/model.py` — MNIST convolutional neural network.
- `src/s3to/data_partition.py` — MNIST loading and Non-IID client partitions.
- `src/s3to/federated.py` — local training, sample-weighted FedAvg,
  evaluation, and a small local-training smoke test.

## Requirements

- Python 3.10 or a compatible newer Python version.
- PyTorch.
- Torchvision.

The development environment originally tested was Ubuntu under WSL,
Python 3.10.21, PyTorch 2.14.1+cu130 and Torchvision 0.29.1+cu130,
with CUDA available on an NVIDIA RTX 4050.

Those CUDA-specific builds should not be pinned for macOS. Install
platform-compatible PyTorch and Torchvision packages using this
directory's `requirements.txt`.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, virtual-environment activation is:

```powershell
.venv\Scripts\Activate.ps1
```

## Download the MNIST training dataset

The local smoke test expects MNIST to be available under `data/`.
Download it with:

```bash
python - <<'PY'
from src.s3to.data_partition import load_mnist

dataset = load_mnist(download=True)
print("MNIST samples:", len(dataset))
PY
```

The downloaded dataset is local data and is intentionally excluded from Git.

## Validation

Run these commands from this directory after setup:

```bash
python -m py_compile src/s3to/model.py src/s3to/data_partition.py src/s3to/federated.py
python -m src.s3to.model
```

To test the sample-weighted FedAvg calculation:

```bash
python - <<'PY'
import torch
from src.s3to.federated import fedavg

states = [
    {"weight": torch.tensor([2.0])},
    {"weight": torch.tensor([6.0])},
]
result = fedavg(states, [2, 6])
print("Weighted average:", result["weight"].item())
assert abs(result["weight"].item() - 5.0) < 1e-6
print("FedAvg numerical test: PASSED")
PY
```

After downloading MNIST, run the one-client smoke test:

```bash
python -m src.s3to.federated
```

This smoke test trains on a small subset; it is not a full federated-learning
experiment.

## Current limitations

- The CNN, partitioner, local-training function and FedAvg have passed
  individual development checks on the original WSL environment.
- The MacBook setup has not yet been tested.
- The current automatic device selection uses CUDA when available and
  otherwise CPU; Apple GPU (MPS) selection has not yet been implemented.
- A full multi-client federated round and integration with the Review-2
  networking/controller implementation remain future work.
- No performance improvement over the baseline should be claimed until
  controlled experiments have been run.

## Git and data policy

Commit source code, documentation, and reproducible setup instructions.
Do not commit virtual environments, downloaded datasets, local logs, or
generated model checkpoints.
