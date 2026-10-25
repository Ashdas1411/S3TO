#!/usr/bin/env python3
"""Standalone Raspberry Pi client: connectivity test or one local FL update."""
import argparse
import socket
import sys
from pathlib import Path

try:
    from .secure_wire import pack_message, recv_message
except ImportError:
    # Support direct execution from a standalone Pi bundle.
    from secure_wire import pack_message, recv_message


def rpc(host, port, message, timeout=180):
    with socket.create_connection((host, port), timeout=15) as sock:
        sock.settimeout(timeout)
        sock.sendall(pack_message(message))
        response = recv_message(sock)

    if not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(f"Receiver rejected the request: {response}")
    return response


def train_and_send(args):
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Training requires PyTorch and NumPy. Use --mode smoke to test "
            "network connectivity without these packages."
        ) from exc

    # Find the copied project directory containing fl/model.py.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    sys.path.insert(0, str(project_root))
    try:
        from fl.model import SimpleMLP
    except ImportError:
        # Standalone Pi bundle: model.py sits beside this script.
        from model import SimpleMLP

    data_path = Path(args.data)
    initial_path = Path(args.initial_model)
    if not data_path.is_file() or not initial_path.is_file():
        raise FileNotFoundError(
            f"Missing client data or initial model: {data_path}, {initial_path}"
        )

    with np.load(data_path) as archive:
        images_array = archive["images"].copy()
        labels_array = archive["labels"].copy()

    images = torch.as_tensor(images_array, dtype=torch.float32)
    if images.numel() and images.max().item() > 1:
        images = images / 255.0
    labels = torch.as_tensor(labels_array, dtype=torch.long)

    dataset = TensorDataset(images, labels)
    loader_generator = torch.Generator().manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but isn't available on this device.")

    model = SimpleMLP().to(device)
    initial_state = torch.load(
        initial_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(initial_state)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            seen += batch_x.size(0)

        print(
            f"[TRAIN] {args.client_id} epoch={epoch+1}/{args.epochs} "
            f"loss={total_loss/max(seen, 1):.4f}",
            flush=True,
        )

    # Send CPU tensors to keep the payload independent of CUDA.
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    response = rpc(
        args.host,
        args.port,
        {
            "type": "model_update",
            "round": args.round,
            "client_id": args.client_id,
            "sample_count": len(dataset),
            "state_dict": state,
        },
    )
    print(f"[UPDATE ACK] {response}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--host", required=True, help="Reachable laptop IPv4 address")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--client-id", default="pi1")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--data", default="client_data.npz")
    parser.add_argument("--initial-model", default="initial_model_mlp.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.mode == "smoke":
        response = rpc(
            args.host,
            args.port,
            {"type": "ping", "client_id": args.client_id},
            timeout=15,
        )
        print(f"[PING ACK] {response}")
    else:
        train_and_send(args)


if __name__ == "__main__":
    main()
