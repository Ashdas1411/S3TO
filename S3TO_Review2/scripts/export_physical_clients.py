#!/usr/bin/env python3
"""Export two deterministic MNIST partitions and an initial MLP checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common.config import load_config
from common.utils import set_seed
from fl.data import load_mnist, partition_noniid
from fl.model_factory import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="/home/arkad/s3to/data")
    parser.add_argument("--output-dir", default="results/hardware")
    parser.add_argument("--samples-per-client", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["fl"]["seed"])

    data_dir = Path(args.data_dir).expanduser()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds, _, fallback = load_mnist(
        str(data_dir), download=False, allow_fallback=False
    )
    if fallback:
        raise RuntimeError("Real MNIST is required for hardware data export.")

    partitions = partition_noniid(
        train_ds,
        num_clients=cfg["fl"]["num_clients"],
        samples_per_client=args.samples_per_client,
        seed=cfg["fl"]["seed"],
        mode=cfg["fl"].get("partition_mode", "fixed"),
        dirichlet_alpha=float(cfg["fl"].get("dirichlet_alpha", 0.5)),
    )

    for cid, name in [(0, "pi1_data.npz"), (1, "pi2_data.npz")]:
        images = []
        labels = []
        for index in partitions[cid]:
            image, label = train_ds[int(index)]
            if torch.is_tensor(image):
                image = image.detach().cpu().numpy()
            else:
                image = np.asarray(image)
            images.append(image)
            labels.append(int(label))

        path = out / name
        np.savez_compressed(
            path,
            images=np.stack(images),
            labels=np.asarray(labels, dtype=np.int64),
        )
        print(f"Exported {len(labels)} samples: {path} ({path.stat().st_size:,} bytes)")

    torch.manual_seed(cfg["fl"]["seed"])
    model = create_model("mlp")
    initial = out / "initial_model_mlp.pt"
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        initial,
    )
    print(f"Initial model saved: {initial}")
    print(f"Seed: {cfg['fl']['seed']}; model: mlp; real MNIST: {not fallback}")


if __name__ == "__main__":
    main()
