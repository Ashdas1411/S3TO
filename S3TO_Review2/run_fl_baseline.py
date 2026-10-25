import argparse
import time
from pathlib import Path

import torch

from common.config import load_config
from common.utils import set_seed, save_json
from fl.data import load_mnist, partition_noniid, make_loader, make_test_loader
from fl.fedavg import fedavg
from fl.local_trainer import train_local, evaluate
from fl.model_factory import create_model


def run(args):
    cfg = load_config(args.config)
    set_seed(cfg["fl"]["seed"])
    data_dir = str(Path(args.data_dir or cfg["fl"]["data_dir"]).expanduser())
    device = args.device or cfg["fl"].get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_ds, test_ds, used_fallback = load_mnist(
        data_dir,
        download=not args.no_download,
        allow_fallback=not args.require_real_mnist,
    )
    partitions = partition_noniid(
        train_ds,
        num_clients=cfg["fl"]["num_clients"],
        samples_per_client=cfg["fl"]["samples_per_client"],
        seed=cfg["fl"]["seed"],
        mode=cfg["fl"].get("partition_mode", "fixed"),
        dirichlet_alpha=float(cfg["fl"].get("dirichlet_alpha", 0.5)),
    )
    test_loader = make_test_loader(test_ds)
    global_state = create_model(args.model).state_dict()
    results = []

    rounds = int(args.rounds or cfg["fl"]["rounds"])
    for round_no in range(1, rounds + 1):
        start = time.time()
        states, counts = [], []
        for cid in range(cfg["fl"]["num_clients"]):
            model = create_model(args.model)
            model.load_state_dict(global_state)
            loader = make_loader(train_ds, partitions[cid], cfg["fl"]["batch_size"])
            state, loss = train_local(
                model,
                loader,
                epochs=cfg["fl"]["local_epochs"],
                lr=cfg["fl"]["learning_rate"],
                device=device,
            )
            states.append(state)
            counts.append(len(partitions[cid]))
            print(f"[BASELINE] Round {round_no} | Client {cid+1} | loss={loss:.4f}")
        global_state = fedavg(states, counts)
        model = create_model(args.model); model.load_state_dict(global_state)
        acc = evaluate(model, test_loader, device)
        elapsed = time.time() - start
        results.append({"round": round_no, "accuracy": acc, "wall_clock_sec": elapsed})
        print(f"[BASELINE] Round {round_no} | accuracy={acc*100:.2f}% | time={elapsed:.2f}s")

    dataset_tag = "synthetic_digit_like" if used_fallback else "real_mnist"
    output = {
        "model": args.model,
        "dataset": dataset_tag,
        "dataset_fallback_used": used_fallback,
        "data_dir": data_dir,
        "device": device,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "results": results,
    }
    default_output = Path("results") / f"baseline_{args.model}_{dataset_tag}.json"
    output_path = Path(args.output) if args.output else default_output
    save_json(output, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--require-real-mnist", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)
