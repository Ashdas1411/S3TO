import argparse
import time
from pathlib import Path

from common.config import load_config
from common.utils import set_seed, save_json
from fl.data import load_mnist, partition_noniid, make_loader, make_test_loader
from fl.fedavg import fedavg
from fl.local_trainer import train_local, evaluate
from fl.model import SimpleMLP


def run(args):
    cfg = load_config(args.config)
    set_seed(cfg["fl"]["seed"])
    train_ds, test_ds, used_fallback = load_mnist(
        cfg["fl"]["data_dir"], download=not args.no_download, allow_fallback=True
    )
    partitions = partition_noniid(
        train_ds,
        num_clients=cfg["fl"]["num_clients"],
        samples_per_client=cfg["fl"]["samples_per_client"],
        seed=cfg["fl"]["seed"],
    )
    test_loader = make_test_loader(test_ds)
    global_state = SimpleMLP().state_dict()
    results = []

    rounds = int(args.rounds or cfg["fl"]["rounds"])
    for round_no in range(1, rounds + 1):
        start = time.time()
        states, counts = [], []
        for cid in range(cfg["fl"]["num_clients"]):
            model = SimpleMLP()
            model.load_state_dict(global_state)
            loader = make_loader(train_ds, partitions[cid], cfg["fl"]["batch_size"])
            state, loss = train_local(
                model,
                loader,
                epochs=cfg["fl"]["local_epochs"],
                lr=cfg["fl"]["learning_rate"],
                device=cfg["fl"]["device"],
            )
            states.append(state)
            counts.append(len(partitions[cid]))
            print(f"[BASELINE] Round {round_no} | Client {cid+1} | loss={loss:.4f}")
        global_state = fedavg(states, counts)
        model = SimpleMLP(); model.load_state_dict(global_state)
        acc = evaluate(model, test_loader, cfg["fl"]["device"])
        elapsed = time.time() - start
        results.append({"round": round_no, "accuracy": acc, "wall_clock_sec": elapsed})
        print(f"[BASELINE] Round {round_no} | accuracy={acc*100:.2f}% | time={elapsed:.2f}s")

    output = {"dataset_fallback_used": used_fallback, "results": results}
    save_json(output, Path("results") / "baseline_results.json")
    print("Saved: results/baseline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    run(args)
