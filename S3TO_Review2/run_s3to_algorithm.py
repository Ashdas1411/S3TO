import argparse
import json

from common.config import load_config
from common.utils import save_json
from fl.data import load_mnist, partition_noniid, label_histogram
from s3to.engine import S3TOEngine


def main(args):
    cfg = load_config(args.config)
    train_ds, _, fallback = load_mnist(
        cfg["fl"]["data_dir"], download=not args.no_download, allow_fallback=True
    )
    partitions = partition_noniid(
        train_ds,
        num_clients=cfg["fl"]["num_clients"],
        samples_per_client=cfg["fl"]["samples_per_client"],
        seed=cfg["fl"]["seed"],
        mode=cfg["fl"].get("partition_mode", "fixed"),
        dirichlet_alpha=float(cfg["fl"].get("dirichlet_alpha", 0.5)),
    )
    histograms = {cid: label_histogram(train_ds, inds).tolist() for cid, inds in partitions.items()}
    result = S3TOEngine(cfg["s3to"]["group_size"]).decide(histograms)
    data = {
        "dataset_fallback_used": fallback,
        "histograms": result.histograms,
        "groups": result.groups,
        "group_scores": result.group_scores,
        "assignment": result.assignment,
        "pairwise_cosine": result.pairwise_cosine,
    }
    save_json(data, "results/s3to_algorithm_result.json")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--no-download", action="store_true")
    main(p.parse_args())
