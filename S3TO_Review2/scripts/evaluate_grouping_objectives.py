import argparse
import itertools

import numpy as np

from common.config import load_config
from fl.data import load_mnist, partition_noniid, label_histogram
from s3to.grouping import (
    _unique_partitions,
    assignment_from_groups,
    choose_complementary_groups,
)
from s3to.similarity import pairwise_cosine


def js_divergence(p, q):
    """Jensen-Shannon divergence using natural logarithms."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def metrics(groups, histograms, pairwise, global_hist):
    group_cosines = []
    weighted_js = 0.0
    total_samples = sum(global_hist)

    for group in groups:
        cosines = [
            pairwise[a][b]
            for a, b in itertools.combinations(group, 2)
        ]
        group_cosines.append(float(np.mean(cosines)) if cosines else 0.0)

        edge_hist = sum(
            (histograms[cid] for cid in group),
            np.zeros_like(global_hist, dtype=float),
        )
        edge_samples = edge_hist.sum()
        weighted_js += (
            edge_samples / total_samples
        ) * js_divergence(edge_hist, global_hist)

    return float(np.mean(group_cosines)), weighted_js


def describe(label, groups, histograms, pairwise, global_hist):
    cosine, divergence = metrics(groups, histograms, pairwise, global_hist)
    print(f"\n{label}")
    print(f"  Groups: {groups}")
    print(f"  Assignment: {assignment_from_groups(groups)}")
    print(f"  Mean within-group cosine (lower = more complementary): {cosine:.6f}")
    print(f"  Weighted edge-to-global JS divergence (lower = more representative): {divergence:.6f}")

    for edge_index, group in enumerate(groups, start=1):
        total = sum(
            (histograms[cid] for cid in group),
            np.zeros_like(global_hist, dtype=float),
        )
        print(
            f"  edge{edge_index} client IDs={group}; "
            f"label proportions={np.round(total / total.sum(), 3).tolist()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="/home/arkad/s3to/data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_ds, _, fallback = load_mnist(
        args.data_dir, download=False, allow_fallback=False
    )
    if fallback:
        raise RuntimeError("Real MNIST is required for this diagnostic.")

    partitions = partition_noniid(
        train_ds,
        num_clients=cfg["fl"]["num_clients"],
        samples_per_client=cfg["fl"]["samples_per_client"],
        seed=cfg["fl"]["seed"],
        mode=cfg["fl"].get("partition_mode", "fixed"),
        dirichlet_alpha=float(cfg["fl"].get("dirichlet_alpha", 0.5)),
    )
    histograms = {
        cid: label_histogram(train_ds, indices).astype(float)
        for cid, indices in partitions.items()
    }
    global_hist = sum(histograms.values(), np.zeros(10, dtype=float))
    pairwise = pairwise_cosine(histograms)

    current_groups, _, _ = choose_complementary_groups(
        histograms, group_size=cfg["s3to"]["group_size"]
    )
    all_groups = list(
        _unique_partitions(histograms, cfg["s3to"]["group_size"])
    )

    scored = [
        (
            metrics(groups, histograms, pairwise, global_hist),
            groups,
        )
        for groups in all_groups
    ]

    best_cosine = min(scored, key=lambda item: (item[0][0], item[1]))
    best_global = min(scored, key=lambda item: (item[0][1], item[1]))

    print(f"Evaluated {len(all_groups)} unique assignments.")
    print("These are label-distribution metrics only; no network improvement is inferred.")

    describe("CURRENT S3TO OBJECTIVE", current_groups, histograms, pairwise, global_hist)
    describe("BEST BY EXISTING COSINE OBJECTIVE", best_cosine[1], histograms, pairwise, global_hist)
    describe("BEST BY EDGE-TO-GLOBAL REPRESENTATIVENESS", best_global[1], histograms, pairwise, global_hist)


if __name__ == "__main__":
    main()
