
#!/usr/bin/env python3

import argparse
import itertools

import numpy as np
from scipy.stats import spearmanr

from common.config import load_config
from fl.data import load_mnist, partition_noniid, label_histogram
from s3to.grouping import (
    _unique_partitions,
    assignment_from_groups,
)
from s3to.similarity import pairwise_cosine


def js_divergence(p, q):
    """Jensen-Shannon divergence using natural logarithms."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def calculate_metrics(groups, histograms, pairwise, global_hist):
    """Return mean within-group cosine and weighted edge-to-global JS."""
    group_cosines = []
    weighted_js = 0.0
    total_samples = float(global_hist.sum())

    if total_samples <= 0:
        raise ValueError("The global label histogram must have a positive sum.")

    for group in groups:
        cosines = [
            pairwise[a][b]
            for a, b in itertools.combinations(group, 2)
        ]

        group_cosines.append(
            float(np.mean(cosines)) if cosines else 0.0
        )

        edge_hist = sum(
            (histograms[cid] for cid in group),
            np.zeros_like(global_hist, dtype=float),
        )

        edge_samples = float(edge_hist.sum())

        if edge_samples <= 0:
            raise ValueError(f"Group {group} has no samples.")

        weighted_js += (
            edge_samples / total_samples
        ) * js_divergence(edge_hist, global_hist)

    mean_cosine = (
        float(np.mean(group_cosines)) if group_cosines else 0.0
    )

    return mean_cosine, float(weighted_js)


def main():
    parser = argparse.ArgumentParser(
        description="Compare S3TO grouping objectives across all assignments."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--data-dir",
        default="/home/arkad/s3to/data",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    train_ds, _, fallback = load_mnist(
        args.data_dir,
        download=False,
        allow_fallback=False,
    )

    if fallback:
        raise RuntimeError(
            "Real MNIST is required; synthetic fallback is not accepted."
        )

    client_count = config["fl"]["num_clients"]
    samples_per_client = config["fl"]["samples_per_client"]
    seed = config["fl"]["seed"]
    group_size = config["s3to"]["group_size"]

    if client_count % group_size != 0:
        raise ValueError(
            "Client count must be divisible by group size."
        )

    partitions = partition_noniid(
        train_ds,
        num_clients=client_count,
        samples_per_client=samples_per_client,
        seed=seed,
        mode=config["fl"].get("partition_mode", "fixed"),
        dirichlet_alpha=float(config["fl"].get("dirichlet_alpha", 0.5)),
    )

    histograms = {
        cid: label_histogram(train_ds, indices).astype(float)
        for cid, indices in partitions.items()
    }

    global_hist = sum(
        histograms.values(),
        np.zeros(10, dtype=float),
    )

    pairwise = pairwise_cosine(histograms)

    all_groups = list(
        _unique_partitions(histograms, group_size)
    )

    rows = []

    for groups in all_groups:
        cosine_score, js_score = calculate_metrics(
            groups,
            histograms,
            pairwise,
            global_hist,
        )

        rows.append({
            "groups": groups,
            "cosine": cosine_score,
            "weighted_js": js_score,
        })

    if not rows:
        raise RuntimeError("No candidate assignments were generated.")

    print(f"Evaluated {len(rows)} unique assignments.")
    print("Lower is better for both metrics.")
    print("No. | Groups | Within cosine | Weighted JS divergence")

    for i, row in enumerate(rows, start=1):
        print(
            f'{i:3} | {str(row["groups"]):25} | '
            f'{row["cosine"]:.12f} | '
            f'{row["weighted_js"]:.12f}'
        )

    best_cosine = min(
        rows,
        key=lambda row: (row["cosine"], row["groups"]),
    )
    best_js = min(
        rows,
        key=lambda row: (row["weighted_js"], row["groups"]),
    )

    print("\nBest by cosine:", best_cosine["groups"])
    print("Best by weighted JS:", best_js["groups"])

    cosine_values = np.array(
        [row["cosine"] for row in rows],
        dtype=float,
    )
    js_values = np.array(
        [row["weighted_js"] for row in rows],
        dtype=float,
    )

    raw_correlation = spearmanr(
        cosine_values,
        js_values,
    ).statistic

    # Diagnostic: remove insignificant differences at the displayed
    # precision before ranking, so effectively tied scores remain tied.
    rounded_correlation = spearmanr(
        np.round(cosine_values, 10),
        np.round(js_values, 10),
    ).statistic

    print(
        "Spearman correlation (unrounded scores): "
        f"{raw_correlation:.4f}"
    )
    print(
        "Spearman correlation (scores rounded to 10 decimals): "
        f"{rounded_correlation:.4f}"
    )

    print(
        "\nInterpretation: these scores compare label-distribution "
        "objectives for this dataset and configuration only. They do "
        "not establish improvements in model accuracy or network performance."
    )


if __name__ == "__main__":
    main()
