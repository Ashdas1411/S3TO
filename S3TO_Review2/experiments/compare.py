import json
from pathlib import Path

import matplotlib.pyplot as plt


def compare_and_plot(baseline_path="results/baseline_results.json", s3to_path="results/s3to_local_results.json"):
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(s3to_path, "r", encoding="utf-8") as f:
        s3to = json.load(f)

    b = {x["round"]: x for x in baseline["results"]}
    s = {x["round"]: x for x in s3to["results"]}
    rounds = sorted(set(b) & set(s))

    print("Round | Baseline Acc. | S3TO Acc.")
    print("------|---------------|----------")
    for r in rounds:
        print(f"{r:5d} | {b[r]['accuracy']*100:12.2f}% | {s[r]['accuracy']*100:8.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [b[r]["accuracy"] * 100 for r in rounds], marker="o", label="Baseline FedAvg")
    plt.plot(rounds, [s[r]["accuracy"] * 100 for r in rounds], marker="o", label="S3TO Hierarchical")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Baseline vs S3TO")
    plt.grid(True, alpha=0.25)
    plt.legend()
    Path("results").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/baseline_vs_s3to_accuracy.png", dpi=160)
    print("Saved: results/baseline_vs_s3to_accuracy.png")


if __name__ == "__main__":
    compare_and_plot()
