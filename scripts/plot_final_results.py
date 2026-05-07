"""Plot final benchmark and benign utility summaries."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label(row):
    parts = [row.get("model_key", "model")]
    if row.get("attack"):
        parts.append(row["attack"])
    if row.get("scenario"):
        parts.append(row["scenario"])
    if row.get("split"):
        parts.append(row["split"])
    return "\n".join(parts)


def plot_attack_metrics(summary, output_dir: Path):
    if not summary:
        return

    labels = [label(row) for row in summary]
    asr = [row.get("asr", 0) for row in summary]
    rr = [row.get("rr", 0) for row in summary]
    sui = [row.get("sui", 0) for row in summary]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))
    ax.bar(x - width, asr, width, label="ASR")
    ax.bar(x, rr, width, label="RR")
    ax.bar(x + width, sui, width, label="SUI")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Attack Safety Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "attack_metrics.png", dpi=200)
    plt.close(fig)


def plot_benign_metrics(summary, output_dir: Path):
    if not summary:
        return

    labels = [label(row) for row in summary]
    over_refusal = [row.get("over_refusal_rate", 0) for row in summary]
    benign_success = [row.get("benign_success_rate", 0) for row in summary]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))
    ax.bar(x - width / 2, over_refusal, width, label="Over-refusal")
    ax.bar(x + width / 2, benign_success, width, label="Benign success")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Benign Utility Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "benign_utility.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot final project results")
    parser.add_argument("--benchmark", default=None, help="Path to final_benchmark JSON")
    parser.add_argument("--benign", default=None, help="Path to benign_utility JSON")
    parser.add_argument("--output-dir", default="results/figures/final")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        benchmark = load_json(args.benchmark)
        plot_attack_metrics(benchmark.get("summary", []), output_dir)
        print(f"Saved attack plot to: {output_dir / 'attack_metrics.png'}")

    if args.benign:
        benign = load_json(args.benign)
        plot_benign_metrics(benign.get("summary", []), output_dir)
        print(f"Saved benign plot to: {output_dir / 'benign_utility.png'}")


if __name__ == "__main__":
    main()
