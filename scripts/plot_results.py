"""
Phase 10: Result Visualization

Generates comparison plots from experiment results saved by run_experiments.py.

Plots generated:
    1. Accuracy comparison bar chart (all 5 experiments)
    2. Federated training curves (accuracy over rounds)
    3. Privacy-utility tradeoff chart

Usage:
    python scripts/plot_results.py

Requires: experiments/*.json files from run_experiments.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs


EXPERIMENTS_DIR = "experiments"


def load_json(filename):
    """Load a JSON result file."""
    filepath = os.path.join(EXPERIMENTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def plot_accuracy_comparison():
    """
    Bar chart comparing final test accuracy across all experiments.
    This is the KEY chart for the project report.
    """
    summary = load_json("summary.json")
    if summary is None:
        print("  [ERROR] Run run_experiments.py first!")
        return

    names = {
        "centralized": "Centralized\nBaseline",
        "fl_iid_no_dp": "FL IID\n(no DP)",
        "fl_iid_dp": "FL IID\n+ DP",
        "fl_noniid_no_dp": "FL Non-IID\n(no DP)",
        "fl_noniid_dp": "FL Non-IID\n+ DP",
    }

    colors = ["#2ECC71", "#3498DB", "#9B59B6", "#E67E22", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = []
    accuracies = []
    bar_colors = []

    for i, (key, label) in enumerate(names.items()):
        if key in summary:
            x_labels.append(label)
            accuracies.append(summary[key]["accuracy"])
            bar_colors.append(colors[i])

    bars = ax.bar(x_labels, accuracies, color=bar_colors, width=0.6,
                  edgecolor="white", linewidth=2)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold",
                fontsize=13)

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff: Accuracy Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(EXPERIMENTS_DIR, "comparison_chart.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {filepath}")


def plot_training_curves():
    """
    Line chart showing accuracy over rounds for all federated experiments.
    """
    experiments = [
        ("fl_iid_no_dp.json", "FL IID (no DP)", "#3498DB", "-"),
        ("fl_iid_dp.json", "FL IID + DP", "#9B59B6", "--"),
        ("fl_noniid_no_dp.json", "FL Non-IID (no DP)", "#E67E22", "-"),
        ("fl_noniid_dp.json", "FL Non-IID + DP", "#E74C3C", "--"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for filename, label, color, linestyle in experiments:
        data = load_json(filename)
        if data is None:
            continue
        rounds = data["round"]
        test_acc = data["test_accuracy"]
        ax.plot(rounds, test_acc, linestyle, color=color, linewidth=2,
                marker="o", markersize=5, label=label)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Federated Training Curves (PathMNIST)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(EXPERIMENTS_DIR, "training_curves.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {filepath}")


def plot_privacy_tradeoff():
    """
    Chart showing accuracy vs privacy setting.
    Demonstrates the core finding: more privacy = less accuracy.
    """
    # Pairs: (label, accuracy, privacy_level_label)
    data_points = []

    summary = load_json("summary.json")
    if summary is None:
        return

    settings = [
        ("centralized", "No Privacy\n(Centralized)", 0),
        ("fl_iid_no_dp", "Data Local\n(FL, no DP)", 1),
        ("fl_iid_dp", "DP-SGD\n(FL + DP)", 2),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = []
    accs = []
    labels = []
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    for key, label, pos in settings:
        if key in summary:
            x_pos.append(pos)
            accs.append(summary[key]["accuracy"])
            labels.append(label)

    bars = ax.bar(labels, accs, color=colors, width=0.5,
                  edgecolor="white", linewidth=2)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold",
                fontsize=13)

    # Add arrow showing tradeoff direction
    ax.annotate("", xy=(2.3, accs[-1] - 5), xytext=(2.3, accs[0] + 5),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=2))
    if len(accs) >= 2:
        gap = accs[0] - accs[-1]
        mid_y = (accs[0] + accs[-1]) / 2
        ax.text(2.45, mid_y, f"Privacy\ncosts\n{gap:.1f}%",
                fontsize=10, color="gray", ha="left", va="center")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(EXPERIMENTS_DIR, "privacy_tradeoff.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {filepath}")


def main():
    print(f"\n{'='*60}")
    print(f"  GENERATING PLOTS")
    print(f"{'='*60}\n")

    plot_accuracy_comparison()
    plot_training_curves()
    plot_privacy_tradeoff()

    print(f"\n{'='*60}")
    print(f"  ALL PLOTS SAVED TO {EXPERIMENTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
