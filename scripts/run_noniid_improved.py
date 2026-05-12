"""
Improved Non-IID Experiments

Runs Non-IID experiments with better settings:
- 4 classes per client (instead of 2) — less extreme heterogeneity
- 20 rounds (instead of 10) — more time to converge

Usage:
    python scripts/run_noniid_improved.py
"""

import json
import os
import copy
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config_loader import load_config
from src.federated.server import run_federated_training


OUTPUT_DIR = "experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_results(results, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [SAVED] {filepath}\n")


def run_experiment(name, filename, config):
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'#'*60}\n")

    start = time.time()
    results = run_federated_training(config=config)
    elapsed = time.time() - start

    results["experiment_name"] = name
    results["elapsed_seconds"] = round(elapsed, 1)

    save_results(results, filename)
    return results


def main():
    base_config = load_config()
    all_results = {}

    # ================================================================
    # Experiment: Non-IID Improved, No DP
    # 4 classes per client, 20 rounds, fraction_fit=0.8
    # ================================================================
    config = copy.deepcopy(base_config)
    config["data"]["iid"] = False
    config["data"]["classes_per_client"] = 4
    config["federated"]["num_rounds"] = 20
    config["federated"]["fraction_fit"] = 0.8
    config["privacy"]["enable_dp"] = False

    results = run_experiment(
        name="FL Non-IID Improved (no DP)",
        filename="fl_noniid_improved_no_dp.json",
        config=config,
    )
    all_results["fl_noniid_improved_no_dp"] = {
        "accuracy": results["test_accuracy"][-1],
        "time": results["elapsed_seconds"],
    }

    # ================================================================
    # Experiment: Non-IID Improved, With DP
    # 4 classes per client, 20 rounds, fraction_fit=0.8, DP enabled
    # ================================================================
    config = copy.deepcopy(base_config)
    config["data"]["iid"] = False
    config["data"]["classes_per_client"] = 4
    config["federated"]["num_rounds"] = 20
    config["federated"]["fraction_fit"] = 0.8
    config["privacy"]["enable_dp"] = True
    config["privacy"]["noise_multiplier"] = 0.5

    results = run_experiment(
        name="FL Non-IID Improved + DP (noise=0.5)",
        filename="fl_noniid_improved_dp.json",
        config=config,
    )
    all_results["fl_noniid_improved_dp"] = {
        "accuracy": results["test_accuracy"][-1],
        "time": results["elapsed_seconds"],
    }

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  IMPROVED NON-IID EXPERIMENTS COMPLETE")
    print(f"{'='*60}\n")

    # Load previous summary and merge
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {}

    summary.update(all_results)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [UPDATED] {summary_path}")

    # Print full comparison
    print(f"\n  {'Experiment':<40} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Non-IID (2 cls, 10 rds) - OLD':<40} {'17.34%':>10}")
    print(f"  {'Non-IID (4 cls, 20 rds) - NEW':<40} {all_results['fl_noniid_improved_no_dp']['accuracy']:>9.2f}%")
    print(f"  {'Non-IID + DP (2 cls, 10 rds) - OLD':<40} {'23.33%':>10}")
    print(f"  {'Non-IID + DP (4 cls, 20 rds) - NEW':<40} {all_results['fl_noniid_improved_dp']['accuracy']:>9.2f}%")
    print()


if __name__ == "__main__":
    main()
