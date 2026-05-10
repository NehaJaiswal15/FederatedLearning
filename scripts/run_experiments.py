"""
Phase 10: Automated Experiment Runner

Runs all experiments sequentially and saves results as JSON files
to the experiments/ directory.

Experiments:
    1. Centralized baseline (all data, no federation)
    2. Federated IID, no DP
    3. Federated IID, with DP (noise=0.5)
    4. Federated Non-IID, no DP
    5. Federated Non-IID, with DP (noise=0.5)

Usage:
    conda activate fedLearn
    python scripts/run_experiments.py

Total runtime: ~30-75 minutes on CPU
"""

import json
import os
import copy
import time
import sys

# Add project root to path so 'src' imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config_loader import load_config
from src.training.train import run_centralized_training
from src.federated.server import run_federated_training


# -- Output directory --
OUTPUT_DIR = "experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_results(results, filename):
    """Save results dict as JSON."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [SAVED] {filepath}\n")


def run_experiment(name, filename, train_fn, config):
    """Run a single experiment with timing."""
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'#'*60}\n")

    start = time.time()
    results = train_fn(config=config)
    elapsed = time.time() - start

    results["experiment_name"] = name
    results["elapsed_seconds"] = round(elapsed, 1)

    save_results(results, filename)
    return results


def main():
    base_config = load_config()

    all_results = {}

    # ================================================================
    # Experiment 1: Centralized Baseline
    # All 90k images trained in one place (no federation)
    # ================================================================
    config = copy.deepcopy(base_config)
    config["training"]["epochs"] = 10  # More epochs for baseline

    results = run_experiment(
        name="Centralized Baseline",
        filename="centralized_baseline.json",
        train_fn=run_centralized_training,
        config=config,
    )
    all_results["centralized"] = {
        "accuracy": results["final_test_accuracy"],
        "time": results["elapsed_seconds"],
    }

    # ================================================================
    # Experiment 2: Federated IID, No DP
    # 5 clients, each has all 9 classes, no privacy noise
    # ================================================================
    config = copy.deepcopy(base_config)
    config["data"]["iid"] = True
    config["privacy"]["enable_dp"] = False

    results = run_experiment(
        name="Federated IID (no DP)",
        filename="fl_iid_no_dp.json",
        train_fn=run_federated_training,
        config=config,
    )
    all_results["fl_iid_no_dp"] = {
        "accuracy": results["test_accuracy"][-1],
        "time": results["elapsed_seconds"],
    }

    # ================================================================
    # Experiment 3: Federated IID, With DP (noise=0.5)
    # Same as above but with DP-SGD privacy protection
    # ================================================================
    config = copy.deepcopy(base_config)
    config["data"]["iid"] = True
    config["privacy"]["enable_dp"] = True
    config["privacy"]["noise_multiplier"] = 0.5

    results = run_experiment(
        name="Federated IID + DP (noise=0.5)",
        filename="fl_iid_dp.json",
        train_fn=run_federated_training,
        config=config,
    )
    all_results["fl_iid_dp"] = {
        "accuracy": results["test_accuracy"][-1],
        "time": results["elapsed_seconds"],
    }

    # ================================================================
    # Experiment 4: Federated Non-IID, No DP
    # Each client has only 2 of 9 classes (realistic heterogeneity)
    # ================================================================
    config = copy.deepcopy(base_config)
    config["data"]["iid"] = False
    config["privacy"]["enable_dp"] = False

    results = run_experiment(
        name="Federated Non-IID (no DP)",
        filename="fl_noniid_no_dp.json",
        train_fn=run_federated_training,
        config=config,
    )
    all_results["fl_noniid_no_dp"] = {
        "accuracy": results["test_accuracy"][-1],
        "time": results["elapsed_seconds"],
    }

    # ================================================================
    # Experiment 5: Federated Non-IID, With DP (noise=0.5)
    # Hardest setting: skewed data + privacy noise
    # ================================================================
    config = copy.deepcopy(base_config)
    config["data"]["iid"] = False
    config["privacy"]["enable_dp"] = True
    config["privacy"]["noise_multiplier"] = 0.5

    results = run_experiment(
        name="Federated Non-IID + DP (noise=0.5)",
        filename="fl_noniid_dp.json",
        train_fn=run_federated_training,
        config=config,
    )
    all_results["fl_noniid_dp"] = {
        "accuracy": results["test_accuracy"][-1],
        "time": results["elapsed_seconds"],
    }

    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}\n")
    print(f"  {'Experiment':<35} {'Accuracy':>10} {'Time':>10}")
    print(f"  {'-'*55}")
    for key, val in all_results.items():
        print(f"  {key:<35} {val['accuracy']:>9.2f}% {val['time']:>8.1f}s")
    print()

    # Save summary
    save_results(all_results, "summary.json")


if __name__ == "__main__":
    main()
