"""
Weights & Biases (W&B) Experiment Logging Utility (Phase 7)

Provides helper functions to log training metrics to W&B for
centralized, federated, and DP-federated training experiments.

W&B creates interactive dashboards automatically from logged data,
making it easy to compare experiments across different configurations.

Setup:
    1. Create a free account at https://wandb.ai
    2. Run `wandb login` in your terminal and paste your API key
    3. Set use_wandb=True when calling init_wandb()

Usage:
    from src.utils.logger import init_wandb, log_epoch, log_round, finish_wandb
"""

import wandb


def init_wandb(config, project_name="federated-learning-privacy", run_name=None, tags=None):
    """
    Initialize a new W&B run for experiment tracking.

    Args:
        config (dict): Full project config dict (from default.yaml).
                       This gets saved with the run so you can reproduce it.
        project_name (str): W&B project name (groups related experiments).
        run_name (str, optional): Human-readable name for this run.
                                   Auto-generated if None.
        tags (list[str], optional): Tags to categorize the run
                                     (e.g., ["centralized", "baseline"]).

    Returns:
        wandb.Run: The initialized W&B run object.
    """
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags or [],
        reinit=True,
    )

    print(f"  [W&B] Run initialized: {run.name}")
    print(f"  [W&B] Dashboard: {run.get_url()}")

    return run


def log_epoch(epoch, train_loss, train_acc, test_loss, test_acc):
    """
    Log metrics for one epoch of centralized training.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Average training loss.
        train_acc (float): Training accuracy (%).
        test_loss (float): Average test loss.
        test_acc (float): Test accuracy (%).
    """
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "test/loss": test_loss,
        "test/accuracy": test_acc,
    }, step=epoch)


def log_round(round_num, avg_train_loss, avg_train_acc, test_loss, test_acc,
              epsilon=None, selected_clients=None):
    """
    Log metrics for one round of federated training.

    Args:
        round_num (int): Current FL round number.
        avg_train_loss (float): Average training loss across selected clients.
        avg_train_acc (float): Average training accuracy across selected clients.
        test_loss (float): Global model test loss.
        test_acc (float): Global model test accuracy (%).
        epsilon (float, optional): Privacy budget spent (if DP enabled).
        selected_clients (list[int], optional): IDs of clients selected this round.
    """
    metrics = {
        "round": round_num,
        "train/avg_loss": avg_train_loss,
        "train/avg_accuracy": avg_train_acc,
        "test/loss": test_loss,
        "test/accuracy": test_acc,
    }

    if epsilon is not None:
        metrics["privacy/epsilon"] = epsilon

    if selected_clients is not None:
        metrics["clients/num_selected"] = len(selected_clients)

    wandb.log(metrics, step=round_num)


def log_summary(results, mode="centralized"):
    """
    Log final summary metrics for the experiment.

    These appear in the W&B run summary table, making it easy
    to compare final results across different experiments.

    Args:
        results (dict): Results dictionary from training.
        mode (str): "centralized" or "federated".
    """
    if mode == "centralized":
        wandb.run.summary["final_test_accuracy"] = results.get("final_test_accuracy", 0)
        wandb.run.summary["final_train_accuracy"] = results["train_accuracies"][-1] if results["train_accuracies"] else 0
        wandb.run.summary["final_test_loss"] = results["test_losses"][-1] if results["test_losses"] else 0
        wandb.run.summary["total_epochs"] = len(results["train_losses"])
    elif mode == "federated":
        wandb.run.summary["final_test_accuracy"] = results["test_accuracy"][-1] if results["test_accuracy"] else 0
        wandb.run.summary["best_test_accuracy"] = max(results["test_accuracy"]) if results["test_accuracy"] else 0
        wandb.run.summary["total_rounds"] = len(results["round"])


def finish_wandb():
    """
    Finish the current W&B run.

    Call this at the end of training to ensure all data is synced.
    """
    wandb.finish()
    print("  [W&B] Run finished and synced.")


# ── Quick test: run this file to verify W&B connection ──
if __name__ == "__main__":
    print("Testing W&B connection...\n")

    # Minimal test config
    test_config = {
        "model": {"name": "SimpleCNN", "num_classes": 10},
        "training": {"epochs": 2, "batch_size": 32, "learning_rate": 0.01},
        "test": True,
    }

    run = init_wandb(
        config=test_config,
        run_name="connection-test",
        tags=["test"],
    )

    # Log some dummy data
    for epoch in range(1, 3):
        log_epoch(epoch, train_loss=1.5/epoch, train_acc=30*epoch,
                  test_loss=1.4/epoch, test_acc=32*epoch)

    finish_wandb()
    print("\nW&B connection test complete! Check your dashboard.")
