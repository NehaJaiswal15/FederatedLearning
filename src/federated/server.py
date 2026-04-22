"""
Federated Learning Server with FedAvg (Phase 5)

Orchestrates the federated training process:
1. Initializes global model weights
2. Each round: selects clients, they train locally, server aggregates
3. Aggregation uses Federated Averaging (FedAvg) -- weighted average by dataset size

Implements the same FedAvg algorithm used in Flower, but runs the simulation
loop directly (without Ray) for Windows compatibility. The algorithm is
identical to the original Google paper:
"Communication-Efficient Learning of Deep Networks from Decentralized Data"
(McMahan et al., 2017)

Usage:
    python -m src.federated.server
"""

import torch
import torch.nn as nn
import random

from src.models.cnn import SimpleCNN
from src.utils.config_loader import load_config
from src.datasets_partition.dataset import get_cifar10
from src.datasets_partition.partition import partition_data
from src.training.train import evaluate
from src.federated.client import FederatedClient
from torch.utils.data import DataLoader


def federated_average(client_results):
    """
    Federated Averaging (FedAvg) -- the core aggregation algorithm.

    Computes a weighted average of client model parameters,
    where each client's contribution is weighted by the number
    of training samples it has.

    This is the exact algorithm from McMahan et al., 2017 --
    the same one Flower uses internally in its FedAvg strategy.

    Args:
        client_results: list of (parameters, num_samples, metrics) tuples

    Returns:
        list[torch.Tensor]: Aggregated global model parameters
    """
    # Step 1: Calculate total samples across all participating clients
    total_samples = sum(num_samples for _, num_samples, _ in client_results)

    # Step 2: Initialize aggregated parameters with zeros
    first_params = client_results[0][0]
    aggregated = [torch.zeros_like(param) for param in first_params]

    # Step 3: Weighted sum -- each client contributes proportionally
    for client_params, num_samples, _ in client_results:
        weight = num_samples / total_samples
        for i, param in enumerate(client_params):
            aggregated[i] += param * weight

    return aggregated


def run_federated_training(config=None, use_wandb=False):
    """
    Execute federated training with FedAvg simulation.

    Args:
        config (dict, optional): Configuration dict. Loads default if None.
        use_wandb (bool): If True, log metrics to Weights & Biases.

    Returns:
        dict: Training history with per-round metrics
    """
    # -- Step 1: Load configuration --
    if config is None:
        config = load_config()

    # -- Optional: Initialize W&B --
    if use_wandb:
        from src.utils.logger import init_wandb, log_round, log_summary, finish_wandb
        dp_tag = "dp-enabled" if config["privacy"]["enable_dp"] else "no-dp"
        iid_tag = "iid" if config["data"]["iid"] else "non-iid"
        init_wandb(
            config=config,
            run_name=f"federated-{iid_tag}-{dp_tag}",
            tags=["federated", iid_tag, dp_tag],
        )

    fed_cfg = config["federated"]
    data_cfg = config["data"]

    num_clients = fed_cfg["num_clients"]
    num_rounds = fed_cfg["num_rounds"]
    fraction_fit = fed_cfg["fraction_fit"]
    num_selected = max(1, int(num_clients * fraction_fit))

    enable_dp = config["privacy"]["enable_dp"]

    print(f"\n{'='*60}")
    print(f"  FEDERATED TRAINING (FedAvg)")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds} | "
          f"Selected/Round: {num_selected}")
    print(f"  Data Distribution: {'IID' if data_cfg['iid'] else 'Non-IID'}")
    print(f"  Differential Privacy: {'ENABLED' if enable_dp else 'DISABLED'}")
    print(f"{'='*60}\n")

    # -- Step 2: Load and partition the dataset --
    print("Loading CIFAR-10 dataset...")
    train_dataset, test_dataset = get_cifar10(data_dir=data_cfg["data_dir"])

    print(f"Partitioning data across {num_clients} clients...")
    client_datasets = partition_data(
        train_dataset,
        num_clients=num_clients,
        iid=data_cfg["iid"],
    )

    # -- Step 3: Create all clients --
    print(f"\nInitializing {num_clients} clients...")
    clients = []
    for i in range(num_clients):
        client = FederatedClient(
            client_id=i,
            train_dataset=client_datasets[i],
            test_dataset=test_dataset,
            config=config,
        )
        clients.append(client)

    # -- Step 4: Initialize global model --
    global_model = SimpleCNN(num_classes=config["model"]["num_classes"])
    global_parameters = [param.data.clone() for param in global_model.parameters()]

    # -- Step 5: Create a test DataLoader for global evaluation --
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # -- Step 6: Federated training loop --
    history = {
        "round": [],
        "test_loss": [],
        "test_accuracy": [],
        "train_losses": [],
        "train_accuracies": [],
    }

    print(f"\n{'-'*60}")
    print(f"  Starting federated training...")
    print(f"{'-'*60}\n")

    for round_num in range(1, num_rounds + 1):
        # 6a: Select a random subset of clients (like Flower's fraction_fit)
        selected_ids = random.sample(range(num_clients), num_selected)

        # 6b: Each selected client trains locally with current global weights
        client_results = []
        round_train_losses = []
        round_train_accs = []

        for cid in selected_ids:
            params, n_samples, metrics = clients[cid].fit(global_parameters)
            client_results.append((params, n_samples, metrics))
            round_train_losses.append(metrics["train_loss"])
            round_train_accs.append(metrics["train_accuracy"])

        # 6c: Aggregate using FedAvg (same algorithm as Flower's FedAvg strategy)
        global_parameters = federated_average(client_results)

        # 6d: Evaluate global model on test set
        for local_param, global_param in zip(
            global_model.parameters(), global_parameters
        ):
            local_param.data = global_param.clone()
        global_model.to(device)

        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)

        # 6e: Record metrics
        avg_train_loss = sum(round_train_losses) / len(round_train_losses)
        avg_train_acc = sum(round_train_accs) / len(round_train_accs)

        # Track epsilon if DP is enabled
        avg_epsilon = None
        if enable_dp:
            epsilons = [m.get("epsilon", 0) for _, _, m in client_results]
            avg_epsilon = sum(epsilons) / len(epsilons) if epsilons else 0

        history["round"].append(round_num)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        history["train_losses"].append(avg_train_loss)
        history["train_accuracies"].append(avg_train_acc)

        # Build round summary
        summary = (
            f"  Round [{round_num:>2}/{num_rounds}]  "
            f"Clients: {selected_ids}  |  "
            f"Avg Train Acc: {avg_train_acc:.2f}%  |  "
            f"Test Acc: {test_acc:.2f}%"
        )
        if avg_epsilon is not None:
            summary += f"  |  Epsilon: {avg_epsilon:.2f}"
        print(summary)

        # Log to W&B
        if use_wandb:
            log_round(
                round_num=round_num,
                avg_train_loss=avg_train_loss,
                avg_train_acc=avg_train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                epsilon=avg_epsilon,
                selected_clients=selected_ids,
            )

    # -- Step 7: Final summary --
    final_acc = history["test_accuracy"][-1]
    best_acc = max(history["test_accuracy"])

    print(f"\n{'='*60}")
    print(f"  FEDERATED TRAINING COMPLETE")
    print(f"  Final Test Accuracy: {final_acc:.2f}%")
    print(f"  Best Test Accuracy:  {best_acc:.2f}%")
    if enable_dp:
        print(f"  Privacy: DP-SGD enabled (noise={config['privacy']['noise_multiplier']}, "
              f"clip={config['privacy']['max_grad_norm']})")
    print(f"{'='*60}\n")

    # Log summary and finish W&B
    if use_wandb:
        log_summary(history, mode="federated")
        finish_wandb()

    return history


# -- Run federated training directly --
if __name__ == "__main__":
    run_federated_training()
