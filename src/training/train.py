"""
Centralized Training Baseline (Phase 4)

Standard (non-federated) training loop for CIFAR-10 using SimpleCNN.
This establishes a performance baseline to compare against federated training.

Expected accuracy: ~70-75% after 10 epochs on CPU.

Usage:
    python -m src.training.train
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config_loader import load_config
from src.datasets_partition.dataset import get_cifar10, CIFAR10_CLASSES
from src.models.cnn import SimpleCNN


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer (e.g., SGD).
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): Device to train on (cpu/cuda).

    Returns:
        tuple: (average_loss, accuracy) for the epoch.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # ── Forward pass ──
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # ── Backward pass + update ──
        loss.backward()
        optimizer.step()

        # ── Track metrics ──
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset (no gradient computation).

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Test/validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.

    Returns:
        tuple: (average_loss, accuracy) for the evaluation set.
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def run_centralized_training(config=None, use_wandb=False):
    """
    Execute centralized (non-federated) training on full CIFAR-10.

    Args:
        config (dict, optional): Configuration dictionary.
        use_wandb (bool): If True, log metrics to Weights & Biases.
    """
    # ── Step 1: Load configuration ──
    if config is None:
        config = load_config()

    # ── Optional: Initialize W&B ──
    if use_wandb:
        from src.utils.logger import init_wandb, log_epoch, log_summary, finish_wandb
        init_wandb(
            config=config,
            run_name="centralized-baseline",
            tags=["centralized", "baseline"],
        )

    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    learning_rate = training_cfg["learning_rate"]
    num_classes = model_cfg["num_classes"]

    # ── Step 2: Set device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 3: Load CIFAR-10 ──
    print("Loading CIFAR-10 dataset...")
    train_dataset, test_dataset = get_cifar10(data_dir=data_cfg["data_dir"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"  Training samples:  {len(train_dataset):,}")
    print(f"  Test samples:      {len(test_dataset):,}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Training batches:  {len(train_loader)}")

    # ── Step 4: Initialize model, loss, optimizer ──
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SimpleCNN ({total_params:,} parameters)")
    print(f"Optimizer: SGD (lr={learning_rate}, momentum=0.9)")
    print(f"Loss: CrossEntropyLoss")

    # ── Step 5: Training loop ──
    print(f"\n{'='*60}")
    print(f"  CENTRALIZED TRAINING - {epochs} epoch(s)")
    print(f"{'='*60}\n")

    results = {
        "train_losses": [],
        "train_accuracies": [],
        "test_losses": [],
        "test_accuracies": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        results["train_losses"].append(train_loss)
        results["train_accuracies"].append(train_acc)
        results["test_losses"].append(test_loss)
        results["test_accuracies"].append(test_acc)

        print(
            f"  Epoch [{epoch:>2}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
            f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%"
        )

        # Log to W&B
        if use_wandb:
            log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)

    # ── Step 6: Final summary ──
    final_test_acc = results["test_accuracies"][-1]
    results["final_test_accuracy"] = final_test_acc

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"{'='*60}\n")

    # Log summary and finish W&B
    if use_wandb:
        log_summary(results, mode="centralized")
        finish_wandb()

    return results


if __name__ == "__main__":
    _ = run_centralized_training()
