"""
Federated Learning Client (Phase 5)

Each client holds a local partition of the dataset and trains
the SimpleCNN model locally. The client communicates model weights
(not data) with the server for aggregation.

Implements the same interface as Flower's NumPyClient but without
the Ray dependency (Ray crashes on Windows with access violations).
The FL simulation loop in server.py calls these methods directly.

Usage:
    This file is imported by server.py -- not run directly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.cnn import SimpleCNN
from src.training.train import train_one_epoch, evaluate


class FederatedClient:
    """
    A federated learning client (Flower-style interface).

    Each client:
    1. Receives global model weights from the server
    2. Trains locally on its own data partition
    3. Sends updated weights back to the server

    Args:
        client_id (int): Unique identifier for this client
        train_dataset: Client's local training data (a Subset)
        test_dataset: Shared test set for evaluation
        config (dict): Training configuration from default.yaml
    """

    def __init__(self, client_id, train_dataset, test_dataset, config):
        self.client_id = client_id
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Create DataLoaders for this client's local data --
        batch_size = config["training"]["batch_size"]

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # -- Initialize local model --
        num_classes = config["model"]["num_classes"]
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.num_samples = len(train_dataset)

        print(f"  [CLIENT] Client {client_id} initialized with {self.num_samples} samples")

    def get_parameters(self):
        """
        Extract model weights as a list of tensors.

        Returns:
            list[torch.Tensor]: Model parameters as a list
        """
        return [param.data.clone() for param in self.model.parameters()]

    def set_parameters(self, global_parameters):
        """
        Replace local model weights with the global model weights.

        Args:
            global_parameters (list[torch.Tensor]): Global model parameters
        """
        for local_param, global_param in zip(
            self.model.parameters(), global_parameters
        ):
            local_param.data = global_param.clone().to(self.device)

    def fit(self, global_parameters):
        """
        Local training on this client's data partition.

        Steps:
        1. Load global weights
        2. Train locally for configured epochs
        3. Return updated weights + metrics

        This mirrors Flower's NumPyClient.fit() interface.

        Args:
            global_parameters (list[torch.Tensor]): Current global model weights

        Returns:
            tuple: (updated_parameters, num_samples, metrics_dict)
        """
        # Step 1: Load the global model weights
        self.set_parameters(global_parameters)

        # Step 2: Set up optimizer for local training
        lr = self.config["training"]["learning_rate"]
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
        )

        # Step 3: Train locally for configured number of epochs
        local_epochs = self.config["training"]["epochs"]
        for epoch in range(local_epochs):
            train_loss, train_acc = train_one_epoch(
                self.model, self.train_loader, optimizer,
                self.criterion, self.device
            )

        # Step 4: Return updated weights and metrics
        return (
            self.get_parameters(),
            self.num_samples,
            {"train_loss": train_loss, "train_accuracy": train_acc},
        )

    def evaluate(self, global_parameters):
        """
        Evaluate the global model on this client's test data.

        This mirrors Flower's NumPyClient.evaluate() interface.

        Args:
            global_parameters (list[torch.Tensor]): Current global model weights

        Returns:
            tuple: (loss, num_samples, metrics_dict)
        """
        self.set_parameters(global_parameters)

        loss, accuracy = evaluate(
            self.model, self.test_loader, self.criterion, self.device
        )

        return (
            float(loss),
            self.num_samples,
            {"test_accuracy": accuracy},
        )
