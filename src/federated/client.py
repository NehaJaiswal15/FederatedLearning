"""
Federated Learning Client with Differential Privacy (Phase 5 + 6)

Each client holds a local partition of the dataset and trains
the SimpleCNN model locally. The client communicates model weights
(not data) with the server for aggregation.

When privacy.enable_dp is True in config, the client applies DP-SGD
(Opacus) during local training:
- Gradients are clipped per-sample (max_grad_norm)
- Calibrated noise is added to gradients (noise_multiplier)
- Privacy budget (epsilon) is tracked per round

Usage:
    This file is imported by server.py -- not run directly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.cnn import SimpleCNN
from src.training.train import train_one_epoch, evaluate
from src.privacy.dp_utils import make_private, get_epsilon


class FederatedClient:
    """
    A federated learning client with optional differential privacy.

    Each client:
    1. Receives global model weights from the server
    2. Trains locally on its own data partition (with or without DP)
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
        self.enable_dp = config["privacy"]["enable_dp"]

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

        dp_status = "DP-SGD ENABLED" if self.enable_dp else "no DP"
        print(f"  [CLIENT] Client {client_id} initialized with {self.num_samples} samples ({dp_status})")

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

        When DP is enabled:
        - Model, optimizer, and dataloader are wrapped by Opacus
        - Each gradient is clipped and noised before the weight update
        - Epsilon (privacy budget) is tracked and returned in metrics

        Args:
            global_parameters (list[torch.Tensor]): Current global model weights

        Returns:
            tuple: (updated_parameters, num_samples, metrics_dict)
        """
        # Step 1: Create a fresh model and load global weights
        # (When DP is enabled, we need a clean model each round
        #  because Opacus adds hooks that can't be re-added)
        if self.enable_dp:
            num_classes = self.config["model"]["num_classes"]
            model = SimpleCNN(num_classes=num_classes).to(self.device)
            # Load global weights into fresh model
            for local_param, global_param in zip(model.parameters(), global_parameters):
                local_param.data = global_param.clone().to(self.device)
        else:
            model = self.model
            self.set_parameters(global_parameters)

        # Step 2: Set up optimizer for local training
        lr = self.config["training"]["learning_rate"]
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
        )

        # Step 3: Conditionally apply differential privacy
        privacy_engine = None
        train_loader = self.train_loader

        if self.enable_dp:
            model, optimizer, train_loader, privacy_engine = make_private(
                model, optimizer, train_loader, self.config
            )

        # Step 4: Train locally for configured number of epochs
        local_epochs = self.config["training"]["epochs"]
        for epoch in range(local_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer,
                self.criterion, self.device
            )

        # Step 5: Build metrics dict
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
        }

        # Step 6: Track privacy budget if DP is enabled
        if privacy_engine is not None:
            epsilon = get_epsilon(privacy_engine)
            metrics["epsilon"] = epsilon
            print(f"    [DP] Client {self.client_id}: epsilon = {epsilon:.2f}")

        # Step 7: Extract parameters from the trained model
        if self.enable_dp:
            # Unwrap Opacus GradSampleModule to get original model params
            params = [param.data.clone() for param in model._module.parameters()]
        else:
            params = self.get_parameters()

        return (params, self.num_samples, metrics)

    def evaluate(self, global_parameters):
        """
        Evaluate the global model on this client's test data.

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
