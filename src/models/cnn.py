"""
SimpleCNN Model for CIFAR-10 Image Classification

A lightweight CNN designed for federated learning with:
- GroupNorm instead of BatchNorm (required for Opacus/DP compatibility)
- No inplace operations (required for Opacus)
- ~590K parameters (small enough for fast FL communication)
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A simple CNN for 32x32 RGB image classification.

    Architecture:
        [Conv → GroupNorm → ReLU] × 2 → MaxPool
        [Conv → GroupNorm → ReLU] × 2 → MaxPool
        Flatten → Linear → ReLU → Dropout → Linear (output)

    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # ── Feature Extraction Block 1 ──
        # Input: (batch, 3, 32, 32) → Output: (batch, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # 3→32 channels, keeps 32x32
            nn.GroupNorm(8, 32),                              # 8 groups over 32 channels
            nn.ReLU(),                                        # Activation (NOT inplace)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # 32→32 channels, keeps 32x32
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                     # 32x32 → 16x16
        )

        # ── Feature Extraction Block 2 ──
        # Input: (batch, 32, 16, 16) → Output: (batch, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 32→64 channels, keeps 16x16
            nn.GroupNorm(8, 64),                              # 8 groups over 64 channels
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # 64→64 channels, keeps 16x16
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                     # 16x16 → 8x8
        )

        # ── Classifier Head ──
        # Input: (batch, 64*8*8=4096) → Output: (batch, num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                     # (batch, 64, 8, 8) → (batch, 4096)
            nn.Linear(64 * 8 * 8, 256),                      # 4096 → 256
            nn.ReLU(),
            nn.Dropout(0.5),                                  # Prevents overfitting
            nn.Linear(256, num_classes),                      # 256 → 10 (raw logits, no softmax)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, 32, 32)

        Returns:
            torch.Tensor: Raw class scores of shape (batch, num_classes)
        """
        x = self.block1(x)       # Extract low-level features (edges, textures)
        x = self.block2(x)       # Extract high-level features (shapes, patterns)
        x = self.classifier(x)   # Map features to class scores
        return x


# ── Quick test: run this file directly to verify it works ──
if __name__ == "__main__":
    # Create the model
    model = SimpleCNN(num_classes=10)

    # Create a dummy input: 1 image, 3 RGB channels, 32x32 pixels
    dummy_input = torch.randn(1, 3, 32, 32)

    # Forward pass
    output = model(dummy_input)

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✅ SimpleCNN created successfully!\n")
    print(f"  Input shape:         {dummy_input.shape}")
    print(f"  Output shape:        {output.shape}")
    print(f"  Total parameters:    {total_params:,}")
    print(f"  Trainable parameters:{trainable_params:,}")
    print(f"\n── Model Architecture ──\n")
    print(model)
