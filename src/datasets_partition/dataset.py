"""
Dataset Loader
Downloads and prepares CIFAR-10 with standard preprocessing transforms.
"""

import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_cifar10(data_dir="./data"):
    """
    Download and load the CIFAR-10 dataset with standard normalization.

    Args:
        data_dir (str): Directory to download/store the dataset.
                        Defaults to './data'.

    Returns:
        tuple: (train_dataset, test_dataset)
            - train_dataset: 50,000 training images (32x32x3)
            - test_dataset:  10,000 test images (32x32x3)
    """

    # ── Define the preprocessing pipeline ──
    # These transforms are applied to every image when it's loaded:
    #   1. ToTensor()   → Converts PIL image (0-255) to tensor (0.0-1.0)
    #   2. Normalize()  → Shifts each RGB channel to mean=0, std=1
    #                     using precomputed CIFAR-10 statistics
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 per-channel means
            std=(0.2023, 0.1994, 0.2010)      # CIFAR-10 per-channel stds
        )
    ])

    # ── Download and load training set (50,000 images) ──
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # ── Download and load test set (10,000 images) ──
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


# ── CIFAR-10 class names for reference ──
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ── Quick test: run this file directly to verify it works ──
if __name__ == "__main__":
    train_data, test_data = get_cifar10()

    print("[OK] CIFAR-10 loaded successfully!\n")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples:     {len(test_data)}")
    print(f"  Image shape:      {train_data[0][0].shape}")  # (3, 32, 32)
    print(f"  Number of classes: {len(CIFAR10_CLASSES)}")
    print(f"  Classes: {CIFAR10_CLASSES}")

    # Show a few sample labels
    print("\n  First 10 labels:", [train_data[i][1] for i in range(10)])
    print("  Mapped to:",       [CIFAR10_CLASSES[train_data[i][1]] for i in range(10)])
