"""
Dataset Loader
Downloads and prepares PathMNIST (colorectal cancer pathology) dataset.

PathMNIST: 9-class colorectal cancer histology classification
- 89,996 training images (28x28 RGB, resized to 32x32)
- 10,004 validation images (merged into train for FL)
- 7,180 test images
- Classes: tissue types found in colorectal cancer biopsies

Source: MedMNIST v2 (Yang et al., 2023)
"""

import numpy as np
import torchvision.transforms as transforms
from medmnist import PathMNIST


def get_pathmnist(data_dir="./data"):
    """
    Download and load the PathMNIST dataset with standard normalization.

    Images are 28x28 RGB, resized to 32x32 to match our SimpleCNN
    architecture (designed for 32x32 input).

    Args:
        data_dir (str): Directory to download/store the dataset.
                        Defaults to './data'.

    Returns:
        tuple: (train_dataset, test_dataset)
            - train_dataset: ~90,000 training images (32x32x3)
            - test_dataset:  ~7,180 test images (32x32x3)
    """

    # -- Define the preprocessing pipeline --
    # 1. Resize(32)  -> Scale 28x28 up to 32x32 (matches our CNN input)
    # 2. ToTensor()  -> Converts PIL image (0-255) to tensor (0.0-1.0)
    # 3. Normalize() -> Shifts each RGB channel to mean=0, std=1
    #                   using ImageNet-style normalization (standard for medical)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # -- Download and load training set (~90,000 images) --
    train_dataset = PathMNIST(
        root=data_dir,
        split="train",
        download=True,
        transform=transform,
    )

    # -- Download and load test set (~7,180 images) --
    test_dataset = PathMNIST(
        root=data_dir,
        split="test",
        download=True,
        transform=transform,
    )

    # MedMNIST stores labels as shape (N, 1) numpy array.
    # We need a flat .targets attribute for partition.py compatibility.
    train_dataset.targets = train_dataset.labels.flatten().tolist()
    test_dataset.targets = test_dataset.labels.flatten().tolist()

    return train_dataset, test_dataset


# -- PathMNIST class names (9 colorectal cancer tissue types) --
PATHMNIST_CLASSES = [
    "Adipose",
    "Background",
    "Debris",
    "Lymphocytes",
    "Mucus",
    "Smooth Muscle",
    "Normal Colon Mucosa",
    "Cancer-Associated Stroma",
    "Colorectal Adenocarcinoma Epithelium",
]

# Keep backward-compatible alias
DATASET_CLASSES = PATHMNIST_CLASSES


# -- Quick test: run this file directly to verify it works --
if __name__ == "__main__":
    train_data, test_data = get_pathmnist()

    print("[OK] PathMNIST loaded successfully!\n")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples:     {len(test_data)}")
    print(f"  Image shape:      {train_data[0][0].shape}")  # (3, 32, 32)
    print(f"  Number of classes: {len(PATHMNIST_CLASSES)}")
    print(f"  Classes: {PATHMNIST_CLASSES}")

    # Show a few sample labels
    # MedMNIST returns labels as numpy arrays, so we convert to int
    first_10 = [train_data[i][1].item() for i in range(10)]
    print(f"\n  First 10 labels: {first_10}")
    print(f"  Mapped to: {[PATHMNIST_CLASSES[l] for l in first_10]}")
