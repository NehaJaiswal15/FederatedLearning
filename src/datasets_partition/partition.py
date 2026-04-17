"""
Data Partitioning for Federated Learning
Splits a dataset into IID or Non-IID partitions across multiple clients.
"""

import numpy as np
from torch.utils.data import Subset


def iid_partition(dataset, num_clients):
    """
    Split dataset into IID (evenly distributed) partitions.
    
    Each client gets a random, equal-sized subset containing
    a balanced mix of all classes.

    Args:
        dataset: PyTorch dataset (e.g., CIFAR-10 training set)
        num_clients (int): Number of clients to split data across

    Returns:
        list[Subset]: One Subset per client, each with ~(total/num_clients) samples
    """
    # Step 1: Create a shuffled list of all indices [0, 1, 2, ..., 49999]
    total_samples = len(dataset)
    indices = np.random.permutation(total_samples)

    # Step 2: Split into num_clients equal chunks
    # np.array_split handles uneven division gracefully
    # e.g., 50000 / 5 = [10000, 10000, 10000, 10000, 10000]
    split_indices = np.array_split(indices, num_clients)

    # Step 3: Wrap each chunk as a Subset (a "view" into the original dataset)
    client_datasets = [
        Subset(dataset, indices.tolist()) for indices in split_indices
    ]

    return client_datasets


def non_iid_partition(dataset, num_clients, classes_per_client=2):
    """
    Split dataset into Non-IID (skewed) partitions.
    
    Each client receives data from only a limited number of classes,
    simulating real-world data heterogeneity (e.g., hospitals seeing 
    different types of diseases).

    Args:
        dataset: PyTorch dataset with a .targets attribute
        num_clients (int): Number of clients
        classes_per_client (int): How many classes each client gets (default: 2)

    Returns:
        list[Subset]: One Subset per client, each containing data
                      from only `classes_per_client` classes
    """
    # Step 1: Extract all labels from the dataset
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    # Step 2: Group indices by class
    # Result: {0: [indices of airplanes], 1: [indices of automobiles], ...}
    class_indices = {}
    for class_id in range(num_classes):
        class_indices[class_id] = np.where(labels == class_id)[0]

    # Step 3: Create shards — divide each class's indices into equal pieces
    # Total shards = num_clients * classes_per_client
    # e.g., 5 clients × 2 classes each = 10 shards (one per class works perfectly)
    total_shards = num_clients * classes_per_client
    shards_per_class = total_shards // num_classes

    all_shards = []
    for class_id in range(num_classes):
        # Split this class's indices into shards_per_class pieces
        class_shards = np.array_split(class_indices[class_id], shards_per_class)
        all_shards.extend(class_shards)

    # Step 4: Assign shards to clients
    # Each client gets `classes_per_client` consecutive shards
    client_datasets = []
    for client_id in range(num_clients):
        # Grab this client's shards
        start = client_id * classes_per_client
        end = start + classes_per_client
        client_indices = np.concatenate(all_shards[start:end])

        # Shuffle within the client so data isn't ordered by class
        np.random.shuffle(client_indices)

        client_datasets.append(Subset(dataset, client_indices.tolist()))

    return client_datasets


def partition_data(dataset, num_clients, iid=True, classes_per_client=2):
    """
    Wrapper function — routes to IID or Non-IID partitioning.

    Args:
        dataset: PyTorch dataset to partition
        num_clients (int): Number of clients
        iid (bool): If True, use IID partitioning; else Non-IID
        classes_per_client (int): Only used for Non-IID

    Returns:
        list[Subset]: One Subset per client
    """
    if iid:
        print(f"📊 Partitioning data: IID across {num_clients} clients")
        return iid_partition(dataset, num_clients)
    else:
        print(f"📊 Partitioning data: Non-IID across {num_clients} clients "
              f"({classes_per_client} classes each)")
        return non_iid_partition(dataset, num_clients, classes_per_client)


# ── Quick test: run this file directly to verify it works ──
if __name__ == "__main__":
    from src.data.dataset import get_cifar10, CIFAR10_CLASSES

    # Load dataset
    train_data, _ = get_cifar10()

    # ── Test IID Partition ──
    print("\n" + "=" * 50)
    print("IID PARTITION TEST")
    print("=" * 50)
    iid_clients = partition_data(train_data, num_clients=5, iid=True)
    for i, client_data in enumerate(iid_clients):
        # Count classes in this client's data
        client_labels = [train_data[idx][1] for idx in client_data.indices]
        unique_classes = len(set(client_labels))
        print(f"  Client {i}: {len(client_data)} samples, {unique_classes} classes")

    # ── Test Non-IID Partition ──
    print("\n" + "=" * 50)
    print("NON-IID PARTITION TEST")
    print("=" * 50)
    non_iid_clients = partition_data(train_data, num_clients=5, iid=False, classes_per_client=2)
    for i, client_data in enumerate(non_iid_clients):
        # Find which classes this client has
        client_labels = [train_data[idx][1] for idx in client_data.indices]
        unique_classes = sorted(set(client_labels))
        class_names = [CIFAR10_CLASSES[c] for c in unique_classes]
        print(f"  Client {i}: {len(client_data)} samples, "
              f"classes: {unique_classes} → {class_names}")
