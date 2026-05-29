import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple
from pathlib import Path

DATA_ROOT = Path("./data/raw")

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return train_tf, test_tf


def load_cifar10(train: bool = True):
    tf, test_tf = get_transforms()
    transform = tf if train else test_tf
    return datasets.CIFAR10(
        root=DATA_ROOT, train=train,
        download=True, transform=transform
    )


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition dataset indices across num_clients using
    Dirichlet(alpha) distribution over class labels.

    Returns a list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices = [
        np.where(labels == c)[0].tolist()
        for c in range(num_classes)
    ]

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls_idx in class_indices:
        rng.shuffle(cls_idx)
        # Sample proportions from Dirichlet distribution
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        # Convert to integer counts that sum to len(cls_idx)
        splits = (proportions * len(cls_idx)).astype(int)
        # Fix rounding — add leftover to largest split
        splits[-1] += len(cls_idx) - splits.sum()

        cursor = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(
                cls_idx[cursor : cursor + count]
            )
            cursor += count

    return client_indices


def make_client_loaders(
    client_indices: List[List[int]],
    dataset,
    batch_size: int = 32,
) -> List[DataLoader]:
    """Wrap each client's index list into a DataLoader."""
    return [
        DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers = 0,
            pin_memory = False,
            # drop_last=False,
            persistent_workers = False,
        )
        for indices in client_indices
    ]


def make_test_loader(batch_size: int = 64) -> DataLoader:
    """Global test set — used by the server for evaluation."""
    test_dataset = load_cifar10(train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_client_class_distribution(
    client_indices: List[List[int]],
    dataset,
    num_classes: int = 10,
) -> np.ndarray:
    """
    Returns a (num_clients, num_classes) matrix of sample counts.
    Useful for visualizing and logging heterogeneity.
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    distribution = np.zeros((len(client_indices), num_classes), dtype=int)
    for cid, indices in enumerate(client_indices):
        for idx in indices:
            distribution[cid, labels[idx]] += 1
    return distribution
