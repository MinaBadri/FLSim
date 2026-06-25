import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
from pathlib import Path
import os
import warnings

DATA_ROOT = Path("./data/raw")

def get_num_workers():
    if os.name == 'nt':        # Windows
        return 2               # Windows supports workers with CUDA
    return 0     

def _extract_labels(dataset) -> np.ndarray:
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].cpu().numpy()
    if hasattr(dataset, "targets"):              # torchvision CIFAR exposes this
        return np.asarray(dataset.targets)
    # Fallback: slow, triggers transforms — only hit for exotic datasets/Subsets
    return np.array([dataset[i][1] for i in range(len(dataset))])


def preload_to_device(dataset, device: torch.device) -> TensorDataset:
    """
    Load entire dataset into GPU memory as tensors.
    Only use when dataset fits in VRAM — CIFAR-10/100 always fits.
    Returns a TensorDataset on the target device.
    """
    

    print(f"  Preloading dataset to {device}...")
    

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = 1024,
        shuffle     = False,
        num_workers = 0,
    )
    all_inputs  = []
    all_targets = []

    for inputs, targets in loader:
        all_inputs.append(inputs)
        all_targets.append(targets)

    all_inputs  = torch.cat(all_inputs).to(device)
    all_targets = torch.cat(all_targets).to(device)

    mb = all_inputs.element_size() * all_inputs.nelement() / 1e6

    print(f"  Loaded: {all_inputs.shape}  ({mb:.1f} MB on {device})")

    return TensorDataset(all_inputs, all_targets)


def get_transforms(dataset: str = "cifar100"):

    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
    else:  # cifar10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)


    # train_tf = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std),
    #     # ((0.4914, 0.4822, 0.4465),
    #                         #  (0.2470, 0.2435, 0.2616)),
    # ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
            # (0.4914, 0.4822, 0.4465),
                            #  (0.2470, 0.2435, 0.2616)),
    ])
    # return train_tf, test_tf
class RandomGpuAugment:
    """
    Per-batch random crop (with reflect padding) + random horizontal flip,
    applied to an already-normalized, already-on-device batch (B, C, H, W).

    Use ONLY during training, AFTER moving the batch to the device:

        if self.model.training:
            inputs = self.augment(inputs)

    Geometric ops commute with normalization, so applying them to normalized
    tensors is fine. This restores fresh augmentation every epoch even when
    the dataset is preloaded to GPU.
    """
    def __init__(self, padding: int = 4, p_flip: float = 0.5):
        self.padding = padding
        self.p_flip  = p_flip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Random horizontal flip (per sample). Batch is a fresh collated tensor,
        # so in-place is safe and won't touch the cached dataset.
        flip_mask = torch.rand(B, device=x.device) < self.p_flip
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

        if self.padding > 0:
            # x = F.pad(x, [self.padding] * 4, mode="reflect")
            # max_off = 2 * self.padding
            # tops  = torch.randint(0, max_off + 1, (B,))
            # lefts = torch.randint(0, max_off + 1, (B,))
            # out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
            # for i in range(B):  # B is small (<=512); fine in practice
            #     t, l = int(tops[i]), int(lefts[i])
            #     out[i] = x[i, :, t:t + H, l:l + W]
            # x = out
            p = self.padding
            padded = torch.nn.functional.pad(x, [p, p, p, p], mode="reflect")
            Wp = W + 2 * p
            max_off = 2 * p
 
            # Per-sample crop offsets, generated on-device (no host sync).
            tops  = torch.randint(0, max_off + 1, (B,), device=x.device)
            lefts = torch.randint(0, max_off + 1, (B,), device=x.device)
 
            ar_h = torch.arange(H, device=x.device)
            ar_w = torch.arange(W, device=x.device)
            rows = tops.view(B, 1)  + ar_h.view(1, H)          # (B, H)
            cols = lefts.view(B, 1) + ar_w.view(1, W)          # (B, W)
 
            # Gather H rows, then W cols — fully vectorized crop.
            rows_idx = rows.view(B, 1, H, 1).expand(B, C, H, Wp)
            gathered = torch.gather(padded, 2, rows_idx)       # (B, C, H, Wp)
            cols_idx = cols.view(B, 1, 1, W).expand(B, C, H, W)
            x = torch.gather(gathered, 3, cols_idx)            # (B, C, H, W)
        return x
    
def load_dataset(train: bool = True, dataset: str = "cifar100"):
    """Load CIFAR-10 or CIFAR-100."""
    transform = get_transforms(dataset)
    # transform = train_tf if train else test_tf

    if dataset == "cifar100":
        return datasets.CIFAR100(
            root=DATA_ROOT, train=train,
            download=True, transform=transform
        )
    return datasets.CIFAR10(
        root=DATA_ROOT, train=train,
        download=True, transform=transform
    )
# def load_cifar10(train: bool = True):
#     tf, test_tf = get_transforms()
#     transform = tf if train else test_tf
#     return datasets.CIFAR10(
#         root=DATA_ROOT, train=train,
#         download=True, transform=transform
    # )


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
    labels = _extract_labels(dataset)            

    num_classes = len(np.unique(labels))
    class_indices = [
        np.where(labels == c)[0].tolist()
        for c in range(num_classes)
    ]

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls_idx in class_indices:
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))

        # Old code did splits[-1] += leftover, dumping every class's
        # rounding remainder onto the last client -> systematic size bias.
        # Distribute each leftover to the clients with the largest
        # fractional parts instead.
        exact   = proportions * len(cls_idx)
        splits  = np.floor(exact).astype(int)
        leftover = len(cls_idx) - int(splits.sum())
        if leftover > 0:
            frac = exact - splits
            recipients = np.argsort(-frac)[:leftover]
            splits[recipients] += 1

        cursor = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(
                cls_idx[cursor: cursor + count]
            )
            cursor += count

    # Warn about empty clients (common at very low alpha) so they don't
    # silently distort heterogeneity results downstream.
    empty = [cid for cid, idx in enumerate(client_indices) if len(idx) == 0]
    if empty:
        warnings.warn(
            f"dirichlet_partition: {len(empty)} client(s) received 0 samples "
            f"at alpha={alpha} (client ids: {empty[:10]}"
            f"{'...' if len(empty) > 10 else ''}). "
            f"These will contribute nothing to training."
        )

    return client_indices
    # Fast label extraction — handles both TensorDataset and regular datasets
    # if isinstance(dataset, TensorDataset):
    #     labels = dataset.tensors[1].cpu().numpy()
    # else:
    #     labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # num_classes = len(np.unique(labels))

    # # Group indices by class
    # class_indices = [
    #     np.where(labels == c)[0].tolist()
    #     for c in range(num_classes)
    # ]

    # client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # for cls_idx in class_indices:
    #     rng.shuffle(cls_idx)
    #     # Sample proportions from Dirichlet distribution
    #     proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
    #     # Convert to integer counts that sum to len(cls_idx)
    #     splits = (proportions * len(cls_idx)).astype(int)
    #     # Fix rounding — add leftover to largest split
    #     splits[-1] += len(cls_idx) - splits.sum()

    #     cursor = 0
    #     for client_id, count in enumerate(splits):
    #         client_indices[client_id].extend(
    #             cls_idx[cursor : cursor + count]
    #         )
    #         cursor += count

    # return client_indices


def make_client_loaders(
    client_indices: List[List[int]],
    dataset,
    batch_size: int = 512,
) -> List[DataLoader]:
    """Wrap each client's index list into a DataLoader.
     With drop_last=True any client holding fewer
    than `batch_size` samples yields ZERO batches and silently vanishes —
    which is exactly the small, highly-skewed clients that low alpha is
    meant to create. drop_last=False keeps every sample of every non-empty
    client."""
    loaders = []
    for cid, indices in enumerate(client_indices):
        if len(indices) == 0:
            warnings.warn(
                f"make_client_loaders: client {cid} is empty; its loader "
                f"will yield no batches."
            )
        loaders.append(
            DataLoader(
                Subset(dataset, indices),
                batch_size         = batch_size,
                shuffle            = True,
                drop_last          = False,   
                num_workers        = 0,
                pin_memory         = False,
                persistent_workers = False,
            )
        )
    return loaders
    # return [
    #     DataLoader(
    #         Subset(dataset, indices),
    #         batch_size         = batch_size,
    #         shuffle            = True,
    #         drop_last          = False,
    #         num_workers        = 0,
    #         pin_memory         = False,
    #         persistent_workers = False,
    #     )
    #     for indices in client_indices
    # ]

# def make_test_loader(batch_size: int = 512, dataset: str = "cifar100") -> DataLoader:
#     test_dataset = load_dataset(train=False, dataset=dataset)
#     return DataLoader(
#         test_dataset,
#         batch_size         = batch_size,
#         shuffle            = False,
#         num_workers        = get_num_workers(),
#         pin_memory         = True,
#         persistent_workers = get_num_workers() > 0,
#     )
# def make_test_loader(batch_size: int = 64) -> DataLoader:
#     """Global test set — used by the server for evaluation."""
#     test_dataset = load_cifar10(train=False)
#     return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
def make_test_loader(
    dataset    : TensorDataset,
    batch_size : int = 512,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = 0,
        pin_memory         = False,
        persistent_workers = False,
    )

def get_client_class_distribution(
    client_indices: List[List[int]],
    dataset,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Returns a (num_clients, num_classes) matrix of sample counts.
    Useful for visualizing and logging heterogeneity.
    """
    labels = _extract_labels(dataset)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    distribution = np.zeros((len(client_indices), num_classes), dtype=int)
    for cid, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue
        counts = np.bincount(labels[np.asarray(indices)], minlength=num_classes)
        distribution[cid] = counts
    return distribution
    # if isinstance(dataset, TensorDataset):
    #     labels = dataset.tensors[1].cpu().numpy()
    # else:
    #     labels = np.array([dataset[i][1] for i in range(len(dataset))])
    # distribution = np.zeros((len(client_indices), num_classes), dtype=int)
    # for cid, indices in enumerate(client_indices):
    #     for idx in indices:
    #         distribution[cid, labels[idx]] += 1
    # return distribution
