import yaml
from pathlib import Path
from client.hardware_profile import HardwareProfileFactory

from data.partitioner import (
    load_cifar10,
    dirichlet_partition,
    make_client_loaders,
    make_test_loader,
)

def load_config(path: str) -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


def build_data_pipeline(cfg: dict):
    """
    Given a loaded config, returns:
      - client_loaders  : list of DataLoader, one per client
      - test_loader     : global test DataLoader
      - client_indices  : raw index lists (needed by registry later)
    """
    train_dataset = load_cifar10(train=True)

    client_indices = dirichlet_partition(
        dataset    = train_dataset,
        num_clients= cfg["simulation"]["num_clients"],
        alpha      = cfg["data"]["dirichlet_alpha"],
        seed       = cfg.get("seed", 42),
    )

    client_loaders = make_client_loaders(
        client_indices,
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
    )

    test_loader = make_test_loader(batch_size=64)

    return client_loaders, test_loader, client_indices

def build_hardware_profiles(cfg: dict) -> list:
    factory = HardwareProfileFactory(seed=cfg.get("seed", 42))
    return factory.from_config(cfg)