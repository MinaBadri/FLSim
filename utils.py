import yaml
from pathlib import Path
from client.hardware_profile import HardwareProfileFactory
from client.data_profile import build_car_fleet
from network.churn_model import ChurnModel
from server.registry import ClientRegistry

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

def build_fleet(cfg: dict, client_loaders: list, hardware_profiles: list) -> list:
    return build_car_fleet(
        client_loaders    = client_loaders,
        hardware_profiles = hardware_profiles,
        num_classes       = cfg["model"]["num_classes"],
        seed              = cfg.get("seed", 42),
    )

def build_registry(cfg: dict, fleet: list) -> ClientRegistry:
    churn    = ChurnModel.from_config(cfg)
    registry = ClientRegistry.from_config(cfg, fleet, churn)
    return registry