from torch.utils.data import DataLoader
from typing import List

from client.car_client import CarClient
from client.hardware_profile import HardwareProfile


def build_car_fleet(
    client_loaders   : List[DataLoader],
    hardware_profiles: List[HardwareProfile],
    num_classes      : int = 10,
    seed             : int = 42,
) -> List[CarClient]:
    """
    Instantiate one CarClient per client_id, pairing each
    with its DataLoader and HardwareProfile.
    """
    assert len(client_loaders) == len(hardware_profiles), (
        "Number of loaders must match number of hardware profiles."
    )

    fleet = [
        CarClient(
            client_id        = i,
            dataloader       = client_loaders[i],
            hardware_profile = hardware_profiles[i],
            num_classes      = num_classes,
            seed             = seed,
        )
        for i in range(len(client_loaders))
    ]

    return fleet