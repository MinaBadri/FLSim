"""
Represents 1 client.
Receives the global model, trains it on private local data, sends the updated weights back.
Handles hardsware failures
Enforces hardware constraints, such as memory limits & compute speed
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time

from torch.utils.data import DataLoader
from typing import Optional, Tuple

from client.hardware_profile import HardwareProfile
from models.cnn import SimpleCNN


# ── Device selection (M4 Mac / CUDA / CPU) ────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Result container ───────────────────────────────────────────────────
class TrainResult:
    """
    Everything the server needs from a client after a round.
    None weights signals the client dropped mid-round.
    """
    def __init__(
        self,
        client_id     : int,
        weights       : Optional[dict],   # state_dict or None if dropped
        num_samples   : int,
        loss          : float,            # final epoch training loss
        accuracy      : float,            # local training accurac
        train_time    : float,            # simulated hardware-adjusted duration
        staleness     : int,              # rounds this client was absent
        hardware_tier : str,
        dropped       : bool,
        effective_batch_size : int = 32,
    ):
        self.client_id     = client_id
        self.weights       = weights
        self.num_samples   = num_samples
        self.loss          = loss
        self.accuracy      = accuracy
        self.train_time    = train_time
        self.staleness     = staleness
        self.hardware_tier = hardware_tier
        self.dropped       = dropped
        self.effective_batch_size = effective_batch_size

    def __repr__(self):
        status = "DROPPED" if self.dropped else f"loss={self.loss:.4f} acc={self.accuracy:.3f}"
        return (f"CarClient({self.client_id}) "
                f"samples={self.num_samples} {status} "
                f"time={self.train_time:.2f}s staleness={self.staleness}")


# ── Car Client ─────────────────────────────────────────────────────────
class CarClient:
    """
    One federated learning client representing a vehicle.

    Responsibilities:
      - Load the global model weights sent by the server
      - Train locally for E epochs on its private data slice
      - Simulate hardware delay and possible mid-round hardware fault
      - Return a TrainResult (or a dropped TrainResult) to the server
    """

    def __init__(
        self,
        client_id       : int,
        dataloader      : DataLoader,
        hardware_profile: HardwareProfile,
        num_classes     : int = 10,
        seed            : int = 42,
    ):
        self.client_id        = client_id
        self.dataloader       = dataloader
        self.hw               = hardware_profile
        self.num_classes      = num_classes
        self.device           = get_device()
        self.rng              = np.random.default_rng(seed + client_id)

        # Local model — rebuilt fresh each round from server weights
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)

    # ── Public API ────────────────────────────────────────────────────
    # Called by registry.run_round()
    def train_round(
        self,
        global_weights  : dict,
        config          : dict,
        current_round   : int,           # reserved — available for Learning rate decay scheduling
        staleness       : int = 0,
    ) -> TrainResult:
        """
        Execute one FL round locally.

        global_weights : server's current model state_dict
        config         : training section of the yaml config
        current_round  : used for logging
        staleness      : how many rounds this client was absent
        """

        # 1. Load global model
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.to(self.device)

        # 2. Resolve effective batch size from hardware cap
        requested_bs = config["batch_size"]
        effective_bs = self.hw.effective_batch_size(requested_bs)

        # Rebuild loader only if batch size needs to change
        loader = self._get_loader(effective_bs)

        # 3. Hardware fault check — fail before any training
        if not self.hw.will_complete(self.rng):
            result = TrainResult(
                client_id     = self.client_id,
                weights       = None,
                num_samples   = 0,
                loss          = 0.0,
                accuracy      = 0.0,
                train_time    = 0.0,
                staleness     = staleness,
                hardware_tier = self._tier_label(),
                dropped       = True,
                effective_batch_size = effective_bs,
            )
            self.hw.record_round(completed=False, duration=0.0)
            return result

        # 4. Local training
        optimizer = self._build_optimizer(config, effective_bs)
        criterion = nn.CrossEntropyLoss()

        t_start = time.time()
        loss, accuracy, num_samples = self._local_train(
            loader     = loader,
            optimizer  = optimizer,
            criterion  = criterion,
            epochs     = config["local_epochs"],
        )
        raw_duration = time.time() - t_start

        # 5. Simulate hardware delay on top of real training time
        simulated_duration = self.hw.simulated_training_delay(raw_duration)
        self.hw.record_round(completed=True, duration=simulated_duration)

        return TrainResult(
            client_id     = self.client_id,
            weights       = copy.deepcopy(self.model.state_dict()),
            num_samples   = num_samples,
            loss          = loss,
            accuracy      = accuracy,
            train_time    = simulated_duration,
            staleness     = staleness,
            hardware_tier = self._tier_label(),
            dropped       = False,
            effective_batch_size = effective_bs,
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _local_train(
        self,
        loader    : DataLoader,
        optimizer : torch.optim.Optimizer,
        criterion : nn.Module,
        epochs    : int,
    ) -> Tuple[float, float, int]:
        """
        Train for E epochs. Returns (avg_loss, accuracy, num_samples).
        """
        self.model.train()
        total_loss    = 0.0
        correct       = 0
        total_samples = 0

        for epoch in range(epochs):
            for inputs, targets in loader:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss    = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if epoch == epochs - 1:
                    total_loss    += loss.item() * inputs.size(0)
                    preds          = outputs.argmax(dim=1)
                    correct       += preds.eq(targets).sum().item()
                    total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct   / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy, total_samples

    def _build_optimizer(self, config: dict, effective_bs: int) -> torch.optim.Optimizer:
        base_lr   = config.get("learning_rate", 0.01)
        ref_bs     = config.get("batch_size", 32)  
        name = config.get("optimizer", "sgd").lower()

        # Linear scaling rule — corrects for batch size mismatch
        scaled_lr  = base_lr * (effective_bs / ref_bs)

        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=scaled_lr)
        return optim.SGD(
            self.model.parameters(),
            lr=scaled_lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 1e-4),
        )

    def _get_loader(self, effective_bs: int) -> DataLoader:
        """
        Return a loader with the hardware-capped batch size.
        Avoids rebuilding if the size matches the existing loader.
        """
        if self.dataloader.batch_size == effective_bs:
            return self.dataloader
        return DataLoader(
            self.dataloader.dataset,
            batch_size  = effective_bs,
            shuffle     = True,
            drop_last   = False,
        )

    def _tier_label(self) -> str:
        s = self.hw.compute_speed
        if s >= 1.5:  return "high"
        if s >= 0.8:  return "mid"
        return "low"