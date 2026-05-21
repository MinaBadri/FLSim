import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HardwareProfile:
    """
    Represents the hardware characteristics of one car-client.

    compute_speed   : multiplier on simulated training delay.
                      1.0 = baseline, 2.0 = twice as fast, 0.5 = half speed.
    memory_cap      : max batch size this client can handle.
                      If the global batch size exceeds this, it is clamped.
    reliability     : probability [0,1] of completing a training round
                      without a hardware fault. Independent of churn.
    client_id       : which car this belongs to.
    """
    client_id     : int
    compute_speed : float  # > 0
    memory_cap    : int    # max batch size in samples
    reliability   : float  # 0.0 – 1.0

    # Runtime state — not set at construction
    total_training_time : float = field(default=0.0, init=False)
    rounds_completed    : int   = field(default=0,   init=False)
    rounds_failed       : int   = field(default=0,   init=False)

    def effective_batch_size(self, requested: int) -> int:
        """Clamp batch size to what this hardware can handle."""
        return min(requested, self.memory_cap)

    def simulated_training_delay(self, base_seconds: float) -> float:
        """
        Return how long training takes for this client.
        Faster hardware = shorter delay.
        A small noise term adds realism.
        """
        noise = np.random.uniform(0.95, 1.05)
        return (base_seconds / self.compute_speed) * noise

    def will_complete(self, rng: Optional[np.random.Generator] = None) -> bool:
        """
        Roll whether this client completes the round without
        a hardware fault.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.random() < self.reliability

    def record_round(self, completed: bool, duration: float):
        """Update runtime stats after each round."""
        self.total_training_time += duration
        if completed:
            self.rounds_completed += 1
        else:
            self.rounds_failed += 1

    def summary(self) -> dict:
        return {
            "client_id"          : self.client_id,
            "compute_speed"      : round(self.compute_speed, 3),
            "memory_cap"         : self.memory_cap,
            "reliability"        : round(self.reliability, 3),
            "rounds_completed"   : self.rounds_completed,
            "rounds_failed"      : self.rounds_failed,
            "total_training_time": round(self.total_training_time, 2),
        }


class HardwareProfileFactory:
    """
    Generates hardware profiles for all clients.
    Three tiers model the real vehicular heterogeneity:

      Tier A (high-end)  : fast compute, large memory, high reliability
      Tier B (mid-range) : moderate everything
      Tier C (embedded)  : slow compute, small memory, lower reliability
    """

    TIERS = {
        "high": dict(
            speed_range=(1.5, 2.5),
            memory_range=(64, 128),
            reliability_range=(0.95, 1.00),
            weight=0.25,  # 25% of cars
        ),
        "mid": dict(
            speed_range=(0.8, 1.5),
            memory_range=(32, 64),
            reliability_range=(0.85, 0.95),
            weight=0.50,  # 50% of cars
        ),
        "low": dict(
            speed_range=(0.3, 0.8),
            memory_range=(8, 32),
            reliability_range=(0.70, 0.85),
            weight=0.25,  # 25% of cars
        ),
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _sample_tier(self, tier_name: str, client_id: int) -> HardwareProfile:
        t = self.TIERS[tier_name]
        return HardwareProfile(
            client_id     = client_id,
            compute_speed = float(self.rng.uniform(*t["speed_range"])),
            memory_cap    = int(self.rng.integers(*t["memory_range"])),
            reliability   = float(self.rng.uniform(*t["reliability_range"])),
        )

    def generate(self, num_clients: int) -> list[HardwareProfile]:
        """
        Assign each client a tier, then sample their profile from it.
        Tier assignment is random but respects the tier weights.
        """
        tier_names  = list(self.TIERS.keys())
        tier_weights = [self.TIERS[t]["weight"] for t in tier_names]

        # Assign tiers
        tiers = self.rng.choice(
            tier_names,
            size=num_clients,
            p=tier_weights,
        )

        profiles = [
            self._sample_tier(tier, client_id=i)
            for i, tier in enumerate(tiers)
        ]

        return profiles

    def from_config(self, cfg: dict) -> list[HardwareProfile]:
        """
        Build profiles driven by config values.
        Falls back to factory defaults if hardware block is absent.
        """
        hw = cfg.get("hardware", {})
        num_clients = cfg["simulation"]["num_clients"]

        # Override tier ranges if config specifies global min/max
        if "min_speed" in hw and "max_speed" in hw:
            span = hw["max_speed"] - hw["min_speed"]
            self.TIERS["high"]["speed_range"] = (
                hw["min_speed"] + 0.6 * span,
                hw["max_speed"],
            )
            self.TIERS["mid"]["speed_range"] = (
                hw["min_speed"] + 0.25 * span,
                hw["min_speed"] + 0.65 * span,
            )
            self.TIERS["low"]["speed_range"] = (
                hw["min_speed"],
                hw["min_speed"] + 0.30 * span,
            )

        return self.generate(num_clients)