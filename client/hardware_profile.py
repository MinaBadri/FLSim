import numpy as np
import time
import copy
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

    def simulated_training_delay(
            self, 
            base_seconds: float,
            rng: Optional[np.random.Generator] = None,) -> float:
        """
        Return how long training takes for this client.
        Faster hardware = shorter delay.
        A small noise term adds realism.
        """
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.uniform(0.95, 1.05)
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
            memory_range=(256, 512),
            reliability_range=(0.95, 1.00),
            weight=0.25,  # 25% of cars
        ),
        "mid": dict(
            speed_range=(0.8, 1.5),
            memory_range=(128, 256),
            reliability_range=(0.85, 0.95),
            weight=0.50,  # 50% of cars
        ),
        "low": dict(
            speed_range=(0.3, 0.8),
            memory_range=(32, 128),
            reliability_range=(0.70, 0.85),
            weight=0.25,  # 25% of cars
        ),
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.TIERS = copy.deepcopy(HardwareProfileFactory.TIERS)

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
        hardware block keys:
        homogeneous       : bool — identical ideal profile for every client
                                    (no speed diff, no batch clamp, no faults).
                                    Use to neutralize hardware as a control.
        speed_range       : [lo, hi]  uniform compute_speed
        memory_range      : [lo, hi]  uniform memory_cap (max batch size)
        reliability_range : [lo, hi]  uniform reliability
        Falls back to legacy min_speed/max_speed tiers, then default TIERS.
        """
        hw = cfg.get("hardware", {})
        num_clients = cfg["simulation"]["num_clients"]

        if hw.get("homogeneous", False):
            return [
                HardwareProfile(client_id=i, compute_speed=1.0,
                                memory_cap=10**9, reliability=1.0)
                for i in range(num_clients)
            ]

        if any(k in hw for k in ("speed_range", "memory_range", "reliability_range",
                         "speed", "memory_cap", "reliability")):
            sr = hw.get("speed_range",       [0.3, 2.5])
            mr = hw.get("memory_range",      [32, 512])
            rr = hw.get("reliability_range", [0.70, 1.00])
           
            if "speed"       in hw: sr = [hw["speed"],       hw["speed"]]
            if "memory_cap"  in hw: mr = [hw["memory_cap"],  hw["memory_cap"]]
            if "reliability" in hw: rr = [hw["reliability"], hw["reliability"]]
            return [
                HardwareProfile(
                    client_id     = i,
                    compute_speed = float(self.rng.uniform(sr[0], sr[1])),
                    memory_cap    = int(self.rng.integers(mr[0], mr[1] + 1)),  # inclusive hi
                    reliability   = float(self.rng.uniform(rr[0], rr[1])),
                )
                for i in range(num_clients)
            ]

       
        if "min_speed" in hw and "max_speed" in hw:
            span = hw["max_speed"] - hw["min_speed"]
            self.TIERS["high"]["speed_range"] = (hw["min_speed"] + 0.6 * span, hw["max_speed"])
            self.TIERS["mid"]["speed_range"]  = (hw["min_speed"] + 0.25 * span, hw["min_speed"] + 0.65 * span)
            self.TIERS["low"]["speed_range"]  = (hw["min_speed"], hw["min_speed"] + 0.30 * span)
            return self.generate(num_clients)

        # Default: use the built-in TIERS
        return self.generate(num_clients)