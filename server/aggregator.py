import torch
import numpy as np
import copy
from typing import List, Optional
from enum import Enum, auto

from client.car_client import TrainResult


# ── Strategy enum ──────────────────────────────────────────────────────
class AggregationStrategy(Enum):
    FEDAVG           = auto()   # standard weighted average by sample count
    STALENESS_AWARE  = auto()   # decay weight by rounds absent
    ADAPTIVE         = auto()   # staleness decay + loss quality gate


# ── Weight computation ─────────────────────────────────────────────────
class WeightComputer:
    """
    Computes per-client aggregation weights given a strategy.
    All weight vectors are normalised to sum to 1.
    """

    def __init__(
        self,
        strategy      : AggregationStrategy,
        staleness_alpha: float = 0.9,    # decay per absent round (0,1)
        loss_threshold : float = 3.0,    # adaptive: drop updates above this
    ):
        self.strategy       = strategy
        self.alpha          = staleness_alpha
        self.loss_threshold = loss_threshold

    def compute(self, results: List[TrainResult]) -> np.ndarray:
        """
        Returns a weight array aligned with results.
        Dropped results are always given weight 0.
        """
        n = len(results)
        weights = np.zeros(n, dtype=np.float64)

        for i, r in enumerate(results):
            if r.dropped or r.num_samples == 0:
                weights[i] = 0.0
                continue

            # Base weight: sample count
            w = float(r.num_samples)

            if self.strategy in (
                AggregationStrategy.STALENESS_AWARE,
                AggregationStrategy.ADAPTIVE,
            ):
                # Exponential staleness penalty
                w *= (self.alpha ** r.staleness)

            if self.strategy == AggregationStrategy.ADAPTIVE:
                # Quality gate: reject updates with loss above threshold
                if r.loss > self.loss_threshold:
                    w = 0.0
                else:
                    # Soft quality score: lower loss → higher weight
                    # score ∈ (0, 1], capped at 1 for very good updates
                    quality = 1.0 / (1.0 + r.loss)
                    w *= quality

            weights[i] = max(w, 0.0)

        total = weights.sum()
        if total > 0:
            weights /= total

        return weights


# ── Aggregator ─────────────────────────────────────────────────────────
class Aggregator:
    """
    Aggregates client model updates into a new global model.

    Handles:
      - Dropped clients (weight = 0, excluded automatically)
      - Stale rejoining clients (downweighted by staleness)
      - Empty rounds (no valid updates → global model unchanged)
    """

    def __init__(
        self,
        strategy        : AggregationStrategy = AggregationStrategy.FEDAVG,
        staleness_alpha : float = 0.9,
        loss_threshold  : float = 3.0,
    ):
        self.strategy = strategy
        self.weight_computer = WeightComputer(
            strategy        = strategy,
            staleness_alpha = staleness_alpha,
            loss_threshold  = loss_threshold,
        )

        # Per-round aggregation log
        self.history: List[dict] = []

    # ── Main aggregation call ──────────────────────────────────────────

    def aggregate(
        self,
        results        : List[TrainResult],
        global_weights : dict,
        current_round  : int,
    ) -> dict:
        """
        Produce new global weights from this round's client results.

        If no valid updates exist (all dropped or filtered),
        the global model is returned unchanged.
        """
        # Filter to clients with usable updates
        valid   = [(i, r) for i, r in enumerate(results)
                   if not r.dropped and r.weights is not None]

        if not valid:
            self._log(current_round, results, [], np.array([]), skipped=True)
            return copy.deepcopy(global_weights)

        weights = self.weight_computer.compute(results)

        # Identify which valid results actually got non-zero weight
        contributing = [(i, r) for i, r in valid if weights[i] > 0]

        if not contributing:
            self._log(current_round, results, [], weights, skipped=True)
            return copy.deepcopy(global_weights)

        # Weighted parameter average
        new_weights = self._weighted_average(contributing, weights)

        self._log(current_round, results, contributing, weights, skipped=False)
        return new_weights

    # ── Weighted average ───────────────────────────────────────────────

    def _weighted_average(
        self,
        contributing : List[tuple],
        weights      : np.ndarray,
    ) -> dict:
        """
        Compute Σ w_i * θ_i for all contributing clients.
        Works layer-by-layer across the state_dict.
        """
        # Initialise accumulator from the first contributor's keys
        first_sd = contributing[0][1].weights
        accum    = {k: torch.zeros_like(v, dtype=torch.float32)
                    for k, v in first_sd.items()}

        for idx, result in contributing:
            w = float(weights[idx])
            for key, param in result.weights.items():
                accum[key] += w * param.float()

        return accum

    # ── Logging ────────────────────────────────────────────────────────

    def _log(
        self,
        current_round : int,
        all_results   : List[TrainResult],
        contributing  : List[tuple],
        weights       : np.ndarray,
        skipped       : bool,
    ):
        successful = [r for r in all_results if not r.dropped]
        dropped    = [r for r in all_results if r.dropped]

        staleness_vals = [r.staleness for r in successful] if successful else [0]
        weight_vals    = [float(weights[i]) for i, _ in contributing]

        self.history.append({
            "round"             : current_round,
            "strategy"          : self.strategy.name,
            "total_selected"    : len(all_results),
            "successful"        : len(successful),
            "dropped_mid_round" : len(dropped),
            "contributing"      : len(contributing),
            "skipped"           : skipped,
            "avg_staleness"     : float(np.mean(staleness_vals)),
            "max_staleness"     : int(np.max(staleness_vals)),
            "avg_weight"        : float(np.mean(weight_vals)) if weight_vals else 0.0,
            "weight_std"        : float(np.std(weight_vals))  if weight_vals else 0.0,
        })

    # ── Evaluation helper ──────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        model         : torch.nn.Module,
        global_weights: dict,
        test_loader,
        device        : torch.device,
    ) -> tuple[float, float]:
        """
        Evaluate the global model on the test set.
        Returns (loss, accuracy).
        """
        criterion = torch.nn.CrossEntropyLoss()
        model.load_state_dict(global_weights)
        model.to(device)
        model.eval()

        total_loss    = 0.0
        correct       = 0
        total_samples = 0

        for inputs, targets in test_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            total_loss    += loss.item() * inputs.size(0)
            preds          = outputs.argmax(dim=1)
            correct       += preds.eq(targets).sum().item()
            total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct   / total_samples
        return avg_loss, accuracy

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "Aggregator":
        agg_cfg  = cfg.get("aggregation", {})
        strategy = AggregationStrategy[
            agg_cfg.get("strategy", "FEDAVG").upper()
        ]
        return cls(
            strategy        = strategy,
            staleness_alpha = agg_cfg.get("staleness_alpha", 0.9),
            loss_threshold  = agg_cfg.get("loss_threshold",  3.0),
        )