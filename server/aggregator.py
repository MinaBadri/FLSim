import torch
import numpy as np
import copy
from typing import List, Optional
from enum import Enum, auto

from client.car_client import TrainResult
# from models import build_model


# ── Strategy enum ──────────────────────────────────────────────────────
class AggregationStrategy(Enum):
    FEDAVG          = auto()   # canonical: weighted average by sample count
    STALENESS_AWARE = auto()   # FedAvg + decay by update staleness
    ADAPTIVE        = auto()   # staleness decay + loss quality gate
    CATCHUP         = auto()   # FedAvg + BOOST returning (stale) clients


# ── Weight computation ─────────────────────────────────────────────────
class WeightComputer:


    def __init__(
        self,
        strategy      : AggregationStrategy,
        staleness_alpha: float = 0.9,    # decay per absent round (0,1)
        loss_threshold : float = 3.0,    # adaptive: drop updates above this
        staleness_boost: float = 0.1,    # CATCHUP: extra weight per absent round
        max_boost      : float = 3.0,    # CATCHUP: cap so one returner can't dominate
    
    ):
        self.strategy       = strategy
        self.alpha          = staleness_alpha
        self.loss_threshold = loss_threshold
        self.boost_beta     = staleness_boost
        self.max_boost      = max_boost

    def compute(
        self,
        results   : List[TrainResult],
        ref_bs    : int = 32,
    ) -> np.ndarray:
        """
        Compute a weight for each client result, based on the configured strategy.
        """

        n = len(results)
        weights = np.zeros(n, dtype=np.float64)

        for i, r in enumerate(results):

            # Always zero for dropped clients
            if r.dropped or r.num_samples == 0 or r.weights is None:
                weights[i] = 0.0
                continue

            
            w = float(r.num_samples)

            if self.strategy in (
                AggregationStrategy.STALENESS_AWARE,
                AggregationStrategy.ADAPTIVE,
            ):
                w *= (self.alpha ** r.staleness)

            elif self.strategy == AggregationStrategy.CATCHUP:
                 w *= min(1.0 + self.boost_beta * r.staleness, self.max_boost)

          
            if self.strategy == AggregationStrategy.ADAPTIVE:
                if r.loss > self.loss_threshold:
                    w = 0.0   # hard reject
                else:
                    # quality = 1.0 / (1.0 + r.loss)
                    # w      *= quality
                    w *= 1.0 / (1.0 + r.loss)  # softer decay by loss value

            weights[i] = max(w, 0.0)

        total = weights.sum()
        if total > 0:
            weights /= total

        return weights


# ── Aggregator ─────────────────────────────────────────────────────────
class Aggregator:


    def __init__(
        self,
        strategy        : AggregationStrategy = AggregationStrategy.FEDAVG,
        staleness_alpha : float = 0.9,
        loss_threshold  : float = 3.0,
        staleness_boost : float = 0.1,
        max_boost       : float = 3.0,
    ):
        self.strategy = strategy
        self.weight_computer = WeightComputer(
            strategy        = strategy,
            staleness_alpha = staleness_alpha,
            loss_threshold  = loss_threshold,
            staleness_boost = staleness_boost,
            max_boost         = max_boost,
        )

   
        self.history: List[dict] = []

 

    def aggregate(
        self,
        results        : List[TrainResult],
        global_weights : dict,
        current_round  : int,
        ref_bs         : int = 32,
    ) -> dict:
        
        valid   = [(i, r) for i, r in enumerate(results)
                   if not r.dropped and r.weights is not None]

        if not valid:
            self._log(current_round, results, [], np.array([]), skipped=True)
            return copy.deepcopy(global_weights)

        # weights = self.weight_computer.compute(results)
        weights = self.weight_computer.compute(results)

       
        contributing = [(i, r) for i, r in valid if weights[i] > 0]

        if not contributing:
            self._log(current_round, results, [], weights, skipped=True)
            return {k: v.clone() for k, v in global_weights.items()}

    
        new_weights = self._weighted_average(contributing, weights)

        self._log(current_round, results, contributing, weights, skipped=False)
        return new_weights

    # ── Weighted average ───────────────────────────────────────────────

    def _weighted_average(
        self,
        contributing : List[tuple],
        weights      : np.ndarray,
    ) -> dict:
       

        first_sd = contributing[0][1].weights
        accum    = {k: torch.zeros_like(v, dtype=torch.float32)
                    for k, v in first_sd.items()}

        for idx, result in contributing:
            w = float(weights[idx])
            for key, param in result.weights.items():
                accum[key] += w * param.float()

        return accum


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

  

    @torch.no_grad()
    def evaluate(
        self,
        model,
        global_weights: dict,
        test_loader,
        device,
        return_per_class: bool = False,
    ) -> tuple[float, float]:
    
        criterion = torch.nn.CrossEntropyLoss()
        model.load_state_dict(global_weights)
        model.to(device)
        model.eval()

        total_loss    = 0.0
        correct       = 0
        total_samples = 0
        pc_correct = pc_total = None     


        for inputs, targets in test_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            total_loss    += loss.item() * inputs.size(0)
            preds          = outputs.argmax(dim=1)
            correct       += preds.eq(targets).sum().item()
            total_samples += inputs.size(0)
            
            if return_per_class:
                if pc_correct is None:
                    K = outputs.size(1)
                    pc_correct = torch.zeros(K, dtype=torch.long, device=device)
                    pc_total   = torch.zeros(K, dtype=torch.long, device=device)
                K = pc_correct.size(0)
                pc_total   += torch.bincount(targets,        minlength=K)
                pc_correct += torch.bincount(targets[preds.eq(targets)], minlength=K)

        avg_loss = total_loss / total_samples
        accuracy = correct   / total_samples
        if return_per_class:
            per_class_acc = (pc_correct.float() / pc_total.clamp(min=1).float()).cpu().numpy()
            return avg_loss, accuracy, per_class_acc
        return avg_loss, accuracy

 

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
            staleness_boost = agg_cfg.get("staleness_boost", 0.1),
            max_boost       = agg_cfg.get("max_boost",       3.0),
        )