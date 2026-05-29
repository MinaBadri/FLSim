import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from client.car_client import CarClient, TrainResult
from network.churn_model import ChurnModel, ClientState

import concurrent.futures
from typing import List
from client.car_client import TrainResult


# ── Per-client persistent state ────────────────────────────────────────
@dataclass
class ClientRecord:
    """
    Everything the server tracks about one client across all rounds.
    """
    client_id        : int
    last_round       : int   = -1    # last round it successfully trained
    total_rounds     : int   =  0    # rounds it has participated in
    total_samples    : int   =  0    # cumulative training samples sent
    avg_loss         : float =  0.0
    avg_accuracy     : float =  0.0
    hardware_tier    : str   = "unknown"

    # Running averages — updated after each successful round
    _loss_sum        : float = field(default=0.0, repr=False)
    _acc_sum         : float = field(default=0.0, repr=False)

    def update(self, result: TrainResult, current_round: int):
        self.last_round    = current_round
        self.total_rounds += 1
        self.total_samples += result.num_samples
        self._loss_sum     += result.loss
        self._acc_sum      += result.accuracy
        self.avg_loss      = self._loss_sum / self.total_rounds
        self.avg_accuracy  = self._acc_sum  / self.total_rounds
        self.hardware_tier = result.hardware_tier

    def staleness(self, current_round: int) -> int:
        """Rounds since this client last successfully trained."""
        if self.last_round < 0:
            return 0
        return current_round - self.last_round


# ── Registry ───────────────────────────────────────────────────────────
class ClientRegistry:
    """
    Single source of truth about every client in the simulation.

    The server calls these methods in order each round:

      1. step(round)         — advance churn, get events
      2. select(round, k)    — pick k active clients to train
      3. run_round(...)      — dispatch training to selected clients
      4. record_results(...) — store results, update stats
    """

    def __init__(
        self,
        fleet    : List[CarClient],
        churn    : ChurnModel,
        seed     : int = 42,
    ):
        self.fleet   = fleet
        self.churn   = churn
        self.rng     = np.random.default_rng(seed)

        # One persistent record per client
        self.records: Dict[int, ClientRecord] = {
            i: ClientRecord(client_id=i)
            for i in range(len(fleet))
        }

        # Full round-by-round history
        self.history: List[dict] = []

    # ── Step 1: Advance churn ──────────────────────────────────────────

    def step(self, current_round: int) -> dict:
        """
        Advance the churn model by one round.
        Returns the churn event dict for logging.
        """
        events = self.churn.step(current_round)
        return events

    # ── Step 2: Select clients for this round ─────────────────────────

    def select(
        self,
        current_round   : int,
        k               : int,
        include_rejoining: bool = True,
    ) -> List[int]:
        """
        Select k clients to participate in this round.

        Active clients are the primary pool.
        REJOINING clients are optionally included — this is
        the key experiment variable: do we let stale cars
        contribute immediately or make them wait?

        Returns a list of client_ids.
        """
        active = self.churn.active_clients()

        if include_rejoining:
            rejoining = [
                cid for cid, rec in self.churn.records.items()
                if rec.is_rejoining()
            ]
            pool = active + rejoining
        else:
            pool = active

        if len(pool) == 0:
            return []

        # Sample min(k, pool_size) without replacement
        k_actual = min(k, len(pool))
        selected = self.rng.choice(pool, size=k_actual, replace=False).tolist()
        return selected

    # ── Step 3: Run training round ────────────────────────────────────

    def run_round(
        self,
        selected_ids  : List[int],
        global_weights: dict,
        config        : dict,
        current_round : int,
    ) -> List[TrainResult]:
        """
    Dispatch local training to selected clients in parallel.
    Uses ThreadPoolExecutor — safe for MPS since PyTorch
    manages MPS context per thread internally.
    """
        def train_one(cid: int) -> TrainResult:
            staleness = self.records[cid].staleness(current_round)
            return self.fleet[cid].train_round(
                global_weights = global_weights,
                config         = config,
                current_round  = current_round,
                staleness      = staleness,
            )

        max_workers = min(len(selected_ids), 4)  # 4 threads on M4

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(train_one, cid): cid
                    for cid in selected_ids}
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    # ── Step 4: Record results ─────────────────────────────────────────

    def record_results(
        self,
        results       : List[TrainResult],
        current_round : int,
        events        : dict,
        global_loss   : float = 0.0,
        global_acc    : float = 0.0,
    ):
        """
        Update client records after a round completes.
        Also appends a full snapshot to self.history.
        """
        successful = [r for r in results if not r.dropped]
        dropped    = [r for r in results if r.dropped]

        for result in successful:
            self.records[result.client_id].update(result, current_round)

        # Mark rejoining clients as active if they contributed
        for result in successful:
            rec = self.churn.records[result.client_id]
            if rec.is_rejoining():
                rec.rejoin()
        
        # Hardware tier distribution of contributors
        tier_counts = {"high": 0, "mid": 0, "low": 0}
        for r in successful:
            tier_counts[r.hardware_tier] += 1

        # Build round snapshot
        snapshot = {
            "round"             : current_round,
            "selected"          : len(results),
            "successful"        : len(successful),
            "dropped_mid_round" : len(dropped),
            "active_pool"       : self.churn.count(ClientState.ACTIVE),
            "dropped_pool"      : self.churn.count(ClientState.DROPPED),
            "rejoining_pool"    : self.churn.count(ClientState.REJOINING),
            "newly_dropped"     : len(events.get("newly_dropped", [])),
            "rejoined"          : len(events.get("rejoined", [])),
            "avg_staleness"     : float(np.mean(
                [r.staleness for r in successful]
            )) if successful else 0.0,
            "avg_train_time"    : float(np.mean(
                [r.train_time for r in successful]
            )) if successful else 0.0,
            "global_loss"       : global_loss,
            "global_accuracy"   : global_acc,
            "high_tier_contributors" : tier_counts["high"],
            "mid_tier_contributors"  : tier_counts["mid"],
            "low_tier_contributors"  : tier_counts["low"],
        }
        self.history.append(snapshot)

    # ── Query helpers ──────────────────────────────────────────────────

    def active_count(self)    -> int:
        return self.churn.count(ClientState.ACTIVE)

    def dropped_count(self)   -> int:
        return self.churn.count(ClientState.DROPPED)

    def rejoining_count(self) -> int:
        return self.churn.count(ClientState.REJOINING)

    def get_record(self, client_id: int) -> ClientRecord:
        return self.records[client_id]

    def most_stale_clients(
        self,
        current_round: int,
        top_n: int = 5,
    ) -> List[Tuple[int, int]]:
        """
        Returns the top_n clients with the highest staleness.
        Useful for debugging and paper analysis.
        """
        stales = [
            (cid, rec.staleness(current_round))
            for cid, rec in self.records.items()
        ]
        return sorted(stales, key=lambda x: x[1], reverse=True)[:top_n]

    def print_round_summary(self, current_round: int):
        if not self.history:
            return
        s = self.history[-1]
        print(
            f"Round {current_round:>3} | "
            f"selected={s['selected']} "
            f"ok={s['successful']} "
            f"mid-drop={s['dropped_mid_round']} | "
            f"pool → active={s['active_pool']} "
            f"dropped={s['dropped_pool']} "
            f"rejoining={s['rejoining_pool']} | "
            f"staleness={s['avg_staleness']:.1f} "
            f"loss={s['global_loss']:.4f} "
            f"acc={s['global_accuracy']:.3f}"
        )

    @classmethod
    def from_config(
        cls,
        cfg   : dict,
        fleet : List[CarClient],
        churn : ChurnModel,
    ) -> "ClientRegistry":
        return cls(
            fleet = fleet,
            churn = churn,
            seed  = cfg.get("seed", 42),
        )