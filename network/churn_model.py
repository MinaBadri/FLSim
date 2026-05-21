import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


class ClientState(Enum):
    ACTIVE    = auto()
    DROPPED   = auto()
    REJOINING = auto()


@dataclass
class ClientChurnRecord:
    """
    Tracks the churn history of a single client across all rounds.
    """
    client_id       : int
    state           : ClientState = ClientState.ACTIVE
    rejoin_round    : int = -1      # round when this client can rejoin
    rounds_dropped  : int = 0       # total rounds spent dropped
    drop_count      : int = 0       # how many times it has dropped
    rejoin_count    : int = 0       # how many times it has rejoined

    def drop(self, current_round: int, delay: int):
        self.state          = ClientState.DROPPED
        self.rejoin_round   = current_round + delay
        self.drop_count    += 1

    def advance(self, current_round: int):
        """
        Called every round. Moves DROPPED → REJOINING when
        the delay has elapsed.
        """
        if self.state == ClientState.DROPPED:
            self.rounds_dropped += 1
            if current_round >= self.rejoin_round:
                self.state = ClientState.REJOINING

    def rejoin(self):
        self.state         = ClientState.ACTIVE
        self.rejoin_count += 1

    def is_active(self)    -> bool: return self.state == ClientState.ACTIVE
    def is_dropped(self)   -> bool: return self.state == ClientState.DROPPED
    def is_rejoining(self) -> bool: return self.state == ClientState.REJOINING


class ChurnModel:
    """
    Manages drop/rejoin dynamics for all clients across rounds.

    drop_prob       : probability an ACTIVE client drops each round.
    min_rejoin_delay: minimum rounds a dropped client must wait.
    max_rejoin_delay: maximum rounds a dropped client must wait.
    rejoin_prob     : probability a REJOINING client is selected
                      back into the active pool each round.
    seed            : for reproducibility.
    """

    def __init__(
        self,
        num_clients      : int,
        drop_prob        : float,
        min_rejoin_delay : int,
        max_rejoin_delay : int,
        rejoin_prob      : float = 0.8,
        seed             : int   = 42,
    ):
        self.num_clients       = num_clients
        self.drop_prob         = drop_prob
        self.min_rejoin_delay  = min_rejoin_delay
        self.max_rejoin_delay  = max_rejoin_delay
        self.rejoin_prob       = rejoin_prob
        self.rng               = np.random.default_rng(seed)

        # One record per client
        self.records: Dict[int, ClientChurnRecord] = {
            i: ClientChurnRecord(client_id=i)
            for i in range(num_clients)
        }

        # Per-round log for analysis / paper plots
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Core step — call this once per round, before selecting clients
    # ------------------------------------------------------------------

    def step(self, current_round: int) -> Dict[str, List[int]]:
        """
        Advance the churn model by one round.

        Returns a dict with four lists:
          newly_dropped   : clients that dropped THIS round
          still_dropped   : clients that remain dropped
          newly_rejoining : clients whose delay just elapsed
          rejoined        : clients moving from REJOINING → ACTIVE
        """
        newly_dropped    = []
        still_dropped    = []
        newly_rejoining  = []
        rejoined         = []

        for cid, rec in self.records.items():

            # 1. Advance delay counters for dropped clients
            prev_state = rec.state
            rec.advance(current_round)

            if rec.is_rejoining() and prev_state == ClientState.DROPPED:
                newly_rejoining.append(cid)

            # 2. ACTIVE clients may drop this round
            if rec.is_active():
                if self.rng.random() < self.drop_prob:
                    delay = int(self.rng.integers(
                        self.min_rejoin_delay,
                        self.max_rejoin_delay + 1
                    ))
                    rec.drop(current_round, delay)
                    newly_dropped.append(cid)

            # 3. REJOINING clients may re-enter the active pool
            elif rec.is_rejoining():
                if self.rng.random() < self.rejoin_prob:
                    rec.rejoin()
                    rejoined.append(cid)
                # if not selected this round, stays REJOINING
                # and will be tried again next round

            # 4. Track still-dropped
            if rec.is_dropped():
                still_dropped.append(cid)

        # Log this round
        snapshot = {
            "round"           : current_round,
            "active"          : self.count(ClientState.ACTIVE),
            "dropped"         : self.count(ClientState.DROPPED),
            "rejoining"       : self.count(ClientState.REJOINING),
            "newly_dropped"   : len(newly_dropped),
            "rejoined"        : len(rejoined),
        }
        self.history.append(snapshot)

        return {
            "newly_dropped"  : newly_dropped,
            "still_dropped"  : still_dropped,
            "newly_rejoining": newly_rejoining,
            "rejoined"       : rejoined,
        }

    # ------------------------------------------------------------------
    # Query helpers — used by the scheduler and server
    # ------------------------------------------------------------------

    def active_clients(self) -> List[int]:
        return [
            cid for cid, rec in self.records.items()
            if rec.is_active()
        ]

    def dropped_clients(self) -> List[int]:
        return [
            cid for cid, rec in self.records.items()
            if rec.is_dropped()
        ]

    def count(self, state: ClientState) -> int:
        return sum(1 for r in self.records.values() if r.state == state)

    def staleness(self, client_id: int, current_round: int) -> int:
        """
        How many rounds has this client been absent?
        Used by the aggregator to weight stale updates.
        """
        rec = self.records[client_id]
        if rec.is_active():
            return 0
        return rec.rounds_dropped

    def round_summary(self, current_round: int) -> dict:
        return self.history[current_round] if self.history else {}

    @classmethod
    def from_config(cls, cfg: dict) -> "ChurnModel":
        churn = cfg.get("churn", {})
        return cls(
            num_clients      = cfg["simulation"]["num_clients"],
            drop_prob        = churn.get("drop_prob", 0.0),
            min_rejoin_delay = churn.get("min_rejoin_delay", 1),
            max_rejoin_delay = churn.get("max_rejoin_delay", 5),
            rejoin_prob      = churn.get("rejoin_prob", 0.8),
            seed             = cfg.get("seed", 42),
        )