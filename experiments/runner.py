import copy
import csv
import json
import time
from itertools import product
from pathlib import Path
from typing import List, Dict, Any

from utils import (
    load_config,
    build_data_pipeline,
    build_hardware_profiles,
    build_fleet,
    build_registry,
    build_server,
)


class ExperimentRunner:
    """
    Sweeps a matrix of hyperparameters and runs one full
    FL simulation per combination.

    Each run gets its own output directory:
      outputs/<experiment_name>/<run_id>/

    After all runs, a summary CSV is written comparing
    final accuracy across all combinations.
    """

    def __init__(
        self,
        base_config_path : str,
        experiment_name  : str,
        sweep            : Dict[str, List[Any]],
        output_root      : str = "./outputs",
    ):
        self.base_cfg         = load_config(base_config_path)
        self.experiment_name  = experiment_name
        self.sweep            = sweep
        self.output_root      = Path(output_root) / experiment_name
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.summary_rows: List[dict] = []

    # ── Main entry point ───────────────────────────────────────────────

    def run_all(self):
        combos = self._build_combos()
        total  = len(combos)

        print(f"\n{'='*55}")
        print(f" Experiment : {self.experiment_name}")
        print(f" Runs       : {total}")
        print(f" Output     : {self.output_root}")
        print(f"{'='*55}\n")

        # Build data pipeline once — shared across all runs
        # (data split is fixed by seed, so this is correct)
        print("Building shared data pipeline ...")
        client_loaders, test_loader, client_indices = \
            build_data_pipeline(self.base_cfg)
        print(f"  {len(client_loaders)} client loaders ready\n")

        for i, combo in enumerate(combos):
            run_id  = self._combo_to_id(combo)
            run_cfg = self._apply_combo(combo)
            run_dir = self.output_root / run_id

            print(f"[{i+1}/{total}] {run_id}")
            self._print_combo(combo)

            t0 = time.time()
            final_loss, final_acc = self._run_one(
                cfg            = run_cfg,
                client_loaders = client_loaders,
                test_loader    = test_loader,
                output_dir     = str(run_dir),
            )
            elapsed = time.time() - t0

            row = {
                "run_id"      : run_id,
                "final_loss"  : round(final_loss, 6),
                "final_acc"   : round(final_acc,  6),
                "elapsed_min" : round(elapsed / 60, 2),
                **{k: v for k, v in combo},
            }
            self.summary_rows.append(row)
            print(f"  → loss={final_loss:.4f}  acc={final_acc:.4f}  "
                  f"time={elapsed/60:.1f}min\n")

        self._save_summary()
        print(f"All runs complete. Summary → {self.output_root}/summary.csv")

    # ── Single run ─────────────────────────────────────────────────────

    def _run_one(
        self,
        cfg            : dict,
        client_loaders : list,
        test_loader,
        output_dir     : str,
    ) -> tuple[float, float]:
        """
        Build all components fresh for this run
        (hardware profiles and churn are re-seeded per run),
        then run the server and return final (loss, acc).
        """
        hardware_profiles = build_hardware_profiles(cfg)
        fleet             = build_fleet(cfg, client_loaders, hardware_profiles)
        registry          = build_registry(cfg, fleet)
        server            = build_server(cfg, registry, test_loader, output_dir)

        history = server.run()

        # Pull final evaluated metrics
        # (last round where evaluation actually ran)
        evaluated = [h for h in history if h["global_loss"] > 0]
        if evaluated:
            last = evaluated[-1]
            return last["global_loss"], last["global_accuracy"]
        return 0.0, 0.0

    # ── Combo helpers ──────────────────────────────────────────────────

    def _build_combos(self) -> List[List[tuple]]:
        """
        Cartesian product of all sweep axes.
        Each combo is a list of (dotted.key.path, value) pairs.
        """
        keys   = list(self.sweep.keys())
        values = list(self.sweep.values())
        return [
            list(zip(keys, combo))
            for combo in product(*values)
        ]

    def _apply_combo(self, combo: List[tuple]) -> dict:
        """
        Deep-copy the base config and override keys using
        dot-notation paths (e.g. 'churn.drop_prob' → cfg['churn']['drop_prob']).
        """
        cfg = copy.deepcopy(self.base_cfg)
        for path, value in combo:
            keys = path.split(".")
            obj  = cfg
            for key in keys[:-1]:
                obj = obj[key]
            obj[keys[-1]] = value
        return cfg

    def _combo_to_id(self, combo: List[tuple]) -> str:
        parts = []
        for path, value in combo:
            short_key = path.split(".")[-1]
            parts.append(f"{short_key}={value}")
        return "__".join(parts)

    def _print_combo(self, combo: List[tuple]):
        for path, value in combo:
            print(f"  {path:<35} = {value}")

    # ── Summary ────────────────────────────────────────────────────────

    def _save_summary(self):
        if not self.summary_rows:
            return
        path = self.output_root / "summary.csv"
        fields = list(self.summary_rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.summary_rows)