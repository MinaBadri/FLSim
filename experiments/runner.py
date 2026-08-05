import copy
import csv
import json
import time
import numpy as np
import hashlib, warnings
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
    seed_everything,
)
from data.partitioner import load_dataset, get_client_class_distribution


class ExperimentRunner:


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
        self._seen_fingerprints: Dict[str, str] = {}


    def run_all(self):
        combos = self._build_combos()
        total  = len(combos)

        print(f"\n{'='*55}")
        print(f" Experiment : {self.experiment_name}")
        print(f" Runs       : {total}")
        print(f" Output     : {self.output_root}")
        print(f"{'='*55}\n")

    

        for i, combo in enumerate(combos):
            run_id  = self._combo_to_id(combo)
            run_cfg = self._apply_combo(combo)
            run_dir = self.output_root / run_id

            print(f"[{i+1}/{total}] {run_id}")
            self._print_combo(combo)

            t0 = time.time()
            final_loss, final_acc, mean_classes = self._run_one(
                cfg            = run_cfg,
                # client_loaders = client_loaders,
                # test_loader    = test_loader,
                output_dir     = str(run_dir),
                run_id         = run_id,
            )
            elapsed = time.time() - t0

            row = {
                "run_id"      : run_id,
                "final_loss"  : round(final_loss, 6),
                "final_acc"   : round(final_acc,  6),
                "mean_classes_per_client" : round(mean_classes, 2),
                "elapsed_min" : round(elapsed / 60, 2),
                # **{k: v for k, v in combo},
                **{k.split(".")[-1]: v for k, v in combo},
            }
            self.summary_rows.append(row)
            print(f"  → loss={final_loss:.4f}  acc={final_acc:.4f}  "
                  f"time={elapsed/60:.1f}min\n")

        self._save_summary()
        print(f"All runs complete. Summary → {self.output_root}/summary.csv")


    def _run_one(
        self,
        cfg            : dict,
        output_dir     : str,
        run_id         : str,
    ) -> tuple[float, float, float]:
        # client_loaders : list,
        # test_loader,

        seed_everything(cfg.get("seed", 42))
 
        mean_classes = self._log_and_check_heterogeneity(cfg, client_indices, run_id)
   
        hardware_profiles = build_hardware_profiles(cfg)
        fleet             = build_fleet(cfg, client_loaders, hardware_profiles)
        registry          = build_registry(cfg, fleet)
        server            = build_server(cfg, registry, test_loader, output_dir)
 
        history = server.run()
 
       
        evaluated = [h for h in history if h["global_loss"] > 0]
        if evaluated:
            last = evaluated[-1]
            return last["global_loss"], last["global_accuracy"], mean_classes
        return 0.0, 0.0, mean_classes

    def _log_and_check_heterogeneity(
        self,
        cfg            : dict,
        client_indices : List[List[int]],
        run_id         : str,
    ) -> float:
       
        ds   = load_dataset(train=True, dataset=cfg["data"].get("dataset", "cifar100"))
        ncls = cfg["model"]["num_classes"]
        dist = get_client_class_distribution(client_indices, ds, num_classes=ncls)
 
        counts       = dist.sum(axis=1)        
        classes_each = (dist > 0).sum(axis=1) 
        nonempty     = int((counts > 0).sum())
 
        print(f"  [het] non-empty clients : {nonempty}/{len(client_indices)}")
        print(f"  [het] samples/client    : min={int(counts.min())} "
              f"med={int(np.median(counts))} max={int(counts.max())}")
        print(f"  [het] classes/client    : min={int(classes_each.min())} "
              f"med={int(np.median(classes_each))} max={int(classes_each.max())} "
              f"(of {ncls})")
 
        fp = hashlib.md5(
            repr([sorted(ix) for ix in client_indices]).encode()
        ).hexdigest()[:10]
 
        for prev_run, prev_fp in self._seen_fingerprints.items():
            if prev_fp == fp:
                warnings.warn(
                    f"Partition for '{run_id}' is IDENTICAL to '{prev_run}'. "
                    f"The data is not changing across runs -- the sweep is "
                    f"likely misconfigured."
                )
        self._seen_fingerprints[run_id] = fp
        print(f"  [het] partition fp      : {fp}")
 
        return float(classes_each.mean())

    def _build_combos(self) -> List[List[tuple]]:
       
        keys   = list(self.sweep.keys())
        values = list(self.sweep.values())
        return [
            list(zip(keys, combo))
            for combo in product(*values)
        ]

    def _apply_combo(self, combo: List[tuple]) -> dict:
     
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


    def _save_summary(self):
        if not self.summary_rows:
            return
        path = self.output_root / "summary.csv"
        fields = list(self.summary_rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.summary_rows)