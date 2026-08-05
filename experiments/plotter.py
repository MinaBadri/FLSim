import json
import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


class ResultsPlotter:

    STYLE = {
        "FEDAVG"          : dict(color="#2196F3", linestyle="-",  linewidth=1.8),
        "STALENESS_AWARE" : dict(color="#FF9800", linestyle="--", linewidth=1.8),
        "ADAPTIVE"        : dict(color="#4CAF50", linestyle="-.", linewidth=1.8),
    }

    def __init__(self, experiment_dir: str):
        self.exp_dir = Path(experiment_dir)
        self.runs    = self._load_runs()


    def _load_runs(self) -> Dict[str, List[dict]]:
        runs = {}
        for history_file in sorted(self.exp_dir.rglob("history.json")):
            run_id = history_file.parent.name
            with open(history_file) as f:
                runs[run_id] = json.load(f)
        print(f"Loaded {len(runs)} runs from {self.exp_dir}")
        return runs
    

    def rebuild_summary(self):
        
        

        rows = []
        for run_id, history in self.runs.items():
          
            params = {}
            for part in run_id.split("__"):
                if "=" in part:
                    k, v = part.split("=", 1)
                
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
                    params[k] = v

          
            evaluated = [h for h in history if h.get("global_loss", 0) > 0]
            if not evaluated:
                continue
            last = evaluated[-1]

            rows.append({
                "run_id"     : run_id,
                "final_loss" : round(last["global_loss"],    6),
                "final_acc"  : round(last["global_accuracy"], 6),
                **params,
            })

        if not rows:
            print("No valid runs found.")
            return

        path   = self.exp_dir / "summary.csv"
        fields = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Rebuilt summary.csv with {len(rows)} rows → {path}")

    def plot_rounds_to_threshold(
    self,
    threshold   : float = 0.60,
    save_path   : str   = None,):
       
        data = defaultdict(dict)

        for run_id, history in self.runs.items():
            evaluated = [h for h in history if h["global_accuracy"] > 0]
            if not evaluated:
                continue

            params = {}
            for part in run_id.split("__"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v

            strategy  = params.get("strategy", "?")
            drop_prob = float(params.get("drop_prob", 0))

  
            rounds_to_threshold = None
            for h in evaluated:
                if h["global_accuracy"] >= threshold:
                    rounds_to_threshold = h["round"]
                    break

            if rounds_to_threshold is None:
                rounds_to_threshold = 9999  # never reached

            if drop_prob not in data[strategy]:
                data[strategy][drop_prob] = []
            data[strategy][drop_prob].append(rounds_to_threshold)

        strategies = sorted(data.keys())
        drop_probs = sorted({
            float(p.get("drop_prob", 0))
            for run_id in self.runs
            for p in [dict(
                part.split("=", 1)
                for part in run_id.split("__")
                if "=" in part
            )]
        })

        x     = np.arange(len(drop_probs))
        width = 0.22

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, strategy in enumerate(strategies):
            vals  = []
            for dp in drop_probs:
                runs = data[strategy].get(dp, [9999])
                mean = np.mean([v for v in runs if v < 9999]) \
                    if any(v < 9999 for v in runs) else None
                vals.append(mean if mean is not None else 150)

            style = self.STYLE.get(strategy, {})
            bars  = ax.bar(
                x + i * width, vals, width,
                label   = strategy,
                color   = style.get("color", "gray"),
                alpha   = 0.85,
            )

        ax.set_xlabel("Drop Probability per Round",          fontsize=11)
        ax.set_ylabel(f"Rounds to reach {threshold*100:.0f}% accuracy", fontsize=11)
        ax.set_title("Convergence Speed Under Client Churn", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(dp) for dp in drop_probs])
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.invert_yaxis()  # lower bar = faster = better

        plt.tight_layout()
        path = save_path or str(self.exp_dir / "rounds_to_threshold.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.show()
    # ── Figure 1: Convergence curves ──────────────────────────────────

    def plot_convergence(
        self,
        # group_by    : str = "strategy",
        metric      : str = "global_accuracy",
        fix_drop_prob : float = None,
        save_path   : str = None,
    ):
       
        grouped = defaultdict(list)
        for run_id, history in self.runs.items():
            evaluated = [h for h in history if h[metric] > 0]
            if not evaluated:
                continue
            drop_prob = None
            for part in run_id.split("__"):
                if part.startswith("drop_prob="):
                    drop_prob = float(part.split("=")[1])

            # Filter by churn level if requested
            if fix_drop_prob is not None and drop_prob != fix_drop_prob:
                continue

            strategy = evaluated[0].get("strategy", run_id)
            label    = f"{strategy} (drop={drop_prob})" if fix_drop_prob is None else strategy
            grouped[label].append(evaluated)
        if not grouped:
            print(f"No runs found for drop_prob={fix_drop_prob}")
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        for label, run_histories in grouped.items():
            L          = min(len(h) for h in run_histories)
            all_rounds = [h["round"] for h in run_histories[0][:L]]
            stacked    = np.array([[h[metric] for h in hist[:L]] for hist in run_histories])
            mean_vals  = stacked.mean(axis=0)
            std_vals   = stacked.std(axis=0)

            strategy_key = label.split(" ")[0]
            style = self.STYLE.get(strategy_key, dict(color="gray", linewidth=1.5))
            ax.plot(all_rounds, mean_vals, label=label, **style)
            if stacked.shape[0] > 1:
                ax.fill_between(all_rounds, mean_vals - std_vals, mean_vals + std_vals,
                                color=style.get("color", "gray"), alpha=0.18, linewidth=0)

        y_label = "Accuracy" if "acc" in metric else "Loss"
        title   = f"Convergence — drop_prob={fix_drop_prob}" \
              if fix_drop_prob is not None else "Convergence Under Client Churn"

        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(f"Global {y_label}",   fontsize=11)
        ax.set_title(title,                  fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        plt.tight_layout()
        suffix    = f"_drop{fix_drop_prob}" if fix_drop_prob is not None else ""
        path      = save_path or str(self.exp_dir / f"convergence_{metric}{suffix}.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.show()

    # ── Figure 2: Churn rate vs final accuracy ─────────────────────────

    def plot_churn_vs_accuracy(self, save_path: str = None):
     
        summary_path = self.exp_dir / "summary.csv"
        if not summary_path.exists():
            print("summary.csv not found — run experiments first.")
            return

        rows = []
        with open(summary_path) as f:
            rows = list(csv.DictReader(f))

     

        data = defaultdict(dict)
        for row in rows:
            strategy  = row.get("strategy", "?")
            drop_prob = float(row.get("drop_prob", 0))
            acc       = float(row["final_acc"])
            data[strategy][drop_prob] = acc

        strategies  = sorted(data.keys())
        drop_probs  = sorted({float(r.get("drop_prob", 0)) for r in rows})
        x           = np.arange(len(drop_probs))
        width       = 0.22

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, strategy in enumerate(strategies):
            vals  = [data[strategy].get(dp, 0) for dp in drop_probs]
            style = self.STYLE.get(strategy, {})
            ax.bar(
                x + i * width, vals, width,
                label=strategy,
                color=style.get("color", "gray"),
                alpha=0.85,
            )

        ax.set_xlabel("Drop Probability per Round", fontsize=11)
        ax.set_ylabel("Final Global Accuracy",      fontsize=11)
        ax.set_title("Effect of Churn Rate on Final Accuracy", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(dp) for dp in drop_probs])
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = save_path or str(self.exp_dir / "churn_vs_accuracy.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.show()

    # ── Figure 3: Client pool dynamics ────────────────────────────────

    def plot_pool_dynamics(self, run_id: str = None, save_path: str = None):
        
        if run_id is None:
            run_id = list(self.runs.keys())[0]

        history = self.runs[run_id]
        rounds    = [h["round"]          for h in history]
        active    = [h["active_pool"]    for h in history]
        dropped   = [h["dropped_pool"]   for h in history]
        rejoining = [h["rejoining_pool"] for h in history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.stackplot(
            rounds,
            active, rejoining, dropped,
            labels=["Active", "Rejoining", "Dropped"],
            colors=["#4CAF50", "#FF9800", "#F44336"],
            alpha=0.75,
        )
        ax.set_xlabel("Round",          fontsize=11)
        ax.set_ylabel("Client Count",   fontsize=11)
        ax.set_title(f"Client Pool Dynamics — {run_id}", fontsize=12)
        ax.legend(loc="upper right",    fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path = save_path or str(self.exp_dir / f"pool_dynamics_{run_id}.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.show()
    
    def plot_convergence_by_param(
        self,
        param_key : str,
        metric    : str = "global_accuracy",
        save_path : str = None,
    ):

        grouped = {}

        for run_id, history in self.runs.items():
            evaluated = [h for h in history if h.get(metric, 0) > 0]
            if not evaluated:
                continue

            param_val = None
            for part in run_id.split("__"):
                if part.startswith(f"{param_key}="):
                    param_val = part.split("=", 1)[1]
                    break

            if param_val is None:
                param_val = run_id

            if param_val not in grouped:
                grouped[param_val] = []
            grouped[param_val].append(evaluated)

        if not grouped:
            print(f"No runs found for param {param_key}")
            return

        def try_float(v):
            try:
                return float(v)
            except ValueError:
                return v

        sorted_keys = sorted(grouped.keys(), key=try_float)

   
        
        colors = cm.viridis(np.linspace(0.1, 0.9, len(sorted_keys)))

        fig, ax = plt.subplots(figsize=(9, 5))

        for color, key in zip(colors, sorted_keys):
            run_histories = grouped[key]
            L          = min(len(h) for h in run_histories)   # align seeds
            all_rounds = [h["round"] for h in run_histories[0][:L]]
            stacked    = np.array(
                [[h[metric] for h in hist[:L]] for hist in run_histories]
            )                                                 # (n_seeds, L)
            mean_vals = stacked.mean(axis=0)
            std_vals  = stacked.std(axis=0)
            n_seeds   = stacked.shape[0]

            ax.plot(all_rounds, mean_vals,
                    label     = f"{param_key}={key}" + (f" (n={n_seeds})" if n_seeds > 1 else ""),
                    color     = color, linewidth = 1.8)
            if n_seeds > 1:
                ax.fill_between(all_rounds, mean_vals - std_vals, mean_vals + std_vals,
                                color=color, alpha=0.18, linewidth=0)

        y_label = "Accuracy" if "acc" in metric else "Loss"
        ax.set_xlabel("Communication Round",   fontsize=11)
        ax.set_ylabel(f"Global {y_label}",     fontsize=11)
        ax.set_title(f"Convergence by {param_key}", fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        plt.tight_layout()
        path = save_path or str(
            self.exp_dir / f"convergence_{metric}_by_{param_key}.pdf"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.show()


    def plot_param_vs_accuracy(
        self,
        param_key   : str,
        param_label : str = None,
        title       : str = "Effect of Parameter on Final Accuracy",
        save_path   : str = None,
    ):
       
        summary_path = self.exp_dir / "summary.csv"
        if not summary_path.exists():
            print("summary.csv not found — calling rebuild_summary first")
            self.rebuild_summary()

        rows = []
        with open(summary_path) as f:
            rows = list(csv.DictReader(f))

        if not rows:
            print("No rows in summary.csv")
            return


        # Extract param values and accuracies
        data = {}
        for row in rows:
            val = row.get(param_key)
            if val is None:
                continue
            try:
                val = float(val)
            except ValueError:
                pass
            acc = float(row["final_acc"])
            if val not in data:
                data[val] = []
            data[val].append(acc)

        if not data:
            print(f"param_key '{param_key}' not found in summary.csv")
            return

        sorted_vals = sorted(data.keys(), key=lambda x: float(x) if isinstance(x, str) else x)
        means       = [np.mean(data[v]) for v in sorted_vals]
        # stds        = [np.std(data[v])  for v in sorted_vals]
        stds        = [np.std(data[v], ddof=1) if len(data[v]) > 1 else 0.0  for v in sorted_vals]
        n_seeds     = max(len(data[v]) for v in sorted_vals)
        x           = np.arange(len(sorted_vals))

        colors = cm.viridis(np.linspace(0.1, 0.9, len(sorted_vals)))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(
            x, means,
            yerr     = stds if n_seeds > 1 else None,
            capsize  = 4,
            color    = colors, alpha=0.85, width=0.6,
            error_kw = dict(ecolor="#333333", lw=1.2),
        )

        for bar, mean, std in zip(bars, means, stds):
            label = f"{mean:.3f}" + (f"±{std:.3f}" if n_seeds > 1 else "")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (std if n_seeds > 1 else 0) + 0.004,
                label, ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel(param_label or param_key, fontsize=11)
        ax.set_ylabel("Final Global Accuracy",  fontsize=11)
        ax.set_title(title,                     fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in sorted_vals])
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = save_path or str(self.exp_dir / f"{param_key}_vs_accuracy.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.show()

        
    def plot_convergence_speed_by_param(
        self,
        param_key : str,
        threshold : float = 0.15,
        save_path : str   = None,
    ):
    

        data = {}
        for run_id, history in self.runs.items():
            evaluated = [h for h in history if h.get("global_accuracy", 0) > 0]
            if not evaluated:
                continue

            param_val = None
            for part in run_id.split("__"):
                if part.startswith(f"{param_key}="):
                    param_val = part.split("=", 1)[1]
                    break
            if param_val is None:
                continue

            rounds_needed = next(
                (h["round"] for h in evaluated if h["global_accuracy"] >= threshold),
                999
            )
            if param_val not in data:
                data[param_val] = []
            data[param_val].append(rounds_needed)

        sorted_vals = sorted(data.keys(), key=lambda x: float(x))
        means, stds, labels = [], [], []
        for v in sorted_vals:
            vals    = np.array(data[v], dtype=float)
            reached = vals[vals < 999]
            if len(reached) == 0:
                means.append(999); stds.append(0.0); labels.append("Never")
            else:
                m = reached.mean()
                means.append(m)
                stds.append(reached.std() if len(reached) > 1 else 0.0)
                frac = f" ({len(reached)}/{len(vals)})" if len(reached) < len(vals) else ""
                labels.append(f"{int(round(m))}{frac}")
        x      = np.arange(len(sorted_vals))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_vals)))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, width=0.6,
                      error_kw=dict(ecolor="#333333", lw=1.2))

        for bar, mean, std, label in zip(bars, means, stds, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.5,
                label, ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel(param_key,                                    fontsize=11)
        ax.set_ylabel(f"Rounds to reach {threshold*100:.0f}% acc", fontsize=11)
        ax.set_title(f"Convergence Speed by {param_key}",          fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in sorted_vals])
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = save_path or str(
            self.exp_dir / f"convergence_speed_{param_key}.pdf"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.show()