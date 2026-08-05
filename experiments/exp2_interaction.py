"""
churn lateness x data heterogeneity.

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import json, re, glob, os
from collections import defaultdict
import numpy as np
from experiments.runner import ExperimentRunner



FIXED_DROP_PROB = 0.3      
FIXED_MIN_DELAY = 1
EXPERIMENT_NAME = "exp2_churn_x_alpha"


def run_interaction(fast: bool = False, seeds: list = None):
    
    delays = [3, 50] if fast else [3, 10, 25, 50]  
    if seeds is None:
        seeds  = [1]  if fast else [1, 2, 3]
    rounds = 30      if fast else 60                 

    runner = ExperimentRunner(
        base_config_path = "configs/exp2_base.yaml",
        experiment_name  = EXPERIMENT_NAME,
        sweep = {
            "data.dirichlet_alpha"   : [0.05, 0.3],
            "churn.max_rejoin_delay" : delays,
            "seed"                   : seeds,
        },
    )
    # Hold the drop rate + min delay fixed across the whole grid.
    runner.base_cfg["churn"]["drop_prob"]        = FIXED_DROP_PROB
    runner.base_cfg["churn"]["min_rejoin_delay"] = FIXED_MIN_DELAY
    runner.base_cfg["simulation"]["num_rounds"]  = rounds
    runner.run_all()


    exp_dir = f"./outputs/{EXPERIMENT_NAME}"
    plot_interaction(exp_dir)
    plot_retention(exp_dir)
    plot_convergence_at_delay(exp_dir, delay=max(delays))

# ── shared helpers ─────────────────────────────────────────────────────

def _parse_rid(rid: str):
    """Return (alpha, delay, seed) from a run-id, or None if it doesn't match."""
    a = re.search(r"dirichlet_alpha=([0-9.]+)", rid)
    d = re.search(r"max_rejoin_delay=([0-9.]+)", rid)
    s = re.search(r"seed=([0-9]+)", rid)
    if not (a and d and s):
        return None
    return float(a.group(1)), float(d.group(1)), int(s.group(1))


def _alpha_label(alpha: float) -> str:
    return f"alpha={alpha}  ({'severe skew' if alpha <= 0.1 else 'mild skew'})"

def _spread(vals, band):
    """Return (mean, half_band). band='std' -> sample std; 'sem' -> std/sqrt(n)."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    m = float(vals.mean()) if n else float("nan")
    if n < 2:
        return m, 0.0
    s = float(vals.std(ddof=1))
    if band == "sem":
        s = s / np.sqrt(n)
    return m, s


def _band_suffix(band, n):
    return f"bands: \u00b1{'SEM' if band == 'sem' else 'std'}, n={n}"

def _load_finals(exp_dir: str):
    """(alpha, delay, seed) -> final eval accuracy."""
    finals = {}
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = _parse_rid(os.path.basename(os.path.dirname(h)))
        if p is None:
            continue
        hist  = json.load(open(h))
        evald = [r["global_accuracy"] for r in hist if r.get("global_accuracy", 0) > 0]
        if evald:
            finals[p] = evald[-1]
    return finals

# ── absolute accuracy vs delay ───────────────────────────────
def plot_interaction(exp_dir: str,band="std", out_name= "interaction_lateness.png"):
    finals = _load_finals(exp_dir)
    if not finals:
        print(f"No runs found under {exp_dir}"); return
    by_ad = defaultdict(list)
    for (alpha, delay, _), acc in finals.items():
        by_ad[(alpha, delay)].append(acc)

    alphas = sorted({a for a, _ in by_ad})
    delays = sorted({d for _, d in by_ad})
    n_typ  = max(len(v) for v in by_ad.values())
    # print(f"Loaded {len(finals)} runs across alpha={alphas} delays={delays}")

    plt.figure(figsize=(8, 5))
    for alpha in alphas:
        xs, means, halfs = [], [], []
        for delay in delays:
            vals = by_ad.get((alpha, delay), [])
            if not vals: continue
            m, hw = _spread(vals, band)
            xs.append(delay); means.append(m); halfs.append(hw)
        xs, means, halfs = map(np.array, (xs, means, halfs))
        line, = plt.plot(xs, means, marker="o", linewidth=2, label=_alpha_label(alpha))
        plt.fill_between(xs, means - halfs, means + halfs, alpha=0.18, color=line.get_color())

    plt.xlabel("Max rejoin delay  (rounds a dropped client stays gone)")
    plt.ylabel("Final global accuracy")
    plt.title(f"Churn lateness x heterogeneity   ({_band_suffix(band, n_typ)})")
    plt.legend(title="Data heterogeneity"); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = os.path.join(exp_dir, out_name); plt.savefig(out, dpi=150)
    print(f"Saved -> {out}"); plt.show()

# ── figure 2: normalized retention vs delay ────────────────────────────

def plot_retention(exp_dir: str, band="std", out_name: str = "interaction_retention.png"):
    finals = _load_finals(exp_dir)
    if not finals:
        print(f"No runs found under {exp_dir}"); return
    alphas = sorted({a for a, _, _ in finals})
    delays = sorted({d for _, d, _ in finals})
    seeds  = sorted({s for _, _, s in finals})
    base_delay = delays[0]

    plt.figure(figsize=(8, 5))
    n_typ = 1
    for alpha in alphas:
        xs, means, halfs = [], [], []
        for delay in delays:
            rets = []
            for seed in seeds:
                num = finals.get((alpha, delay, seed))
                den = finals.get((alpha, base_delay, seed))
                if num is not None and den:
                    rets.append(num / den)
            if not rets: continue
            n_typ = max(n_typ, len(rets))
            m, hw = _spread(rets, band)
            xs.append(delay); means.append(m); halfs.append(hw)
        xs, means, halfs = map(np.array, (xs, means, halfs))
        line, = plt.plot(xs, means, marker="o", linewidth=2, label=_alpha_label(alpha))
        plt.fill_between(xs, means - halfs, means + halfs, alpha=0.18, color=line.get_color())

    plt.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    plt.xlabel("Max rejoin delay  (rounds a dropped client stays gone)")
    plt.ylabel(f"Accuracy retained  (fraction of delay={int(base_delay)} baseline)")
    plt.title("Churn robustness: accuracy retained vs rejoin lateness")
    plt.legend(title="Data heterogeneity"); plt.grid(True, alpha=0.3)
    plt.ylim(top=1.05); plt.tight_layout()
    out = os.path.join(exp_dir, out_name); plt.savefig(out, dpi=150)
    print(f"Saved -> {out}"); plt.show()


# ── figure 3: convergence overlay at one delay ─────────────────────────

def plot_convergence_at_delay(exp_dir: str, delay: int = 50, band="std", out_name: str = None):
    """
    Accuracy-vs-round for both alphas at a single (long) delay, averaged over
    seeds. Shows the mechanism: under mild skew the curve keeps climbing and
    largely recovers; under severe skew it stalls because absent classes are
    never trained.
    """
    by_alpha = defaultdict(lambda: defaultdict(list))
    seeds_seen = defaultdict(set)   
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = _parse_rid(os.path.basename(os.path.dirname(h)))
        if p is None: continue
        alpha, d, seed = p
        if int(d) != int(delay): continue
        hist   = json.load(open(h))
        contributed = False
        for r in hist:
            acc = r.get("global_accuracy", 0)
            if acc > 0:
                by_alpha[alpha][r["round"]].append(acc)
                contributed = True
        if contributed:
            seeds_seen[alpha].add(seed)

    if not by_alpha:
        print(f"No runs at delay={delay} under {exp_dir}"); return

    plt.figure(figsize=(8, 5))
    n_typ = max((len(s) for s in seeds_seen.values()), default=1)
    for alpha in sorted(by_alpha):
        rounds = sorted(by_alpha[alpha])
        means, halfs = [], []
        for r in rounds:
            m, hw = _spread(by_alpha[alpha][r], band)
            means.append(m); halfs.append(hw)
        rounds = np.array(rounds); means = np.array(means); halfs = np.array(halfs)
        line, = plt.plot(rounds, means, linewidth=2, label=_alpha_label(alpha))
        plt.fill_between(rounds, means - halfs, means + halfs, alpha=0.18, color=line.get_color())

    plt.xlabel("Communication round"); plt.ylabel("Global accuracy")
    plt.title(f"Convergence under long absence  (delay={int(delay)}, {_band_suffix(band, n_typ)})")
    plt.legend(title="Data heterogeneity"); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = os.path.join(exp_dir, out_name or f"convergence_delay{int(delay)}.png")
    plt.savefig(out, dpi=150); print(f"Saved -> {out}"); plt.show()



if __name__ == "__main__":
    exp = f"./outputs/{EXPERIMENT_NAME}"
    args = sys.argv[1:]
    if args and args[0] == "plot":
        band = args[1] if len(args) > 1 and args[1] in ("std", "sem") else "std"
        plot_interaction(exp, band=band)
        plot_retention(exp, band=band)
        plot_convergence_at_delay(exp, delay=50, band=band)
    elif args and args[0] == "addseeds":
        extra = [int(s) for s in args[1:]] or [4, 5]
        print(f"Adding seeds {extra} to {EXPERIMENT_NAME} (existing seeds untouched)")
        run_interaction(fast=False, seeds=extra)
    else:
        run_interaction(fast=False)