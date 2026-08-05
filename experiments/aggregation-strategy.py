"""
RQ6: aggregation strategies under churn (Path A — synchronous, absence-aware).

Compares FEDAVG vs STALENESS_AWARE vs ADAPTIVE in the RQ2 worst case (severe
skew, drop_prob=0.3), at low churn (delay=3, strategies differ only by selection
cadence) and high churn (delay=50, where returning rare-class clients matter).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import json, re, glob, os
from collections import defaultdict
import numpy as np
from experiments.runner import ExperimentRunner

EXPERIMENT_NAME = "exp6_aggregation"
STRATEGIES = ["FEDAVG", "STALENESS_AWARE", "ADAPTIVE", "CATCHUP"]


def run_rq6():
    runner = ExperimentRunner(
        base_config_path = "configs/aggregation-strategy.yaml",
        experiment_name  = EXPERIMENT_NAME,
        sweep = {
            "aggregation.strategy"   :  STRATEGIES, #["FEDAVG"],
            "churn.max_rejoin_delay" : [3, 50], # [50],       # low churn vs worst case
            "seed"                   : [1, 2, 3], # [1], 
        },
    )
    runner.run_all()
    exp_dir = f"./outputs/{EXPERIMENT_NAME}"
    plot_strategy_convergence(exp_dir, delay=50)
    plot_strategy_bars(exp_dir)


# ── helpers ────────────────────────────────────────────────────────────

def _parse_rid(rid):
    
    s = re.search(r"strategy=([A-Z]+(?:_[A-Z]+)*)", rid)
    d = re.search(r"max_rejoin_delay=([0-9.]+)", rid)
    sd = re.search(r"seed=([0-9]+)", rid)
    if not (s and d and sd):
        return None
    return s.group(1), float(d.group(1)), int(sd.group(1))


def _spread(vals, band):
    vals = np.asarray(vals, dtype=float); n = len(vals)
    m = float(vals.mean()) if n else float("nan")
    if n < 2: return m, 0.0
    sd = float(vals.std(ddof=1))
    return m, (sd / np.sqrt(n) if band == "sem" else sd)


def _band_suffix(band, n):
    return f"bands: \u00b1{'SEM' if band == 'sem' else 'std'}, n={n}"


# ── convergence per strategy at one delay ────────────────────

def plot_strategy_convergence(exp_dir, delay=50, band="sem", out_name=None):
    by_strat = defaultdict(lambda: defaultdict(list))   # strat -> round -> [acc]
    seeds_seen = defaultdict(set)
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = _parse_rid(os.path.basename(os.path.dirname(h)))
        if p is None: continue
        strat, d, seed = p
        if int(d) != int(delay): continue
        for r in json.load(open(h)):
            if r.get("global_accuracy", 0) > 0:
                by_strat[strat][r["round"]].append(r["global_accuracy"])
        seeds_seen[strat].add(seed)
    if not by_strat:
        print(f"No runs at delay={delay} under {exp_dir}"); return

    plt.figure(figsize=(8, 5))
    n_typ = max((len(s) for s in seeds_seen.values()), default=1)
    for strat in [s for s in STRATEGIES if s in by_strat]:
        rounds = sorted(by_strat[strat])
        means, halfs = zip(*[_spread(by_strat[strat][r], band) for r in rounds])
        rounds = np.array(rounds); means = np.array(means); halfs = np.array(halfs)
        line, = plt.plot(rounds, means, linewidth=2, label=strat)
        plt.fill_between(rounds, means - halfs, means + halfs, alpha=0.18, color=line.get_color())

    plt.xlabel("Communication round"); plt.ylabel("Global accuracy")
    plt.title(f"Aggregation strategies under churn  (delay={int(delay)}, {_band_suffix(band, n_typ)})")
    plt.legend(title="Strategy"); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = os.path.join(exp_dir, out_name or f"rq6_convergence_delay{int(delay)}.png")
    plt.savefig(out, dpi=150); print(f"Saved -> {out}"); plt.show()


# ── final accuracy bars, strategy x delay ────────────────────

def plot_strategy_bars(exp_dir, band="sem", out_name="rq6_strategy_bars.png"):
    finals = defaultdict(list)   # (strat, delay) -> [final acc]
    seeds_seen = set()
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = _parse_rid(os.path.basename(os.path.dirname(h)))
        if p is None: continue
        strat, delay, seed = p
        evald = [r["global_accuracy"] for r in json.load(open(h)) if r.get("global_accuracy", 0) > 0]
        if evald:
            finals[(strat, delay)].append(evald[-1]); seeds_seen.add(seed)
    if not finals:
        print(f"No runs under {exp_dir}"); return

    delays = sorted({d for _, d in finals})
    strats = [s for s in STRATEGIES if any((s, d) in finals for d in delays)]
    x = np.arange(len(delays)); width = 0.8 / max(len(strats), 1)

    plt.figure(figsize=(8, 5))
    for i, strat in enumerate(strats):
        means, errs = [], []
        for d in delays:
            m, hw = _spread(finals.get((strat, d), [np.nan]), band)
            means.append(m); errs.append(hw)
        plt.bar(x + i * width, means, width, yerr=errs, capsize=4, label=strat)

    plt.xticks(x + width * (len(strats) - 1) / 2, [f"delay={int(d)}" for d in delays])
    plt.ylabel("Final global accuracy")
    plt.title(f"Final accuracy by aggregation strategy  ({_band_suffix(band, len(seeds_seen))})")
    plt.legend(title="Strategy"); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    out = os.path.join(exp_dir, out_name); plt.savefig(out, dpi=150)
    print(f"Saved -> {out}"); plt.show()


if __name__ == "__main__":
    exp = f"./outputs/{EXPERIMENT_NAME}"
    args = sys.argv[1:]
    if args and args[0] == "plot":
        band = args[1] if len(args) > 1 and args[1] in ("std", "sem") else "sem"
        plot_strategy_convergence(exp, delay=50, band=band)
        plot_strategy_bars(exp, band=band)
    else:
        run_rq6()