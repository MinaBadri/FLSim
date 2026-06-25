"""
RQ2 long-horizon check: does the severe-skew long-absence damage RECOVER given a
large round budget, or PLATEAU below the low-churn baseline?

We rerun churn at delay=3 (near-baseline) and delay=50 (damaged) under severe
skew for 200 rounds (vs the 80 used in the main sweep) and ask whether the
delay=50 curve catches the delay=3 curve. If the gap keeps shrinking -> slow
recovery; if it flattens after the absent clients have had ample time back in
the pool -> not recovered ("permanent" within a large budget).

Add alpha=0.3 to DATA_ALPHAS for the mild-skew recovery contrast.

Usage:
  python experiments/run_exp2_longhorizon.py          # run, then plot
  python experiments/run_exp2_longhorizon.py plot      # replot (SEM)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import json, re, glob, os
from collections import defaultdict
import numpy as np
from experiments.runner import ExperimentRunner

EXPERIMENT  = "exp2_longhorizon"
BASE        = "configs/exp2_longhorizon.yaml"
DATA_ALPHAS = [0.05]          # add 0.3 for the mild-skew recovery contrast
DELAYS      = [3, 50]
SEEDS       = [1, 2, 3]


def run():
    runner = ExperimentRunner(
        base_config_path=BASE, experiment_name=EXPERIMENT,
        sweep={"data.dirichlet_alpha": DATA_ALPHAS,
               "churn.max_rejoin_delay": DELAYS, "seed": SEEDS},
    )
    runner.run_all()
    plot_longhorizon()


def _parse(rid):
    a = re.search(r"dirichlet_alpha=([0-9.]+)", rid)
    d = re.search(r"max_rejoin_delay=([0-9.]+)", rid)
    s = re.search(r"seed=([0-9]+)", rid)
    return (float(a.group(1)), float(d.group(1)), int(s.group(1))) if (a and d and s) else None


def _sem(x):
    x = np.asarray(x, float)
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (float(x.mean()), 0.0)


def _curves(exp_dir):
    # (alpha, delay) -> round -> [acc across seeds]
    by = defaultdict(lambda: defaultdict(list))
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = _parse(os.path.basename(os.path.dirname(h)))
        if not p: continue
        a, d, _ = p
        for r in json.load(open(h)):
            if r.get("global_accuracy", 0) > 0:
                by[(a, d)][r["round"]].append(r["global_accuracy"])
    return by


def plot_longhorizon(band="sem"):
    exp_dir = f"./outputs/{EXPERIMENT}"
    by = _curves(exp_dir)
    if not by:
        print(f"No runs under {exp_dir}"); return
    alphas = sorted({a for a, _ in by})

    # Figure 1: convergence, delay 3 vs 50, one panel per alpha
    fig, axes = plt.subplots(1, len(alphas), figsize=(7*len(alphas), 5), squeeze=False)
    for ax, a in zip(axes[0], alphas):
        for d in sorted({dd for aa, dd in by if aa == a}):
            rounds = sorted(by[(a, d)])
            m, e = zip(*[_sem(by[(a, d)][r]) for r in rounds])
            rounds, m, e = np.array(rounds), np.array(m), np.array(e)
            line, = ax.plot(rounds, m, lw=2, label=f"delay={int(d)}")
            ax.fill_between(rounds, m-e, m+e, alpha=0.18, color=line.get_color())
        ax.set_title(f"alpha={a:g}"); ax.set_xlabel("Communication round")
        ax.set_ylabel("Global accuracy"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("RQ2 long-horizon: does delay=50 catch delay=3?")
    fig.tight_layout()
    out1 = os.path.join(exp_dir, "rq2_longhorizon_convergence.png")
    fig.savefig(out1, dpi=150); print(f"Saved -> {out1}"); plt.close(fig)

    # Figure 2: gap (delay3 - delay50) vs round per alpha -> plateau vs closing
    plt.figure(figsize=(8, 5))
    for a in alphas:
        if (a, 3.0) not in by or (a, 50.0) not in by: continue
        rounds = sorted(set(by[(a, 3.0)]) & set(by[(a, 50.0)]))
        gap = [np.mean(by[(a, 3.0)][r]) - np.mean(by[(a, 50.0)][r]) for r in rounds]
        plt.plot(rounds, gap, lw=2, marker="o", ms=3, label=f"alpha={a:g}")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Communication round")
    plt.ylabel("Accuracy gap:  delay=3  \u2212  delay=50")
    plt.title("RQ2 long-horizon: recovery (gap \u2192 0) vs plateau (gap flat)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    out2 = os.path.join(exp_dir, "rq2_longhorizon_gap.png")
    plt.savefig(out2, dpi=150); print(f"Saved -> {out2}"); plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_longhorizon()
    else:
        run()