"""
RQ5: sync vs semi-async vs async under increasing compute-speed heterogeneity,
at an equal wall-clock (virtual-time) budget.

The synchronization mode is really one knob -- the buffer size M, i.e. how many
arrivals you wait for before applying:
    async = M=1  ...  semi = M in {2,4,8}  ...  sync = M=C (full round barrier).
We sweep M so "semi" is placed fairly rather than judged at a single buffer size,
and so we can read accuracy vs M (the async<->sync spectrum) at each spread.

Bonus (Option 3): async with staleness-aware weighting ON vs OFF -- does the
weighting that was inert in RQ6's synchronous setting help now staleness is live?

Usage:
  python experiments/run_exp5_async.py            # run sweep + bonus, then plot
  python experiments/run_exp5_async.py plot       # replot from saved records
  python experiments/run_exp5_async.py smoke      # 1 quick wiring check
"""
import sys, os, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from async_fl import run_async_fl
from utils import load_config


EXPERIMENT  = "exp5_async"
BASE        = "configs/exp5_async.yaml"
# (label, orchestrator_mode, buffer_size M).  M is the async<->sync dial.
RUNS = [
    ("async",  "async", 1),
    ("semi2",  "semi",  2),
    ("semi4",  "semi",  4),
    ("semi8",  "semi",  8),
    ("sync",   "sync",  10),
]
SPREADS     = [0.0, 0.3, 0.6, 0.9]     # reduced first pass; expand to [0.0,0.3,0.6,0.9] for full
SEEDS       = [1, 2]                 # expand to [1, 2] for error bars
TIME_BUDGET = 80.0
OUT         = f"./outputs/{EXPERIMENT}"
ORDER       = [r[0] for r in RUNS]


def _save(records):
    os.makedirs(OUT, exist_ok=True)
    json.dump(records, open(f"{OUT}/rq5_records.json", "w"))


def run():
    cfg = load_config(BASE)
    records = []
    for (label, mode, M) in RUNS:
        for spread in SPREADS:
            for seed in SEEDS:
                print(f"\n>>> RUN {label} (mode={mode} M={M}) spread={spread} seed={seed}", flush=True)
                res, _ = run_async_fl(cfg, mode, M, spread, seed,
                                      time_budget=TIME_BUDGET, staleness_a=0.5)
                h = res["history"]
                records.append(dict(label=label, mode=mode, buffer_size=M, spread=spread,
                                    seed=seed, staleness_a=0.5, final_acc=h[-1]["acc"],
                                    applied=res["applied"], end_t=res["end_t"],
                                    avg_staleness=float(np.mean([x["avg_staleness"] for x in h])),
                                    history=h))
                _save(records)
                print(f"    -> final={h[-1]['acc']:.4f} applied={res['applied']} "
                      f"avg_stale={records[-1]['avg_staleness']:.1f}", flush=True)

    # bonus: async staleness-aware OFF (a=0) across spreads
    for spread in SPREADS:
        for seed in SEEDS:
            print(f"\n>>> RUN async_nostale spread={spread} seed={seed}", flush=True)
            res, _ = run_async_fl(cfg, "async", 1, spread, seed,
                                  time_budget=TIME_BUDGET, staleness_a=0.0)
            h = res["history"]
            records.append(dict(label="async_nostale", mode="async", buffer_size=1, spread=spread,
                                seed=seed, staleness_a=0.0, final_acc=h[-1]["acc"],
                                applied=res["applied"], end_t=res["end_t"],
                                avg_staleness=float(np.mean([x["avg_staleness"] for x in h])),
                                history=h))
            _save(records)
            print(f"    -> final={h[-1]['acc']:.4f}", flush=True)
    plot_all()


def _sem(x):
    x = np.asarray(x, float)
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (float(x.mean()), 0.0)


def plot_all():
    records = json.load(open(f"{OUT}/rq5_records.json"))
    spreads = sorted({r["spread"] for r in records})
    labels  = [l for l in ORDER if any(r["label"] == l for r in records)]

    def series(label):
        ms, es = [], []
        for sp in spreads:
            vals = [r["final_acc"] for r in records if r["label"] == label and r["spread"] == sp]
            m, e = _sem(vals) if vals else (np.nan, 0.0); ms.append(m); es.append(e)
        return np.array(ms), np.array(es)

    # Figure 1: final accuracy vs speed spread, one line per mode/buffer
    plt.figure(figsize=(8.5, 5))
    for label in labels:
        m, e = series(label)
        plt.errorbar(spreads, m, yerr=e, marker="o", capsize=4, lw=2, label=label)
    plt.xlabel("Compute-speed heterogeneity (lognormal spread; 0 = homogeneous)")
    plt.ylabel("Final global accuracy (equal wall-clock budget)")
    plt.title("RQ5: sync vs semi(M=2,4,8) vs async as speed heterogeneity grows")
    plt.legend(title="mode (buffer M)"); plt.grid(True, alpha=0.3); plt.tight_layout()
    p1 = f"{OUT}/rq5_acc_vs_spread.png"; plt.savefig(p1, dpi=150); print("Saved ->", p1); plt.close()

    # Figure 2 (NEW): accuracy vs buffer size M -- the async<->sync spectrum, one line per spread
    plt.figure(figsize=(8.5, 5))
    Mvals = sorted({r["buffer_size"] for r in records if r["label"] != "async_nostale"})
    for sp in spreads:
        ms, es = [], []
        for M in Mvals:
            vals = [r["final_acc"] for r in records
                    if r["buffer_size"] == M and r["spread"] == sp and r["label"] != "async_nostale"]
            m, e = _sem(vals) if vals else (np.nan, 0.0); ms.append(m); es.append(e)
        plt.errorbar(Mvals, ms, yerr=es, marker="o", capsize=4, lw=2, label=f"spread={sp}")
    plt.xscale("log", base=2); plt.xticks(Mvals, [str(m) for m in Mvals])
    plt.xlabel("Buffer size M   (1 = async  \u2192  10 = sync)")
    plt.ylabel("Final global accuracy")
    plt.title("RQ5: accuracy across the async\u2013sync spectrum, by speed heterogeneity")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p2 = f"{OUT}/rq5_acc_vs_buffer.png"; plt.savefig(p2, dpi=150); print("Saved ->", p2); plt.close()

    # Figure 3: throughput + staleness vs spread
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label in labels:
        ap = [np.mean([r["applied"] for r in records if r["label"] == label and r["spread"] == sp]) for sp in spreads]
        st = [np.mean([r["avg_staleness"] for r in records if r["label"] == label and r["spread"] == sp]) for sp in spreads]
        a1.plot(spreads, ap, marker="o", lw=2, label=label)
        a2.plot(spreads, st, marker="o", lw=2, label=label)
    a1.set_xlabel("speed spread"); a1.set_ylabel("global updates applied"); a1.set_title("Throughput"); a1.grid(True, alpha=0.3); a1.legend()
    a2.set_xlabel("speed spread"); a2.set_ylabel("avg staleness"); a2.set_title("Staleness cost"); a2.grid(True, alpha=0.3); a2.legend()
    fig.tight_layout(); p3 = f"{OUT}/rq5_throughput_staleness.png"; fig.savefig(p3, dpi=150); print("Saved ->", p3); plt.close(fig)

    # Figure 4: staleness-aware ON vs OFF for async (bonus)
    if any(r["label"] == "async_nostale" for r in records):
        plt.figure(figsize=(8, 5))
        for label, lbl in [("async", "async (staleness-aware)"), ("async_nostale", "async (no weighting)")]:
            m, e = series(label)
            plt.errorbar(spreads, m, yerr=e, marker="o", capsize=4, lw=2, label=lbl)
        plt.xlabel("Compute-speed heterogeneity"); plt.ylabel("Final global accuracy")
        plt.title("RQ5 bonus: does staleness-aware weighting help once staleness is live?")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        p4 = f"{OUT}/rq5_staleness_aware.png"; plt.savefig(p4, dpi=150); print("Saved ->", p4); plt.close()


def smoke():
    cfg = load_config(BASE)
    res, _ = run_async_fl(cfg, "sync", 10, 0.0, 1, time_budget=4.0, eval_dt=2.0)
    print("smoke OK:", res["mode"], "applied=", res["applied"], "final_acc=", res["history"][-1]["acc"])


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "run"
    if arg == "plot":   plot_all()
    elif arg == "smoke": smoke()
    else:               run()