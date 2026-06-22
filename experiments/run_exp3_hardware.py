"""
RQ3: hardware constraints on local learning (synchronous, churn-free).

Channels (vary one, hold others ideal, drop_prob=0, alpha=0.3 unless noted):
  SPEED         exact null (compute_speed only feeds a logged delay).
  MEMORY        memory_cap clamps batch (effective=min(req,cap)).
  RELIABILITY   fault prob = 1-R, not backfilled => thins contributor-rounds.
  MEMORY_SEVERE memory sweep rerun at alpha=0.05 (exp3_hardware_severe.yaml).
  LR_BASELINE   CONTROL: unconstrained batch=64 (the base config) swept over
                learning rate. Tests whether the batch-64 baseline is merely
                under-tuned: if a higher LR lifts it to the batch-32 sweet-spot,
                the memory effect was a tuning artifact; if it can't catch up,
                the more-updates effect is real beyond LR. See plot_lr_baseline().

Usage:
  python experiments/run_exp3_hardware.py                 # speed/memory/reliability + plot
  python experiments/run_exp3_hardware.py memory          # one channel
  python experiments/run_exp3_hardware.py memory_severe   # alpha=0.05 follow-up
  python experiments/run_exp3_hardware.py lr_baseline     # the LR under-tuning control
  python experiments/run_exp3_hardware.py compare         # replot both comparison figures
  python experiments/run_exp3_hardware.py plot [std|sem]  # replot everything
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import json, re, glob, os
from collections import defaultdict
import numpy as np
from experiments.runner import ExperimentRunner

BASE        = "configs/exp3_hardware.yaml"          # mild skew (alpha=0.3)
BASE_SEVERE = "configs/exp3_hardware_severe.yaml"   # severe skew (alpha=0.05)

CHANNELS = {
    "speed": dict(
        exp="exp3_speed", key="hardware.speed",
        values=[0.5, 1.0, 2.0], seeds=[1, 2], base=BASE,
        label="Compute speed", fmt=lambda v: f"speed={v:g}",
    ),
    "memory": dict(
        exp="exp3_memory", key="hardware.memory_cap",
        values=[64, 32, 16, 8], seeds=[1, 2, 3], base=BASE,
        label="Effective batch size (memory cap)", fmt=lambda v: f"batch={int(v)}",
    ),
    "reliability": dict(
        exp="exp3_reliability", key="hardware.reliability",
        values=[1.0, 0.9, 0.7, 0.5], seeds=[1, 2, 3], base=BASE,
        label="Reliability (1 - fault prob)", fmt=lambda v: f"R={v:g}",
    ),
    "memory_severe": dict(
        exp="exp3_memory_severe", key="hardware.memory_cap",
        values=[64, 32, 16, 8], seeds=[1, 2, 3], base=BASE_SEVERE,
        label="Effective batch size (memory cap)", fmt=lambda v: f"batch={int(v)}",
    ),
    # CONTROL: batch=64 (base config, unconstrained) swept over learning rate.
    "lr_baseline": dict(
        exp="exp3_lr_baseline", key="training.learning_rate",
        values=[0.01, 0.02, 0.04], seeds=[1, 2, 3], base=BASE,
        label="Learning rate (batch=64 baseline)", fmt=lambda v: f"lr={v:g}",
    ),
}


def _run_channel(name):
    c = CHANNELS[name]
    runner = ExperimentRunner(
        base_config_path = c.get("base", BASE),
        experiment_name  = c["exp"],
        sweep = { c["key"]: c["values"], "seed": c["seeds"] },
    )
    runner.run_all()
    plot_channel(name)
    if name == "memory_severe":
        plot_memory_skew()
    if name == "lr_baseline":
        plot_lr_baseline()


def _add_seeds(name, extra_seeds):
    """Append extra seeds to an ALREADY-RUN channel (same experiment dir).
    New seeds create fresh run folders alongside the existing ones; the plots
    glob every history.json, so they automatically become n=(old+new)."""
    c = CHANNELS[name]
    print(f"Appending seeds {list(extra_seeds)} to ./outputs/{c['exp']} ...")
    runner = ExperimentRunner(
        base_config_path = c.get("base", BASE),
        experiment_name  = c["exp"],          # SAME dir -> appends, never overwrites 1-3
        sweep = { c["key"]: c["values"], "seed": list(extra_seeds) },
    )
    runner.run_all()
    plot_channel(name)
    if name in ("memory", "memory_severe"): plot_memory_skew()
    if name == "lr_baseline":               plot_lr_baseline()


# ── parsing / stats ────────────────────────────────────────────────────

def _short_key(full_key):
    return full_key.split(".")[-1]


def _parse(rid, short_key):
    v  = re.search(rf"{re.escape(short_key)}=([0-9.]+)", rid)
    sd = re.search(r"seed=([0-9]+)", rid)
    if not (v and sd):
        return None
    return float(v.group(1)), int(sd.group(1))


def _spread(vals, band):
    vals = np.asarray(vals, float); n = len(vals)
    m = float(vals.mean()) if n else float("nan")
    if n < 2: return m, 0.0
    sd = float(vals.std(ddof=1))
    return m, (sd / np.sqrt(n) if band == "sem" else sd)


def _finals_by_value(exp_dir, short_key):
    finals = defaultdict(list)
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = _parse(os.path.basename(os.path.dirname(h)), short_key)
        if p is None: continue
        val, _ = p
        evald = [r["global_accuracy"] for r in json.load(open(h))
                 if r.get("global_accuracy", 0) > 0]
        if evald:
            finals[val].append(evald[-1])
    return finals


# ── per-channel plots ──────────────────────────────────────────────────

def plot_channel(name, band="sem"):
    c       = CHANNELS[name]
    exp_dir = f"./outputs/{c['exp']}"
    short   = _short_key(c["key"])

    by_val = defaultdict(lambda: defaultdict(list))
    seeds  = set()
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = _parse(os.path.basename(os.path.dirname(h)), short)
        if p is None: continue
        val, seed = p; seeds.add(seed)
        for r in json.load(open(h)):
            if r.get("global_accuracy", 0) > 0:
                by_val[val][r["round"]].append(r["global_accuracy"])
    finals = _finals_by_value(exp_dir, short)
    if not by_val:
        print(f"No runs under {exp_dir}"); return

    bs    = f"\u00b1{'SEM' if band=='sem' else 'std'}, n={len(seeds)}"
    order = sorted(by_val, reverse=(name not in ("reliability", "lr_baseline")))

    plt.figure(figsize=(8, 5))
    for val in order:
        rounds = sorted(by_val[val])
        means, halfs = zip(*[_spread(by_val[val][r], band) for r in rounds])
        rounds = np.array(rounds); means = np.array(means); halfs = np.array(halfs)
        line, = plt.plot(rounds, means, lw=2, label=c["fmt"](val))
        plt.fill_between(rounds, means-halfs, means+halfs, alpha=0.18, color=line.get_color())
    plt.xlabel("Communication round"); plt.ylabel("Global accuracy")
    plt.title(f"RQ3 {name}: convergence under constraint  (bands: {bs})")
    plt.legend(title=c["label"]); plt.grid(True, alpha=0.3); plt.tight_layout()
    out1 = os.path.join(exp_dir, f"rq3_{name}_convergence.png")
    plt.savefig(out1, dpi=150); print(f"Saved -> {out1}")

    plt.figure(figsize=(8, 5))
    vals_sorted = sorted(finals)
    xs    = [c["fmt"](v) for v in vals_sorted]
    means = [_spread(finals[v], band)[0] for v in vals_sorted]
    errs  = [_spread(finals[v], band)[1] for v in vals_sorted]
    plt.bar(range(len(xs)), means, yerr=errs, capsize=5)
    plt.xticks(range(len(xs)), xs)
    plt.ylabel("Final global accuracy"); plt.xlabel(c["label"])
    plt.title(f"RQ3 {name}: final accuracy vs constraint  (bands: {bs})")
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    out2 = os.path.join(exp_dir, f"rq3_{name}_final.png")
    plt.savefig(out2, dpi=150); print(f"Saved -> {out2}")
    plt.close("all")


# ── memory x skew comparison ───────────────────────────────────────────

def plot_memory_skew(band="sem"):
    mild   = _finals_by_value("./outputs/exp3_memory",        "memory_cap")
    severe = _finals_by_value("./outputs/exp3_memory_severe", "memory_cap")
    if not severe:
        print("No severe-skew runs yet -- run: memory_severe"); return
    if not mild:
        print("No alpha=0.3 memory runs found (./outputs/exp3_memory)"); return
    batches = sorted(set(mild) | set(severe))
    x = np.arange(len(batches))
    plt.figure(figsize=(8, 5))
    for finals, lbl, fmt in [(mild,   "alpha=0.3  (mild)",   "o-"),
                             (severe, "alpha=0.05 (severe)", "s--")]:
        means = [_spread(finals.get(b, [np.nan]), band)[0] for b in batches]
        errs  = [_spread(finals.get(b, [np.nan]), band)[1] for b in batches]
        plt.errorbar(x, means, yerr=errs, fmt=fmt, lw=2, capsize=4, label=lbl)
    plt.xticks(x, [f"batch={int(b)}" for b in batches])
    plt.xlabel("Effective batch size (memory cap)"); plt.ylabel("Final global accuracy")
    band_lbl = "SEM" if band == "sem" else "std"
    plt.title(f"RQ3 memory \u00d7 skew: does the batch sweet-spot flip?  (bands: \u00b1{band_lbl})")
    plt.legend(title="Data heterogeneity"); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = "./outputs/rq3_memory_skew_interaction.png"
    plt.savefig(out, dpi=150); print(f"Saved -> {out}"); plt.close("all")


# ── LR under-tuning control (the decisive memory check) ────────────────

def plot_lr_baseline(band="sem"):
    lr_fin  = _finals_by_value("./outputs/exp3_lr_baseline", "learning_rate")
    mem_fin = _finals_by_value("./outputs/exp3_memory",      "memory_cap")
    if not lr_fin:
        print("No lr_baseline runs yet -- run: lr_baseline"); return
    lrs   = sorted(lr_fin)
    means = [_spread(lr_fin[v], band)[0] for v in lrs]
    errs  = [_spread(lr_fin[v], band)[1] for v in lrs]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(lrs)), means, yerr=errs, capsize=5, label="batch=64 (unconstrained)")
    plt.xticks(range(len(lrs)), [f"lr={v:g}" for v in lrs])
    # reference line: the batch-32 sweet-spot at the default lr=0.01
    if 32 in mem_fin:
        ref_m, _ = _spread(mem_fin[32], band)
        plt.axhline(ref_m, ls="--", lw=2, color="red",
                    label=f"batch=32 sweet-spot (lr=0.01) = {ref_m:.3f}")
    if 64 in mem_fin:
        base_m, _ = _spread(mem_fin[64], band)
        plt.axhline(base_m, ls=":", lw=1.5, color="gray",
                    label=f"batch=64 at lr=0.01 = {base_m:.3f}")
    plt.ylabel("Final global accuracy"); plt.xlabel("Learning rate at batch=64")
    plt.title("RQ3 control: is the batch=64 baseline just under-tuned?")
    plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    out = "./outputs/rq3_lr_baseline_check.png"
    plt.savefig(out, dpi=150); print(f"Saved -> {out}"); plt.close("all")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "plot":
        band = args[1] if len(args) > 1 and args[1] in ("std", "sem") else "sem"
        for nm in CHANNELS: plot_channel(nm, band=band)
        plot_memory_skew(band=band); plot_lr_baseline(band=band)
    elif args and args[0] == "compare":
        plot_memory_skew(); plot_lr_baseline()
    elif args and args[0] == "harden":
        # the whole hardening pass: LR-at-batch-64 control + seeds 4,5 on memory
        _run_channel("lr_baseline")
        _add_seeds("memory", [4, 5])
    elif args and args[0] == "addseeds":
        # usage: addseeds <channel> <seed> [<seed> ...]
        _add_seeds(args[1], [int(s) for s in args[2:]])
    elif args and args[0] in CHANNELS:
        _run_channel(args[0])
    else:
        for nm in ("speed", "memory", "reliability"):
            _run_channel(nm)