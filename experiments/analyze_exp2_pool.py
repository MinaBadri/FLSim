"""
RQ2 post-hoc analysis from EXISTING run logs (no new training).

(1) Unifying check: final accuracy vs pool starvation. For every run we read
    the per-round 'contributing' count and compute the mean contributors/round
    and the fraction of rounds below the per-round quota. Plotting accuracy
    against that shows it tracks "available pool below quota" -- the variable
    that BOTH drop rate and absence duration feed. Rate-sweep runs cluster at
    zero starvation (hence flat); delay-sweep runs trace the curve down.

(2) Retention with seed spread: per (alpha, delay), retention = acc(delay)/
    acc(delay=3) paired WITHIN each seed, reported mean +/- SEM, so the headline
    72%/88%-type numbers carry their own error bar.

Usage:
  python experiments/analyze_exp2_pool.py DIR [DIR ...] [--quota 10]
  e.g. python experiments/analyze_exp2_pool.py ./outputs/exp2_interaction ./outputs/exp2_churn
"""
import sys, os, re, glob, json, argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def parse_rid(rid):
    g = lambda pat: (re.search(pat, rid) or [None, None])[1] if re.search(pat, rid) else None
    a = re.search(r"dirichlet_alpha=([0-9.]+)", rid)
    d = re.search(r"max_rejoin_delay=([0-9.]+)", rid)
    p = re.search(r"drop_prob=([0-9.]+)", rid)
    s = re.search(r"seed=([0-9]+)", rid)
    return dict(
        alpha=float(a.group(1)) if a else None,
        delay=float(d.group(1)) if d else None,
        drop=float(p.group(1)) if p else None,
        seed=int(s.group(1)) if s else None,
    )


def load_runs(dirs, quota):
    runs = []
    for D in dirs:
        for h in glob.glob(os.path.join(D, "*", "history.json")):
            meta = parse_rid(os.path.basename(os.path.dirname(h)))
            hist = json.load(open(h))
            contrib = [r.get("contributing", r.get("successful", quota)) for r in hist]
            accs = [r["global_accuracy"] for r in hist if r.get("global_accuracy", 0) > 0]
            if not accs or not contrib:
                continue
            meta.update(
                src=os.path.basename(D.rstrip("/\\")),
                final_acc=accs[-1],
                mean_contrib=float(np.mean(contrib)),
                frac_below=float(np.mean([c < quota for c in contrib])),
            )
            runs.append(meta)
    return runs


def _sem(x):
    x = np.asarray(x, float)
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (float(x.mean()), 0.0)


def retention_table(runs):
    # key on (alpha, delay, drop, seed) so rate-sweep and delay-sweep runs that
    # differ only in drop_prob never collide on the delay=3 baseline.
    acc = {(r["alpha"], r["delay"], r["drop"], r["seed"]): r["final_acc"]
           for r in runs if None not in (r["delay"], r["seed"])}  # alpha may be None (rate sweep, fixed alpha)
    groups = sorted({(a, dp) for a, _, dp, _ in acc},
                    key=lambda t: (t[0] if t[0] is not None else 1e9, t[1] if t[1] is not None else -1))
    delays = sorted({d for _, d, _, _ in acc})
    seeds  = sorted({s for _, _, _, s in acc})
    print(f"\n{'alpha':>6} {'drop':>6} {'delay':>6} {'final_acc (mean+/-SEM)':>26} "
          f"{'retention vs d=3 (mean+/-SEM)':>30} {'mean_contrib':>12} {'frac<quota':>11}")
    rows = []
    for a, dp in groups:
        for d in delays:
            finals = [acc[(a, d, dp, s)] for s in seeds if (a, d, dp, s) in acc]
            if not finals:
                continue
            fm, fe = _sem(finals)
            rets = [acc[(a, d, dp, s)] / acc[(a, 3.0, dp, s)]
                    for s in seeds if (a, d, dp, s) in acc and (a, 3.0, dp, s) in acc]
            rm, re_ = _sem(rets) if rets else (float("nan"), 0.0)
            sub = [r for r in runs if r["alpha"] == a and r["delay"] == d and r["drop"] == dp]
            mc = np.mean([r["mean_contrib"] for r in sub])
            fb = np.mean([r["frac_below"] for r in sub])
            dps = f"{dp:.2f}" if dp is not None else "fixed"
            a_lbl = f"{a}" if a is not None else "fixed"
            print(f"{a_lbl:>6} {dps:>6} {int(d):>6} {fm:>10.3f} +/- {fe:<11.3f} "
                  f"{rm*100:>14.1f}% +/- {re_*100:<10.1f}% {mc:>12.2f} {fb:>11.2f}")
            rows.append((a, dp, d, fm, fe, rm, re_, mc, fb))
    return rows


def plot_unifying(runs, quota, out="./outputs/rq2_pool_unifying.png"):
    alphas = sorted({r["alpha"] for r in runs}, key=lambda a: (a is None, a))
    plt.figure(figsize=(8, 5))
    cmap = {a: c for a, c in zip(alphas, plt.cm.viridis(np.linspace(0.15, 0.8, max(len(alphas), 2))))}
    for a in alphas:
        pts = [(r["mean_contrib"], r["final_acc"], r["delay"]) for r in runs if r["alpha"] == a]
        if not pts: continue
        pts.sort()
        xs, ys, ds = zip(*pts)
        lbl = f"alpha={a:g} (duration sweep)" if a is not None else "alpha fixed (rate sweep)"
        mk  = "o" if a is not None else "^"
        plt.scatter(xs, ys, color=cmap[a], s=55, marker=mk, edgecolor="k", linewidth=0.4,
                    label=lbl, zorder=3)
        for x, y, d in pts:
            if d is not None:
                plt.annotate(f"d{int(d)}", (x, y), fontsize=7, xytext=(3, 3),
                             textcoords="offset points")
    plt.axvline(quota, ls="--", color="red", lw=1.5, label=f"per-round quota = {quota}")
    plt.xlabel(f"Mean contributors per round  (quota = {quota}; lower = pool starved)")
    plt.ylabel("Final global accuracy")
    plt.title("Final accuracy tracks contributors relative to the per-round quota")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150); print(f"\nSaved -> {out}"); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--quota", type=int, default=10)
    args = ap.parse_args()
    runs = load_runs(args.dirs, args.quota)
    if not runs:
        print("No runs found under:", args.dirs); sys.exit(1)
    print(f"Loaded {len(runs)} runs from {args.dirs}")
    retention_table(runs)
    plot_unifying(runs, args.quota)