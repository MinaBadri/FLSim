"""
Endpoint-sensitivity check: recompute the PAIRED metric (retention vs delay=3,
or delta vs FedAvg) at each of the last K evaluation rounds. If the paired number
is stable across those rounds, the "history.json records round N-5, not round N"
issue does not affect it -- the offset cancels because the metric is paired at a
common round across conditions.

Usage:
  python verify_endpoint.py retention ./outputs/exp2_churn_x_alpha [--k 4]
  python verify_endpoint.py delta     ./outputs/exp6_aggregation  [--k 4]
"""
import sys, os, re, glob, json, argparse
from collections import defaultdict
import numpy as np

def acc_traj(h):
    """round -> accuracy, for evaluated rounds only (acc>0)."""
    return {r["round"]: r["global_accuracy"] for r in h if r.get("global_accuracy", 0) > 0}

def load(exp_dir, parse):
    runs = {}
    for hp in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        key = parse(os.path.basename(os.path.dirname(hp)))
        if key is None: continue
        runs[key] = acc_traj(json.load(open(hp)))
    return runs

def last_k_common_rounds(runs, k):
    common = None
    for tr in runs.values():
        rs = set(tr)
        common = rs if common is None else (common & rs)
    return sorted(common)[-k:] if common else []

def p_ret(rid):
    a = re.search(r"dirichlet_alpha=([0-9.]+)", rid)
    d = re.search(r"max_rejoin_delay=([0-9.]+)", rid)
    s = re.search(r"seed=([0-9]+)", rid)
    return (float(a.group(1)), float(d.group(1)), int(s.group(1))) if (a and d and s) else None

def p_delta(rid):
    s = re.search(r"strategy=([A-Z]+(?:_[A-Z]+)*)", rid)
    d = re.search(r"max_rejoin_delay=([0-9.]+)", rid)
    sd = re.search(r"seed=([0-9]+)", rid)
    return (s.group(1), float(d.group(1)), int(sd.group(1))) if (s and d and sd) else None

def retention(exp_dir, k):
    runs = load(exp_dir, p_ret)
    rounds = last_k_common_rounds(runs, k)
    alphas = sorted({a for a,_,_ in runs}); delays = sorted({d for _,d,_ in runs})
    seeds  = sorted({s for _,_,s in runs})
    print(f"Retention vs delay=3, paired within seed, at the last {len(rounds)} eval rounds: {rounds}\n")
    print(f"{'alpha':>6} {'delay':>6} | " + " ".join(f"r={r:<4d}" for r in rounds))
    for a in alphas:
        for d in delays:
            if d == 3: continue
            row = []
            for r in rounds:
                rets = [runs[(a,d,s)][r]/runs[(a,3.0,s)][r]
                        for s in seeds
                        if (a,d,s) in runs and (a,3.0,s) in runs
                        and r in runs[(a,d,s)] and r in runs[(a,3.0,s)]]
                row.append(np.mean(rets)*100 if rets else float('nan'))
            print(f"{a:>6} {int(d):>6} | " + " ".join(f"{v:5.1f}%" for v in row))
    print("\n-> if each row is flat across columns, the endpoint choice is immaterial.")

def delta(exp_dir, k):
    runs = load(exp_dir, p_delta)
    rounds = last_k_common_rounds(runs, k)
    strats = sorted({s for s,_,_ in runs}); delays = sorted({d for _,d,_ in runs})
    seeds  = sorted({s for _,_,s in runs})
    print(f"Delta vs FedAvg, paired within seed, at the last {len(rounds)} eval rounds: {rounds}\n")
    print(f"{'strategy':>16} {'delay':>6} | " + " ".join(f"r={r:<4d}" for r in rounds))
    for st in strats:
        if st == "FEDAVG": continue
        for d in delays:
            row = []
            for r in rounds:
                ds = [runs[(st,d,s)][r]-runs[("FEDAVG",d,s)][r]
                      for s in seeds
                      if (st,d,s) in runs and ("FEDAVG",d,s) in runs
                      and r in runs[(st,d,s)] and r in runs[("FEDAVG",d,s)]]
                row.append(np.mean(ds) if ds else float('nan'))
            print(f"{st:>16} {int(d):>6} | " + " ".join(f"{v:+.4f}" for v in row))
    print("\n-> if each row is flat across columns, the endpoint choice is immaterial.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retention","delta"])
    ap.add_argument("exp_dir")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    (retention if a.mode=="retention" else delta)(a.exp_dir, a.k)