"""


(1) Null-result confirmation: per-strategy final accuracy mean +/- SEM at each
    delay, plus the paired (within-seed) difference vs FedAvg with its SEM and a
    flag for whether 0 lies inside +/-2 SEM

(2) How often can the strategies even differ from FedAvg? 

"""
import sys, os, re, glob, json, argparse
import numpy as np

ORDER = ["FEDAVG", "STALENESS_AWARE", "ADAPTIVE", "CATCHUP"]


def parse_rid(rid):
    s  = re.search(r"strategy=([A-Z]+(?:_[A-Z]+)*)", rid)
    d  = re.search(r"max_rejoin_delay=([0-9.]+)", rid)
    sd = re.search(r"seed=([0-9]+)", rid)
    return (s.group(1), float(d.group(1)), int(sd.group(1))) if (s and d and sd) else None


def sem(x):
    x = np.asarray(x, float)
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (float(x.mean()), 0.0)


_TCRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
          8: 2.31, 9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}

def tcrit(n):
    df = max(n - 1, 1)
    return _TCRIT.get(df, 2.04)


def load(exp_dir):
    runs = {}
    for h in glob.glob(os.path.join(exp_dir, "*", "history.json")):
        p = parse_rid(os.path.basename(os.path.dirname(h)))
        if not p:
            continue
        hist = json.load(open(h))
        accs = [r["global_accuracy"] for r in hist if r.get("global_accuracy", 0) > 0]
        if accs:
            runs[p] = {"final": accs[-1], "history": hist}
    return runs


def final_table(runs):
    delays = sorted({d for (_, d, _) in runs})
    seeds  = sorted({sd for (_, _, sd) in runs})
    strats = [s for s in ORDER if any(st == s for (st, _, _) in runs)]
    for d in delays:
        fed = {sd: runs[("FEDAVG", d, sd)]["final"] for sd in seeds if ("FEDAVG", d, sd) in runs}
        print(f"\n=== delay={int(d)} ===")
        print(f"{'strategy':<16}{'final_acc (mean+/-SEM)':>24}{'Δ vs FedAvg (paired)':>24}{'vs FedAvg @95%':>13}")
        for s in strats:
            finals = [runs[(s, d, sd)]["final"] for sd in seeds if (s, d, sd) in runs]
            if not finals:
                continue
            m, e = sem(finals)
            if s == "FEDAVG":
                print(f"{s:<16}{m:>14.4f} +/- {e:<6.4f}{'(reference)':>24}{'':>13}")
            else:
                deltas = [runs[(s, d, sd)]["final"] - fed[sd]
                          for sd in seeds if (s, d, sd) in runs and sd in fed]
                dm, de = sem(deltas)
                tc = tcrit(len(deltas))
                if abs(dm) <= tc * max(de, 1e-12):
                    verdict = "tied"
                else:
                    verdict = "WORSE" if dm < 0 else "BETTER"
                print(f"{s:<16}{m:>14.4f} +/- {e:<6.4f}{dm:>+14.4f} +/- {de:<6.4f}{verdict:>13}")


def staleness_freq(runs, delay=50):
    rej, spike, meds, ns = [], [], [], []
    for (s, d, sd), v in runs.items():
        if s != "FEDAVG" or int(d) != int(delay):  
            continue
        hist = v["history"]
        av = [r.get("avg_staleness", 0.0) for r in hist]
        nz = [x for x in av if x > 0]
        if not nz:
            continue
        med = np.median(nz)
        rej.append(np.mean([1 if r.get("rejoined", 0) > 0 else 0 for r in hist]))
        spike.append(np.mean([1 if x > 2 * med else 0 for x in nz]))
        meds.append(med); ns.append(len(hist))
    if not meds:
        print(f"\n(no FEDAVG runs at delay={delay} for staleness check)"); return
    print(f"\n=== delay={delay}: how often can strategies diverge from FedAvg? (FedAvg runs, n={len(meds)}) ===")
    print(f"median cohort avg-staleness            : {np.mean(meds):.1f} rounds")
    print(f"rounds with a rejoin event             : {np.mean(rej)*100:.1f}%")
    print(f"rounds with avg-staleness > 2x median  : {np.mean(spike)*100:.1f}%  (UPPER BOUND on divergent rounds)")
    print("Strategies can only differ from FedAvg when within-cohort staleness varies; the")
    print("spike fraction bounds that above (elevated mean is necessary, not sufficient, for variance>0).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", nargs="?", default="./outputs/exp6_aggregation")
    ap.add_argument("--delay", type=int, default=50)
    a = ap.parse_args()
    runs = load(a.exp_dir)
    if not runs:
        print("No runs under", a.exp_dir); sys.exit(1)
    print(f"Loaded {len(runs)} runs from {a.exp_dir}")
    final_table(runs)
    staleness_freq(runs, a.delay)