"""
RQ4: does hardware heterogeneity induce per-class BIAS -- and only when the
hardware quality is CORRELATED with which classes a client owns?

Mechanism (from RQ3): an unreliable client faults and its update is dropped, so
the classes it owns receive fewer gradient updates and end up under-trained.
Under severe skew classes are concentrated on specific clients, so if the poor
hardware lands on those clients, their classes are systematically under-served.

Conditions (severe skew, churn OFF, FedAvg) -- all share the same partition per
seed, so the disadvantaged class set D is identical across conditions:
  homogeneous      : every client good hardware (the no-bias baseline)
  correlated       : poor RELIABILITY on the owners of D (the headline mechanism)
  random           : poor reliability on an equally-sized RANDOM client set
                     (same hardware budget, uncorrelated with class -> control)
  both_correlated  : poor reliability + small memory on the owners of D
                     (realism robustness: a full low-end device)

Metric: per-class test accuracy. Bias = mean acc on D minus mean acc on the rest
(<0 means D is under-served). homogeneous/random ~ 0; correlated < 0.

Usage:
  python experiments/run_exp4_bias.py            # run all conditions, then plot
  python experiments/run_exp4_bias.py plot       # replot from saved records
"""
import sys, os, json, copy
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from client.hardware_profile import HardwareProfile
from data.partitioner import load_dataset, get_client_class_distribution
from utils import (load_config, build_data_pipeline, build_fleet,
                               build_registry, build_server, seed_everything)

EXPERIMENT       = "exp4_bias"
BASE             = "configs/exp4_bias.yaml"
CONDITIONS       = ["homogeneous", "correlated", "random", "both_correlated"]
SEEDS            = [1, 2, 3]
ALPHA            = 0.05      # severe skew -> concentrated ownership
POOR_FRACTION    = 0.30     # share of clients placed in the poor tier
POOR_RELIABILITY = 0.50     # poor-tier reliability (RQ3: ~15% global cost at R=0.5)
POOR_MEMORY      = 8        # poor-tier batch clamp, only used by 'both_correlated'


# ── condition construction ─────────────────────────────────────────────────

def select_poor_and_disadvantaged(dist, frac, seed):
    """
    dist : (num_clients, num_classes) sample-count matrix.
    Returns (P, P_rand, D, rest):
      owner[c] = client holding the most of class c (its primary owner).
      P        = poor-tier clients, drawn from actual owners so D is non-trivial.
      D        = classes whose primary owner is in P (the disadvantaged set).
      rest     = classes whose primary owner is NOT in P.
      P_rand   = equally-sized random client set disjoint from P (the control).
    """
    num_clients, num_classes = dist.shape
    owner = dist.argmax(0)
    actual_owners = np.unique(owner)                      # clients that own >=1 class
    n_poor = int(round(frac * num_clients))
    n_poor = max(1, min(n_poor, len(actual_owners) - 1))  # leave >=1 owner for 'rest'

    rng = np.random.default_rng(seed * 7919 + 17)
    P = set(rng.choice(actual_owners, size=n_poor, replace=False).tolist())
    D    = [c for c in range(num_classes) if owner[c] in P]
    rest = [c for c in range(num_classes) if owner[c] not in P]

    non_P = [int(i) for i in actual_owners if i not in P]
    P_rand = set(rng.choice(non_P, size=min(n_poor, len(non_P)), replace=False).tolist())
    return P, P_rand, D, rest


def make_profiles(condition, num_clients, P, P_rand):
    good      = lambda i: HardwareProfile(client_id=i, compute_speed=1.0, memory_cap=10**9, reliability=1.0)
    poor_rel  = lambda i: HardwareProfile(client_id=i, compute_speed=1.0, memory_cap=10**9, reliability=POOR_RELIABILITY)
    poor_both = lambda i: HardwareProfile(client_id=i, compute_speed=1.0, memory_cap=POOR_MEMORY, reliability=POOR_RELIABILITY)
    out = []
    for i in range(num_clients):
        if   condition == "homogeneous":     out.append(good(i))
        elif condition == "correlated":      out.append(poor_rel(i)  if i in P      else good(i))
        elif condition == "random":          out.append(poor_rel(i)  if i in P_rand else good(i))
        elif condition == "both_correlated": out.append(poor_both(i) if i in P      else good(i))
        else: raise ValueError(condition)
    return out


# ── run ─────────────────────────────────────────────────────────────────────

def run():
    base     = load_config(BASE)
    out_root = f"./outputs/{EXPERIMENT}"
    os.makedirs(out_root, exist_ok=True)
    K        = base["model"]["num_classes"]
    train_ds = load_dataset(train=True, dataset=base["data"].get("dataset", "cifar100"))
    records  = []

    for seed in SEEDS:
        cfg = copy.deepcopy(base); cfg["seed"] = seed; cfg["data"]["dirichlet_alpha"] = ALPHA

        # Partition once per seed -> P / D shared across this seed's conditions.
        seed_everything(seed)
        client_loaders, test_loader, client_indices = build_data_pipeline(cfg)
        dist = get_client_class_distribution(client_indices, train_ds, K)
        P, P_rand, D, rest = select_poor_and_disadvantaged(dist, POOR_FRACTION, seed)
        print(f"\n=== seed {seed}: |P|={len(P)} poor clients, "
              f"|D|={len(D)} disadvantaged classes, |rest|={len(rest)} ===")

        for cond in CONDITIONS:
            seed_everything(seed)   # identical training noise across conditions; only HW differs
            profiles = make_profiles(cond, cfg["simulation"]["num_clients"], P, P_rand)
            fleet    = build_fleet(cfg, client_loaders, profiles)
            registry = build_registry(cfg, fleet)
            odir     = f"{out_root}/{cond}__seed={seed}"
            server   = build_server(cfg, registry, test_loader, odir)
            server.run()

            _, acc, per_class = server.aggregator.evaluate(
                server.model, server.global_weights, server.test_loader, server.device,
                return_per_class=True)

            d_acc, r_acc = float(np.mean(per_class[D])), float(np.mean(per_class[rest]))
            rec = dict(condition=cond, seed=seed, global_acc=float(acc),
                       per_class=per_class.tolist(), D=list(map(int, D)),
                       rest=list(map(int, rest)),
                       class_std=float(np.std(per_class)))
            records.append(rec)
            os.makedirs(odir, exist_ok=True)
            json.dump(rec, open(f"{odir}/per_class.json", "w"))
            print(f"[{cond:<15} seed={seed}] global={acc:.4f}  "
                  f"D={d_acc:.4f}  rest={r_acc:.4f}  gap={d_acc - r_acc:+.4f}")

    json.dump(records, open(f"{out_root}/rq4_records.json", "w"))
    plot_bias(out_root)


# ── analysis / plots ────────────────────────────────────────────────────────

def _sem(x):
    x = np.asarray(x, float)
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (float(x.mean()), 0.0)


def plot_bias(out_root=f"./outputs/{EXPERIMENT}"):
    records = json.load(open(f"{out_root}/rq4_records.json"))
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in records)]
    label = {"homogeneous": "homogeneous\n(baseline)", "correlated": "correlated\n(poor reliability)",
             "random": "random\n(control)", "both_correlated": "both\n(rel.+memory)"}

    gap = {c: [] for c in conds}          # per-seed D-minus-rest gap
    dacc = {c: [] for c in conds}         # pooled per-class accs (D)
    racc = {c: [] for c in conds}         # pooled per-class accs (rest)
    gacc = {c: [] for c in conds}         # global acc
    for r in records:
        c = r["condition"]; pc = np.array(r["per_class"])
        gap[c].append(pc[r["D"]].mean() - pc[r["rest"]].mean())
        dacc[c].extend(pc[r["D"]].tolist()); racc[c].extend(pc[r["rest"]].tolist())
        gacc[c].append(r["global_acc"])

    # ---- paired analysis vs the homogeneous baseline (removes intrinsic D-vs-rest
    # difficulty AND global shifts; cancels between-seed D-composition variance) ----
    rec_by = {(r["condition"], r["seed"]): r for r in records}
    seeds  = sorted({r["seed"] for r in records})
    did   = {c: [] for c in conds}   # difference-in-differences vs homogeneous
    dropD = {c: [] for c in conds}   # absolute acc(D) change vs homogeneous
    dropR = {c: [] for c in conds}   # absolute acc(rest) change vs homogeneous
    for sd in seeds:
        if ("homogeneous", sd) not in rec_by:
            continue
        h = rec_by[("homogeneous", sd)]; hpc = np.array(h["per_class"])
        gap_h = hpc[h["D"]].mean() - hpc[h["rest"]].mean()
        for c in conds:
            if (c, sd) not in rec_by:
                continue
            r = rec_by[(c, sd)]; pc = np.array(r["per_class"])
            gap_c = pc[r["D"]].mean() - pc[r["rest"]].mean()
            did[c].append(gap_c - gap_h)
            dropD[c].append(pc[r["D"]].mean()    - hpc[h["D"]].mean())
            dropR[c].append(pc[r["rest"]].mean() - hpc[h["rest"]].mean())

    print("\n--- paired vs homogeneous baseline (within-seed) ---")
    print(f"{'condition':<16}{'DiD (bias added)':>22}{'acc(D) drop':>18}{'acc(rest) drop':>18}")
    for c in conds:
        if c == "homogeneous":
            print(f"{c:<16}{'0 (reference)':>22}{'0':>18}{'0':>18}"); continue
        dm, de = _sem(did[c]); ddm, dde = _sem(dropD[c]); drm, dre = _sem(dropR[c])
        print(f"{c:<16}{dm:>+12.4f} +/- {de:<6.4f}{ddm:>+10.4f}±{dde:<5.4f}{drm:>+10.4f}±{dre:<5.4f}")

    # Figure 0 (new headline): paired DiD vs homogeneous
    nz = [c for c in conds if c != "homogeneous"]
    if nz and all(len(did[c]) for c in nz):
        plt.figure(figsize=(8, 5))
        xs = np.arange(len(nz))
        m = [_sem(did[c])[0] for c in nz]; e = [_sem(did[c])[1] for c in nz]
        cols = ["#C0392B" if c == "correlated" else "#7F8C8D" if c == "random" else "#922B21" for c in nz]
        plt.bar(xs, m, yerr=e, capsize=5, color=cols, edgecolor="k", linewidth=0.5)
        plt.axhline(0, color="k", lw=0.8)
        plt.xticks(xs, [label[c] for c in nz])
        plt.ylabel("Bias added vs homogeneous (paired DiD)")
        plt.title("RQ4: per-class bias attributable to hardware (baseline-subtracted, paired)")
        plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        out0 = f"{out_root}/rq4_bias_did.png"; plt.savefig(out0, dpi=150); print(f"Saved -> {out0}"); plt.close()

    # Figure 1: headline bias bar (mean D-minus-rest gap +/- SEM)
    plt.figure(figsize=(8, 5))
    xs = np.arange(len(conds))
    means = [(_sem(gap[c])[0]) for c in conds]
    errs  = [(_sem(gap[c])[1]) for c in conds]
    colors = ["#4C9F70" if c == "homogeneous" else "#C0392B" if c == "correlated"
              else "#7F8C8D" if c == "random" else "#922B21" for c in conds]
    plt.bar(xs, means, yerr=errs, capsize=5, color=colors, edgecolor="k", linewidth=0.5)
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(xs, [label[c] for c in conds])
    plt.ylabel("Per-class bias:  mean acc(D) \u2212 mean acc(rest)")
    plt.title("RQ4: hardware induces per-class bias only when correlated with class ownership")
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    out1 = f"{out_root}/rq4_bias_gap.png"; plt.savefig(out1, dpi=150); print(f"Saved -> {out1}"); plt.close()

    # Figure 2: per-class accuracy distributions, D vs rest, per condition
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = 0; ticks = []; tlabels = []
    for c in conds:
        b = ax.boxplot([dacc[c], racc[c]], positions=[pos, pos + 0.7], widths=0.55,
                       patch_artist=True, showfliers=False)
        for patch, col in zip(b["boxes"], ["#C0392B", "#4C9F70"]):
            patch.set_facecolor(col); patch.set_alpha(0.65)
        ticks.append(pos + 0.35); tlabels.append(label[c]); pos += 2
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels)
    ax.set_ylabel("Per-class test accuracy")
    ax.set_title("RQ4: disadvantaged (red) vs rest (green) per-class accuracy by condition")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#C0392B", alpha=0.65, label="disadvantaged classes (D)"),
                       Patch(facecolor="#4C9F70", alpha=0.65, label="rest")], loc="lower left")
    ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    out2 = f"{out_root}/rq4_per_class_box.png"; fig.savefig(out2, dpi=150); print(f"Saved -> {out2}"); plt.close(fig)

    # console summary
    print("\ncondition         gap(D-rest)        global_acc      across-class std")
    for c in conds:
        gm, ge = _sem(gap[c]); am, ae = _sem(gacc[c])
        stds = [r["class_std"] for r in records if r["condition"] == c]
        print(f"{c:<16} {gm:+.4f} +/- {ge:.4f}    {am:.4f} +/- {ae:.4f}    {np.mean(stds):.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_bias()
    else:
        run()