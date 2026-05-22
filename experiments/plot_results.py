import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.plotter import ResultsPlotter

EXPERIMENTS = {
    "exp_A_baseline"    : ["convergence"],
    "exp_B_churn_sweep" : ["convergence", "churn_vs_accuracy", "pool_dynamics"],
    "exp_C_late_rejoin" : ["convergence", "churn_vs_accuracy"],
}

for exp_name, plots in EXPERIMENTS.items():
    exp_dir = Path("./outputs") / exp_name
    if not exp_dir.exists():
        print(f"Skipping {exp_name} — not run yet")
        continue

    print(f"\nPlotting {exp_name} ...")
    plotter = ResultsPlotter(str(exp_dir))

    if "convergence" in plots:
        plotter.plot_convergence(metric="global_accuracy")
        plotter.plot_convergence(metric="global_loss")

    if "churn_vs_accuracy" in plots:
        plotter.plot_churn_vs_accuracy()

    if "pool_dynamics" in plots:
        # Plot pool dynamics for highest-churn run
        matching = [
            r for r in plotter.runs.keys()
            if "drop_prob=0.6" in r or "drop_prob=0.4" in r
        ]
        if matching:
            plotter.plot_pool_dynamics(run_id=matching[0])

print("\nAll figures saved to outputs/<experiment>/")