import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentRunner

def main():
    runner = ExperimentRunner(
        base_config_path = "configs/fast.yaml",           #baseline.yaml",
        experiment_name  = "exp_A_baseline",
        sweep = {
            "aggregation.strategy"   : ["FEDAVG", "STALENESS_AWARE", "ADAPTIVE"],
            "training.learning_rate" : [0.01, 0.05],
            "data.dirichlet_alpha"   : [0.1, 0.5, 1.0],
        },
    )
    runner.run_all()

    # Auto-plot after run
    from experiments.plotter import ResultsPlotter
    plotter = ResultsPlotter(f"./outputs/exp_A_baseline")
    plotter.plot_convergence(metric="global_accuracy")
    plotter.plot_convergence(metric="global_loss")

if __name__ == "__main__":
    main()