import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentRunner

def main():
    runner = ExperimentRunner(
        base_config_path = "configs/churn.yaml",
        experiment_name  = "exp_B_churn_sweep",
        sweep = {
            "churn.drop_prob"        : [0.0, 0.2, 0.4, 0.6],
            "aggregation.strategy"   : ["FEDAVG", "STALENESS_AWARE", "ADAPTIVE"],
            "training.learning_rate" : [0.01, 0.05],
        },
    )
    runner.run_all()

    from experiments.plotter import ResultsPlotter
    plotter = ResultsPlotter("./outputs/exp_B_churn_sweep")
    plotter.plot_convergence()
    plotter.plot_churn_vs_accuracy()
    plotter.plot_pool_dynamics()

if __name__ == "__main__":
    main()