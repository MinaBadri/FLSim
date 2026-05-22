import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentRunner

def main():
    runner = ExperimentRunner(
        base_config_path = "configs/churn.yaml",
        experiment_name  = "exp_C_late_rejoin",
        sweep = {
            "churn.min_rejoin_delay"     : [1,  5, 15],
            "churn.max_rejoin_delay"     : [5, 15, 30],
            "aggregation.staleness_alpha": [0.7, 0.85, 1.0],
            "aggregation.strategy"       : ["STALENESS_AWARE", "ADAPTIVE"],
        },
    )
    runner.run_all()

    from experiments.plotter import ResultsPlotter
    plotter = ResultsPlotter("./outputs/exp_C_late_rejoin")
    plotter.plot_convergence()
    plotter.plot_churn_vs_accuracy()

if __name__ == "__main__":
    main()