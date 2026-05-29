import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentRunner

def main():
    # Define delay pairs explicitly instead of sweeping independently
    delay_pairs = [
        (1,  5),    # short absence
        (5,  15),   # medium absence
        (15, 30),   # long absence
    ]

    for min_delay, max_delay in delay_pairs:
        runner = ExperimentRunner(
            base_config_path = "configs/fast.yaml",      #churn.yaml
            experiment_name  = "exp_C_late_rejoin",
            sweep = {
                "churn.min_rejoin_delay"     : [min_delay],
                "churn.max_rejoin_delay"     : [max_delay],
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