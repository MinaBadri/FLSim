import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentRunner

# def main():
#     runner = ExperimentRunner(
#         base_config_path = "configs/fast.yaml",          #churn.yaml",
#         experiment_name  = "exp_B_churn_sweep",
#         sweep = {
#             "churn.drop_prob"        : [0.0, 0.2, 0.4, 0.6],
#             "aggregation.strategy"   : ["FEDAVG", "STALENESS_AWARE", "ADAPTIVE"],
#             "training.learning_rate" : [0.01],            #, 0.05],
#         },
#     )
#     runner.run_all()

#     from experiments.plotter import ResultsPlotter
#     plotter = ResultsPlotter("./outputs/exp_B_churn_sweep")
#     plotter.plot_convergence()
#     plotter.plot_churn_vs_accuracy()
#     plotter.plot_pool_dynamics()

# if __name__ == "__main__":
#     main()
def main():
    runner = ExperimentRunner(
        base_config_path = "configs/hard_paper.yaml",
        experiment_name  = "exp_B_hardv2",
        sweep = {
            "churn.drop_prob"      : [0.0, 0.3, 0.5, 0.7],
            "aggregation.strategy" : ["FEDAVG", "STALENESS_AWARE", "ADAPTIVE"],
            "data.dirichlet_alpha"   : [0.1, 0.5],
        },
    )
    runner.run_all()

    from experiments.plotter import ResultsPlotter
    plotter = ResultsPlotter("./outputs/exp_B_churn_sweep_v2")

    # Key plot: each churn level separately, all 3 strategies on one chart
    for drop_prob in [0.0, 0.3, 0.5, 0.7]:
        plotter.plot_convergence(
            metric        = "global_accuracy",
            fix_drop_prob = drop_prob,
        )

    # Summary bar chart
    plotter.plot_churn_vs_accuracy()

    # Pool dynamics for highest churn run
    high_churn = [r for r in plotter.runs.keys() if "drop_prob=0.6" in r]
    if high_churn:
        plotter.plot_pool_dynamics(run_id=high_churn[0])

if __name__ == "__main__":
    main()