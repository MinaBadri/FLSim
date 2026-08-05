import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentRunner
from experiments.plotter import ResultsPlotter



def main():
    runner = ExperimentRunner(
        base_config_path = "configs/exp1_base.yaml",
        experiment_name  = "exp1_data_heterogeneity",
        sweep = {
            "data.dirichlet_alpha" : [0.01, 0.05, 0.5, 1.0],
            "seed"                 : [1, 2, 3], # [0.05, 0.1, 0.3, 0.5, 1.0]
        },
    )
    # runner.run_all()
    plot_results()


def plot_results():
    plotter = ResultsPlotter("./outputs/exp1_data_heterogeneity")
    # plotter.rebuild_summary()

    # Convergence curves
    plotter.plot_convergence_by_param(
        param_key = "dirichlet_alpha",
        metric    = "global_accuracy",
    )
    plotter.plot_convergence_by_param(
        param_key = "dirichlet_alpha",
        metric    = "global_loss",
    )

  
    plotter.plot_param_vs_accuracy(
        param_key   = "dirichlet_alpha",
        param_label = "Dirichlet Alpha (data heterogeneity)",
        title       = "Effect of Data Heterogeneity on Final Accuracy",
    )

   
    plotter.plot_convergence_speed_by_param(
    param_key = "dirichlet_alpha",
    threshold = 0.15,
)


if __name__ == "__main__":
    main()