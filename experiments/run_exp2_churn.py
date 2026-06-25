import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import ExperimentRunner
from experiments.plotter import ResultsPlotter


def run_rate_sweep():
    """
    RQ2a — churn RATE: how OFTEN clients drop.
    Delays are short & fixed (min=1, max=3 from the base config) so clients
    return quickly; the only thing changing is how frequently they leave.
    """
    runner = ExperimentRunner(
        base_config_path = "configs/exp2_base.yaml",
        experiment_name  = "exp2_churn_rate",
        sweep = {
            "churn.drop_prob" : [0.0, 0.1, 0.2, 0.3],   # 0.0 = no-churn baseline
            "seed"            : [1, 2, 3],
        },
    )
    runner.run_all()

    p = ResultsPlotter("./outputs/exp2_churn_rate")
    p.plot_convergence_by_param("drop_prob", metric="global_accuracy")
    p.plot_convergence_by_param("drop_prob", metric="global_loss")
    p.plot_param_vs_accuracy(
        "drop_prob",
        param_label="Drop probability per round",
        title="Effect of Churn Rate on Final Accuracy",
    )
    p.plot_convergence_speed_by_param("drop_prob", threshold=0.15)


def run_lateness_sweep():
    """
    RQ2b — rejoin LATENESS: how LONG clients stay gone.
    drop_prob is held fixed at a moderate level; only the rejoin delay grows,
    so a client's data is absent from aggregation for longer and longer.
    """
    runner = ExperimentRunner(
        base_config_path = "configs/exp2_base.yaml",
        experiment_name  = "exp2_rejoin_lateness",
        sweep = {
            "churn.max_rejoin_delay" : [3, 10, 25, 50],   # rounds gone (uniform[1, max])
            "seed"                   : [1, 2, 3],
        },
    )
    # Hold the rate fixed for this sweep (base config has drop_prob=0.0).
    runner.base_cfg["churn"]["drop_prob"] = 0.2
    runner.run_all()

    p = ResultsPlotter("./outputs/exp2_rejoin_lateness")
    p.plot_convergence_by_param("max_rejoin_delay", metric="global_accuracy")
    p.plot_param_vs_accuracy(
        "max_rejoin_delay",
        param_label="Max rejoin delay (rounds absent)",
        title="Effect of Rejoin Lateness on Final Accuracy",
    )
    # A pool-dynamics view of the most extreme run shows the churn behaviour.
    longest = [r for r in p.runs if "max_rejoin_delay=50" in r]
    if longest:
        p.plot_pool_dynamics(run_id=longest[0])


if __name__ == "__main__":
    run_rate_sweep()
    run_lateness_sweep()