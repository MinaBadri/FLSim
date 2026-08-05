# FLSim

A controlled simulation of federated learning under heterogeneity, client churn and
system-design choices. Supporting code for *"How Heterogeneity, Churn and System Design
Affect Federated Learning: A Simulation Study."*

The simulator holds the model, dataset, optimizer and client-selection policy fixed
while varying one stressor at a time — data skew, churn, hardware constraints,
synchronization mode, aggregation strategy — so that observed degradation can be
attributed to a specific cause. All experiments use CIFAR-100, a compact ResNet-style
model and 50 clients with 10 selected per round unless a config overrides it.

The headline finding is that degradation is best explained by **information
availability**: severe skew narrows what updates contain, long absences thin the
contributor pool, and correlated poor hardware produces targeted per-class loss.
Reweighting received updates does not repair damage caused by updates that never
arrived.

## Install

```bash
git clone https://github.com/MinaBadri/FLSim.git
cd FLSim
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 and PyTorch 2.2. A CUDA GPU is strongly recommended — runs were done on
dual RTX 5000 Ada; CPU-only will work but is slow.

## Run

Experiments are driven by YAML configs:

```bash
python main.py --config configs/exp4_bias.yaml          # TODO: confirm entry point + flag
```

<!-- TODO: how are multiple seeds run? One of:
     python main.py --config configs/exp4_bias.yaml --seeds 5
     for s in 1 2 3 4 5; do python main.py --config ... --seed $s; done
     Replace this block with the real command. -->

Each run writes checkpoints that embed the fully resolved config that produced them,
so every reported setting is recoverable from the run artifacts rather than from the
YAML, which may have changed since.

### Experiments

| Config | Budget | Varies |
|---|---|---|
| `TODO` | 100r, E=10, n=3 | Dirichlet α ∈ {0.01, 0.05, 0.5, 1.0} |
| `TODO` | 80r, E=5, n=2–5 | speed, batch cap (+LR controls), reliability R |
| `TODO` | 80r/60r, E=10/5 | drop probability, max rejoin delay, α |
| `TODO` | 60r, E=5, n=3 | matched cohort control (5 clients/round) |
| `TODO` | 200r, E=5, n=3 | long-horizon churn |
| `exp4_bias.yaml` | 80r, E=5, n=5 | poor reliability: homogeneous / random / class-owners |
| `TODO` | equal virtual-time, n=2 | sync mode, buffer M ∈ {1, 2, 4, 8, 10} |
| `TODO` | 60r, E=5, n=3 | FedAvg vs. staleness-aware / adaptive / CatchUp |

### Analysis scripts

```bash
python audit_runs.py        # check runs against their embedded configs
python rebuild_summary.py   # regenerate summary tables from checkpoints
python mechanism_test.py    # rare-information (n_eff) diagnostic
```

## Results

<!-- TODO: replace filenames with the actual ones in Results/ -->

**Churn lateness × heterogeneity** — accuracy is flat to a rejoin delay of 10 rounds,
then declines from 25. Severe skew loses a larger fraction of a lower baseline.

![Churn lateness](Results/FIGURE_1.png)

**Per-class bias attributable to hardware** — difference-in-differences against a
homogeneous baseline. Poor reliability placed on the *owners* of tracked classes
produces localized loss (−0.042 ± 0.011, p = 0.02); the same amount of poor hardware
placed randomly does not.

![Per-class bias](Results/FIGURE_2.png)

**Synchronization mode under speed heterogeneity** — at an equal wall-clock budget,
asynchronous execution retains ~90% of its homogeneous-speed accuracy at spread 0.9
against ~57% for synchronous, because the barrier leaves the pipeline idle.

![Synchronization](Results/FIGURE_3.png)

**Aggregation under churn** — no reweighting strategy improves on FedAvg at either
rejoin delay. These methods only reweight updates that arrived.

![Aggregation](Results/FIGURE_4.png)

## Citation

<!-- TODO: add once the paper has a venue and DOI. -->

## License

<!-- TODO: add a LICENSE file. MIT is the usual choice for research code. -->
