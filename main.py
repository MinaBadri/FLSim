import torch
from models.cnn import SimpleCNN
from client.car_client import TrainResult
from server.aggregator import Aggregator, AggregationStrategy
from utils import load_config

cfg    = load_config("configs/churn.yaml")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model          = SimpleCNN(num_classes=10).to(device)
global_weights = model.state_dict()

# Fake three results: one normal, one stale, one dropped
fake_results = [
    TrainResult(0, global_weights, 512, loss=1.8, accuracy=0.42,
                train_time=1.2, staleness=0,  hardware_tier="high", dropped=False),
    TrainResult(1, global_weights, 256, loss=2.5, accuracy=0.31,
                train_time=2.8, staleness=7,  hardware_tier="low",  dropped=False),
    TrainResult(2, None,           0,   loss=0.0, accuracy=0.00,
                train_time=0.0, staleness=0,  hardware_tier="mid",  dropped=True),
]

for strategy in AggregationStrategy:
    cfg["aggregation"]["strategy"] = strategy.name
    agg = Aggregator.from_config(cfg)

    new_weights = agg.aggregate(fake_results, global_weights, current_round=0)
    log         = agg.history[-1]

    print(f"\n{strategy.name}")
    print(f"  contributing={log['contributing']}  "
          f"avg_staleness={log['avg_staleness']:.1f}  "
          f"avg_weight={log['avg_weight']:.4f}  "
          f"weight_std={log['weight_std']:.4f}")