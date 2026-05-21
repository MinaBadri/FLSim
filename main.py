from network.churn_model import ChurnModel, ClientState
from utils import load_config

cfg   = load_config("configs/churn.yaml")
churn = ChurnModel.from_config(cfg)

print(f"{'Round':<6} {'Active':>7} {'Dropped':>8} {'Rejoining':>10} "
      f"{'New drops':>10} {'Rejoined':>9}")
print("-" * 55)

for r in range(20):
    events = churn.step(current_round=r)
    s      = churn.round_summary(r)
    print(f"{r:<6} {s['active']:>7} {s['dropped']:>8} "
          f"{s['rejoining']:>10} {s['newly_dropped']:>10} "
          f"{s['rejoined']:>9}")