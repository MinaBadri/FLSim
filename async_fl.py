"""
RQ5 integration: drive the sync/semi/async orchestrator with real CIFAR/ResNet
local training. Speed is isolated (reliability=1.0, memory=inf), so the only
thing that differs across clients is compute_speed, and the only thing that
differs across runs is the synchronization mode and the speed spread.

run_async_fl(cfg, mode, buffer_size, speed_spread, seed, ...) -> (result, speeds)
where result = {history, applied, contrib, end_t, mode} from the orchestrator.

Local training is single-client (async trains one job at a time); we reuse the
client's loader + augmentation so the per-update training matches the sync path.
"""
import sys, copy
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from async_core import AsyncOrchestrator
from utils import (load_config, build_data_pipeline, build_fleet,
                               seed_everything)
from models import build_model            # matches server.py
from server.aggregator import Aggregator
from client.hardware_profile import HardwareProfile


def make_speeds(num_clients, spread, seed):
    """Per-client compute speed, mean pinned to 1.0; spread=0 -> homogeneous."""
    if spread <= 0:
        return [1.0] * num_clients
    rng = np.random.default_rng(seed + 555)
    s = rng.lognormal(mean=0.0, sigma=spread, size=num_clients)
    s = s / s.mean()                 # pin mean speed to 1.0
    s = np.clip(s, 0.15, None)       # cap the worst straggler so durations stay finite
    return s.tolist()


def run_async_fl(cfg, mode, buffer_size, speed_spread, seed,
                 time_budget=80.0, concurrency=None, alpha=0.6, staleness_a=0.5,
                 work=1.0, speed_noise=0.10, eval_dt=4.0):
    seed_everything(seed)
    cfg = copy.deepcopy(cfg); cfg["seed"] = seed
    client_loaders, test_loader, _ = build_data_pipeline(cfg)
    num_clients = cfg["simulation"]["num_clients"]
    C = concurrency or cfg["simulation"]["clients_per_round"]
    speeds = make_speeds(num_clients, speed_spread, seed)

    # isolate speed: reliability=1, memory unconstrained; compute_speed carries the spread
    profiles = [HardwareProfile(client_id=i, compute_speed=speeds[i],
                                memory_cap=10**9, reliability=1.0)
                for i in range(num_clients)]
    fleet = build_fleet(cfg, client_loaders, profiles)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model = build_model(cfg).to(device)
    eval_model  = build_model(cfg).to(device)
    agg = Aggregator.from_config(cfg)
    tcfg = cfg["training"]; epochs = tcfg["local_epochs"]
    lr = tcfg.get("learning_rate", 0.01); bs = tcfg["batch_size"]
    crit = torch.nn.CrossEntropyLoss()

    def train_fn(weights, cid):
        client = fleet[cid]
        train_model.load_state_dict(weights); train_model.train()
        opt = torch.optim.SGD(train_model.parameters(), lr=lr,
                              momentum=tcfg.get("momentum", 0.9),
                              weight_decay=tcfg.get("weight_decay", 0.001))
        loader = client._get_loader(bs); augment = client.augment; n = 0
        for ep in range(epochs):
            for x, y in loader:
                x = x.to(device); y = y.to(device); x = augment(x)
                opt.zero_grad(); loss = crit(train_model(x), y); loss.backward(); opt.step()
                if ep == 0: n += x.size(0)
        return ({k: v.detach().clone() for k, v in train_model.state_dict().items()}, n)

    def fedavg_fn(items):
        tot = sum(n for _, n in items) or 1
        out = None
        for w, n in items:
            f = n / tot
            if out is None: out = {k: v.clone() * f for k, v in w.items()}
            else:
                for k in out: out[k] += w[k] * f
        return out

    def blend_fn(gw, cw, lam):
        return {k: (1.0 - lam) * gw[k] + lam * cw[k] for k in gw}

    def eval_fn(weights):
        _, acc = agg.evaluate(eval_model, weights, test_loader, device)
        return acc

    init = {k: v.detach().clone() for k, v in build_model(cfg).to(device).state_dict().items()}
    orch = AsyncOrchestrator(
        range(num_clients), speeds, init, train_fn, fedavg_fn, blend_fn, eval_fn,
        mode=mode, buffer_size=buffer_size, concurrency=C, alpha=alpha,
        staleness_a=staleness_a, work=work, speed_noise=speed_noise,
        time_budget=time_budget, eval_dt=eval_dt, seed=seed)
    return orch.run(), speeds