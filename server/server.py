import torch
import copy
import json
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import copy
from client.car_client import TrainResult
import time

# from models.cnn import SimpleCNN
from models import build_model
from server.aggregator import Aggregator
from server.registry import ClientRegistry
from client.car_client import get_device


class FLServer:
    """
    Central FL server. Owns the global model and orchestrates
    every round of federated training.

    Usage:
        server = FLServer.from_config(cfg, registry, aggregator, test_loader)
        server.run()
    """

    def __init__(
        self,
        model        : torch.nn.Module,
        registry     : ClientRegistry,
        aggregator   : Aggregator,
        test_loader,
        config       : dict,
        output_dir   : str = "./outputs",
    ):
        self.model       = model
        self.registry    = registry
        self.aggregator  = aggregator
        self.test_loader = test_loader
        self.cfg         = config
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device         = get_device()
        self.global_weights = {k: v.clone() for k, v in model.state_dict().items()}

        # Training config shortcuts
        sim_cfg             = config["simulation"]
        self.num_rounds     = sim_cfg["num_rounds"]
        self.clients_per_round = sim_cfg["clients_per_round"]

        # Eval frequency — evaluate every N rounds
        self.eval_every     = config.get("eval_every", 5)

        # Checkpoint frequency
        self.checkpoint_every = config.get("checkpoint_every", 20)

        # Full run history — exported at the end
        self.history = []

        print(f"FLServer ready")
        print(f"  device          : {self.device}")
        print(f"  rounds          : {self.num_rounds}")
        print(f"  clients/round   : {self.clients_per_round}")
        print(f"  strategy        : {self.aggregator.strategy.name}")
        print(f"  output dir      : {self.output_dir}\n")

    # ── Main training loop ─────────────────────────────────────────────

    def run(self):
        """Run the full FL training loop."""
        t_start = time.time()

        for rnd in tqdm(range(self.num_rounds), desc="FL Rounds"):
            round_log = self._run_one_round(rnd)
            self.history.append(round_log)

            # Console summary
            if rnd % self.eval_every == 0:
                self.registry.print_round_summary(rnd)

            # Checkpoint
            if rnd % self.checkpoint_every == 0 and rnd > 0:
                self._save_checkpoint(rnd)

        total_time = time.time() - t_start
        print(f"\nTraining complete in {total_time/60:.1f} min")

        # Final evaluation
        loss, acc = self._evaluate()
        print(f"Final global model — loss={loss:.4f}  acc={acc:.4f}")

        # Save final outputs
        self._save_checkpoint("final")
        self._save_history()

        return self.history

    # ── Single round ───────────────────────────────────────────────────

    def _run_one_round(self, rnd: int) -> dict:

        # 1. Advance churn — who drops, who rejoins
        events = self.registry.step(current_round=rnd)

        # 2. Select k clients from the active pool
        selected = self.registry.select(
            current_round    = rnd,
            k                = self.clients_per_round,
            include_rejoining= True,     #True
        )

        # Edge case: no clients available
        if not selected:
            return self._empty_round_log(rnd, events)

        # 3 + 4. Broadcast global model + run local training
        if self.device.type == "cuda":
            results = self._run_round_batched(selected, rnd)
        else:
            results = self.registry.run_round(
                    selected_ids   = selected,
                    global_weights = self.global_weights,
                    config         = self.cfg["training"],
                    current_round  = rnd,
                )

        # 5. Aggregate into new global model
        self.global_weights = self.aggregator.aggregate(
            results        = results,
            global_weights = self.global_weights,
            current_round  = rnd,
            ref_bs         = self.cfg["training"]["batch_size"],
        )

        # 6. Evaluate every eval_every rounds
        loss, acc = 0.0, 0.0
        if rnd % self.eval_every == 0:
            loss, acc = self._evaluate()

        # 7. Record results in registry
        self.registry.record_results(
            results       = results,
            current_round = rnd,
            events        = events,
            global_loss   = loss,
            global_acc    = acc,
        )

        # 8. Build round log
        agg_log = self.aggregator.history[-1]
        return {
            "round"              : rnd,
            "selected"           : len(selected),
            "successful"         : agg_log["successful"],
            "dropped_mid_round"  : agg_log["dropped_mid_round"],
            "contributing"       : agg_log["contributing"],
            "active_pool"        : self.registry.active_count(),
            "dropped_pool"       : self.registry.dropped_count(),
            "rejoining_pool"     : self.registry.rejoining_count(),
            "newly_dropped"      : len(events.get("newly_dropped", [])),
            "rejoined"           : len(events.get("rejoined", [])),
            "avg_staleness"      : agg_log["avg_staleness"],
            "global_loss"        : loss,
            "global_accuracy"    : acc,
            "strategy"           : agg_log["strategy"],
        }
    def _run_round_batched(self, selected_ids, rnd):
        """
        Train all selected clients in one batched GPU operation per epoch.
        Each client gets its own model copy but all forward/backward passes
        run back to back without Python overhead between them.
        """
        

        cfg      = self.cfg["training"]
        epochs   = cfg["local_epochs"]
        lr       = cfg.get("learning_rate", 0.01)
        bs       = cfg["batch_size"]
        results  = []

        # Build per-client model copies and optimizers
        client_models = []
        client_optims = []
        client_loaders= []
        client_ids    = []
        client_hw     = []

        criterion = torch.nn.CrossEntropyLoss()

        for cid in selected_ids:
            client   = self.registry.fleet[cid]
            hw       = client.hw

            # Hardware fault check
            if not hw.will_complete(client.rng):
                staleness = self.registry.records[cid].staleness(rnd)
                results.append(TrainResult(
                    client_id            = cid,
                    weights              = None,
                    num_samples          = 0,
                    loss                 = 0.0,
                    accuracy             = 0.0,
                    train_time           = 0.0,
                    staleness            = staleness,
                    hardware_tier        = client._tier_label(),
                    dropped              = True,
                    effective_batch_size = bs,
                ))
                continue

            # Clone global model for this client
            # m = copy.deepcopy(self.model)
            m = build_model(self.cfg)
            m.load_state_dict({k: v.clone() for k, v in self.global_weights.items()})
            m.to(self.device)
            m.train()

            opt = torch.optim.SGD(
                m.parameters(),
                lr           = lr, #* (min(hw.effective_batch_size(bs), bs) / bs),
                momentum     = cfg.get("momentum", 0.9),
                weight_decay = cfg.get("weight_decay", 0.001),
            )

            effective_bs = hw.effective_batch_size(bs)
            loader       = client._get_loader(effective_bs)

            client_models.append(m)
            client_optims.append(opt)
            client_loaders.append(loader)
            client_ids.append(cid)
            client_hw.append(hw)

        # Train all clients — back to back with no Python gaps
        
        t_start = time.time()

        for epoch in range(epochs):
            for idx in range(len(client_models)):
                m   = client_models[idx]
                opt = client_optims[idx]
                ldr = client_loaders[idx]

                for inputs, targets in ldr:
                    if inputs.device != self.device:
                        inputs  = inputs.to(self.device)
                        targets = targets.to(self.device)
                    opt.zero_grad()
                    loss = criterion(m(inputs), targets)
                    loss.backward()
                    opt.step()

        train_time = time.time() - t_start
        # if len(client_models) > 0:
        #     global_vec = torch.cat([v.flatten().cpu() for v in self.global_weights.values()])
        #     for idx, m in enumerate(client_models):
        #         client_vec = torch.cat([v.flatten().cpu() for v in m.state_dict().values()])
        #         diff = (client_vec - global_vec).norm().item()
        #         print(f"  Client {client_ids[idx]} weight divergence: {diff:.4f}")
        # Collect results from final epoch metrics
        for idx, cid in enumerate(client_ids):
            m      = client_models[idx]
            ldr    = client_loaders[idx]
            hw     = client_hw[idx]
            client = self.registry.fleet[cid]

            m.eval()
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in ldr:
                    if inputs.device != self.device:
                        inputs  = inputs.to(self.device)
                        targets = targets.to(self.device)
                    out     = m(inputs)
                    loss    = criterion(out, targets)
                    total_loss += loss.item() * inputs.size(0)
                    correct    += out.argmax(1).eq(targets).sum().item()
                    total      += inputs.size(0)

            staleness    = self.registry.records[cid].staleness(rnd)
            eff_bs       = hw.effective_batch_size(bs)
            sim_duration = hw.simulated_training_delay(train_time / len(client_ids))
            hw.record_round(completed=True, duration=sim_duration)

            results.append(TrainResult(
                client_id            = cid,
                weights              = copy.deepcopy(m.state_dict()),
                num_samples          = total,
                loss                 = total_loss / total if total > 0 else 0.0,
                accuracy             = correct / total    if total > 0 else 0.0,
                train_time           = sim_duration,
                staleness            = staleness,
                hardware_tier        = client._tier_label(),
                dropped              = False,
                effective_batch_size = eff_bs,
            ))

        return results
    # ── Evaluation ─────────────────────────────────────────────────────

    def _evaluate(self) -> tuple[float, float]:
        return self.aggregator.evaluate(
            model          = self.model,
            global_weights = self.global_weights,
            test_loader    = self.test_loader,
            device         = self.device,
        )

    # ── Persistence ────────────────────────────────────────────────────

    def _save_checkpoint(self, label):
        path = self.output_dir / f"checkpoint_{label}.pt"
        torch.save({
            "global_weights" : self.global_weights,
            "history"        : self.history,
            "config"         : self.cfg,
        }, path)

    def _save_history(self):
        path = self.output_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved → {path}")

    # ── Edge case ──────────────────────────────────────────────────────

    def _empty_round_log(self, rnd: int, events: dict) -> dict:
        return {
            "round"             : rnd,
            "selected"          : 0,
            "successful"        : 0,
            "dropped_mid_round" : 0,
            "contributing"      : 0,
            "active_pool"       : self.registry.active_count(),
            "dropped_pool"      : self.registry.dropped_count(),
            "rejoining_pool"    : self.registry.rejoining_count(),
            "newly_dropped"     : len(events.get("newly_dropped", [])),
            "rejoined"          : len(events.get("rejoined", [])),
            "avg_staleness"     : 0.0,
            "global_loss"       : 0.0,
            "global_accuracy"   : 0.0,
            "strategy"          : self.aggregator.strategy.name,
        }

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg        : dict,
        registry   : ClientRegistry,
        aggregator : Aggregator,
        test_loader,
        output_dir : Optional[str] = None,
    ) -> "FLServer":
        
        # model = SimpleCNN(num_classes=cfg["model"]["num_classes"])
        model = build_model(cfg)

        out   = output_dir or f"./outputs/{cfg['aggregation']['strategy'].lower()}"
        return cls(
            model       = model,
            registry    = registry,
            aggregator  = aggregator,
            test_loader = test_loader,
            config      = cfg,
            output_dir  = out,
        )