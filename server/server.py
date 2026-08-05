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

        sim_cfg             = config["simulation"]
        self.num_rounds     = sim_cfg["num_rounds"]
        self.clients_per_round = sim_cfg["clients_per_round"]

      
        self.eval_every     = config.get("eval_every", 5)

   
        self.checkpoint_every = config.get("checkpoint_every", 20)

        self.history = []

        print(f"FLServer ready")
        print(f"  device          : {self.device}")
        print(f"  rounds          : {self.num_rounds}")
        print(f"  clients/round   : {self.clients_per_round}")
        print(f"  strategy        : {self.aggregator.strategy.name}")
        print(f"  output dir      : {self.output_dir}\n")


    def run(self):
        
        t_start = time.time()

        for rnd in tqdm(range(self.num_rounds), desc="FL Rounds"):
            round_log = self._run_one_round(rnd)
            self.history.append(round_log)

            if rnd % self.eval_every == 0:
                self.registry.print_round_summary(rnd)


            if rnd % self.checkpoint_every == 0 and rnd > 0:
                self._save_checkpoint(rnd)

        total_time = time.time() - t_start
        print(f"\nTraining complete in {total_time/60:.1f} min")

        
        loss, acc = self._evaluate()
        print(f"Final global model — loss={loss:.4f}  acc={acc:.4f}")

      
        if self.history and self.history[-1].get("global_accuracy", 0) == 0:
            self.history[-1] = {**self.history[-1],
                                "global_loss": loss, "global_accuracy": acc}

        self._save_checkpoint("final")
        self._save_history()

        return self.history



    def _run_one_round(self, rnd: int) -> dict:

        events = self.registry.step(current_round=rnd)

        selected = self.registry.select(
            current_round    = rnd,
            k                = self.clients_per_round,
            include_rejoining= True,     #True
        )

    
        if not selected:
            return self._empty_round_log(rnd, events)

        if self.device.type == "cuda":
            results = self._run_round_batched(selected, rnd)
        else:
            results = self.registry.run_round(
                    selected_ids   = selected,
                    global_weights = self.global_weights,
                    config         = self.cfg["training"],
                    current_round  = rnd,
                )

        self.global_weights = self.aggregator.aggregate(
            results        = results,
            global_weights = self.global_weights,
            current_round  = rnd,
            ref_bs         = self.cfg["training"]["batch_size"],
        )

       
        loss, acc = 0.0, 0.0
        if rnd % self.eval_every == 0 or rnd == self.num_rounds - 1:
            loss, acc = self._evaluate()

        
        self.registry.record_results(
            results       = results,
            current_round = rnd,
            events        = events,
            global_loss   = loss,
            global_acc    = acc,
        )

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
     

        cfg      = self.cfg["training"]
        epochs   = cfg["local_epochs"]
        lr       = cfg.get("learning_rate", 0.01)
        bs       = cfg["batch_size"]
        results  = []

 
        client_models = []
        client_optims = []
        client_loaders= []
        client_ids    = []
        client_hw     = []

        criterion = torch.nn.CrossEntropyLoss()

 
        if getattr(self, "_model_pool", None) is None:
            self._model_pool = [
                build_model(self.cfg).to(self.device)
                for _ in range(self.clients_per_round)
            ]

        for cid in selected_ids:
            client   = self.registry.fleet[cid]
            hw       = client.hw

           
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

            m = self._model_pool[len(client_models)]
            m.load_state_dict({k: v.clone() for k, v in self.global_weights.items()})
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

        
        metrics = [[0.0, 0, 0] for _ in range(len(client_models))]

        t_start = time.time()

        for epoch in range(epochs):
            last_epoch = (epoch == epochs - 1)
            for idx in range(len(client_models)):
                m   = client_models[idx]
                opt = client_optims[idx]
                ldr = client_loaders[idx]
                augment = self.registry.fleet[client_ids[idx]].augment
                for inputs, targets in ldr:
                    if inputs.device != self.device:
                        inputs  = inputs.to(self.device)
                        targets = targets.to(self.device)
                    inputs = augment(inputs)
                    opt.zero_grad()
                    out  = m(inputs)
                    loss = criterion(out, targets)
                    loss.backward()
                    opt.step()
                    if last_epoch:
                        metrics[idx][0] += loss.item() * inputs.size(0)
                        metrics[idx][1] += out.argmax(1).eq(targets).sum().item()
                        metrics[idx][2] += inputs.size(0)

        train_time = time.time() - t_start

        for idx, cid in enumerate(client_ids):
            m      = client_models[idx]
            hw     = client_hw[idx]
            client = self.registry.fleet[cid]

            loss_sum, correct, total = metrics[idx]
            staleness    = self.registry.records[cid].staleness(rnd)
            eff_bs       = hw.effective_batch_size(bs)
            sim_duration = hw.simulated_training_delay(train_time / len(client_ids), client.rng)
            hw.record_round(completed=True, duration=sim_duration)

            results.append(TrainResult(
                client_id            = cid,
                weights              = copy.deepcopy(m.state_dict()),
                num_samples          = total,
                loss                 = loss_sum / total if total > 0 else 0.0,
                accuracy             = correct / total  if total > 0 else 0.0,
                train_time           = sim_duration,
                staleness            = staleness,
                hardware_tier        = client._tier_label(),
                dropped              = False,
                effective_batch_size = eff_bs,
            ))

        return results
   

    def _evaluate(self) -> tuple[float, float]:
        return self.aggregator.evaluate(
            model          = self.model,
            global_weights = self.global_weights,
            test_loader    = self.test_loader,
            device         = self.device,
        )


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