"""
RQ5: trainer-agnostic sync / semi-async / async FL orchestration in virtual time.

The three modes are one knob -- how many arrivals you wait for before applying:
  sync  : full round barrier; aggregate all C, stragglers extend the round.
  semi  : buffer M arrivals (FedBuff); slots restart independently.
  async : apply each arrival (M=1, FedAsync), staleness-scaled blend.

Speed is live: a job takes virtual time = work / speed (x noise). Staleness is
live: while a client trains, other slots advance the global version, so its
update is applied stale by tau = g_now - g_start global steps.

Fair budget = equal wall-clock: all modes run to the same virtual time_budget.
Sync wastes time waiting for stragglers (fewer applies under heterogeneity);
async keeps slots busy (throughput robust) but pays staleness.

The orchestrator is weight-type agnostic -- caller supplies:
  train_fn(weights, cid)            -> (new_weights, num_samples)
  fedavg_fn(list[(weights, n)])     -> sample-weighted mean (sync round + buffer mean)
  blend_fn(global_w, client_w, lam) -> (1-lam)*global_w + lam*client_w
  eval_fn(weights)                  -> scalar (e.g. accuracy)
so the same code drives the numpy mock and the real torch state_dicts.
"""
import heapq
import numpy as np


def staleness_weight(tau, a=0.5):
    """FedAsync polynomial discount; a=0 disables staleness weighting (s=1)."""
    return (1.0 + tau) ** (-a)


class AsyncOrchestrator:
    def __init__(self, client_ids, speeds, init_weights,
                 train_fn, fedavg_fn, blend_fn, eval_fn,
                 mode="async", buffer_size=1, concurrency=10,
                 alpha=0.6, staleness_a=0.5, work=1.0, speed_noise=0.10,
                 time_budget=200.0, eval_dt=5.0, seed=0, active_fn=None):
        self.client_ids = list(client_ids)
        self.speeds = dict(zip(self.client_ids, speeds)) if not isinstance(speeds, dict) else speeds
        self.init_weights = init_weights
        self.train_fn, self.fedavg_fn, self.blend_fn, self.eval_fn = train_fn, fedavg_fn, blend_fn, eval_fn
        self.mode, self.M, self.C = mode, buffer_size, concurrency
        self.alpha, self.sa, self.work, self.noise = alpha, staleness_a, work, speed_noise
        self.time_budget, self.eval_dt, self.seed = time_budget, eval_dt, seed
        self.active_fn = active_fn or (lambda cid, g: True)

    def _dur(self, cid, rng):
        d = self.work / self.speeds[cid] * (1.0 + self.noise * rng.standard_normal())
        return max(d, 1e-3)

    def _apply_buffer(self, w, buf, g):
        # buf: list of (w_client, n, g_start, cid). lam_eff = alpha * mean staleness discount.
        ws = [s for (_, _, gs, _) in buf for s in [staleness_weight(g - gs, self.sa)]]
        ns = [n for (_, n, _, _) in buf]
        mean_client = self.fedavg_fn([(wc, n) for (wc, n, _, _) in buf])
        num = sum(s * n for s, n in zip(ws, ns)); den = sum(ns)
        lam = self.alpha * (num / den if den else 1.0)
        return self.blend_fn(w, mean_client, min(max(lam, 0.0), 1.0))

    def run(self):
        rng = np.random.default_rng(self.seed)
        w, g, t = self.init_weights, 0, 0.0
        hist, next_eval, total_applied = [], 0.0, 0
        contrib = {cid: 0 for cid in self.client_ids}   # finishes per client
        stale_log = []

        def record():
            hist.append(dict(t=t, g=g, acc=float(self.eval_fn(w)),
                             avg_staleness=float(np.mean(stale_log[-50:])) if stale_log else 0.0))

        if self.mode == "sync":
            while t < self.time_budget:
                active = [c for c in self.client_ids if self.active_fn(c, g)]
                if not active: break
                k = min(self.C, len(active))
                cohort = [int(c) for c in rng.choice(active, size=k, replace=False)]
                trained = [(self.train_fn(w, c)) for c in cohort]       # (wc, n) each, all from w
                durs = [self._dur(c, rng) for c in cohort]
                t += max(durs)                                          # wait for slowest
                for c in cohort: contrib[c] += 1
                w = self.fedavg_fn(trained)                            # FedAvg, staleness 0
                g += 1; total_applied += 1; stale_log.append(0.0)
                if t >= next_eval: record(); next_eval += self.eval_dt
            record(); 
            return dict(history=hist, applied=total_applied, contrib=contrib, end_t=t, mode=self.mode)

        # ---- async / semi: independent slots, event-driven ----
        heap, buf, seq = [], [], 0
        active = [c for c in self.client_ids if self.active_fn(c, g)]
        k = min(self.C, len(active))
        cohort = [int(c) for c in rng.choice(active, size=k, replace=False)] if active else []
        for c in cohort:
            wc, n = self.train_fn(w, c)
            heapq.heappush(heap, (t + self._dur(c, rng), seq, c, wc, g, n)); seq += 1
        while heap and t < self.time_budget:
            ft, _, cid, wc, g_start, n = heapq.heappop(heap)
            t = ft
            buf.append((wc, n, g_start, cid)); contrib[cid] += 1
            if len(buf) >= self.M:
                for (_, _, gs, _) in buf: stale_log.append(g - gs)
                w = self._apply_buffer(w, buf, g); g += 1; total_applied += 1; buf = []
                if t >= next_eval: record(); next_eval += self.eval_dt
            # Refill the freed slot with a NEW uniform-random client from the
            # active pool not currently in flight, so the C slots always hold C
            # distinct clients and, over the budget, all clients participate.
            # Was: nxt = cid -> the same initial C clients restarted forever.
            nxt = self._draw(rng, g, heap)
            if nxt is not None:
                nwc, nn = self.train_fn(w, nxt)
                heapq.heappush(heap, (t + self._dur(nxt, rng), seq, nxt, nwc, g, nn)); seq += 1
        record()
        return dict(history=hist, applied=total_applied, contrib=contrib, end_t=t, mode=self.mode)
    
    def _draw(self, rng, g, heap):
        """Uniformly pick an active client not currently in flight, using the
        run's rng (reproducible). Sampling without replacement across the C
        concurrent slots; the just-freed client is eligible again (it has been
        popped from the heap), so draws stay uniform over the active pool."""
        inflight = {item[2] for item in heap}
        pool = [c for c in self.client_ids if self.active_fn(c, g) and c not in inflight]
        if not pool:
            return None
        return int(pool[rng.integers(len(pool))])

    # def _replacement(self, g, heap):
    #     inflight = {item[2] for item in heap}
    #     for c in self.client_ids:
    #         if self.active_fn(c, g) and c not in inflight:
    #             return c
    #     return None