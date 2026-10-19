"""Development-set evaluation used only for checkpoint and action-mode selection.

Dev markets: 20 base-scenario markets (seeds 1000-1019) plus 20 randomised-regime markets
(seeds 2000-2019, drawn from the training randomisation distribution). Each market is
played by the candidate in the focal seat in two modes: "sampled" (average of two seeded
draws) and "greedy" (most likely action). These seeds never appear in training or testing.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

DEV_BASE = list(range(1000, 1020))
DEV_DR = list(range(2000, 2020))
DRAWS = (12345, 1)

_net = None


def _init(cfg_dict, hidden, emb):
    import torch
    torch.set_num_threads(1)
    global _net
    from economy.config import ScenarioConfig
    from training.checkpoints import build_net
    _net = build_net(ScenarioConfig.model_validate(cfg_dict), hidden, emb).eval()


def _market(args):
    state, cfg_dict, seed, dr, rule_only = args
    from agents.runner import PolicyRunner
    from economy.config import ScenarioConfig
    from training.evaluate import run_focal_episode
    from training.randomize import config_for_seed
    base = ScenarioConfig.model_validate(cfg_dict)
    cfg = config_for_seed(base, seed) if dr else base
    if rule_only:
        return run_focal_episode(cfg, seed, "best_npv", None)["raw_return"]
    _net.load_state_dict(state)
    sampled = np.mean([run_focal_episode(cfg, seed, "policy", PolicyRunner(_net, "cpu", seed=k))["raw_return"]
                       for k in DRAWS])
    greedy = run_focal_episode(cfg, seed, "policy", PolicyRunner(_net, "cpu", deterministic=True))["raw_return"]
    return float(sampled), float(greedy)


class DevEvaluator:
    def __init__(self, cfg, hidden: int, emb: int, workers: int = 6):
        self.cfg_dict = cfg.model_dump()
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(self.cfg_dict, hidden, emb))
        self.markets = [(s, False) for s in DEV_BASE] + [(s, True) for s in DEV_DR]
        rule = self.pool.map(_market, [(None, self.cfg_dict, s, dr, True) for s, dr in self.markets])
        self.rule = np.array(rule)

    def __call__(self, net) -> dict:
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        out = np.array(self.pool.map(_market, [(state, self.cfg_dict, s, dr, False) for s, dr in self.markets]))
        sampled, greedy = out[:, 0], out[:, 1]
        nb = len(DEV_BASE)
        res = {"sampled": float(sampled.mean()), "greedy": float(greedy.mean()), "rule": float(self.rule.mean()),
               "sampled_base": float(sampled[:nb].mean()), "sampled_dr": float(sampled[nb:].mean()),
               "greedy_base": float(greedy[:nb].mean()), "greedy_dr": float(greedy[nb:].mean()),
               "rule_base": float(self.rule[:nb].mean()), "rule_dr": float(self.rule[nb:].mean())}
        res["mode"] = "greedy" if res["greedy"] > res["sampled"] else "sampled"
        res["score"] = max(res["greedy"], res["sampled"])
        return res

    def close(self):
        self.pool.close()
        self.pool.join()
