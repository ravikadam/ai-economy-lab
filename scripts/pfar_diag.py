"""Diagnostic D (paper/PROTOCOL_v2.md, plan R3): does shared randomness reduce the noise of the fork advantage?

On markets 800-819, the v2 seed-0 agent plays the focal seat against the fixed rule background. At up to 10
decisions per market where the agent and the best-expected-value rule disagree, we estimate the fork advantage
8 times with the two copies sharing their random streams (common random numbers), and 8 times with independent
streams in the two copies. The variance across the 8 repeats is compared decision by decision.

Usage: python -m scripts.pfar_diag [--workers 4]  -> results/v2/pfar_diag.json
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

SEEDS = list(range(800, 820))
K = 8
PER_MARKET = 10
HORIZON = 12


def _reseed(env, seed: int):
    from economy.rng import RngStreams
    env.world.rng = RngStreams(seed)
    env.rng = np.random.default_rng([seed, 7919])


def _market(seed):
    import torch
    torch.set_num_threads(1)
    from agents.runner import PolicyRunner
    from economy.config import load_config
    from envs.multi_env import MultiFirmEnv
    from training.checkpoints import load_checkpoint
    from training.evaluate import BACKGROUND, ENTRANT_MIX
    from training.pfar import _branch, disagrees, rule_action
    net, _, _ = load_checkpoint("results/v2/seed0/best.pt")
    cfg = load_config("config/base.yaml")
    env = MultiFirmEnv(cfg, ENTRANT_MIX, fixed_controllers={0: "policy", **BACKGROUND}, min_policy_seats=0)
    batch = env.reset(seed)
    runner = PolicyRunner(net, "cpu", seed=seed)
    rows = []
    while len(rows) < PER_MARKET:
        seats = [s for s in range(env.S) if batch.occupied[s] and batch.controller[s] == "policy"]
        acts = {s: r["action"] for s, r in runner.act(batch, seats).items()} if seats else {}
        if 0 in acts and env.seat_firm[0] >= 0:
            rule = rule_action(env, 0, np.random.default_rng([seed, env.world.month]))
            if disagrees(env, 0, acts[0], rule):
                diffs = {"crn": [], "independent": []}
                for k in range(K):
                    for mode in diffs:
                        a_env, b_env = copy.deepcopy(env), copy.deepcopy(env)
                        sa = 10_000 * seed + 100 * env.world.month + k
                        _reseed(a_env, sa)
                        _reseed(b_env, sa if mode == "crn" else sa + 5_000_000)
                        first_b = dict(acts)
                        first_b[0] = rule
                        ga, va, _ = _branch(a_env, 0, dict(acts), net, sa, HORIZON, 0.99, 1.0)
                        gb, vb, _ = _branch(b_env, 0, first_b, net, sa if mode == "crn" else sa + 5_000_000,
                                            HORIZON, 0.99, 1.0)
                        diffs[mode].append((ga - gb) + 0.99 ** HORIZON * (va - vb))
                rows.append({"seed": seed, "month": int(env.world.month),
                             "var_crn": float(np.var(diffs["crn"], ddof=1)),
                             "var_independent": float(np.var(diffs["independent"], ddof=1)),
                             "mean_crn": float(np.mean(diffs["crn"])),
                             "mean_independent": float(np.mean(diffs["independent"]))})
        out = env.step(acts)
        if out["episode_done"] or env.seat_firm[0] < 0:
            break
        batch = env.observe()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = [r for rs in pool.map(_market, SEEDS) for r in rs]
    vc = np.array([r["var_crn"] for r in rows])
    vi = np.array([r["var_independent"] for r in rows])
    ratio = vi / np.maximum(vc, 1e-12)
    res = {"decisions": len(rows), "repeats": K, "horizon": HORIZON,
           "mean_var_crn": float(vc.mean()), "mean_var_independent": float(vi.mean()),
           "variance_reduction_factor_pooled": float(vi.mean() / max(vc.mean(), 1e-12)),
           "median_ratio_per_decision": float(np.median(ratio)),
           "share_decisions_crn_lower": float(np.mean(vc < vi)),
           "share_crn_exactly_zero": float(np.mean(vc < 1e-12)), "rows": rows}
    Path("results/v2/pfar_diag.json").write_text(json.dumps(res, indent=2))
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
