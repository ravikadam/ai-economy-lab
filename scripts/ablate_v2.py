"""Ablations for the v2 agents (paper/PROTOCOL_v2.md): switch one mechanism off at a time.

Usage: python -m scripts.ablate_v2 --runs 'runs/v2-seed*' [--seeds 400:449] [--workers 6]
Each v2 agent (in its dev-chosen action mode) and best_npv play the focal seat of the same markets with the
mechanism removed. Writes results/v2/ablations.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from training.evaluate import ABLATIONS

SAMPLE_SEEDS = (12345, 1, 2)


def _cfg(ab):
    from economy.config import load_config
    cfg = load_config("config/base.yaml")
    if ab == "full":
        return cfg
    return load_config(None, overrides={**cfg.model_dump(), "ablation": {**cfg.ablation.model_dump(), ab: False}})


def _job(args):
    run, ab, seed = args
    import torch
    torch.set_num_threads(1)
    from agents.runner import PolicyRunner
    from training.checkpoints import load_checkpoint
    from training.evaluate import run_focal_episode
    cfg = _cfg(ab)
    if run == "best_npv":
        return run, ab, seed, run_focal_episode(cfg, seed, "best_npv", None)["raw_return"]
    net, _, ck = load_checkpoint(Path(run) / "best.pt")
    runners = ([PolicyRunner(net, "cpu", deterministic=True)] if ck["meta"]["eval"]["mode"] == "greedy"
               else [PolicyRunner(net, "cpu", seed=k) for k in SAMPLE_SEEDS])
    return run, ab, seed, float(np.mean([run_focal_episode(cfg, seed, "policy", r)["raw_return"] for r in runners]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", default="400:449")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split(":"))
    seeds = list(range(lo, hi + 1))
    runs = [r for pat in a.runs for r in sorted(glob.glob(pat)) if (Path(r) / "best.pt").exists()]
    abl = ["full"] + list(ABLATIONS)
    jobs = [(r, ab, s) for ab in abl for s in seeds for r in runs + ["best_npv"]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        out = pool.map(_job, jobs, chunksize=4)
    tab = {(r, ab, s): v for r, ab, s, v in out}
    res = {"seeds": [lo, hi], "ablations": {}}
    for ab in abl:
        rule = np.array([tab[("best_npv", ab, s)] for s in seeds])
        agent = np.array([[tab[(r, ab, s)] for s in seeds] for r in runs]).mean(0)   # mean over the 5 agents
        d = agent - rule
        h = float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))
        res["ablations"][ab] = {"agent_mean": float(agent.mean()), "rule_mean": float(rule.mean()),
                                "diff": float(d.mean()), "ci95": h,
                                "per_run_diff": [float(np.mean([tab[(r, ab, s)] for s in seeds]) - rule.mean())
                                                 for r in runs]}
        print(f"{ab:22s} agents {agent.mean():6.2f}  rule {rule.mean():6.2f}  diff {d.mean():+.2f} ± {h:.2f}")
    Path("results/v2/ablations.json").write_text(json.dumps(res, indent=2))
    print(f"[ablate_v2] {time.time() - t0:.0f}s -> results/v2/ablations.json")


if __name__ == "__main__":
    main()
