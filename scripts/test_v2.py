"""One-time test of the v2 agents on fresh markets (paper/PROTOCOL_v2.md).

Usage: python -m scripts.test_v2 --runs 'runs/v2-seed*' [--workers 6]
Base scenario on seeds 400-499; each named regime on seeds 400-449. Each agent plays the focal seat in the
action mode it chose on the dev set (stored in best.pt); best_npv plays the same seat in the same markets.
Writes results/v2/test.json.
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

SAMPLE_SEEDS = (12345, 1, 2)
SCENARIOS = {"base": ("config/base.yaml", range(400, 500))}
for _n in ["compute_crunch", "entrant_wave", "fast_tech", "slow_tech", "harsh_failures"]:
    SCENARIOS[_n] = (f"config/scenarios/{_n}.yaml", range(400, 450))


def _job(args):
    run, scen, seed = args
    import torch
    torch.set_num_threads(1)
    from agents.runner import PolicyRunner
    from economy.config import load_config
    from training.checkpoints import load_checkpoint
    from training.evaluate import run_focal_episode
    cfg = load_config(SCENARIOS[scen][0])
    if run == "best_npv":
        r = run_focal_episode(cfg, seed, "best_npv", None)
        return run, scen, seed, r["raw_return"], float(r["survived"]), r["launches"]
    net, _, ck = load_checkpoint(Path(run) / "best.pt")
    mode = ck["meta"]["eval"]["mode"]
    runners = ([PolicyRunner(net, "cpu", deterministic=True)] if mode == "greedy"
               else [PolicyRunner(net, "cpu", seed=k) for k in SAMPLE_SEEDS])
    reps = [run_focal_episode(cfg, seed, "policy", rn) for rn in runners]
    return (run, scen, seed, float(np.mean([x["raw_return"] for x in reps])),
            float(np.mean([x["survived"] for x in reps])), float(np.mean([x["launches"] for x in reps])))


def ci95(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    runs = [r for pat in a.runs for r in sorted(glob.glob(pat)) if (Path(r) / "best.pt").exists()]
    jobs = [(r, sc, s) for sc, (_, seeds) in SCENARIOS.items() for s in seeds for r in runs + ["best_npv"]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        out = pool.map(_job, jobs, chunksize=4)
    import torch
    modes = {r: torch.load(Path(r) / "best.pt", weights_only=False)["meta"] for r in runs}
    res = {"protocol": "paper/PROTOCOL_v2.md", "sample_seeds": list(SAMPLE_SEEDS),
           "agents": {Path(r).name: {"mode": m["eval"]["mode"], "env_steps": m["env_steps"], "dev": m["eval"]}
                      for r, m in modes.items()}, "scenarios": {}}
    for sc, (_, seeds) in SCENARIOS.items():
        tab = {(r, s): (v, sv, la) for r, c, s, v, sv, la in out if c == sc}
        rule = np.array([tab[("best_npv", s)][0] for s in seeds])
        per = []
        for r in runs:
            pol = np.array([tab[(r, s)][0] for s in seeds])
            d, h = ci95(pol - rule)
            per.append({"run": Path(r).name, "agent_mean": float(pol.mean()), "diff": d, "ci95": h,
                        "better": bool(d - h > 0), "worse": bool(d + h < 0),
                        "survival": float(np.mean([tab[(r, s)][1] for s in seeds])),
                        "launches": float(np.mean([tab[(r, s)][2] for s in seeds])),
                        "per_market": pol.tolist()})
        diffs = np.array([p["diff"] for p in per])
        res["scenarios"][sc] = {
            "seeds": [seeds[0], seeds[-1]], "n_markets": len(seeds), "rule_mean": float(rule.mean()),
            "rule_survival": float(np.mean([tab[("best_npv", s)][1] for s in seeds])),
            "rule_launches": float(np.mean([tab[("best_npv", s)][2] for s in seeds])),
            "rule_per_market": rule.tolist(), "runs": per, "diff_mean": float(diffs.mean()),
            "diff_sd": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "diff_min": float(diffs.min()), "diff_max": float(diffs.max()),
            "n_better": int(sum(p["better"] for p in per)), "n_worse": int(sum(p["worse"] for p in per))}
    Path("results/v2").mkdir(parents=True, exist_ok=True)
    Path("results/v2/test.json").write_text(json.dumps(res, indent=2))
    for sc, r in res["scenarios"].items():
        print(f"== {sc} ({r['n_markets']} markets): rule {r['rule_mean']:.2f} $M, survival {r['rule_survival']:.0%}")
        for p in r["runs"]:
            tag = "better" if p["better"] else ("WORSE" if p["worse"] else "n.s.")
            print(f"  {p['run']:10s} agent {p['agent_mean']:6.2f}  diff {p['diff']:+.2f} ± {p['ci95']:.2f} {tag:6s} "
                  f"survival {p['survival']:.0%}")
        print(f"  mean diff {r['diff_mean']:+.2f} (range {r['diff_min']:+.2f} to {r['diff_max']:+.2f}); "
              f"{r['n_better']} better, {r['n_worse']} worse")
    print(f"[test_v2] {time.time() - t0:.0f}s -> results/v2/test.json")


if __name__ == "__main__":
    main()
