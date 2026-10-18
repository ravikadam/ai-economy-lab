"""Evaluate several independently trained agents on the same unseen markets (training-seed robustness).

Usage: python -m scripts.multiseed_eval --runs runs/ppo-20261018-113906 runs/ppo-seed1-* ... --seeds 300:399
Each agent plays the focal seat in every market; the best-expected-value rule plays the same seat in the
same markets once, so every agent's difference is paired market by market. Writes results/multiseed.json.
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
SCENARIOS = {"base": "config/base.yaml", "compute_crunch": "config/scenarios/compute_crunch.yaml"}


def _job(args):
    run, scen_path, seeds = args
    import torch
    torch.set_num_threads(1)
    from agents.runner import PolicyRunner
    from economy.config import load_config
    from training.checkpoints import load_checkpoint
    from training.evaluate import run_focal_episode
    cfg = load_config(scen_path)
    if run == "best_npv":
        return run, scen_path, [run_focal_episode(cfg, s, "best_npv", None) for s in seeds]
    net, _, _ = load_checkpoint(run)
    rows = []
    for s in seeds:
        # the agent samples its actions: average over SAMPLE_SEEDS repeats, each with a fresh seeded stream
        reps = [run_focal_episode(cfg, s, "policy", PolicyRunner(net, "cpu", seed=k)) for k in SAMPLE_SEEDS]
        rows.append({"seed": s, "raw_return": float(np.mean([r["raw_return"] for r in reps])),
                     "survived": float(np.mean([r["survived"] for r in reps])),
                     "reps": [r["raw_return"] for r in reps]})
    return run, scen_path, rows


def ci95(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", default="300:399")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split(":"))
    seeds = list(range(lo, hi + 1))
    runs = [r for pat in a.runs for r in sorted(glob.glob(pat))]
    jobs = [(r, p, seeds) for p in SCENARIOS.values() for r in runs + ["best_npv"]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        out = pool.map(_job, jobs)
    res = {}
    for name, path in SCENARIOS.items():
        base = {r: rows for r, p, rows in out if p == path}
        rule = np.array([x["raw_return"] for x in base["best_npv"]])
        per_run = []
        for r in runs:
            pol = np.array([x["raw_return"] for x in base[r]])
            d, h = ci95(pol - rule)
            per_run.append({"run": Path(r).name, "agent_mean": float(pol.mean()), "diff": d, "ci95": h,
                            "significant": bool(d - h > 0),
                            "survival": float(np.mean([x["survived"] for x in base[r]]))})
        diffs = np.array([p["diff"] for p in per_run])
        agents = np.array([p["agent_mean"] for p in per_run])
        res[name] = {"n_markets": len(seeds), "rule_mean": float(rule.mean()),
                     "rule_survival": float(np.mean([x["survived"] for x in base["best_npv"]])),
                     "runs": per_run, "n_runs": len(runs),
                     "diff_mean_across_runs": float(diffs.mean()), "diff_sd_across_runs": float(diffs.std(ddof=1)),
                     "diff_min": float(diffs.min()), "diff_max": float(diffs.max()),
                     "agent_mean_across_runs": float(agents.mean()), "agent_sd_across_runs": float(agents.std(ddof=1)),
                     "runs_significantly_better": int(sum(p["significant"] for p in per_run))}
    for name, path in SCENARIOS.items():
        base = {r: rows for r, p, rows in out if p == path}
        within = [np.mean([np.std(x["reps"], ddof=1) for x in base[r]]) for r in runs]
        res[name]["sampling_sd_within_market"] = float(np.mean(within))
        res[name]["sample_seeds"] = list(SAMPLE_SEEDS)
    Path("results").mkdir(exist_ok=True)
    Path("results/multiseed.json").write_text(json.dumps(res, indent=2))
    for name, r in res.items():
        print(f"== {name}: rule {r['rule_mean']:.2f} $M (survival {r['rule_survival']:.0%})")
        for p in r["runs"]:
            print(f"  {p['run']:32s} agent {p['agent_mean']:6.2f}  diff {p['diff']:+.2f} ± {p['ci95']:.2f} "
                  f"{'*' if p['significant'] else ' '} survival {p['survival']:.0%}")
        print(f"  across {r['n_runs']} runs: diff {r['diff_mean_across_runs']:+.2f} (sd {r['diff_sd_across_runs']:.2f}, "
              f"range {r['diff_min']:+.2f} to {r['diff_max']:+.2f}); {r['runs_significantly_better']}/{r['n_runs']} "
              f"significantly better")
    print(f"[multiseed] {time.time() - t0:.0f}s -> results/multiseed.json")


if __name__ == "__main__":
    main()
