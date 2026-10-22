"""Revision plan R1, items 3-5, on fresh markets 500-549 (paper/PROTOCOL_v2.md).

  yardstick   -- oracle (best expected value given the true hidden difficulty), the four rules and the five
                 agents in the focal seat, base scenario and compute crunch.
  sensitivity -- the five agents and best expected value with one parameter changed at a time.
  crunch      -- month-by-month diagnostics of the focal company (agents vs best expected value) in the crunch.

Usage: python -m scripts.review_r1 [--workers 3]  -> results/v2/review_r1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

SEEDS = list(range(500, 550))
SAMPLE_SEEDS = (12345, 1, 2)
RULES = ["oracle", "best_npv", "cash_preserving", "easy_first", "random"]
SENS = {"competition_discount_0.06": ("demand", "competition_discount", 0.06),
        "competition_discount_0.24": ("demand", "competition_discount", 0.24),
        "credit_cost_0.006": ("builders", "interest_rate", 0.006),
        "credit_cost_0.024": ("builders", "interest_rate", 0.024),
        "price_share_0.24": ("demand", "price_share", 0.24),
        "price_share_0.40": ("demand", "price_share", 0.40),
        "overhead_15": ("builders", "overhead", 15.0),
        "overhead_40": ("builders", "overhead", 40.0)}
SCEN = {"base": "config/base.yaml", "compute_crunch": "config/scenarios/compute_crunch.yaml"}


def _cfg(name):
    from economy.config import load_config
    if name in SCEN:
        return load_config(SCEN[name])
    sec, key, val = SENS[name]
    base = load_config(SCEN["base"]).model_dump()
    base[sec][key] = val
    return load_config(None, overrides=base)


def _runners(run):
    from agents.runner import PolicyRunner
    from training.checkpoints import load_checkpoint
    net, _, ck = load_checkpoint(Path(run) / "best.pt")
    if ck["meta"]["eval"]["mode"] == "greedy":
        return [PolicyRunner(net, "cpu", deterministic=True)]
    return [PolicyRunner(net, "cpu", seed=k) for k in SAMPLE_SEEDS]


def _diag_episode(cfg, seed, focal, runner):
    """run_focal_episode with a month-by-month record of the focal company."""
    from entities.firms import DEV_STAGES
    from envs.multi_env import MultiFirmEnv
    from training.evaluate import BACKGROUND, ENTRANT_MIX
    ctrl = "policy" if focal == "policy" else f"heuristic:{focal}"
    env = MultiFirmEnv(cfg, ENTRANT_MIX, fixed_controllers={0: ctrl, **BACKGROUND}, min_policy_seats=0)
    batch = env.reset(seed)
    fid = int(env.seat_firm[0])
    months = []
    while True:
        seats = [s for s in range(env.S) if batch.occupied[s] and batch.controller[s] == "policy"]
        acts = {s: r["action"] for s, r in (runner.act(batch, seats).items() if runner and seats else [])}
        out = env.step(acts)
        w = env.world
        f = w.firms[fid]
        dev = [p for p in f.projects if p.stage in DEV_STAGES]
        months.append({"m": w.month, "alive": bool(f.alive), "dev": len(dev), "live": sum(p.is_live for p in f.projects),
                       "premium": sum(p.tier == 2 for p in dev), "frontier": sum(p.model_id == 0 for p in dev),
                       "burn": float(w.monthly_burn(f)) if f.alive else 0.0,
                       "runway": float(w.runway(f)) if f.alive else 0.0, "debt": float(f.debt),
                       "price": float(w.compute_price)})
        if out["episode_done"] or not f.alive:
            break
        batch = env.observe()
    return {"raw": float(f.cum_raw_reward), "survived": bool(f.alive), "death": f.death_month, "months": months}


def _job(args):
    kind, cand, scen, seed = args
    import torch
    torch.set_num_threads(1)
    from training.evaluate import run_focal_episode
    cfg = _cfg(scen)
    if kind == "diag":
        rs = [None] if cand == "best_npv" else _runners(cand)
        reps = [_diag_episode(cfg, seed, "best_npv" if cand == "best_npv" else "policy", r) for r in rs]
        return kind, cand, scen, seed, reps
    if cand in RULES:
        r = run_focal_episode(cfg, seed, cand, None)
        return kind, cand, scen, seed, (r["raw_return"], float(r["survived"]))
    reps = [run_focal_episode(cfg, seed, "policy", rn) for rn in _runners(cand)]
    return kind, cand, scen, seed, (float(np.mean([x["raw_return"] for x in reps])),
                                    float(np.mean([x["survived"] for x in reps])))


def boot2(D, rng, B=10_000):
    """Two-level bootstrap (runs x markets) of the mean of a runs x markets matrix."""
    n, k = D.shape[1], D.shape[0]
    vals = [D[rng.integers(0, k, k)][:, rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    agents = sorted(glob.glob("runs/v2-seed*"))
    jobs = []
    for scen in SCEN:
        jobs += [("yard", c, scen, s) for c in RULES + agents for s in SEEDS]
    for scen in SENS:
        jobs += [("sens", c, scen, s) for c in ["best_npv"] + agents for s in SEEDS]
    jobs += [("diag", c, "compute_crunch", s) for c in ["best_npv"] + agents for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            out.append(r)
            if (i + 1) % 250 == 0:
                print(f"[review_r1] {i + 1}/{len(jobs)} games, {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(20261022)
    res = {"seeds": [SEEDS[0], SEEDS[-1]], "yardstick": {}, "sensitivity": {}, "crunch": {}}
    tab = {(k, c, sc, s): v for k, c, sc, s, v in out}
    for scen in SCEN:
        rows = {}
        for c in RULES:
            v = np.array([tab[("yard", c, scen, s)] for s in SEEDS])
            rows[c] = {"mean": float(v[:, 0].mean()), "survival": float(v[:, 1].mean())}
        A = np.array([[tab[("yard", r, scen, s)][0] for s in SEEDS] for r in agents])
        rows["agents"] = {"mean": float(A.mean()), "survival": float(np.mean([[tab[("yard", r, scen, s)][1]
                                                                                for s in SEEDS] for r in agents]))}
        rule = np.array([tab[("yard", "best_npv", scen, s)][0] for s in SEEDS])
        orc = np.array([tab[("yard", "oracle", scen, s)][0] for s in SEEDS])
        d = orc - rule
        idx = rng.integers(0, len(d), (10_000, len(d)))
        rows["oracle_minus_rule"] = {"mean": float(d.mean()), "interval": [float(np.percentile(d[idx].mean(1), 2.5)),
                                                                           float(np.percentile(d[idx].mean(1), 97.5))]}
        rows["agents_minus_oracle"] = {"mean": float((A - orc).mean()), "interval": boot2(A - orc, rng)}
        rows["agents_minus_rule"] = {"mean": float((A - rule).mean()), "interval": boot2(A - rule, rng)}
        res["yardstick"][scen] = rows
    for scen in SENS:
        rule = np.array([tab[("sens", "best_npv", scen, s)][0] for s in SEEDS])
        A = np.array([[tab[("sens", r, scen, s)][0] for s in SEEDS] for r in agents])
        res["sensitivity"][scen] = {"rule": float(rule.mean()), "agents": float(A.mean()),
                                    "lead": float((A - rule).mean()), "interval": boot2(A - rule, rng),
                                    "per_run_lead": [float((A[i] - rule).mean()) for i in range(len(agents))]}
    base = res["yardstick"]["base"]
    res["sensitivity"]["base_0.12_0.012_0.32_25"] = {"rule": base["best_npv"]["mean"], "agents": base["agents"]["mean"],
                                                    "lead": base["agents_minus_rule"]["mean"],
                                                    "interval": base["agents_minus_rule"]["interval"]}
    # crunch diagnostics: averages over months alive, split by survived / bankrupt
    for who in ["best_npv", "agents"]:
        eps = [e for c in (["best_npv"] if who == "best_npv" else agents) for s in SEEDS
               for e in tab[("diag", c, "compute_crunch", s)]]
        def summ(group):
            ms = [m for e in group for m in e["months"] if m["alive"]]
            if not ms:
                return {}
            early = [m for e in group for m in e["months"] if m["alive"] and m["m"] <= 36]
            return {"companies": len(group), "dev_projects": float(np.mean([m["dev"] for m in ms])),
                    "premium_share": float(np.sum([m["premium"] for m in ms]) / max(1, np.sum([m["dev"] for m in ms]))),
                    "frontier_share": float(np.sum([m["frontier"] for m in ms]) / max(1, np.sum([m["dev"] for m in ms]))),
                    "burn_k": float(np.mean([m["burn"] for m in ms])), "runway_months": float(np.mean([m["runway"] for m in ms])),
                    "debt_k": float(np.mean([m["debt"] for m in ms])),
                    "dev_projects_first36": float(np.mean([m["dev"] for m in early])) if early else None,
                    "burn_k_first36": float(np.mean([m["burn"] for m in early])) if early else None,
                    "death_month_median": float(np.median([e["death"] for e in group if e["death"] is not None]))
                    if any(e["death"] is not None for e in group) else None}
        res["crunch"][who] = {"survival": float(np.mean([e["survived"] for e in eps])),
                              "all": summ(eps), "survived": summ([e for e in eps if e["survived"]]),
                              "bankrupt": summ([e for e in eps if not e["survived"]])}
    Path("results/v2/review_r1.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("yardstick", "sensitivity")}, indent=1)[:4000])
    print(json.dumps(res["crunch"], indent=1))
    print(f"[review_r1] {time.time() - t0:.0f}s -> results/v2/review_r1.json")


if __name__ == "__main__":
    main()
