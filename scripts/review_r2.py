"""Revision plan R2 (paper/PROTOCOL_v2.md).

  market     -- robustness of the market findings: one structural parameter changed at a time, in the base
                scenario and the compute crunch, on fresh markets 600-629 with the v2 seed-0 agent in the same mixed
                population as the analysis markets (scripts/experiments.py).
  holm       -- Holm correction of the per-agent and pooled test comparisons, from results/v2/test.json only.

Usage: python -m scripts.review_r2 [--workers 4]  -> results/v2/review_r2.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = list(range(600, 630))
CKPT = "results/v2/seed0/best.pt"
VARIANTS = {"normal": None,
            "adoption_steepness_2": ("demand", "adoption_steepness", 2.0),
            "adoption_steepness_8": ("demand", "adoption_steepness", 8.0),
            "competition_discount_0.06": ("demand", "competition_discount", 0.06),
            "competition_discount_0.24": ("demand", "competition_discount", 0.24),
            "entry_base_prob_0.02": ("builders", "entry_base_prob", 0.02),
            "entry_base_prob_0.08": ("builders", "entry_base_prob", 0.08),
            "price_ceiling_0.015": ("compute_price", "ceiling", 0.015),
            "price_ceiling_0.06": ("compute_price", "ceiling", 0.06),
            "confidence_diffusion_0.15": ("demand", "confidence_diffusion", 0.15),
            "confidence_diffusion_0.6": ("demand", "confidence_diffusion", 0.6),
            "fab_lead_time_12": ("fab", "capex_lead_time", 12),
            "fab_lead_time_24": ("fab", "capex_lead_time", 24)}
SCEN = {"base": "config/base.yaml", "crunch": "config/scenarios/compute_crunch.yaml"}


def write_configs() -> dict:
    """One YAML per (variant, scenario) under config/sensitivity/, so experiments.run_episode can load them."""
    import yaml
    out = {}
    d = Path("config/sensitivity")
    d.mkdir(parents=True, exist_ok=True)
    for v, change in VARIANTS.items():
        for sc, path in SCEN.items():
            if change is None:
                out[(v, sc)] = path
                continue
            sec, key, val = change
            p = d / f"{sc}_{v}.yaml"
            p.write_text(yaml.safe_dump({"extends": f"../{'base.yaml' if sc == 'base' else 'scenarios/compute_crunch.yaml'}",
                                         "name": f"{sc}_{v}", sec: {key: val}}))
            out[(v, sc)] = str(p)
    return out


def market_metrics(eps):
    """The market findings for a list of experiments.run_episode results."""
    ad = np.array([[m["sector_adoption"] for m in r["history"][1:]] for r in eps]).mean(0).T   # sectors x months
    final = ad[:, -1]
    total = ad.sum(0)
    half = int(np.argmax(total >= total[-1] / 2)) + 1
    new = np.diff(np.concatenate([[0.0], total]))
    smooth = np.convolve(new, np.ones(12) / 12, mode="same")
    peak_month = int(np.argmax(smooth)) + 1
    fm, launchers = [], []
    for r in eps:
        cust = {(f["id"], int(o)): n for f in r["firms"] for o, n in f["customers_by_opp"].items()}
        by = defaultdict(list)
        for fid, opp, m in r["launches"]:
            by[opp].append((m, fid))
        for opp, L in by.items():
            L = sorted(set(L))
            if len(L) < 2:
                continue
            tot = sum(cust.get((fid, opp), 0) for _, fid in L)
            if tot:
                fm.append(cust.get((L[0][1], opp), 0) / tot)
                launchers.append(len(L))
    F = [f for r in eps for f in r["firms"] if f["controller"] != "?"]
    inc = [f["raw"] for f in F if not f["entrant"]]
    ent = [f["raw"] for f in F if f["entrant"]]
    price = np.array([[m["compute_price"] for m in r["history"][1:]] for r in eps]).mean(0)
    return {"final_adoption": final.round(1).tolist(),
            "regulated_vs_others": float(final[2] / max(1e-9, final[:2].mean())),
            "months_to_half_adoption": half, "new_adoption_peak_month": peak_month,
            "first_mover_share": float(np.mean(fm)) if fm else None,
            "even_split_share": float(np.mean([1 / k for k in launchers])) if launchers else None,
            "founders_value": float(np.mean(inc)), "entrants_value": float(np.mean(ent)) if ent else None,
            "founders_over_entrants": float(np.mean(inc) / np.mean(ent)) if ent and np.mean(ent) > 0 else None,
            "price_peak": float(price.max() * 1000), "price_end": float(price[-1] * 1000),
            "total_value_M": float(np.mean([sum(m["welfare"] for m in r["history"]) for r in eps]) / 1000)}


def holm(pvals: dict, alpha=0.05) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, still = {}, True
    for i, k in enumerate(keys):
        thr = alpha / (m - i)
        still = still and pvals[k] <= thr
        out[k] = {"p": pvals[k], "holm_threshold": thr, "clear_after_holm": bool(still)}
    return out


def holm_analysis():
    t = json.load(open("results/v2/test.json"))["scenarios"]
    from math import erf, sqrt
    def p_two_sided(d, ci):
        se = ci / 1.959964
        z = abs(d) / se if se > 0 else math.inf
        return float(2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))))
    per = {}
    for sc, d in t.items():
        for r in d["runs"]:
            if r["run"].startswith("v2-seed"):
                per[f"{sc}|{r['run'].split('-2026')[0]}"] = p_two_sided(r["diff"], r["ci95"])
    rng = np.random.default_rng(20261022)
    pooled = {}
    for sc, d in t.items():
        rule = np.array(d["rule_per_market"])
        D = np.stack([np.array(r["per_market"]) - rule for r in d["runs"] if r["run"].startswith("v2-seed")])
        k, n = D.shape
        vals = np.array([D[rng.integers(0, k, k)][:, rng.integers(0, n, n)].mean() for _ in range(20_000)])
        pooled[sc] = float(min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()) + 1 / 20_000))
    hp = holm(per)
    primary = {k: v for k, v in per.items() if k.startswith("base|")}
    return {"per_agent": hp, "pooled": holm(pooled),
            "primary_endpoint_unadjusted_clear": int(sum(1 for k, p in primary.items() if p < 0.05)),
            "per_agent_clear_unadjusted": int(sum(p < 0.05 for p in per.values())),
            "per_agent_clear_after_holm": int(sum(v["clear_after_holm"] for v in hp.values())),
            "per_agent_clear_after_holm_by_condition": {sc: int(sum(v["clear_after_holm"] for k, v in hp.items()
                                                                    if k.startswith(sc + "|"))) for sc in t}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from scripts.experiments import _init, run_episode
    cfgs = write_configs()
    jobs = [(cfgs[(v, sc)], s) for v in VARIANTS for sc in SCEN for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(CKPT,)) as pool:
        out = pool.map(run_episode, jobs, chunksize=2)
    by = defaultdict(list)
    for r in out:
        by[r["scenario"]].append(r)
    res = {"seeds": [SEEDS[0], SEEDS[-1]], "market": {}, "holm": holm_analysis()}
    for v in VARIANTS:
        b, c = market_metrics(by[cfgs[(v, "base")]]), market_metrics(by[cfgs[(v, "crunch")]])
        res["market"][v] = {"base": b, "crunch": c,
                            "crunch_price_peak_ratio": c["price_peak"] / b["price_peak"],
                            "crunch_value_change": c["total_value_M"] / b["total_value_M"] - 1}
        print(f"{v:28s} half@{b['months_to_half_adoption']:3d} peak@{b['new_adoption_peak_month']:3d} "
              f"reg/others {b['regulated_vs_others']:.2f} first {b['first_mover_share']:.2f} vs {b['even_split_share']:.2f} "
              f"founders/entrants {b['founders_over_entrants'] or float('nan'):.2f} | crunch price x{res['market'][v]['crunch_price_peak_ratio']:.1f} "
              f"value {res['market'][v]['crunch_value_change']:+.0%}", flush=True)
    h = res["holm"]
    print(f"[holm] per-agent clear: {h['per_agent_clear_unadjusted']} unadjusted -> {h['per_agent_clear_after_holm']} "
          f"after Holm; by condition {h['per_agent_clear_after_holm_by_condition']}")
    print("[holm] pooled:", {k: (round(v['p'], 4), v['clear_after_holm']) for k, v in h['pooled'].items()})
    Path("results/v2/review_r2.json").write_text(json.dumps(res, indent=2))
    print(f"[review_r2] {time.time() - t0:.0f}s -> results/v2/review_r2.json")


if __name__ == "__main__":
    main()
