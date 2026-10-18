"""Held-out evaluation against baselines (spec §9, §14, M4).

Protocol "focal seat": in each seed, seat 0 is controlled by the candidate
(policy or a baseline) while seats 1-4 and all entrants follow a FIXED
background of rule-based firms. The world RNG streams depend only on the seed,
so every candidate faces the same latent world (common random numbers) and
differences are paired.

Usage:
  python -m training.evaluate --checkpoint runs/latest --seeds 100:129
  python -m training.evaluate --checkpoint runs/latest --seeds 100:129 --scenarios config/scenarios/*.yaml --ablations
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

from agents.runner import PolicyRunner
from economy.config import ScenarioConfig, load_config
from envs.multi_env import MultiFirmEnv

BACKGROUND = {1: "heuristic:best_npv", 2: "heuristic:easy_first", 3: "heuristic:cash_preserving",
              4: "heuristic:best_npv"}
ENTRANT_MIX = {"heuristic:best_npv": 0.4, "heuristic:easy_first": 0.4, "heuristic:cash_preserving": 0.2}
BASELINES = ["random", "easy_first", "best_npv", "cash_preserving"]
ABLATIONS = ["competition", "technical_failure", "confidence_feedback", "supply_bottlenecks", "delayed_capex",
             "risk_heterogeneity"]


def parse_seeds(s: str) -> list[int]:
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float(x.mean()) if x.size else 0.0, 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(x.size))


def run_focal_episode(cfg: ScenarioConfig, seed: int, focal: str, runner: PolicyRunner | None) -> dict:
    ctrl = "policy" if focal == "policy" else f"heuristic:{focal}"
    env = MultiFirmEnv(cfg, ENTRANT_MIX, fixed_controllers={0: ctrl, **BACKGROUND}, min_policy_seats=0)
    batch = env.reset(seed)
    focal_fid = int(env.seat_firm[0])
    disc_profit, g = 0.0, 1.0
    while True:
        seats = [s for s in range(env.S) if batch.occupied[s] and batch.controller[s] == "policy"]
        acts = {s: r["action"] for s, r in (runner.act(batch, seats).items() if runner and seats else [])}
        out = env.step(acts)
        f = env.world.firms[focal_fid]
        if f.monthly_profit_hist:
            disc_profit += g * f.monthly_profit_hist[-1]
        g *= 0.99
        if out["episode_done"]:
            break
        batch = env.observe()
    w = env.world
    f = w.firms[focal_fid]
    dev = [p for p in getattr(f, "closed_projects", [])] + list(f.projects)
    explored_d = [w.opps[p.opportunity_id].latent_difficulty for p in dev]
    return {"seed": seed, "focal": focal, "raw_return": f.cum_raw_reward, "shaped_return": f.cum_shaped_reward,
            "survived": f.alive, "net_worth": w.net_worth(f), "discounted_profit": disc_profit,
            "launches": f.launches, "tech_failures": f.tech_failures, "commercial_failures": f.commercial_failures,
            "deferrals": f.deferrals, "illegal": f.illegal_actions, "profile": f.profile.name,
            "mean_difficulty_attempted": float(np.mean(explored_d)) if explored_d else float("nan"),
            "projects_attempted": len(dev), "market_welfare": float(sum(r["welfare"] for r in w.price_history)),
            "hhi_final": w.price_history[-1]["hhi"]}


def evaluate_focal(net, cfg: ScenarioConfig, seeds: list[int], focal: str = "policy", device: str = "cpu",
                   deterministic: bool = False) -> dict:
    runner = PolicyRunner(net, device, deterministic=deterministic, seed=12345) if net is not None else None
    rows = [run_focal_episode(cfg, s, focal, runner) for s in seeds]
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    r = [x["raw_return"] for x in rows]
    m, h = ci95(r)
    out = {"n": len(rows), "raw_return_mean": m, "raw_return_ci95": h,
           "shaped_return_mean": float(np.mean([x["shaped_return"] for x in rows])),
           "survival": float(np.mean([x["survived"] for x in rows])),
           "discounted_profit_mean": float(np.mean([x["discounted_profit"] for x in rows])),
           "downside_p10": float(np.percentile(r, 10)) if r else 0.0,
           "launches_mean": float(np.mean([x["launches"] for x in rows])),
           "tech_fail_rate": _rate(rows, "tech_failures"), "commercial_fail_rate": _rate(rows, "commercial_failures"),
           "mean_difficulty_attempted": float(np.nanmean([x["mean_difficulty_attempted"] for x in rows]))
           if any(not math.isnan(x["mean_difficulty_attempted"]) for x in rows) else float("nan"),
           "illegal_actions": int(sum(x["illegal"] for x in rows))}
    return out


def _rate(rows, key):
    att = sum(x["projects_attempted"] for x in rows)
    return float(sum(x[key] for x in rows) / att) if att else 0.0


def compare(net, cfg, seeds, device="cpu", include_oracle=False) -> dict:
    cands = (["policy"] if net is not None else []) + BASELINES + (["oracle"] if include_oracle else [])
    rows = {c: [run_focal_episode(cfg, s, c, PolicyRunner(net, device, seed=12345) if c == "policy" else None)
                for s in seeds] for c in cands}
    summ = {c: summarize(v) for c, v in rows.items()}
    if net is not None:
        base = np.array([x["raw_return"] for x in rows["policy"]])
        for c in cands:
            if c == "policy":
                continue
            d = base - np.array([x["raw_return"] for x in rows[c]])
            m, h = ci95(d)
            summ[c]["policy_minus_this"] = m
            summ[c]["policy_minus_this_ci95"] = h
        best = max((c for c in BASELINES), key=lambda c: summ[c]["raw_return_mean"])
        summ["policy"]["regret_vs_best_heuristic"] = summ[best]["raw_return_mean"] - summ["policy"]["raw_return_mean"]
        summ["policy"]["best_heuristic"] = best
    return {"summary": summ, "rows": rows}


def fmt_table(summ: dict) -> str:
    lines = ["| controller | raw return $M (±95% CI) | survival | disc. profit $k | p10 return | launches | "
             "tech-fail rate | comm-fail rate | policy − this |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c, s in summ.items():
        diff = f"{s['policy_minus_this']:+.2f} ± {s['policy_minus_this_ci95']:.2f}" if "policy_minus_this" in s else "—"
        lines.append(f"| {c} | {s['raw_return_mean']:.2f} ± {s['raw_return_ci95']:.2f} | {s['survival']:.2f} | "
                     f"{s['discounted_profit_mean']:.0f} | {s['downside_p10']:.2f} | {s['launches_mean']:.2f} | "
                     f"{s['tech_fail_rate']:.2f} | {s['commercial_fail_rate']:.2f} | {diff} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/latest")
    ap.add_argument("--seeds", default="100:129")
    ap.add_argument("--config", default=None, help="scenario YAML (default: the checkpoint's own)")
    ap.add_argument("--scenarios", nargs="*", default=[], help="held-out scenario YAMLs (globs ok)")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--no-policy", action="store_true", help="baselines only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)

    net, cfg = None, load_config(args.config) if args.config else None
    ck_dir = None
    if not args.no_policy:
        from training.checkpoints import load_checkpoint, resolve
        ck_path = resolve(args.checkpoint)
        net, ck_cfg, _ = load_checkpoint(ck_path)
        cfg = cfg or ck_cfg
        ck_dir = ck_path.parent
    cfg = cfg or load_config("config/base.yaml")
    out_dir = Path(args.out) if args.out else (ck_dir or Path("runs")) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    report = {"seeds": seeds, "base": compare(net, cfg, seeds)["summary"]}
    md = [f"# AI Economy Lab — evaluation report", "",
          f"Checkpoint: `{args.checkpoint}`  ·  seeds {seeds[0]}–{seeds[-1]} (n={len(seeds)}, unseen in training)  ·  "
          f"generated {time.strftime('%Y-%m-%d %H:%M')}", "",
          "Protocol: focal seat 0 is the candidate; seats 1–4 and entrants are a fixed rule-based background. "
          "Same seed ⇒ same latent world for every candidate (paired comparison). Raw return = Δ net worth in $M "
          "(cash − debt + PV of live products).", "",
          f"## Base scenario: `{cfg.name}`", "", fmt_table(report["base"]), ""]
    for pattern in args.scenarios:
        for path in sorted(glob.glob(pattern)):
            sc = load_config(path)
            r = compare(net, sc, seeds)["summary"]
            report[f"scenario:{sc.name}"] = r
            md += [f"## Held-out scenario: `{sc.name}` ({path})", "", fmt_table(r), ""]
    if args.ablations:
        md += ["## Ablations (one mechanism removed at a time)", "",
               "| ablation | policy raw return | best_npv raw return | policy − best_npv |", "|---|---|---|---|"]
        for ab in ABLATIONS:
            sc = load_config(None, overrides={**cfg.model_dump(), "ablation": {**cfg.ablation.model_dump(), ab: False}})
            subset = ["policy", "best_npv"] if net is not None else ["best_npv"]
            rows = {c: [run_focal_episode(sc, s, c, PolicyRunner(net, "cpu", seed=12345) if c == "policy" else None)
                        for s in seeds] for c in subset}
            sm = {c: summarize(v) for c, v in rows.items()}
            pr = sm.get("policy", {}).get("raw_return_mean", float("nan"))
            br = sm["best_npv"]["raw_return_mean"]
            report[f"ablation:no_{ab}"] = sm
            md.append(f"| no {ab} | {pr:.2f} | {br:.2f} | {pr - br:+.2f} |")
        md.append("")
    md += ["## Reading this report", "",
           "* Results are from a stylised simulator with illustrative parameters — not evidence about real AI markets.",
           "* The RL policy is not required to beat baselines everywhere; regret vs the best heuristic is reported "
           "as-is.", "* Shaped (training) reward is reported separately in JSON; the table shows raw economic return.",
           f"* Wall time: {time.time() - t0:.0f}s."]
    (out_dir / "report.md").write_text("\n".join(md))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\n".join(md))
    print(f"\n[evaluate] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
