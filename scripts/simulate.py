"""Headless simulation with replay logging.

Usage:
  python -m scripts.simulate --config config/base.yaml --seed 42
  python -m scripts.simulate --config config/base.yaml --seed 42 --controllers best_npv,easy_first,policy \
      --checkpoint runs/latest
Controllers are assigned to seats in order (cycled); entrants use --entrants.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from agents.actions import decode
from analysis.replay import ReplayRecorder, decision_record
from economy.config import load_config
from envs.multi_env import MultiFirmEnv


def run(config: str, seed: int, controllers: list[str], entrants: str, checkpoint: str | None, out: str | None,
        quiet: bool = False) -> Path:
    cfg = load_config(config)
    def ctrl(c):
        return c if c in ("policy",) else f"heuristic:{c}"
    fixed = {i: ctrl(controllers[i % len(controllers)]) for i in range(cfg.builders.initial_firms)}
    runner = None
    if any(c == "policy" for c in fixed.values()) or entrants == "policy":
        from agents.runner import PolicyRunner
        from training.checkpoints import load_checkpoint
        net, _, _ = load_checkpoint(checkpoint or "runs/latest")
        runner = PolicyRunner(net, seed=seed)
    env = MultiFirmEnv(cfg, {ctrl(entrants): 1.0}, fixed_controllers=fixed, min_policy_seats=0)
    batch = env.reset(seed)
    path = Path(out) if out else Path("runs/replays") / f"sim-{cfg.name}-{seed}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    rec = ReplayRecorder(path, cfg, seed, fixed, {"mode": "simulate", "entrants": entrants})
    t0 = time.time()
    while True:
        w = env.world
        seats = [s for s in range(env.S) if batch.occupied[s] and batch.controller[s] == "policy"]
        pol = runner.act(batch, seats) if runner and seats else {}
        decisions = []
        for s in range(env.S):
            if not batch.occupied[s]:
                continue
            f = w.firms[int(batch.firm_ids[s])]
            if s in pol:
                a = decode(pol[s]["action"], env.candidates(s))
                decisions.append(decision_record(w, f, env.candidates(s), a, pol[s]["value"], pol[s]["verb_probs"],
                                                 "policy"))
        out_ = env.step({s: r["action"] for s, r in pol.items()})
        # heuristic decisions are taken inside env.step; log their realised actions from the world diagnostics
        rec.month(w, decisions, out_["result"].events)
        if out_["episode_done"]:
            break
        batch = env.observe()
    rec.final(env.world)
    if not quiet:
        w = env.world
        print(f"[simulate] {cfg.name} seed={seed} months={w.month} in {time.time() - t0:.1f}s -> {path}")
        print(f"[simulate] state hash {w.state_hash()[:16]}")
        for f in sorted(w.firms.values(), key=lambda g: -g.cum_raw_reward):
            print(f"  firm {f.firm_id:2d} {f.controller:28s} {'alive' if f.alive else 'bankrupt m' + str(f.death_month):14s}"
                  f" value {f.cum_raw_reward:7.2f} $M  launches {f.launches}  customers "
                  f"{sum(p.customers for p in f.projects)}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--controllers", default="best_npv,easy_first,cash_preserving,best_npv,random")
    ap.add_argument("--entrants", default="best_npv")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plots", action="store_true", help="also write PNG dashboards next to the replay")
    args = ap.parse_args(argv)
    path = run(args.config, args.seed, args.controllers.split(","), args.entrants, args.checkpoint, args.out)
    if args.plots:
        from analysis.plots import save_dashboard
        print(f"[simulate] plots -> {save_dashboard(path)}")


if __name__ == "__main__":
    main()
