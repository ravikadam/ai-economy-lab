"""Experiments behind the research paper (market dynamics, learned behaviour, shocks, first movers).

Usage: python -m scripts.experiments --checkpoint runs/latest --seeds 200:229 --workers 8
Writes runs/paper/experiments.json. Seeds 200+ are disjoint from training (1e6+), BC (3e6+),
dev (1000-1004) and evaluation (100-129) seeds.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

POP = {0: "policy", 1: "best_npv", 2: "policy", 3: "easy_first", 4: "cash_preserving"}
ENTRANTS = ["policy", "best_npv"]

_net = None


def _init(ck):
    global _net
    import torch
    torch.set_num_threads(1)
    from training.checkpoints import load_checkpoint
    _net, _, _ = load_checkpoint(ck)


def run_episode(args):
    scenario, seed = args
    import torch
    from agents.actions import decode
    from agents.heuristics import HEURISTICS
    from agents.masks import build_masks
    from agents.observation import build_observation
    from economy.actions import Verb
    from economy.config import load_config
    from economy.world import World
    cfg = load_config(scenario)
    w = World(cfg, seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    ctrl = {}
    for i, f in enumerate(w.alive_firms()):
        ctrl[f.firm_id] = POP[i % len(POP)]
    starts = []          # (controller, profile, sector, model, tier, d_signal, d_true, month)
    verbs = defaultdict(lambda: defaultdict(int))
    launches = []        # (firm, opp, month)
    orders = []
    prev_backlog = w.fab.backlog
    while True:
        for f in w.alive_firms():
            if f.firm_id not in ctrl:
                ctrl[f.firm_id] = ENTRANTS[int(rng.integers(len(ENTRANTS)))]
        acts = {}
        pol = [f for f in w.alive_firms() if ctrl[f.firm_id] == "policy"]
        if pol:
            data = [build_masks(w, f) for f in pol]
            obs = np.stack([build_observation(w, f, c) for f, (c, _) in zip(pol, data)])
            with torch.no_grad():
                out = _net(torch.as_tensor(obs), torch.as_tensor(np.stack([m["verb"] for _, m in data])),
                           torch.as_tensor(np.stack([m["candidate"] for _, m in data])),
                           torch.as_tensor(np.stack([m["slot"] for _, m in data])),
                           torch.as_tensor(np.stack([m["tier"] for _, m in data])), generator=gen)
            for f, (c, _), a in zip(pol, data, out.actions.numpy()):
                acts[f.firm_id] = decode(tuple(int(x) for x in a), c)
        for f in w.alive_firms():
            if f.firm_id not in acts:
                acts[f.firm_id] = HEURISTICS[ctrl[f.firm_id]](w, f, rng)
        for f in w.alive_firms():
            a = acts[f.firm_id]
            c = ctrl[f.firm_id]
            verbs[c][a.verb.name] += 1
            if a.verb == Verb.START and a.opportunity_id >= 0:
                o = w.opps[a.opportunity_id]
                starts.append([c, f.profile.name, o.sector, a.model_id, a.tier, w.difficulty_signal(f, o.opportunity_id),
                               o.latent_difficulty, w.month + 1])
        r = w.step(acts)
        for e in r.events:
            if e.kind.value == "PROJECT_GATE_RESULT" and e.payload.get("gate") == "launch":
                launches.append([e.payload["firm"], e.payload["opportunity"], w.month])
        orders.append(max(0.0, w.fab.backlog - prev_backlog + w.fab.output_last))
        prev_backlog = w.fab.backlog
        if r.done:
            break
    firms = []
    for f in w.firms.values():
        firms.append({"id": f.firm_id, "controller": ctrl.get(f.firm_id, "?"), "profile": f.profile.name,
                      "entrant": f.birth_month > 0, "birth": f.birth_month, "death": f.death_month,
                      "raw": f.cum_raw_reward, "alive": f.alive, "launches": f.launches,
                      "tech_fail": f.tech_failures, "comm_fail": f.commercial_failures, "deferrals": f.deferrals,
                      "customers_by_opp": {str(p.opportunity_id): p.customers for p in f.projects if p.is_live}})
    hist = [{k: v for k, v in rec.items() if k not in ("launches", "model_capability")} |
            {"model_capability": rec["model_capability"]} for rec in w.price_history]
    gates = [{k: g[k] for k in ("gate", "ok", "month") if k in g} |
             {"d": w.opps[g["opportunity"]].latent_difficulty, "sector": w.opps[g["opportunity"]].sector,
              "rel_ok": g.get("passes_reliability"), "val_ok": g.get("passes_value"), "p": g.get("p")}
             for g in w.gate_audit]
    return {"scenario": scenario, "seed": seed, "history": hist, "firms": firms, "starts": starts,
            "verbs": {k: dict(v) for k, v in verbs.items()}, "launches": launches, "orders": orders, "gates": gates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/latest")
    ap.add_argument("--seeds", default="200:229")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios", nargs="*", default=["config/base.yaml", "config/scenarios/compute_crunch.yaml"])
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split(":"))
    jobs = [(s, seed) for s in a.scenarios for seed in range(lo, hi + 1)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.checkpoint,)) as pool:
        res = pool.map(run_episode, jobs)
    out = Path("runs/paper")
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiments.json").write_text(json.dumps(res, default=float))
    print(f"[experiments] {len(res)} episodes in {time.time() - t0:.0f}s -> {out / 'experiments.json'}")


if __name__ == "__main__":
    main()
