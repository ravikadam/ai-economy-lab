"""Behaviour-cloning warm start from a rule-based teacher (optional, before PPO).

Sparse, delayed rewards (spend now, revenue after a START→PILOT→LAUNCH chain)
make pure-PPO exploration collapse toward "do nothing". Cloning a competent
baseline first puts the policy in a sensible basin; PPO then optimises the
shaped reward from there. Reports must say the policy is BC-initialised.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import torch

from agents.actions import encode


def _collect(args):
    cfg_dict, teacher, seeds, randomize = args
    from agents.heuristics import HEURISTICS
    from agents.masks import build_masks
    from agents.observation import build_observation
    from economy.config import ScenarioConfig
    from economy.world import World
    from training.randomize import config_for_seed
    base = ScenarioConfig.model_validate(cfg_dict)
    obs, vm, cm, sm, tm, act = [], [], [], [], [], []
    skipped = 0
    for seed in seeds:
        dr = randomize > 0 and np.random.default_rng([seed, 17]).random() < randomize
        cfg = config_for_seed(base, seed) if dr else base
        w = World(cfg, seed)
        rng = np.random.default_rng(seed)
        for _ in range(cfg.episode.months):
            acts = {}
            for f in w.alive_firms():
                cand, masks = build_masks(w, f)
                a = HEURISTICS[teacher](w, f, rng)
                idx = encode(a, cand, f.reserve_level)
                acts[f.firm_id] = a
                if idx is None or not masks["verb"][idx[0]]:
                    skipped += 1
                    continue
                obs.append(build_observation(w, f, cand)); vm.append(masks["verb"]); cm.append(masks["candidate"])
                sm.append(masks["slot"]); tm.append(masks["tier"]); act.append(idx)
            if w.step(acts).done:
                break
    return (np.array(obs, np.float32), np.array(vm), np.array(cm), np.array(sm), np.array(tm),
            np.array(act, np.int64), skipped)


def collect_dataset(cfg, teacher: str, episodes: int, workers: int, seed_base: int = 3_000_000,
                    randomize: float = 0.0):
    seeds = [seed_base + i for i in range(episodes)]
    chunks = [(cfg.model_dump(), teacher, seeds[i::workers], randomize) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_collect, chunks)
    data = [np.concatenate([p[k] for p in parts if len(p[k])]) for k in range(6)]
    return data, sum(p[6] for p in parts)


def behaviour_clone(net, cfg, device: str, teacher: str = "best_npv", episodes: int = 64, workers: int = 8,
                    seed_base: int = 3_000_000, randomize: float = 0.0,
                    epochs: int = 6, batch: int = 4096, lr: float = 1e-3, log=print) -> dict:
    t0 = time.time()
    (obs, vm, cm, sm, tm, act), skipped = collect_dataset(cfg, teacher, episodes, workers, seed_base, randomize)
    log(f"[bc] collected {len(obs)} teacher decisions ({skipped} skipped: START target outside candidate set) "
        f"in {time.time() - t0:.0f}s")
    T = [torch.as_tensor(x, device=device) for x in (obs, vm, cm, sm, tm, act)]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = len(obs)
    net.train()
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        losses, accs = [], []
        for i in range(0, N, batch):
            mb = perm[i:i + batch]
            out = net(T[0][mb], T[1][mb], T[2][mb], T[3][mb], T[4][mb], actions=T[5][mb])
            loss = -out.logp.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            accs.append((out.verb_probs.argmax(-1) == T[5][mb][:, 0]).float().mean().item())
        hist.append({"epoch": ep, "nll": float(np.mean(losses)), "verb_acc": float(np.mean(accs))})
        log(f"[bc] epoch {ep} nll={hist[-1]['nll']:.3f} verb_acc={hist[-1]['verb_acc']:.3f}")
    return {"samples": N, "skipped": skipped, "history": hist, "teacher": teacher, "seconds": time.time() - t0}
