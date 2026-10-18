"""Benchmark headless throughput and PPO update speed on CPU vs Apple-Silicon MPS (spec §15 perf target).

Usage: python -m scripts.benchmark [--months 2400] [--workers 8]
Reports actual numbers; nothing is assumed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agents.heuristics import HEURISTICS
from economy.config import load_config
from economy.world import World
from training.checkpoints import build_net
from training.train import TrainConfig, ppo_update
from training.vec import SubprocWorlds


def bench_sim(cfg, months: int) -> dict:
    rng = np.random.default_rng(0)
    w = World(cfg, 1)
    t = time.time()
    n = 0
    while n < months:
        acts = {f.firm_id: HEURISTICS["best_npv"](w, f, rng) for f in w.alive_firms()}
        r = w.step(acts)
        n += 1
        if r.done:
            w = World(cfg, n)
    dt = time.time() - t
    return {"single_process_world_months_per_s": n / dt}


def bench_vec(cfg, workers: int, steps: int = 60) -> dict:
    vec = SubprocWorlds(cfg, {"policy": 0.6, "heuristic:best_npv": 0.4}, workers, 4, 5_000_000)
    b = vec.reset()
    rng = np.random.default_rng(0)
    t = time.time()
    for _ in range(steps):
        acts = []
        for wi in range(vec.n):
            d = {}
            for s in np.flatnonzero(b.ctrl[wi] == 1):
                v = int(rng.choice(np.flatnonzero(b.verb_mask[wi, s] & np.isin(np.arange(8), [0]))))
                d[int(s)] = (v, 0, 0, 0, 0, 1)
            acts.append(d)
        _, _, _, _, b = vec.step(acts)
    dt = time.time() - t
    vec.close()
    return {"vectorised_world_months_per_s": vec.n * steps / dt, "workers": workers, "worlds": vec.n}


def bench_update(cfg, device: str, n: int = 16384) -> dict:
    net = build_net(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    D = net.lay.total
    V, K, P, T = 8, cfg.builders.candidates, cfg.builders.max_projects, cfg.n_tiers
    rng = np.random.default_rng(0)
    vm = np.ones((n, V), bool)
    data = {"obs": rng.normal(size=(n, D)).astype(np.float32), "verb_mask": vm, "cand_mask": np.ones((n, K), bool),
            "slot_mask": np.ones((n, V, P), bool), "tier_mask": np.ones((n, V, T), bool),
            "actions": np.stack([rng.integers(0, 8, n), rng.integers(0, K, n), rng.integers(0, P, n),
                                 rng.integers(0, cfg.n_models, n), rng.integers(0, T, n), rng.integers(0, 3, n)], -1),
            "logp": np.full(n, -5.0, np.float32), "values": np.zeros(n, np.float32),
            "adv": rng.normal(size=n).astype(np.float32), "returns": rng.normal(size=n).astype(np.float32)}
    tc = TrainConfig(epochs=4, minibatch=4096, target_kl=1e9)
    ppo_update(net, opt, data, tc, device)  # warm-up
    if device == "mps":
        torch.mps.synchronize()
    t = time.time()
    ppo_update(net, opt, data, tc, device)
    if device == "mps":
        torch.mps.synchronize()
    dt = time.time() - t
    return {f"update_{device}_s_per_{n}_samples_x4_epochs": dt}


def bench_inference(cfg, device: str, batch: int) -> dict:
    net = build_net(cfg).to(device).eval()
    D = net.lay.total
    x = torch.randn(batch, D, device=device)
    vm = torch.ones(batch, 8, dtype=torch.bool, device=device)
    cm = torch.ones(batch, cfg.builders.candidates, dtype=torch.bool, device=device)
    sm = torch.ones(batch, 8, cfg.builders.max_projects, dtype=torch.bool, device=device)
    tm = torch.ones(batch, 8, cfg.n_tiers, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(3):
            net(x, vm, cm, sm, tm)
        if device == "mps":
            torch.mps.synchronize()
        t = time.time()
        for _ in range(20):
            out = net(x, vm, cm, sm, tm)
            out.actions.cpu()
        dt = (time.time() - t) / 20
    return {f"inference_{device}_ms_batch{batch}": dt * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-vec", action="store_true")
    args = ap.parse_args()
    cfg = load_config("config/base.yaml")
    res = {"torch": torch.__version__, "mps_available": torch.backends.mps.is_available()}
    res.update(bench_sim(cfg, args.months))
    if not args.skip_vec:
        res.update(bench_vec(cfg, args.workers))
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for d in devices:
        res.update(bench_update(cfg, d))
        res.update(bench_inference(cfg, d, 256))
    if "vectorised_world_months_per_s" in res:
        res["est_seconds_per_100k_world_months"] = 100_000 / res["vectorised_world_months_per_s"]
    Path("runs").mkdir(exist_ok=True)
    Path("runs/benchmark.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:55s} {v:.3f}" if isinstance(v, float) else f"{k:55s} {v}")


if __name__ == "__main__":
    main()
