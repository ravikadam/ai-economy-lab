"""Vectorised, independently seeded worlds across subprocesses.

Each worker owns several worlds and steps them sequentially; the main process
batches policy inference across all of them. Episode-level seed separation:
worker w, world j, episode e uses seed  base + (w * per_worker + j) * 100_000 + e.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass

import numpy as np

CTRL_CODES = {"": 0, "policy": 1, "frozen": 2, "human": 3}   # heuristics -> 4


def ctrl_code(c: str) -> int:
    return CTRL_CODES.get(c, 4)


@dataclass
class VecBatch:
    obs: np.ndarray        # [W, S, D]
    verb_mask: np.ndarray  # [W, S, V]
    cand_mask: np.ndarray  # [W, S, K]
    slot_mask: np.ndarray  # [W, S, V, P]
    tier_mask: np.ndarray  # [W, S, V, T]
    ctrl: np.ndarray       # [W, S] int codes


def _stack(batches) -> VecBatch:
    return VecBatch(
        obs=np.stack([b.obs for b in batches]),
        verb_mask=np.stack([b.verb_mask for b in batches]),
        cand_mask=np.stack([b.cand_mask for b in batches]),
        slot_mask=np.stack([b.slot_mask for b in batches]),
        tier_mask=np.stack([b.tier_mask for b in batches]),
        ctrl=np.stack([np.array([ctrl_code(c) if o else 0 for c, o in zip(b.controller, b.occupied)])
                       for b in batches]),
    )


def _episode_stats(env) -> list[dict]:
    w = env.world
    out = []
    for f in w.firms.values():
        out.append({"controller": f.controller, "raw_return": f.cum_raw_reward, "shaped_return": f.cum_shaped_reward,
                    "survived": f.alive, "launches": f.launches, "tech_failures": f.tech_failures,
                    "commercial_failures": f.commercial_failures, "profile": f.profile.name,
                    "entrant": f.birth_month > 0, "net_worth": w.net_worth(f)})
    return out


def _worker(remote, cfg_dict, mix, n_worlds, seed_base, worker_idx, randomize=0.0):
    from economy.config import ScenarioConfig
    from envs.multi_env import MultiFirmEnv
    from training.randomize import config_for_seed
    try:
        cfg = ScenarioConfig.model_validate(cfg_dict)
        envs = [MultiFirmEnv(cfg, mix, min_policy_seats=1) for _ in range(n_worlds)]
        episode = [0] * n_worlds

        def seed_for(j):
            s = seed_base + (worker_idx * n_worlds + j) * 100_000 + episode[j]
            # domain randomisation: a share of episodes draw a fresh regime (deterministic in the seed)
            use_dr = randomize > 0 and np.random.default_rng([s, 17]).random() < randomize
            envs[j].cfg = config_for_seed(cfg, s) if use_dr else cfg
            return s

        batches = [None] * n_worlds
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                batches = [e.reset(seed_for(j)) for j, e in enumerate(envs)]
                remote.send(batches)
            elif cmd == "step":
                outs = []
                for j, e in enumerate(envs):
                    o = e.step(data[j])
                    rec = {"raw": o["raw"], "shaped": o["shaped"], "done": o["done"], "episode": None}
                    if o["episode_done"]:
                        rec["episode"] = _episode_stats(e)
                        episode[j] += 1
                        batches[j] = e.reset(seed_for(j))
                    else:
                        batches[j] = e.observe()
                    outs.append(rec)
                remote.send((outs, batches))
            elif cmd == "close":
                remote.close()
                return
    except Exception:  # pragma: no cover - surfaced to parent
        remote.send(("__error__", traceback.format_exc()))


class SubprocWorlds:
    def __init__(self, cfg, mix: dict, num_workers: int, worlds_per_worker: int, seed_base: int,
                 randomize: float = 0.0):
        ctx = mp.get_context("spawn")
        self.n = num_workers * worlds_per_worker
        self.per = worlds_per_worker
        self.remotes, self.procs = [], []
        for w in range(num_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, cfg.model_dump(), mix, worlds_per_worker, seed_base, w, randomize),
                            daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)

    def _recv(self, r):
        msg = r.recv()
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__error__":
            raise RuntimeError("worker failed:\n" + msg[1])
        return msg

    def reset(self) -> VecBatch:
        for r in self.remotes:
            r.send(("reset", None))
        batches = [b for r in self.remotes for b in self._recv(r)]
        return _stack(batches)

    def step(self, actions: list[dict]):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.per:(i + 1) * self.per]))
        outs, batches = [], []
        for r in self.remotes:
            o, b = self._recv(r)
            outs += o
            batches += b
        raw = np.stack([o["raw"] for o in outs])
        shaped = np.stack([o["shaped"] for o in outs])
        done = np.stack([o["done"] for o in outs])
        episodes = [o["episode"] for o in outs if o["episode"] is not None]
        return raw, shaped, done, episodes, _stack(batches)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
