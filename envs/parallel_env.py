"""PettingZoo ParallelEnv adapter (dynamic agent set) and Gymnasium single-agent adapter.

Agents are seats ("seat_0".."seat_{max_firms-1}") so observation/action shapes stay
fixed while firms enter and exit. Observations are dicts with a flat action mask:
[verb | candidate | slot(verb x slot) | tier(verb x tier) | model | reserve].
"""
from __future__ import annotations

import functools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from agents.masks import N_RESERVE
from economy.actions import N_VERBS
from economy.config import ScenarioConfig, load_config
from envs.multi_env import MultiFirmEnv


def flat_mask(batch, s: int, n_models: int) -> np.ndarray:
    return np.concatenate([batch.verb_mask[s], batch.cand_mask[s], batch.slot_mask[s].ravel(),
                           batch.tier_mask[s].ravel(), np.ones(n_models, bool), np.ones(N_RESERVE, bool)]).astype(np.int8)


def action_space(cfg: ScenarioConfig) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([N_VERBS, cfg.builders.candidates, cfg.builders.max_projects, cfg.n_models,
                                 cfg.n_tiers, N_RESERVE])


class AIEconomyParallelEnv(ParallelEnv):
    metadata = {"name": "ai_economy_lab_v1", "render_modes": []}

    def __init__(self, scenario_config: ScenarioConfig | None = None, controller_mix: dict | None = None):
        self.cfg = scenario_config or load_config()
        self.core = MultiFirmEnv(self.cfg, controller_mix or {"policy": 1.0}, min_policy_seats=1)
        self.possible_agents = [f"seat_{i}" for i in range(self.core.S)]
        self.agents: list[str] = []
        self._batch = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        n_mask = len(flat_mask(_DummyBatch(self.cfg), 0, self.cfg.n_models))
        return spaces.Dict({"observation": spaces.Box(-10, 10, (self.core.D,), np.float32),
                            "action_mask": spaces.MultiBinary(n_mask)})

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return action_space(self.cfg)

    def _obs(self):
        b = self._batch
        out, infos = {}, {}
        self.agents = []
        for s in range(self.core.S):
            if b.occupied[s] and b.controller[s] == "policy":
                a = f"seat_{s}"
                self.agents.append(a)
                out[a] = {"observation": b.obs[s], "action_mask": flat_mask(b, s, self.cfg.n_models)}
                infos[a] = {"firm_id": int(b.firm_ids[s]), "candidate_ids": self.core.candidates(s)}
        return out, infos

    def reset(self, seed: int | None = None, options: dict | None = None):
        if options and "scenario_config" in options:
            self.cfg = options["scenario_config"]
            self.core = MultiFirmEnv(self.cfg, self.core.mix, min_policy_seats=1)
        self._batch = self.core.reset(0 if seed is None else seed)
        return self._obs()

    def step(self, actions: dict[str, np.ndarray]):
        seat_actions = {int(a.split("_")[1]): tuple(int(x) for x in act) for a, act in actions.items()}
        acting = list(self.agents)
        out = self.core.step(seat_actions)
        self._batch = self.core.observe()
        rewards = {a: float(out["shaped"][int(a.split("_")[1])]) for a in acting}
        terms = {a: bool(out["done"][int(a.split("_")[1])]) and not out["episode_done"] for a in acting}
        truncs = {a: bool(out["episode_done"]) for a in acting}
        obs, infos = self._obs()
        for a in acting:
            infos.setdefault(a, {})["raw_reward"] = float(out["raw"][int(a.split("_")[1])])
        if out["episode_done"]:
            self.agents = []
        return obs, rewards, terms, truncs, infos


class _DummyBatch:
    def __init__(self, cfg):
        from envs.multi_env import empty_masks
        self.verb_mask, self.cand_mask, self.slot_mask, self.tier_mask = empty_masks(cfg, 1)


class SingleFirmGymEnv(gym.Env):
    """Debugging adapter: one learning firm (seat 0); all others run a fixed heuristic."""

    def __init__(self, scenario_config: ScenarioConfig | None = None, opponent: str = "best_npv"):
        self.cfg = scenario_config or load_config()
        self.core = MultiFirmEnv(self.cfg, {f"heuristic:{opponent}": 1.0}, fixed_controllers={0: "policy"},
                                 min_policy_seats=0)
        self.action_space = action_space(self.cfg)
        n_mask = len(flat_mask(_DummyBatch(self.cfg), 0, self.cfg.n_models))
        self.observation_space = spaces.Dict({"observation": spaces.Box(-10, 10, (self.core.D,), np.float32),
                                              "action_mask": spaces.MultiBinary(n_mask)})
        self._batch = None

    def _o(self):
        b = self._batch
        return {"observation": b.obs[0], "action_mask": flat_mask(b, 0, self.cfg.n_models)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._batch = self.core.reset(0 if seed is None else seed)
        return self._o(), {"firm_id": int(self._batch.firm_ids[0])}

    def step(self, action):
        out = self.core.step({0: tuple(int(x) for x in action)})
        self._batch = self.core.observe()
        terminated = bool(out["done"][0]) and not out["episode_done"]
        truncated = bool(out["episode_done"])
        if self._batch.firm_ids[0] < 0:
            terminated = True
        return self._o(), float(out["shaped"][0]), terminated, truncated, {"raw_reward": float(out["raw"][0])}
