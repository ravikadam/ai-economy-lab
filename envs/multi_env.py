"""Seat-based multi-firm environment: the array interface used by training.

Firms are born and die; seats keep tensor shapes fixed. Seat k holds at most one
firm at a time. When a firm goes bankrupt its seat reports done=True; an entrant
may later occupy the seat as a brand-new trajectory. Each occupied seat has a
controller:
  policy        — learning policy (actions supplied by the caller)
  frozen        — frozen checkpoint opponent (actions supplied by the caller)
  human         — human player (actions supplied by the caller)
  heuristic:<n> — computed here from agents.heuristics
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.actions import decode
from agents.heuristics import HEURISTICS
from agents.masks import N_RESERVE, build_masks
from agents.observation import build_observation, layout
from economy.actions import N_VERBS, Action
from economy.config import ScenarioConfig
from economy.world import World

EXTERNAL_CONTROLLERS = ("policy", "frozen", "human")


@dataclass
class SeatBatch:
    """Per-seat arrays for one world at one decision point."""
    obs: np.ndarray          # [S, D] float32
    verb_mask: np.ndarray    # [S, V] bool
    cand_mask: np.ndarray    # [S, K]
    slot_mask: np.ndarray    # [S, V, P]
    tier_mask: np.ndarray    # [S, V, T]
    occupied: np.ndarray     # [S] bool
    controller: list[str]    # [S]
    firm_ids: np.ndarray     # [S] int (-1 empty)


def empty_masks(cfg: ScenarioConfig, S: int):
    K, P, T = cfg.builders.candidates, cfg.builders.max_projects, cfg.n_tiers
    return (np.zeros((S, N_VERBS), bool), np.zeros((S, K), bool), np.zeros((S, N_VERBS, P), bool),
            np.zeros((S, N_VERBS, T), bool))


class MultiFirmEnv:
    def __init__(self, cfg: ScenarioConfig, controller_mix: dict[str, float] | None = None,
                 fixed_controllers: dict[int, str] | None = None, min_policy_seats: int = 1):
        self.cfg = cfg
        self.S = cfg.builders.max_firms
        self.D = layout(cfg).total
        self.mix = controller_mix or {"policy": 1.0}
        self.fixed = fixed_controllers or {}
        self.min_policy_seats = min_policy_seats
        self.world: World | None = None
        self.seat_firm = np.full(self.S, -1, dtype=np.int64)
        self.seat_ctrl = ["" for _ in range(self.S)]
        self._cands: list[list[int]] = [[] for _ in range(self.S)]
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def reset(self, seed: int) -> SeatBatch:
        self.world = World(self.cfg, seed)
        self.rng = np.random.default_rng([seed, 7919])
        self.seat_firm[:] = -1
        self.seat_ctrl = ["" for _ in range(self.S)]
        for f in self.world.alive_firms():
            self._assign(f.firm_id, initial=True)
        self._ensure_policy_seat()
        return self.observe()

    def _pick_controller(self, seat: int, initial: bool) -> str:
        """Fixed controllers apply to a seat's initial occupant only; entrants always draw from the mix."""
        if initial and seat in self.fixed:
            return self.fixed[seat]
        names = list(self.mix)
        p = np.array([self.mix[n] for n in names], dtype=float)
        return names[int(self.rng.choice(len(names), p=p / p.sum()))]

    def _assign(self, firm_id: int, initial: bool = False) -> int:
        free = np.flatnonzero(self.seat_firm < 0)
        if free.size == 0:
            raise RuntimeError("no free seat — max_firms must bound the population")
        seat = int(free[0])
        self.seat_firm[seat] = firm_id
        ctrl = self._pick_controller(seat, initial)
        self.seat_ctrl[seat] = ctrl
        self.world.firms[firm_id].controller = ctrl
        return seat

    def _ensure_policy_seat(self) -> None:
        n_ext = sum(1 for s in range(self.S) if self.seat_firm[s] >= 0 and self.seat_ctrl[s] == "policy")
        for s in range(self.S):
            if n_ext >= self.min_policy_seats:
                break
            if self.seat_firm[s] >= 0 and self.seat_ctrl[s] != "policy" and s not in self.fixed:  # initial only
                self.seat_ctrl[s] = "policy"
                self.world.firms[int(self.seat_firm[s])].controller = "policy"
                n_ext += 1

    def seat_of(self, firm_id: int) -> int:
        idx = np.flatnonzero(self.seat_firm == firm_id)
        return int(idx[0]) if idx.size else -1

    # ------------------------------------------------------------------
    def observe(self) -> SeatBatch:
        w = self.world
        obs = np.zeros((self.S, self.D), np.float32)
        vm, cm, sm, tm = empty_masks(self.cfg, self.S)
        occ = self.seat_firm >= 0
        for s in range(self.S):
            if not occ[s]:
                self._cands[s] = []
                continue
            f = w.firms[int(self.seat_firm[s])]
            cand, masks = build_masks(w, f)
            self._cands[s] = cand
            if self.seat_ctrl[s] in EXTERNAL_CONTROLLERS:
                obs[s] = build_observation(w, f, cand)
            vm[s], cm[s], sm[s], tm[s] = masks["verb"], masks["candidate"], masks["slot"], masks["tier"]
        return SeatBatch(obs, vm, cm, sm, tm, occ.copy(), list(self.seat_ctrl), self.seat_firm.copy())

    def candidates(self, seat: int) -> list[int]:
        return self._cands[seat]

    def step(self, seat_actions: dict[int, tuple | Action]) -> dict:
        """seat_actions: seat -> index tuple (policy heads) or world Action. Heuristic seats are filled here."""
        w = self.world
        acts: dict[int, Action] = {}
        for s in range(self.S):
            fid = int(self.seat_firm[s])
            if fid < 0:
                continue
            f = w.firms[fid]
            ctrl = self.seat_ctrl[s]
            if s in seat_actions:       # caller-supplied (also lets a viewer pre-compute and log heuristic moves)
                a = seat_actions[s]
                acts[fid] = a if isinstance(a, Action) else decode(a, self._cands[s])
            elif ctrl.startswith("heuristic:"):
                acts[fid] = HEURISTICS[ctrl.split(":", 1)[1]](w, f, self.rng)
        res = w.step(acts)
        raw = np.zeros(self.S, np.float32)
        shaped = np.zeros(self.S, np.float32)
        done = np.zeros(self.S, bool)
        acted = self.seat_firm >= 0
        prev_firm = self.seat_firm.copy()
        prev_ctrl = list(self.seat_ctrl)
        for s in range(self.S):
            fid = int(self.seat_firm[s])
            if fid < 0:
                continue
            raw[s] = res.raw_rewards.get(fid, 0.0)
            shaped[s] = res.shaped_rewards.get(fid, 0.0)
            if fid in res.terminated or res.done:
                done[s] = True
            if fid in res.terminated:
                self.seat_firm[s] = -1
                self.seat_ctrl[s] = ""
        for fid in res.born:
            self._assign(fid)
        return {"raw": raw, "shaped": shaped, "done": done, "acted": acted, "episode_done": res.done,
                "prev_firm": prev_firm, "prev_ctrl": prev_ctrl, "result": res}


__all__ = ["MultiFirmEnv", "SeatBatch", "N_RESERVE"]
