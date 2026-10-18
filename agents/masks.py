"""Action masks for the factorised / autoregressive action interface.

Heads: verb(8) -> [candidate(12) | slot(3)] -> model(n_models), tier(n_tiers); reserve(3).
Masks are derived from economy.actions so legality has one source of truth.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from economy.actions import (N_VERBS, SLOT_VERBS, Action, Verb, legal_tiers, legal_verbs, slot_ok,
                             spend_cost, spending_capacity, startable_opportunities)
from entities.firms import RESERVE_MONTHS, FirmState

if TYPE_CHECKING:
    from economy.world import World

N_RESERVE = len(RESERVE_MONTHS)


def candidate_ids(world: "World", f: FirmState) -> list[int]:
    """Up to K startable opportunities, ranked by public market size (value signal × need count)."""
    K = world.cfg.builders.candidates
    opps = startable_opportunities(world, f)
    opps.sort(key=lambda o: (-world.opps[o].value_signal * world.opps[o].need_count, o))
    return opps[:K]


def build_masks(world: "World", f: FirmState, cand: list[int] | None = None) -> tuple[list[int], dict]:
    cfg = world.cfg
    K, S = cfg.builders.candidates, cfg.builders.max_projects
    cand = candidate_ids(world, f) if cand is None else cand
    verbs = np.array(legal_verbs(world, f), dtype=bool)
    cap = spending_capacity(world, f)
    min_start = min(spend_cost(world, f, Action(Verb.START, tier=t)) for t in range(cfg.n_tiers))
    cmask = np.zeros(K, dtype=bool)
    if cap >= min_start:
        cmask[: len(cand)] = True
    if not cmask.any():
        verbs[Verb.START] = False
    smask = np.zeros((N_VERBS, S), dtype=bool)
    for v in SLOT_VERBS:
        for s in range(min(S, len(f.projects))):
            smask[v, s] = slot_ok(f, v, s)
        if not smask[v].any():
            verbs[v] = False
    tmask = np.ones((N_VERBS, cfg.n_tiers), dtype=bool)
    for v in (Verb.START, Verb.RETRY):
        tmask[v] = np.array(legal_tiers(world, f, v), dtype=bool)
        if not tmask[v].any():
            verbs[v] = False
    masks = {
        "verb": verbs,
        "candidate": cmask,
        "slot": smask,
        "tier": tmask,
        "model": np.ones(cfg.n_models, dtype=bool),
        "reserve": np.ones(N_RESERVE, dtype=bool),
    }
    return cand, masks


def sample_from_masks(world: "World", f: FirmState, cand: list[int], masks: dict, rng: np.random.Generator) -> Action:
    from agents.actions import decode
    verbs = np.flatnonzero(masks["verb"])
    v = int(rng.choice(verbs))
    c = int(rng.choice(np.flatnonzero(masks["candidate"]))) if v == Verb.START else 0
    s = int(rng.choice(np.flatnonzero(masks["slot"][v]))) if v in SLOT_VERBS else 0
    m = int(rng.integers(world.cfg.n_models))
    t = int(rng.choice(np.flatnonzero(masks["tier"][v])))
    r = int(rng.integers(N_RESERVE))
    return decode((v, c, s, m, t, r), cand)
