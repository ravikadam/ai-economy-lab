"""PFAR: Paired-Fork Advantage against a Rule (paper/PROTOCOL_v2.md, revision plan R3).

At a decision where the learning company's action differs from the best-expected-value rule's action, copy the
market twice. Copy A takes the agent's action, copy B the rule's; every other company's action that month is the
same. Both copies run `horizon` months with learning and frozen seats acting by the current policy (a sampling
generator seeded identically in both copies) and every random stream shared (common random numbers):

    A = sum_{k<H} gamma^k (r^A_k - r^B_k) / scale + gamma^H (V(s^A_H) - V(s^B_H))

Q(s, rule(s)) does not depend on the agent's action, so it is a valid policy-gradient baseline.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from agents.actions import decode
from agents.heuristics import HEURISTICS
from economy.actions import Action, Verb

EXTERNAL = ("policy", "frozen")


def _key(a: Action) -> tuple:
    """What makes two actions different decisions (the reserve level is ignored)."""
    v = a.verb
    if v == Verb.START:
        return (int(v), a.opportunity_id, a.model_id, a.tier)
    if v == Verb.RETRY:
        return (int(v), a.slot, a.model_id, a.tier)
    if v == Verb.NOOP:
        return (int(v),)
    return (int(v), a.slot)


def rule_action(env, seat: int, rng: np.random.Generator) -> Action:
    """The rule's choice for this seat now, using a private random stream so the market's own streams are untouched."""
    f = env.world.firms[int(env.seat_firm[seat])]
    return HEURISTICS["best_npv"](env.world, f, rng)


def disagrees(env, seat: int, idx: tuple, rule: Action) -> bool:
    mine = decode(idx, env.candidates(seat))
    if mine.verb == Verb.NOOP and rule.verb == Verb.NOOP:
        return False
    return _key(mine) != _key(rule)


@torch.no_grad()
def _branch(env, seat: int, first: dict, net, gen_seed: int, horizon: int, gamma: float, scale: float):
    """Run one copy: `first` this month, then the current policy for `horizon - 1` months.
    Returns (discounted normalised reward sum, bootstrap value or 0, bankrupt?)."""
    fid = int(env.seat_firm[seat])
    gen = torch.Generator().manual_seed(gen_seed)
    total, g = 0.0, 1.0
    acts = first
    for k in range(horizon):
        out = env.step(acts)
        total += g * float(out["shaped"][seat]) / scale
        g *= gamma
        res = out["result"]
        if fid in res.terminated:
            return total, 0.0, True
        if out["episode_done"]:
            return total, 0.0, False
        batch = env.observe()
        if k == horizon - 1:
            break
        seats = [s for s in range(env.S) if batch.occupied[s] and batch.controller[s] in EXTERNAL]
        acts = {}
        if seats:
            ix = np.array(seats)
            o = net(torch.as_tensor(batch.obs[ix]), torch.as_tensor(batch.verb_mask[ix]),
                    torch.as_tensor(batch.cand_mask[ix]), torch.as_tensor(batch.slot_mask[ix]),
                    torch.as_tensor(batch.tier_mask[ix]), generator=gen)
            for i, s in enumerate(seats):
                acts[s] = tuple(int(x) for x in o.actions[i])
    v = float(net.value(torch.as_tensor(batch.obs[seat:seat + 1]))[0])
    return total, v, False


def fork_advantage(env, seat: int, actions: dict, rule: Action, net, gen_seed: int, horizon: int = 12,
                   gamma: float = 0.99, scale: float = 1.0) -> dict:
    """Fork `env` (before this month's step) into agent-action and rule-action copies; return the PFAR advantage."""
    a_env, b_env = copy.deepcopy(env), copy.deepcopy(env)
    first_b = dict(actions)
    first_b[seat] = rule
    ga, va, bank_a = _branch(a_env, seat, dict(actions), net, gen_seed, horizon, gamma, scale)
    gb, vb, bank_b = _branch(b_env, seat, first_b, net, gen_seed, horizon, gamma, scale)
    adv = (ga - gb) + gamma ** horizon * (va - vb)
    return {"adv": float(adv), "bankrupt_agent": bank_a, "bankrupt_rule": bank_b}
