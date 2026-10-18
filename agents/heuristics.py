"""Rule-based baselines (spec §9) and the firm-side belief estimator.

`estimate()` uses ONLY information a firm legitimately has: its own noisy
difficulty signal, public value signals, public (lagged) model capability and
prices, and its own skill. The oracle baseline is the single exception and is
for tiny diagnostic instances only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from economy.actions import Action, Verb, check_action, legal_tiers, slot_ok, startable_opportunities
from entities.firms import FirmState, Stage

if TYPE_CHECKING:
    from economy.world import World


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Estimate:
    p_tech: float
    p_comm: float
    margin: float          # expected monthly margin per customer
    customers: float       # expected customers after 12 months live
    dev_cost: float
    ev: float              # rough expected value over remaining horizon ($k)
    cap_gap: float         # public capability − own difficulty signal


def estimate(world: "World", f: FirmState, opp_id: int, model_id: int, tier: int,
             use_truth: bool = False, retries: int = 0) -> Estimate:
    cfg = world.cfg
    t, b, dcfg = cfg.technology, cfg.builders, cfg.demand
    opp = world.opps[opp_id]
    pub = world.public
    cap = pub["model_capability"][model_id][opp.sector]
    rel = pub["model_reliability"][model_id]
    d = opp.latent_difficulty if use_truth else world.difficulty_signal(f, opp_id)
    effort = b.tier_effort[tier] + b.retry_effort_bonus * retries
    x = t.alpha * (cap - d) + t.beta * f.skill + t.gamma * effort - t.delta * opp.integration_burden + t.bias
    p_tech = min(t.p_max, max(t.p_min, _sig(x)))
    sec = dcfg.sectors[opp.sector]
    req = min(0.99, sec.reliability_floor + opp.reliability_req)
    rel_ok = _sig((rel - req) / 0.02)
    q = float(np.clip(1.0 + sec.capability_weight * (cap - d), 0.1, 1.6))
    rf = float(np.clip(1.0 - sec.reliability_weight * max(0.0, req - rel) * 5.0, 0.0, 1.0))
    price = dcfg.price_share * opp.value_signal
    benefit = opp.value_signal * q * rf
    value_ok = _sig((benefit / max(price, 1e-6) - 1.0 - sec.hurdle_roi) * 3.0)
    p_comm = rel_ok * value_ok
    unit = pub["model_unit_price"][model_id]
    margin = price - opp.inference_units * unit - b.support_cost_per_customer
    conf = pub["sector_confidence"][opp.sector]
    customers = min(opp.need_count * 0.6, 12 * dcfg.sales_capacity_deployed * conf * value_ok)
    months_dev = 2 * opp.duration / b.tier_speed[tier] + b.pilot_months
    dev_cost = months_dev * b.tier_burn[tier] + b.start_cost + b.pilot_cost + b.launch_cost
    remaining = max(0.0, cfg.episode.months - world.month - months_dev)
    live_months = min(remaining, 36.0)
    ev = p_tech * p_comm * margin * customers * 0.5 * live_months / 1.0 - dev_cost
    return Estimate(p_tech, p_comm, margin, customers, dev_cost, ev, cap - d)


# --------------------------------------------------------------------------
def _advance_existing(world: "World", f: FirmState, max_retries: int, scale_ok: bool) -> Action | None:
    """Common project hygiene shared by the heuristics."""
    for i, p in enumerate(f.projects):
        if p.stage == Stage.PILOT_READY:
            return Action(Verb.LAUNCH, slot=i)
    for i, p in enumerate(f.projects):
        if p.stage == Stage.PROTOTYPE_READY:
            return Action(Verb.PILOT, slot=i)
    for i, p in enumerate(f.projects):
        if p.stage == Stage.FAILED:
            if p.retries < max_retries:
                best_m = max(range(world.cfg.n_models),
                             key=lambda m: estimate(world, f, p.opportunity_id, m, p.tier, retries=p.retries + 1).ev)
                return Action(Verb.RETRY, slot=i, model_id=best_m, tier=p.tier)
            return Action(Verb.ABANDON, slot=i)
    for i, p in enumerate(f.projects):
        if p.is_live and len(p.profit_hist) >= 6 and p.trailing_profit(6) < 0 and p.customers < 3:
            return Action(Verb.ABANDON, slot=i)
    if scale_ok:
        for i, p in enumerate(f.projects):
            if p.stage == Stage.DEPLOYED and p.customers >= 6 and p.trailing_profit() > 5:
                return Action(Verb.SCALE, slot=i)
    return None


def _legal_or_noop(world: "World", f: FirmState, a: Action | None) -> Action:
    if a is None:
        return Action()
    ok, _ = check_action(world, f, a)
    return a if ok else Action()


def random_policy(world: "World", f: FirmState, rng: np.random.Generator) -> Action:
    from agents.masks import build_masks, sample_from_masks
    obs_ids, masks = build_masks(world, f)
    return sample_from_masks(world, f, obs_ids, masks, rng)


def easy_first(world: "World", f: FirmState, rng: np.random.Generator) -> Action:
    a = _advance_existing(world, f, max_retries=2, scale_ok=True)
    if a is not None:
        return _legal_or_noop(world, f, a)
    if len(f.projects) >= world.cfg.builders.max_projects:
        return Action()
    opps = startable_opportunities(world, f)
    if not opps:
        return Action()
    easiest = min(opps, key=lambda o: world.difficulty_signal(f, o))
    return _legal_or_noop(world, f, Action(Verb.START, opportunity_id=easiest, model_id=0, tier=1))


def best_npv(world: "World", f: FirmState, rng: np.random.Generator, use_truth: bool = False) -> Action:
    a = _advance_existing(world, f, max_retries=1, scale_ok=True)
    if a is not None:
        return _legal_or_noop(world, f, a)
    if len(f.projects) >= world.cfg.builders.max_projects:
        return Action()
    best, best_ev = None, 0.0
    for o in startable_opportunities(world, f):
        for m in range(world.cfg.n_models):
            for t in range(world.cfg.n_tiers):
                e = estimate(world, f, o, m, t, use_truth=use_truth)
                if e.ev > best_ev:
                    best, best_ev = Action(Verb.START, opportunity_id=o, model_id=m, tier=t), e.ev
    return _legal_or_noop(world, f, best)


def cash_preserving(world: "World", f: FirmState, rng: np.random.Generator) -> Action:
    a = _advance_existing(world, f, max_retries=0, scale_ok=world.runway(f) > 12)
    if a is not None:
        return _legal_or_noop(world, f, Action(a.verb, a.opportunity_id, a.slot, a.model_id, a.tier, reserve=2))
    if world.runway(f) < 12 or len([p for p in f.projects if not p.is_live]) >= 1:
        return Action(reserve=2)
    best, best_ev = None, 0.0
    for o in startable_opportunities(world, f):
        e = estimate(world, f, o, 1, 0)
        if e.p_tech * e.p_comm > 0.45 and e.ev > best_ev:
            best, best_ev = Action(Verb.START, opportunity_id=o, model_id=1, tier=0, reserve=2), e.ev
    return _legal_or_noop(world, f, best) if best else Action(reserve=2)


def oracle(world: "World", f: FirmState, rng: np.random.Generator) -> Action:
    """Upper bound using hidden difficulty. Diagnostic only — never a training opponent."""
    return best_npv(world, f, rng, use_truth=True)


HEURISTICS: dict[str, Callable] = {
    "random": random_policy,
    "easy_first": easy_first,
    "best_npv": best_npv,
    "cash_preserving": cash_preserving,
    "oracle": oracle,
}
