"""Enterprise behaviour: pilots, adoption, churn, confidence (spec §6).

Customers evaluate net value = expected benefit × expected reliability − costs.
Adoption is probabilistic and bounded by budget, trust (confidence),
procurement delay and the builder's sales capacity. Benefit is heterogeneous
per enterprise and hidden from builders. Aggregate demand growth is never
hard-coded: it emerges from products, prices and confidence.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from economy.events import EventType
from entities.firms import Stage

if TYPE_CHECKING:
    from economy.world import World
    from entities.firms import FirmState, Project


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def product_quality(world: "World", p: "Project") -> tuple[float, float, float]:
    """(capability quality multiplier, reliability multiplier, reliability requirement)."""
    opp = world.opps[p.opportunity_id]
    sec = world.cfg.demand.sectors[opp.sector]
    m = world.models[p.model_id]
    q = float(np.clip(1.0 + sec.capability_weight * (m.capability[opp.sector] - opp.latent_difficulty), 0.1, 1.6))
    req = min(0.99, sec.reliability_floor + opp.reliability_req)
    short = max(0.0, req - m.reliability)
    r = float(np.clip(1.0 - sec.reliability_weight * short * 5.0, 0.0, 1.0)) * (0.9 + 0.1 * m.reliability)
    return q, r, req


def realised_benefit(world: "World", p: "Project", ent_idx: np.ndarray, served_frac: float = 1.0) -> np.ndarray:
    opp = world.opps[p.opportunity_id]
    q, r, _ = product_quality(world, p)
    return opp.value * world.ent.benefit[ent_idx, opp.opportunity_id] * q * r * served_frac


def list_price(world: "World", opp_id: int, exclude_firm: int) -> float:
    dc = world.cfg.demand
    opp = world.opps[opp_id]
    n_comp = sum(1 for f in world.alive_firms() if f.firm_id != exclude_firm
                 for p in f.projects if p.opportunity_id == opp_id and p.is_live)
    if not world.cfg.ablation.competition:
        n_comp = 0
    return dc.price_share * opp.value_signal * max(0.4, 1.0 - dc.competition_discount * n_comp)


def recruit_pilots(world: "World", firm: "FirmState", p: "Project") -> list[int]:
    ent = world.ent
    o = p.opportunity_id
    cand = np.flatnonzero(ent.need[:, o] & (ent.provider[:, o] < 0) & (ent.piloting[:, o] < 0))
    if cand.size == 0:
        return []
    rng = world.rng["alloc"]
    willing = ent.confidence[cand] * (0.5 + ent.risk_tolerance[cand]) * (0.6 + 0.8 * firm.reputation)
    draws = rng.random(cand.size) < np.clip(willing, 0, 0.95)
    chosen = cand[draws]
    if chosen.size > world.cfg.builders.pilot_customers:
        chosen = rng.choice(chosen, size=world.cfg.builders.pilot_customers, replace=False)
    ent.piloting[chosen, o] = firm.firm_id
    return [int(i) for i in chosen]


def pilot_gate(world: "World", firm: "FirmState", p: "Project") -> tuple[bool, dict]:
    """Commercial gate: realised benefit must beat total cost AND reliability must clear compliance."""
    ent = world.ent
    o = p.opportunity_id
    ids = np.array(p.pilot_ids, dtype=np.int64)
    ent.piloting[ids, o] = -1
    q, r, req = product_quality(world, p)
    m = world.models[p.model_id]
    rng = world.rng["gates"]
    observed_rel = m.reliability + rng.normal(0, 0.02)
    price = list_price(world, o, firm.firm_id)
    benefit = realised_benefit(world, p, ids)
    hurdle = ent.hurdle[ids]
    passes_value = float(np.mean(benefit - price * (1.0 + hurdle))) > 0 if ids.size else False
    passes_rel = observed_rel >= req
    ok = bool(passes_value and passes_rel)
    detail = {"benefit_mean": float(np.mean(benefit)) if ids.size else 0.0, "price": price,
              "observed_reliability": float(observed_rel), "required_reliability": req,
              "passes_value": bool(passes_value), "passes_reliability": bool(passes_rel)}
    s = world.opps[o].sector
    if world.cfg.ablation.confidence_feedback:
        dc = world.cfg.demand
        world.sector_conf[s] += dc.confidence_pilot_success if ok else -dc.confidence_pilot_failure
        world.sector_conf[s] = float(np.clip(world.sector_conf[s], 0.02, 0.98))
    return ok, detail


def convert_pilots(world: "World", firm: "FirmState", p: "Project") -> int:
    """Successful pilot customers become paying customers at launch (if budget allows)."""
    ent = world.ent
    o = p.opportunity_id
    ids = np.array(p.pilot_ids, dtype=np.int64)
    p.pilot_ids = []
    if ids.size == 0:
        return 0
    ok = ids[(ent.provider[ids, o] < 0) & (ent.budget[ids] - ent.spend[ids] >= p.price)]
    for e in ok:
        _sign(world, firm, p, int(e))
    return int(ok.size)


def _sign(world: "World", firm: "FirmState", p: "Project", e: int) -> None:
    ent = world.ent
    o = p.opportunity_id
    ent.provider[e, o] = firm.firm_id
    ent.price_paid[e, o] = p.price
    ent.spend[e] += p.price
    lo, hi = world.cfg.demand.procurement_delay
    ent.procurement_ready[e] = world.month + int(world.rng["demand"].integers(lo, hi + 1))
    p.customers += 1
    # one-off implementation / integration fee paid by the enterprise to the builder
    fee = world.cfg.demand.integration_cost_months * p.price * (1.0 + world.opps[o].integration_burden)
    world.ledger.transfer("ext:enterprise", firm.account, fee, "integration_fee")
    p.revenue_last += fee
    world.metrics_month["adoptions"] = world.metrics_month.get("adoptions", 0) + 1


def adoption(world: "World") -> None:
    ent = world.ent
    dcfg = world.cfg.demand
    rng = world.rng["demand"]
    offers: dict[int, list[tuple]] = {}
    for f in world.alive_firms():
        for p in f.projects:
            if p.is_live:
                offers.setdefault(p.opportunity_id, []).append((f, p))
    for o, offs in sorted(offers.items()):
        opp = world.opps[o]
        sec = dcfg.sectors[opp.sector]
        elig = np.flatnonzero(ent.need[:, o] & (ent.provider[:, o] < 0) & (ent.piloting[:, o] < 0)
                              & (ent.procurement_ready <= world.month))
        if elig.size == 0:
            continue
        scores = []
        for f, p in offs:
            q, r, _ = product_quality(world, p)
            exp_rel = r * (0.8 + 0.4 * f.reputation)
            eb = opp.value_signal * q * exp_rel
            integ = dcfg.integration_cost_months * p.price * (1 + opp.integration_burden) / 12.0
            cost = sec.price_sensitivity * (p.price + integ)
            ratio = eb / max(cost, 1e-6)
            afford = (ent.budget[elig] - ent.spend[elig]) >= p.price
            prob = _sigmoid(dcfg.adoption_steepness * (ratio - 1.0 - ent.hurdle[elig])) * ent.confidence[elig]
            scores.append(np.where(afford, prob, 0.0))
        S = np.stack(scores)                         # [n_offers, n_elig]
        best = np.argmax(S, axis=0)
        pbest = S[best, np.arange(elig.size)]
        adopt = rng.random(elig.size) < pbest * 0.5  # monthly hazard
        for k, (f, p) in enumerate(offs):
            cap = dcfg.sales_capacity_scaled if p.stage == Stage.SCALED else dcfg.sales_capacity_deployed
            winners = elig[adopt & (best == k)]
            if winners.size > cap:
                winners = rng.choice(winners, size=cap, replace=False)
            for e in winners:
                _sign(world, f, p, int(e))
            if winners.size:
                world.emit(EventType.ENTERPRISE_ADOPTION, {"firm": f.firm_id, "opportunity": o,
                                                           "count": int(winners.size)}, public=False)


def churn_and_bill(world: "World", served_frac: float) -> None:
    """Bill served customers, then churn those whose realised value disappoints."""
    ent = world.ent
    dcfg = world.cfg.demand
    rng = world.rng["demand"]
    for f in world.alive_firms():
        for p in f.projects:
            if not p.is_live or p.customers == 0:
                continue
            o = p.opportunity_id
            ids = np.flatnonzero(ent.provider[:, o] == f.firm_id)
            prices = ent.price_paid[ids, o]
            bill = float(prices.sum()) * served_frac
            world.ledger.transfer("ext:enterprise", f.account, bill, "subscription")
            p.revenue_last += bill
            benefit = realised_benefit(world, p, ids, served_frac)
            world.metrics_month["enterprise_benefit"] = world.metrics_month.get("enterprise_benefit", 0.0) \
                + float(benefit.sum())
            world.metrics_month["enterprise_payments"] = world.metrics_month.get("enterprise_payments", 0.0) + bill
            sec = world.opps[o].sector
            world.sector_revenue[sec] += bill
            bad = benefit < prices * (1.0 + 0.5 * ent.hurdle[ids])
            pch = np.where(bad, dcfg.churn_bad_value, dcfg.churn_base)
            leave = ids[rng.random(ids.size) < pch]
            if leave.size:
                drop_customers(world, f, p, leave, reason="value")


def drop_customers(world: "World", f: "FirmState", p: "Project", ids: np.ndarray, reason: str) -> None:
    ent = world.ent
    o = p.opportunity_id
    ent.spend[ids] -= ent.price_paid[ids, o]
    ent.spend[ids] = np.maximum(ent.spend[ids], 0.0)
    ent.price_paid[ids, o] = 0.0
    ent.provider[ids, o] = -1
    p.customers -= int(ids.size)
    world.metrics_month["churn"] = world.metrics_month.get("churn", 0) + int(ids.size)
    if world.cfg.ablation.confidence_feedback and reason in ("value", "bankruptcy"):
        s = world.opps[o].sector
        hit = world.cfg.demand.confidence_churn * ids.size
        world.sector_conf[s] = float(np.clip(world.sector_conf[s] - hit, 0.02, 0.98))
    world.emit(EventType.CUSTOMER_CHURN, {"firm": f.firm_id, "opportunity": o, "count": int(ids.size),
                                          "reason": reason}, public=False)


def update_confidence(world: "World") -> None:
    if not world.cfg.ablation.confidence_feedback:
        return
    dcfg = world.cfg.demand
    for s, sc in enumerate(dcfg.sectors):
        world.sector_conf[s] += dcfg.confidence_recovery * (sc.initial_confidence - world.sector_conf[s])
    ent = world.ent
    target = world.sector_conf[ent.sector]
    ent.confidence += dcfg.confidence_diffusion * (target - ent.confidence)
    np.clip(ent.confidence, 0.02, 0.98, out=ent.confidence)
