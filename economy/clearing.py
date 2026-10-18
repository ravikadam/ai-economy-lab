"""Compute market clearing and bounded price transmission (spec §4, §6).

Builders' served customers create inference demand, which model providers turn
into compute demand on data centres. If demand exceeds installed capacity the
shortfall is shared proportionally (identical served fraction for everyone),
and the unmet part is recorded as lost demand — no sale exceeds capacity.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from economy.world import World


def compute_demand_by_project(world: "World") -> list[tuple]:
    """[(firm, project, opp, inference_units, compute_units)] for live projects and pilots."""
    rows = []
    for f in world.alive_firms():
        for p in f.projects:
            opp = world.opps[p.opportunity_id]
            users = p.customers if p.is_live else (len(p.pilot_ids) if p.stage.name == "PILOT" else 0)
            if users <= 0:
                continue
            inf = users * opp.inference_units
            comp = inf * world.models[p.model_id].compute_per_unit
            rows.append((f, p, opp, inf, comp))
    return rows


def clear_compute(world: "World", rows: list[tuple]) -> float:
    """Allocate capacity, settle compute payments, return served fraction."""
    demand = sum(r[4] for r in rows)
    capacity = sum(dc.capacity for dc in world.dcs)
    if not world.cfg.ablation.supply_bottlenecks:
        capacity = max(capacity, demand)
    served_frac = 1.0 if demand <= capacity or demand == 0 else capacity / demand
    served = demand * served_frac
    world.metrics_month.update(compute_demand=demand, compute_capacity=capacity, served_frac=served_frac,
                               unmet_compute=demand - served,
                               utilization=served / capacity if capacity > 0 else 0.0,
                               demand_ratio=demand / capacity if capacity > 0 else 0.0)
    price = world.compute_price
    # builders pay model providers per inference unit actually served
    for f, p, opp, inf, comp in rows:
        m = world.models[p.model_id]
        cost = inf * served_frac * m.unit_price(price)
        world.firm_pay(f, m.account, cost, "inference")
        p.cost_last += cost
        m.load_last += inf * served_frac
    # model providers pay data centres for compute, split by capacity share
    for m in world.models:
        need = sum(r[4] for r in rows if r[1].model_id == m.model_id) * served_frac
        if need <= 0:
            continue
        for dc in world.dcs:
            share = dc.capacity / capacity if capacity > 0 else 0.0
            pay = need * share * price
            short = pay - world.ledger.balance(m.account)
            if short > 0:
                world.ledger.transfer("ext:investors", m.account, short, "upstream_equity")
            world.ledger.transfer(m.account, dc.account, pay, "compute_sale")
    for dc in world.dcs:
        share = dc.capacity / capacity if capacity > 0 else 0.0
        dc.served_last = served * share
        dc.util_hist.append(dc.served_last / dc.capacity if dc.capacity > 0 else 0.0)
        power = dc.served_last * world.cfg.datacenters.power_cost_per_unit
        short = power - world.ledger.balance(dc.account)
        if short > 0:
            world.ledger.transfer("ext:investors", dc.account, short, "upstream_equity")
        world.ledger.transfer(dc.account, "ext:power", power, "power")
    return served_frac


def update_compute_price(world: "World") -> None:
    pc = world.cfg.compute_price
    if not world.cfg.ablation.supply_bottlenecks:
        return
    ratio = min(1.5, world.metrics_month.get("demand_ratio", 0.0))
    target = world.cfg.datacenters.target_utilization
    new = world.compute_price * (1.0 + pc.adjust_rate * (ratio - target) * 0.25)
    world.compute_price = min(pc.ceiling, max(pc.floor, new))
