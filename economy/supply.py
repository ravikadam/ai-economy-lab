"""Upstream supply chain dynamics: chip orders, fab production, delayed capacity.

Demand propagates backward (spec §6): data-centre utilisation -> chip orders ->
wafer orders -> fab backlog/utilisation -> chip price and fab capex. Capacity
additions arrive through the event queue after explicit lead times, so
investment today can never produce capacity before its configured delay.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from economy.events import EventType, MarketEvent

if TYPE_CHECKING:
    from economy.world import World


def _fund(world: "World", account: str, amount: float) -> None:
    """Scripted upstream firms are kept solvent by equity injections when short."""
    short = amount - world.ledger.balance(account)
    if short > 0:
        world.ledger.transfer("ext:investors", account, short, "upstream_equity")


def apply_supply_event(world: "World", ev: MarketEvent) -> None:
    p = ev.payload
    if ev.kind == EventType.CAPACITY_ONLINE:
        if p["target"] == "fab":
            world.fab.wafer_capacity = min(world.cfg.fab.max_capacity, world.fab.wafer_capacity + p["amount"])
            world.fab.pending_capacity -= p["amount"]
        else:
            dc = world.dcs[p["dc"]]
            dc.capacity += p["amount"]
            dc.pending_capacity -= p["amount"]
    elif ev.kind == EventType.CHIP_DELIVERY:
        dc = world.dcs[p["dc"]]
        designer = world.chips[p["designer"]]
        cost = p["chips"] * designer.unit_price
        _fund(world, dc.account, cost)
        world.ledger.transfer(dc.account, designer.account, cost, "chip_purchase")
        units = p["chips"] * designer.efficiency
        dc.chips_on_order -= units
        delay = world.cfg.datacenters.install_delay if world.cfg.ablation.delayed_capex else 0
        if delay == 0:
            dc.capacity += units
        else:
            dc.pending_capacity += units
            world.schedule(MarketEvent(world.month + delay, EventType.CAPACITY_ONLINE,
                                       {"target": "dc", "dc": dc.dc_id, "amount": units}))
    elif ev.kind == EventType.FAB_RECOVERY:
        world.fab.shock_factor = 1.0
        world.fab.shock_until = -1


def maybe_fab_shock(world: "World") -> None:
    sc = world.cfg.shocks
    if not world.cfg.ablation.supply_bottlenecks or world.fab.shock_until >= world.month:
        return
    rng = world.rng["supply"]
    if rng.random() < sc.fab_shortage_prob:
        months = int(rng.integers(sc.fab_shortage_months[0], sc.fab_shortage_months[1] + 1))
        sev = float(rng.uniform(*sc.fab_shortage_severity))
        world.fab.shock_factor = 1.0 - sev
        world.fab.shock_until = world.month + months
        world.emit(EventType.FAB_SHORTAGE, {"months": months, "severity": round(sev, 3)})
        world.schedule(MarketEvent(world.month + months, EventType.FAB_RECOVERY, {}))


def place_chip_orders(world: "World") -> None:
    """Data centres order chips when trailing utilisation is high."""
    dcc = world.cfg.datacenters
    for dc in world.dcs:
        util = dc.trailing_util()
        in_flight = dc.chips_on_order + dc.pending_capacity
        if util < dcc.order_trigger_utilization or in_flight > 0.5 * dcc.order_step_frac * dc.capacity:
            continue
        units = dcc.order_step_frac * dc.capacity
        # split across designers by compute-per-dollar
        scores = [c.efficiency / c.unit_price for c in world.chips]
        tot = sum(scores)
        for c, sc in zip(world.chips, scores):
            u = units * sc / tot
            chips = u / c.efficiency
            wafers = chips / (world.cfg.fab.chips_per_wafer * world.fab.yield_rate)
            world.fab_orders.append({"designer": c.designer_id, "dc": dc.dc_id, "wafers": wafers,
                                     "chips_per_wafer": chips / wafers})
            world.fab.backlog += wafers
            dc.chips_on_order += u


def run_fab(world: "World") -> None:
    """Produce wafers against the backlog; oversubscription resolved proportionally."""
    fab = world.fab
    cap = fab.effective_capacity if world.cfg.ablation.supply_bottlenecks else max(fab.backlog, 1.0)
    produce = min(cap, fab.backlog)
    fab.output_last = produce
    fab.utilization = produce / cap if cap > 0 else 0.0
    if produce <= 0:
        return
    frac = produce / fab.backlog
    deliveries: dict[tuple[int, int], float] = {}
    remaining = []
    for o in world.fab_orders:
        w = o["wafers"] * frac
        o["wafers"] -= w
        key = (o["designer"], o["dc"])
        deliveries[key] = deliveries.get(key, 0.0) + w * o["chips_per_wafer"]
        if o["wafers"] > 1e-6:
            remaining.append(o)
        # designer pays fab for wafers at production
        designer = world.chips[o["designer"]]
        cost = w * world.cfg.fab.wafer_price
        _fund(world, designer.account, cost)
        world.ledger.transfer(designer.account, fab.account, cost, "wafer_purchase")
    world.fab_orders = remaining
    fab.backlog = max(0.0, fab.backlog - produce)
    for (d, dc), chips in deliveries.items():
        world.schedule(MarketEvent(world.month + 1, EventType.CHIP_DELIVERY,
                                   {"designer": d, "dc": dc, "chips": chips}))


def fab_invest(world: "World") -> None:
    fc = world.cfg.fab
    fab = world.fab
    if fab.backlog <= fc.invest_backlog_ratio * fab.effective_capacity or fab.pending_capacity > 0:
        return
    if fab.wafer_capacity >= fc.max_capacity:
        return
    add = fc.invest_step_frac * fab.wafer_capacity
    cost = add * fc.capex_per_wafer_capacity
    _fund(world, fab.account, cost)
    world.ledger.transfer(fab.account, "ext:capex", cost, "fab_capex")
    fab.capex_total += cost
    world.capex_month += cost
    lead = fc.capex_lead_time if world.cfg.ablation.delayed_capex else 0
    if lead == 0:
        fab.wafer_capacity += add
    else:
        fab.pending_capacity += add
        world.schedule(MarketEvent(world.month + lead, EventType.CAPACITY_ONLINE, {"target": "fab", "amount": add}))


def update_chip_prices(world: "World") -> None:
    sens = world.cfg.chips.price_sensitivity
    pressure = world.fab.backlog / max(world.fab.effective_capacity, 1e-6)
    for c in world.chips:
        c.unit_price = c.base_price * min(3.0, max(0.7, 1.0 + sens * (pressure - 1.0)))


def retire_capacity(world: "World") -> None:
    for dc in world.dcs:
        dc.capacity *= (1.0 - world.cfg.datacenters.depreciation)
