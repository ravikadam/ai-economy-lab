"""Pure simulation core: WorldState + the monthly 9-phase tick (spec §4).

No policy code lives here. `World.step(actions_by_firm_id)` advances one month:
  1 shocks & queued arrivals   2 publish lagged indicators   3 collect actions
  4 reserve budgets / resolve  5 advance experiments, gates  6 demand & compute
  7 opex, financing, bankruptcy  8 prices, confidence, supply, entry  9 rewards
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from economy import clearing, demand, supply, technology
from economy.accounting import Ledger
from economy.actions import Action, Verb, check_action
from economy.config import ScenarioConfig
from economy.events import PUBLIC_EVENTS, EventQueue, EventType, MarketEvent
from economy.rng import RngStreams
from entities.enterprises import generate_enterprises
from entities.firms import DEV_STAGES, LIVE_STAGES, RESERVE_MONTHS, FirmState, Project, RiskProfile, Stage
from entities.opportunities import OppStatus, generate_opportunities
from entities.suppliers import ChipDesigner, DataCenter, Fabricator, ModelProvider

REWARD_SCALE = 1000.0   # reward unit = $1M
PV_HORIZON = 24         # months of trailing profit credited to a live product
PV_DISCOUNT = 0.01      # monthly
# accounting convention for capitalised R&D: share of investment carried as an asset, by stage
BOOK_WEIGHT = {Stage.RESEARCH: 0.6, Stage.PROTOTYPE: 0.6, Stage.PROTOTYPE_READY: 0.8, Stage.PILOT: 0.8,
               Stage.PILOT_READY: 0.9, Stage.FAILED: 0.3}


@dataclass
class StepResult:
    month: int
    raw_rewards: dict[int, float] = field(default_factory=dict)
    shaped_rewards: dict[int, float] = field(default_factory=dict)
    terminated: set[int] = field(default_factory=set)     # bankrupt this month
    born: list[int] = field(default_factory=list)
    events: list[MarketEvent] = field(default_factory=list)
    diagnostics: dict[int, list[str]] = field(default_factory=dict)
    done: bool = False


class World:
    def __init__(self, cfg: ScenarioConfig, seed: int, keep_ledger_log: bool = False):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = RngStreams(seed)
        self.month = 0
        self.ledger = Ledger(keep_log=keep_ledger_log)
        self.queue = EventQueue()
        self.event_log: list[MarketEvent] = []
        self._month_events: list[MarketEvent] = []
        self.metrics_month: dict[str, Any] = {}
        self.price_history: list[dict] = []
        self.public: dict = {}
        self.capex_month = 0.0
        self.fab_orders: list[dict] = []
        self.gate_audit: list[dict] = []          # hidden: probabilities and draws
        self._diag: dict[int, list[str]] = {}
        self._next_firm_id = 0
        self.entries = 0
        self.exits = 0
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        cfg = self.cfg
        fc = cfg.fab
        self.fab = Fabricator(wafer_capacity=fc.wafer_capacity, yield_rate=fc.yield_rate)
        self.chips = [ChipDesigner(i, cfg.chips.efficiency[i], cfg.chips.base_price[i], cfg.chips.base_price[i])
                      for i in range(cfg.chips.count)]
        self.dcs = [DataCenter(i, cfg.datacenters.initial_capacity[i]) for i in range(cfg.datacenters.count)]
        for dc in self.dcs:
            self.ledger.transfer("ext:investors", dc.account, cfg.datacenters.initial_cash, "seed_capital")
        self.ledger.transfer("ext:investors", self.fab.account, 20000.0, "seed_capital")
        self.models = [ModelProvider(i, m.name, m.open_weight, list(m.capability), m.reliability,
                                     m.compute_per_unit, m.markup) for i, m in enumerate(cfg.technology.models)]
        self.compute_price = cfg.compute_price.initial
        technology.schedule_initial_releases(self)
        sizes = [s.enterprises for s in cfg.demand.sectors]
        self.opps = generate_opportunities(cfg, sizes, self.rng["opps"])
        order = self.rng["opps"].permutation(len(self.opps))
        for i in order[: cfg.opportunities.initially_available]:
            self.opps[int(i)].status = OppStatus.AVAILABLE
            self.opps[int(i)].discovered_month = 0
        self.ent = generate_enterprises(cfg, self.opps, self.rng["demand"])
        self.sector_conf = np.array([s.initial_confidence for s in cfg.demand.sectors], dtype=float)
        self.sector_revenue = np.zeros(cfg.n_sectors)
        self.firms: dict[int, FirmState] = {}
        for _ in range(cfg.builders.initial_firms):
            self._spawn_firm(entrant=False)
        self._publish_indicators()
        self.public = dict(self.price_history[-1])

    def _profile(self, rng) -> RiskProfile:
        rc = self.cfg.risk
        if not self.cfg.ablation.risk_heterogeneity:
            p = next((p for p in rc.profiles if p.name == "balanced"), rc.profiles[0])
        else:
            w = np.array(rc.profile_weights, dtype=float)
            p = rc.profiles[int(rng.choice(len(rc.profiles), p=w / w.sum()))]
        return RiskProfile(**p.model_dump())

    def _spawn_firm(self, entrant: bool, controller: str = "policy") -> FirmState:
        b = self.cfg.builders
        rng = self.rng["entry"]
        fid = self._next_firm_id
        self._next_firm_id += 1
        skill = float(rng.uniform(*b.skill_range)) * (0.9 if entrant else 1.0)
        f = FirmState(firm_id=fid, birth_month=self.month, profile=self._profile(rng), skill=skill,
                      sector_affinity=int(rng.integers(self.cfg.n_sectors)), controller=controller)
        f.signal_z = self.rng["signals"].normal(size=len(self.opps))  # type: ignore[attr-defined]
        cash = b.initial_cash * (float(rng.uniform(0.6, 1.0)) if entrant else 1.0)
        self.ledger.transfer("ext:investors", f.account, cash, "seed_capital")
        f.last_net_worth = self.net_worth(f)
        self.firms[fid] = f
        if entrant:
            self.entries += 1
            self.emit(EventType.FIRM_ENTRY, {"firm": fid, "profile": f.profile.name, "cash": round(cash, 1)})
        return f

    # --------------------------------------------------------------- helpers
    def emit(self, kind: EventType, payload: dict, public: bool | None = None) -> MarketEvent:
        ev = MarketEvent(self.month, kind, payload)
        ev.payload.setdefault("_public", kind in PUBLIC_EVENTS if public is None else public)
        self._month_events.append(ev)
        return ev

    def schedule(self, ev: MarketEvent) -> None:
        self.queue.schedule(ev)

    def alive_firms(self) -> list[FirmState]:
        return [f for f in self.firms.values() if f.alive]

    def available_opportunities(self):
        return [o for o in self.opps if o.status == OppStatus.AVAILABLE]

    def claimed_opportunities(self) -> set[int]:
        return {p.opportunity_id for f in self.alive_firms() for p in f.projects}

    def cash(self, f: FirmState) -> float:
        return self.ledger.balance(f.account)

    def monthly_burn(self, f: FirmState) -> float:
        b = self.cfg.builders
        burn = b.overhead + b.interest_rate * f.debt
        for p in f.projects:
            if p.stage in DEV_STAGES:
                burn += b.tier_burn[p.tier]
            elif p.is_live:
                burn += max(0.0, -(p.trailing_profit()))
        return burn

    def runway(self, f: FirmState) -> float:
        net = self.monthly_burn(f) - sum(max(0.0, p.trailing_profit()) for p in f.projects if p.is_live)
        cash = self.cash(f) + max(0.0, self.cfg.builders.credit_limit - f.debt)
        if net <= 0:
            return 36.0
        return float(min(36.0, cash / net))

    def project_pv(self, p: Project) -> float:
        """Live product: PV of trailing profit. In development: capitalised investment (book value)
        weighted by a stage-based success convention and faded to zero over the final 12 months,
        so terminal value only counts what can still earn. Failures are impaired; abandonment
        writes the book value off. This is potential-based shaping: the episode sum still
        telescopes to terminal value − initial value."""
        remaining = self.cfg.episode.months - self.month
        if p.is_live:
            n = max(0, min(PV_HORIZON, remaining))
            annuity = (1 - (1 + PV_DISCOUNT) ** -n) / PV_DISCOUNT if n > 0 else 0.0
            return max(0.0, p.trailing_profit()) * annuity
        stage = p.paused_from if p.stage == Stage.PAUSED and p.paused_from is not None else p.stage
        w = BOOK_WEIGHT.get(stage, 0.0) * (0.8 if p.stage == Stage.PAUSED else 1.0)
        return p.investment * w * max(0.0, min(1.0, remaining / 12.0))

    def net_worth(self, f: FirmState) -> float:
        """Firm value: cash − debt + PV of live products + risk-weighted book value of projects in development.
        A bankrupt firm's projects are worthless: its value is cash − debt (cash is liquidated, so −debt)."""
        if not f.alive:
            return self.cash(f) - f.debt
        return self.cash(f) - f.debt + sum(self.project_pv(p) for p in f.projects)

    def firm_pay(self, f: FirmState, dst: str, amount: float, memo: str) -> float:
        """Pay, drawing on the credit line if needed. Shortfall marks the firm insolvent."""
        if amount <= 0:
            return 0.0
        cash = self.cash(f)
        if cash < amount:
            draw = min(amount - cash, max(0.0, self.cfg.builders.credit_limit - f.debt))
            if draw > 0:
                self.ledger.transfer("ext:lender", f.account, draw, "credit_draw")
                f.debt += draw
        paid = self.ledger.transfer(f.account, dst, amount, memo, allow_partial=True)
        if paid < amount - 1e-9:
            f._insolvent = True  # type: ignore[attr-defined]
        return paid

    def difficulty_signal(self, f: FirmState, opp_id: int) -> float:
        sd = self.cfg.opportunities.signal_noise * f.knowledge.get(opp_id, 1.0)
        return float(self.opps[opp_id].latent_difficulty + sd * f.signal_z[opp_id])  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ tick
    def step(self, actions: dict[int, Action]) -> StepResult:
        self.month += 1
        self.ledger.start_month(self.month)
        self._month_events = []
        self.metrics_month = {}
        self.capex_month = 0.0
        self._diag = {}
        self.sector_revenue = np.zeros(self.cfg.n_sectors)
        for m in self.models:
            m.load_last = 0.0
        for f in self.alive_firms():
            f._insolvent = False  # type: ignore[attr-defined]
            f._spent = 0.0  # type: ignore[attr-defined]
            for p in f.projects:
                p.revenue_last = 0.0
                p.cost_last = 0.0

        # 1. exogenous shocks and queued arrivals
        for ev in self.queue.pop_due(self.month):
            supply.apply_supply_event(self, ev)
            self._month_events.append(ev)
        supply.maybe_fab_shock(self)
        technology.advance_models(self)
        self._discover_opportunities()

        # 2. publish lagged indicators (last completed month)
        self.public = dict(self.price_history[-1])

        # 3-4. collect simultaneous actions, validate, reserve budgets (random fair order)
        alive = self.alive_firms()
        order = self.rng["alloc"].permutation(len(alive))
        illegal: dict[int, int] = {}
        for i in order:
            f = alive[int(i)]
            a = actions.get(f.firm_id, Action())
            ok, reason = check_action(self, f, a)
            if not ok:
                f.illegal_actions += 1
                illegal[f.firm_id] = 1
                self._diag.setdefault(f.firm_id, []).append(f"illegal {a.to_dict()}: {reason}")
                self.emit(EventType.ILLEGAL_ACTION, {"firm": f.firm_id, "action": a.to_dict(), "reason": reason},
                          public=False)
                a = Action(reserve=a.reserve if a.reserve is not None and 0 <= int(a.reserve) < 3 else None)
            self._execute(f, a)

        # 5. advance experiments and sample gates
        for f in self.alive_firms():
            self._advance_projects(f)

        # 6. demand: adoption, compute clearing, billing, churn
        rows = clearing.compute_demand_by_project(self)
        served = clearing.clear_compute(self, rows)
        demand.churn_and_bill(self, served)
        demand.adoption(self)

        # 7. operating costs, financing, bankruptcy
        result = StepResult(month=self.month)
        for f in self.alive_firms():
            self._settle(f)
        for f in list(self.alive_firms()):
            if getattr(f, "_insolvent", False):
                self._bankrupt(f)
                result.terminated.add(f.firm_id)

        # 8. prices, confidence, upstream investment, entry
        clearing.update_compute_price(self)
        supply.update_chip_prices(self)
        supply.run_fab(self)
        supply.place_chip_orders(self)
        supply.fab_invest(self)
        supply.retire_capacity(self)
        demand.update_confidence(self)
        born = self._entry()
        result.born = born

        # 9. rewards
        from agents.reward import compute_rewards  # local import keeps core free of policy code at import
        for fid in list(self.firms):
            f = self.firms[fid]
            if not f.alive and fid not in result.terminated:
                continue
            raw, shaped = compute_rewards(self, f, bankrupt=fid in result.terminated,
                                          illegal=illegal.get(fid, 0))
            result.raw_rewards[fid] = raw
            result.shaped_rewards[fid] = shaped
        self._publish_indicators()
        result.events = list(self._month_events)
        self.event_log.extend(self._month_events)
        result.diagnostics = self._diag
        result.done = self.month >= self.cfg.episode.months
        self.ledger.check_invariants()
        return result

    # ------------------------------------------------------------- phase 4
    def _execute(self, f: FirmState, a: Action) -> None:
        b = self.cfg.builders
        if a.reserve is not None:
            f.reserve_level = int(a.reserve)
        v = a.verb
        if v == Verb.NOOP:
            return
        if v == Verb.START:
            p = Project(firm_id=f.firm_id, opportunity_id=a.opportunity_id, model_id=a.model_id, tier=a.tier)
            self.firm_pay(f, "ext:labor", b.start_cost, "project_start")
            p.investment += b.start_cost
            f.projects.append(p)
            return
        p = f.projects[a.slot]
        if v == Verb.PILOT:
            ids = demand.recruit_pilots(self, f, p)
            if not ids:
                f.deferrals += 1
                self._diag.setdefault(f.firm_id, []).append("pilot deferred: no enterprise willing (low confidence)")
                self.emit(EventType.PROJECT_GATE_RESULT, {"firm": f.firm_id, "opportunity": p.opportunity_id,
                                                          "gate": "pilot_recruit", "result": "deferred"},
                          public=False)
                return
            self.firm_pay(f, "ext:labor", b.pilot_cost, "pilot")
            p.investment += b.pilot_cost
            p.pilot_ids = ids
            p.stage, p.progress, p.stage_age = Stage.PILOT, 0.0, 0
        elif v == Verb.LAUNCH:
            self.firm_pay(f, "ext:labor", b.launch_cost, "launch")
            p.investment += b.launch_cost
            p.price = demand.list_price(self, p.opportunity_id, f.firm_id)
            p.stage, p.stage_age = Stage.DEPLOYED, 0
            f.launches += 1
            f.skill = min(1.0, f.skill + b.skill_gain_per_launch)
            f.reputation = min(1.0, f.reputation + b.reputation_gain)
            converted = demand.convert_pilots(self, f, p)
            self.emit(EventType.PROJECT_GATE_RESULT, {"firm": f.firm_id, "opportunity": p.opportunity_id,
                                                      "gate": "launch", "converted": converted}, public=True)
        elif v == Verb.SCALE:
            self.firm_pay(f, "ext:labor", b.scale_cost, "scale")
            p.investment += b.scale_cost
            p.stage = Stage.SCALED
        elif v == Verb.ABANDON:
            self._close_project(f, a.slot, "abandoned")
        elif v == Verb.RETRY:
            self.firm_pay(f, "ext:labor", b.retry_cost, "retry")
            p.investment += b.retry_cost
            p.retries += 1
            p.model_id, p.tier = a.model_id, a.tier
            if p.failure_kind == "commercial":
                p.stage = Stage.PROTOTYPE_READY      # go back to pilot with (possibly) a new model
            else:
                p.stage = Stage.PROTOTYPE
            p.progress, p.stage_age, p.failure_kind = 0.0, 0, ""
        elif v == Verb.PAUSE:
            if p.stage == Stage.PAUSED:
                p.stage, p.paused_from = p.paused_from or Stage.RESEARCH, None
            else:
                if p.stage == Stage.PILOT:
                    self.ent.piloting[np.array(p.pilot_ids, dtype=np.int64), p.opportunity_id] = -1
                    p.pilot_ids = []
                    p.paused_from, p.stage, p.progress = Stage.PROTOTYPE_READY, Stage.PAUSED, 0.0
                else:
                    p.paused_from, p.stage = p.stage, Stage.PAUSED

    def _close_project(self, f: FirmState, slot: int, outcome: str) -> None:
        p = f.projects.pop(slot)
        p.outcome = outcome
        if p.customers:
            ids = np.flatnonzero(self.ent.provider[:, p.opportunity_id] == f.firm_id)
            demand.drop_customers(self, f, p, ids, reason="shutdown")
        if p.pilot_ids:
            self.ent.piloting[np.array(p.pilot_ids, dtype=np.int64), p.opportunity_id] = -1
        f.closed_projects = getattr(f, "closed_projects", []) + [p]  # type: ignore[attr-defined]

    # ------------------------------------------------------------- phase 5
    def _advance_projects(self, f: FirmState) -> None:
        b = self.cfg.builders
        for p in f.projects:
            p.age += 1
            p.stage_age += 1
            if p.stage not in DEV_STAGES:
                continue
            if p.stage != Stage.PILOT:
                burn = b.tier_burn[p.tier]
                self.firm_pay(f, "ext:labor", burn, "rnd_burn")
                p.investment += burn
                p.progress += b.tier_speed[p.tier]
            else:
                p.progress += 1.0
            opp = self.opps[p.opportunity_id]
            if p.stage == Stage.RESEARCH and p.progress >= opp.duration:
                f.knowledge[opp.opportunity_id] = f.knowledge.get(opp.opportunity_id, 1.0) * b.research_signal_shrink
                p.stage, p.progress, p.stage_age = Stage.PROTOTYPE, 0.0, 0
            elif p.stage == Stage.PROTOTYPE and p.progress >= opp.duration:
                effort = b.tier_effort[p.tier] + b.retry_effort_bonus * p.retries
                prob = technology.success_probability(self, opp, p.model_id, f.skill, effort)
                draw = float(self.rng["gates"].random())
                ok = draw < prob if self.cfg.ablation.technical_failure else True
                self.gate_audit.append({"month": self.month, "firm": f.firm_id, "opportunity": opp.opportunity_id,
                                        "gate": "technical", "p": prob, "draw": draw, "ok": ok})
                f.knowledge[opp.opportunity_id] = f.knowledge.get(opp.opportunity_id, 1.0) * 0.7
                if ok:
                    p.stage = Stage.PROTOTYPE_READY
                else:
                    p.stage, p.failure_kind = Stage.FAILED, "technical"
                    f.tech_failures += 1
                    f.reputation = max(0.0, f.reputation - b.reputation_loss * 0.5)
                p.progress, p.stage_age = 0.0, 0
                self.emit(EventType.PROJECT_GATE_RESULT, {"firm": f.firm_id, "opportunity": opp.opportunity_id,
                                                          "gate": "technical", "result": "pass" if ok else "fail"},
                          public=False)
                self.metrics_month["tech_pass" if ok else "tech_fail"] = \
                    self.metrics_month.get("tech_pass" if ok else "tech_fail", 0) + 1
                self._record_gate(opp, ok, "technical")
            elif p.stage == Stage.PILOT and p.progress >= b.pilot_months:
                ok, detail = demand.pilot_gate(self, f, p)
                self.gate_audit.append({"month": self.month, "firm": f.firm_id, "opportunity": opp.opportunity_id,
                                        "gate": "commercial", **detail, "ok": ok})
                if ok:
                    p.stage = Stage.PILOT_READY
                else:
                    p.stage, p.failure_kind = Stage.FAILED, "commercial"
                    p.pilot_ids = []
                    f.commercial_failures += 1
                    f.reputation = max(0.0, f.reputation - b.reputation_loss)
                p.progress, p.stage_age = 0.0, 0
                self.emit(EventType.PROJECT_GATE_RESULT, {"firm": f.firm_id, "opportunity": opp.opportunity_id,
                                                          "gate": "commercial", "result": "pass" if ok else "fail",
                                                          "reliability_ok": detail["passes_reliability"],
                                                          "value_ok": detail["passes_value"]}, public=True)
                self.metrics_month["pilot_pass" if ok else "pilot_fail"] = \
                    self.metrics_month.get("pilot_pass" if ok else "pilot_fail", 0) + 1
                self._record_gate(opp, ok, "commercial")

    def _record_gate(self, opp, ok: bool, gate: str) -> None:
        bucket = min(2, int(opp.latent_difficulty * 3))     # low/med/high complexity
        key = f"{gate}_{'pass' if ok else 'fail'}_by_complexity"
        arr = self.metrics_month.setdefault(key, [0, 0, 0])
        arr[bucket] += 1

    # ------------------------------------------------------------- phase 7
    def _settle(self, f: FirmState) -> None:
        b = self.cfg.builders
        self.firm_pay(f, "ext:labor", b.overhead, "overhead")
        for p in f.projects:
            if p.is_live:
                support = b.support_cost_per_customer * p.customers
                self.firm_pay(f, "ext:labor", support, "support")
                p.cost_last += support
                p.profit_hist.append(p.revenue_last - p.cost_last)
        if f.debt > 0:
            self.firm_pay(f, "ext:lender", f.debt * b.interest_rate, "interest")
            excess = self.cash(f) - RESERVE_MONTHS[f.reserve_level] * self.monthly_burn(f) * 1.5
            if excess > 0:
                repay = min(excess, f.debt)
                self.ledger.transfer(f.account, "ext:lender", repay, "credit_repay")
                f.debt -= repay
        profit = sum(p.revenue_last - p.cost_last for p in f.projects if p.is_live)
        f.monthly_profit_hist.append(profit)

    def _bankrupt(self, f: FirmState) -> None:
        for slot in reversed(range(len(f.projects))):
            p = f.projects[slot]
            if p.customers:
                ids = np.flatnonzero(self.ent.provider[:, p.opportunity_id] == f.firm_id)
                demand.drop_customers(self, f, p, ids, reason="bankruptcy")
            if p.pilot_ids:
                self.ent.piloting[np.array(p.pilot_ids, dtype=np.int64), p.opportunity_id] = -1
                p.pilot_ids = []
        if self.cfg.ablation.confidence_feedback:
            s = f.sector_affinity
            self.sector_conf[s] = max(0.02, self.sector_conf[s] - self.cfg.demand.confidence_bankruptcy)
        left = self.cash(f)
        if left > 0:
            self.ledger.transfer(f.account, "ext:lender", left, "liquidation")
        f.alive = False
        f.death_month = self.month
        self.exits += 1
        self.emit(EventType.BANKRUPTCY, {"firm": f.firm_id, "debt": round(f.debt, 1), "age": f.age(self.month)})

    # ------------------------------------------------------------- phase 8
    def _discover_opportunities(self) -> None:
        rng = self.rng["opps"]
        oc = self.cfg.opportunities
        for o in self.opps:
            if o.status != OppStatus.UNDISCOVERED:
                continue
            reachable = technology.best_capability(self, o.sector) >= o.capability_threshold
            rate = oc.discovery_rate * (2.0 if reachable else 0.3)
            if rng.random() < rate:
                o.status = OppStatus.AVAILABLE
                o.discovered_month = self.month
                self.emit(EventType.OPPORTUNITY_DISCOVERED, {"opportunity": o.opportunity_id, "sector": o.sector})

    def _entry(self) -> list[int]:
        b = self.cfg.builders
        born: list[int] = []
        alive = self.alive_firms()
        if b.replace_bankrupt:
            while len(self.alive_firms()) < b.initial_firms:
                born.append(self._spawn_firm(entrant=True).firm_id)
        if self.month < b.entry_start_month or len(self.alive_firms()) >= b.max_firms:
            return born
        trailing = [np.mean(f.monthly_profit_hist[-6:]) for f in alive if f.monthly_profit_hist]
        mean_profit = float(np.mean(trailing)) if trailing else 0.0
        p = min(b.entry_max_prob, max(0.0, b.entry_base_prob + b.entry_profit_sensitivity * mean_profit))
        if self.rng["entry"].random() < p:
            born.append(self._spawn_firm(entrant=True).firm_id)
        return born

    # ----------------------------------------------------------- indicators
    def _publish_indicators(self) -> None:
        mm = self.metrics_month
        cap_total = sum(dc.capacity for dc in self.dcs)
        rev = {}
        for f in self.alive_firms():
            rev[f.firm_id] = sum(p.revenue_last for p in f.projects)
        tot = sum(rev.values())
        hhi = sum((r / tot) ** 2 for r in rev.values()) if tot > 0 else 0.0
        adoption = np.zeros(self.cfg.n_sectors)
        provided = (self.ent.provider >= 0)
        for s in range(self.cfg.n_sectors):
            rows = self.ent.sector == s
            adoption[s] = provided[rows].sum()
        builder_profit = sum(f.monthly_profit_hist[-1] for f in self.alive_firms() if f.monthly_profit_hist)
        benefit = mm.get("enterprise_benefit", 0.0)
        payments = mm.get("enterprise_payments", 0.0)
        launches = [e.payload for e in self._month_events
                    if e.kind == EventType.PROJECT_GATE_RESULT and e.payload.get("gate") == "launch"]
        rec = {
            "month": self.month,
            "compute_price": self.compute_price,
            "compute_capacity": cap_total,
            "compute_pending": sum(dc.pending_capacity for dc in self.dcs),
            "compute_demand": mm.get("compute_demand", 0.0),
            "utilization": mm.get("utilization", 0.0),
            "served_frac": mm.get("served_frac", 1.0),
            "unmet_compute": mm.get("unmet_compute", 0.0),
            "fab_capacity": self.fab.effective_capacity,
            "fab_nominal_capacity": self.fab.wafer_capacity,
            "fab_pending": self.fab.pending_capacity,
            "fab_backlog": self.fab.backlog,
            "fab_utilization": self.fab.utilization,
            "fab_shock": 1.0 - self.fab.shock_factor,
            "chip_price": float(np.mean([c.unit_price for c in self.chips])),
            "supplier_capex": self.capex_month,
            "model_capability": [list(m.capability) for m in self.models],
            "model_reliability": [m.reliability for m in self.models],
            "model_unit_price": [m.unit_price(self.compute_price) for m in self.models],
            "sector_confidence": self.sector_conf.tolist(),
            "sector_adoption": adoption.tolist(),
            "sector_revenue": self.sector_revenue.tolist(),
            "n_firms": len(self.alive_firms()),
            "entries_total": self.entries,
            "exits_total": self.exits,
            "hhi": hhi,
            "builder_revenue": tot,
            "builder_profit": builder_profit,
            "enterprise_benefit": benefit,
            "enterprise_payments": payments,
            "welfare": benefit - payments + builder_profit,
            "adoptions": mm.get("adoptions", 0),
            "churn": mm.get("churn", 0),
            "tech_pass": mm.get("tech_pass", 0), "tech_fail": mm.get("tech_fail", 0),
            "pilot_pass": mm.get("pilot_pass", 0), "pilot_fail": mm.get("pilot_fail", 0),
            "technical_pass_by_complexity": mm.get("technical_pass_by_complexity", [0, 0, 0]),
            "technical_fail_by_complexity": mm.get("technical_fail_by_complexity", [0, 0, 0]),
            "commercial_pass_by_complexity": mm.get("commercial_pass_by_complexity", [0, 0, 0]),
            "commercial_fail_by_complexity": mm.get("commercial_fail_by_complexity", [0, 0, 0]),
            "launches": launches,
            "available_opps": len(self.available_opportunities()),
        }
        self.price_history.append(rec)

    # ------------------------------------------------------- audit / replay
    def state_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps({k: round(v, 6) for k, v in sorted(self.ledger.balances.items())}).encode())
        for f in sorted(self.firms.values(), key=lambda x: x.firm_id):
            h.update(json.dumps([f.firm_id, f.alive, round(f.debt, 6),
                                 [(p.opportunity_id, int(p.stage), p.customers) for p in f.projects]]).encode())
        h.update(self.ent.provider.tobytes())
        h.update(np.round(self.ent.confidence, 9).tobytes())
        h.update(json.dumps([e.to_dict() for e in self.event_log], default=str).encode())
        return h.hexdigest()

    def firm_summary(self, f: FirmState) -> dict:
        return {
            "firm_id": f.firm_id, "alive": f.alive, "controller": f.controller, "profile": f.profile.name,
            "cash": round(self.cash(f), 2), "debt": round(f.debt, 2), "net_worth": round(self.net_worth(f), 2),
            "runway": round(self.runway(f), 2), "skill": round(f.skill, 3), "reputation": round(f.reputation, 3),
            "launches": f.launches, "tech_failures": f.tech_failures, "commercial_failures": f.commercial_failures,
            "deferrals": f.deferrals, "customers": sum(p.customers for p in f.projects),
            "profit": round(f.monthly_profit_hist[-1], 2) if f.monthly_profit_hist else 0.0,
            "projects": [{"opportunity": p.opportunity_id, "sector": self.opps[p.opportunity_id].sector,
                          "stage": p.stage.name, "model": self.models[p.model_id].name, "tier": p.tier,
                          "customers": p.customers, "investment": round(p.investment, 1),
                          "profit": round(p.profit_hist[-1], 2) if p.profit_hist else 0.0} for p in f.projects],
            "cum_raw_reward": round(f.cum_raw_reward, 4), "cum_shaped_reward": round(f.cum_shaped_reward, 4),
        }
