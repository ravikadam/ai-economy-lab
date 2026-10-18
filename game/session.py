"""Human-in-the-loop game session: an implementation team runs one builder firm.

Seat 0 is the team. Seats 1-4 are rivals (trained RL policy, heuristics, or a
mix); entrants follow the same rival mix. The trained policy doubles as the
AI coach: it shows what it would do and its value estimate, and a what-if
lookahead replays the next months under both choices with common random
numbers. At the end a "ghost" run puts the coach in the team's seat on the
same seed so the debrief can show the counterfactual.
"""
from __future__ import annotations

import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from agents.actions import decode, encode
from agents.heuristics import estimate
from analysis.replay import ReplayRecorder, decision_record
from economy.actions import SLOT_VERBS, Action, Verb, check_action, spend_cost
from economy.config import load_config
from economy.world import World
from entities.firms import RESERVE_MONTHS, RiskProfile, Stage
from envs.multi_env import MultiFirmEnv
from game.coach import PRINCIPLES, review_decision, review_state

RIVAL_FIELDS = {
    "heuristics": ({1: "heuristic:best_npv", 2: "heuristic:easy_first", 3: "heuristic:cash_preserving",
                    4: "heuristic:best_npv"},
                   {"heuristic:best_npv": 0.5, "heuristic:easy_first": 0.3, "heuristic:cash_preserving": 0.2}),
    "rl": ({1: "policy", 2: "policy", 3: "policy", 4: "policy"}, {"policy": 1.0}),
    "mixed": ({1: "policy", 2: "heuristic:best_npv", 3: "policy", 4: "heuristic:easy_first"},
              {"policy": 0.5, "heuristic:best_npv": 0.5}),
}


def _load_coach(checkpoint: str | None):
    if not checkpoint:
        return None
    try:
        from training.checkpoints import load_checkpoint
        net, _, _ = load_checkpoint(checkpoint)
        return net
    except Exception as e:  # noqa: BLE001 — game still works without a coach
        print(f"[game] coach unavailable: {e}")
        return None


class GameSession:
    def __init__(self, team: str, scenario: str = "config/base.yaml", seed: int = 42, rivals: str = "heuristics",
                 checkpoint: str | None = "runs/latest", months: int | None = None, months_per_turn: int = 3,
                 profile: str = "balanced", mission_id: str = "custom", replay_dir: str = "runs/replays"):
        overrides = {"episode": {"months": months}} if months else None
        self.cfg = load_config(scenario, overrides=overrides)
        self.team, self.seed, self.mission_id = team, seed, mission_id
        self.months_per_turn = months_per_turn
        self.profile_name = profile
        self.net = _load_coach(checkpoint)
        if rivals in ("rl", "mixed") and self.net is None:
            rivals = "heuristics"
        self.rivals = rivals
        fixed, mix = RIVAL_FIELDS[rivals]
        self.fixed = {0: "human", **fixed}
        self.mix = mix
        self.env = self._make_env(self.fixed)
        self.batch = self.env.reset(seed)
        self.fid = int(self.env.seat_firm[0])
        self._set_profile(self.env.world, self.fid)
        from agents.runner import PolicyRunner
        self.rival_runner = PolicyRunner(self.net, seed=seed) if self.net is not None else None
        self.coach_runner = PolicyRunner(self.net, deterministic=True, seed=seed) if self.net is not None else None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.replay_path = Path(replay_dir) / f"game-{team.replace(' ', '_')}-{mission_id}-{seed}-{stamp}.jsonl"
        self.rec = ReplayRecorder(self.replay_path, self.cfg, seed, self.fixed,
                                  {"mode": "game", "team": team, "mission": mission_id, "rivals": rivals})
        self.turns: list[dict] = []
        self.lesson_counts: Counter = Counter()
        self.good_counts: Counter = Counter()
        self.agree = 0
        self.decisions = 0
        self.finished = False
        self.debrief_cache: dict | None = None

    # ------------------------------------------------------------ setup
    def _make_env(self, fixed):
        return MultiFirmEnv(self.cfg, self.mix, fixed_controllers=fixed, min_policy_seats=0)

    def _set_profile(self, world: World, fid: int) -> None:
        p = next(p for p in self.cfg.risk.profiles if p.name == self.profile_name)
        world.firms[fid].profile = RiskProfile(**p.model_dump())

    @property
    def world(self) -> World:
        return self.env.world

    @property
    def firm(self):
        return self.world.firms[self.fid]

    # ------------------------------------------------------------ views
    def candidates(self) -> list[dict]:
        w, f = self.world, self.firm
        rows = []
        for o in self.env.candidates(0):
            opp = w.opps[o]
            sec = self.cfg.demand.sectors[opp.sector]
            ests = {w.models[m].name: estimate(w, f, o, m, 1) for m in range(self.cfg.n_models)}
            competitors = sum(1 for g in w.alive_firms() if g.firm_id != f.firm_id
                              for p in g.projects if p.opportunity_id == o and p.is_live)
            row = {"id": o, "sector": sec.name, "value_signal": round(opp.value_signal, 1),
                   "customers_with_need": opp.need_count, "difficulty_signal": round(w.difficulty_signal(f, o), 2),
                   "reliability_needed": round(sec.reliability_floor + opp.reliability_req, 3),
                   "months_per_stage": opp.duration, "integration_burden": round(opp.integration_burden, 2),
                   "live_competitors": competitors}
            for name, e in ests.items():
                row[f"p_success[{name}]"] = round(e.p_tech * e.p_comm, 2)
                row[f"margin[{name}]"] = round(e.margin, 1)
            rows.append(row)
        return rows

    def projects(self) -> list[dict]:
        w = self.world
        rows = []
        for i, p in enumerate(self.firm.projects):
            opp = w.opps[p.opportunity_id]
            rows.append({"slot": i, "use_case": p.opportunity_id, "sector": self.cfg.demand.sectors[opp.sector].name,
                         "stage": p.stage.name, "model": w.models[p.model_id].name, "tier": ["lean", "standard",
                                                                                              "premium"][p.tier],
                         "progress": f"{p.progress:.1f}/{opp.duration}" if p.stage in (Stage.RESEARCH,
                                                                                         Stage.PROTOTYPE) else "",
                         "customers": p.customers, "profit_last": round(p.profit_hist[-1], 1) if p.profit_hist else 0,
                         "invested": round(p.investment, 0), "retries": p.retries, "failure": p.failure_kind})
        return rows

    def kpis(self) -> dict:
        w, f = self.world, self.firm
        ranked = sorted(w.firms.values(), key=lambda g: -w.net_worth(g))
        alive = [g for g in ranked if g.alive]
        return {"month": w.month, "months_total": self.cfg.episode.months, "cash": w.cash(f), "debt": f.debt,
                "net_worth": w.net_worth(f), "runway": w.runway(f), "customers": sum(p.customers for p in f.projects),
                "profit": f.monthly_profit_hist[-1] if f.monthly_profit_hist else 0.0, "reputation": f.reputation,
                "skill": f.skill, "alive": f.alive, "value_created": f.cum_raw_reward,
                "rank": (alive.index(f) + 1) if f in alive else None, "n_firms": len(alive),
                "reserve_months": RESERVE_MONTHS[f.reserve_level], "profile": f.profile.name}

    def rivals_table(self) -> list[dict]:
        w = self.world
        out = []
        for g in w.firms.values():
            if g.firm_id == self.fid:
                continue
            live = [p for p in g.projects if p.is_live]
            out.append({"firm": g.firm_id, "who": g.controller.replace("heuristic:", "rule:"), "alive": g.alive,
                        "live_products": len(live), "customers": sum(p.customers for p in live),
                        "launches": g.launches, "status": "active" if g.alive else f"bankrupt m{g.death_month}"})
        return out

    def news(self, n_months: int = 3) -> list[dict]:
        lo = self.world.month - n_months + 1
        out = []
        for e in self.world.event_log:
            if e.month < lo:
                continue
            mine = e.payload.get("firm") == self.fid
            if e.payload.get("_public") or mine:
                out.append({"month": e.month, "kind": e.kind.value, "yours": mine,
                            **{k: v for k, v in e.payload.items() if not k.startswith("_")}})
        return out[::-1]

    def legal(self) -> dict:
        """Structured legal options for the UI (derived from the same masks the policy uses)."""
        b = self.batch
        verbs = [Verb(i).name for i in range(len(Verb)) if b.verb_mask[0][i]]
        slots = {Verb(v).name: [int(s) for s in np.flatnonzero(b.slot_mask[0][v])] for v in SLOT_VERBS}
        tiers = {Verb(v).name: [int(t) for t in np.flatnonzero(b.tier_mask[0][v])] for v in (Verb.START, Verb.RETRY)}
        cands = [self.env.candidates(0)[k] for k in np.flatnonzero(b.cand_mask[0])]
        return {"verbs": verbs, "slots": slots, "tiers": tiers, "candidates": cands}

    def cost_of(self, a: Action) -> float:
        return spend_cost(self.world, self.firm, a)

    def warnings(self) -> list[dict]:
        return [l.to_dict() for l in review_state(self.world, self.firm)]

    def preview(self, a: Action) -> list[dict]:
        ok, reason = check_action(self.world, self.firm, a)
        if not ok:
            return [{"code": "illegal", "principle": "illegal", "severity": "critical", "message": reason}]
        return [l.to_dict() for l in review_decision(self.world, self.firm, a)]

    # ------------------------------------------------------------ coach
    def coach(self) -> dict | None:
        if self.coach_runner is None or not self.firm.alive:
            return None
        r = self.coach_runner.act(self.batch, [0])[0]
        a = decode(r["action"], self.env.candidates(0))
        probs = {Verb(i).name: round(p, 3) for i, p in enumerate(r["verb_probs"]) if p > 0.02}
        return {"action": a, "label": self.describe(a), "value_estimate": r["value"], "verb_probs": probs}

    def describe(self, a: Action) -> str:
        w = self.world
        v = a.verb
        if v == Verb.NOOP:
            return "Hold — no new commitments this turn"
        if v == Verb.START:
            o = w.opps[a.opportunity_id]
            return (f"Start use case #{a.opportunity_id} ({self.cfg.demand.sectors[o.sector].name}) with "
                    f"{w.models[a.model_id].name} model, {['lean', 'standard', 'premium'][a.tier]} budget")
        p = self.firm.projects[a.slot] if 0 <= a.slot < len(self.firm.projects) else None
        tgt = f"use case #{p.opportunity_id}" if p else f"slot {a.slot}"
        if v == Verb.RETRY:
            return f"Retry {tgt} with {w.models[a.model_id].name}, {['lean', 'standard', 'premium'][a.tier]} budget"
        return f"{v.name.title()} {tgt}"

    def what_if(self, a: Action, horizon: int = 12, rollouts: int = 6) -> dict | None:
        """Compare the team's move with the coach's move: same futures (common random numbers)."""
        c = self.coach()
        if c is None:
            return None
        res = {}
        for label, act in (("yours", a), ("coach", c["action"])):
            vals = []
            for i in range(rollouts):
                env = copy.deepcopy(self.env)
                env.world.rng.__init__(self.seed * 1000 + self.world.month * 31 + i)
                runner = copy.copy(self.coach_runner)
                vals.append(self._rollout(env, act, horizon, runner))
            res[label] = {"mean": float(np.mean(vals)), "p10": float(np.percentile(vals, 10)),
                          "p90": float(np.percentile(vals, 90))}
        res["coach_label"] = c["label"]
        res["horizon"] = horizon
        return res

    def _rollout(self, env, first: Action, horizon: int, runner) -> float:
        fid = self.fid
        start = env.world.net_worth(env.world.firms[fid])
        batch = env.observe()
        act = first
        for _ in range(horizon):
            seats = [s for s in range(env.S) if batch.occupied[s] and batch.controller[s] == "policy"]
            acts = {s: r["action"] for s, r in runner.act(batch, seats).items()} if seats else {}
            s0 = env.seat_of(fid)
            if s0 >= 0:
                acts[s0] = act if act is not None else runner.act(batch, [s0])[s0]["action"]
            out = env.step(acts)
            if out["episode_done"] or not env.world.firms[fid].alive:
                break
            batch = env.observe()
            act = None
        return (env.world.net_worth(env.world.firms[fid]) - start) / 1000.0

    # ------------------------------------------------------------ turn
    def submit(self, a: Action) -> dict:
        """Apply the team's action, then autopilot ('momentum': only pilot/launch ready projects) for the rest
        of the turn. Returns a summary of what happened."""
        if self.finished:
            return {"finished": True}
        f = self.firm
        lessons = review_decision(self.world, f, a) if check_action(self.world, f, a)[0] else []
        coach = self.coach()
        agreed = None
        if coach is not None:
            agreed = coach["action"].verb == a.verb and (
                a.verb != Verb.START or coach["action"].opportunity_id == a.opportunity_id)
            self.agree += int(bool(agreed))
        self.decisions += 1
        for l in lessons:
            (self.good_counts if l.severity == "good" else self.lesson_counts)[l.principle] += 1
        nw0 = self.world.net_worth(f)
        events = []
        for k in range(self.months_per_turn):
            if self.finished:
                break
            act = a if k == 0 else self._autopilot()
            events += self._step_month(act, coach if k == 0 else None, lessons if k == 0 else [])
        turn = {"month": self.world.month, "action": a.to_dict(), "label": self.describe(a),
                "lessons": [l.to_dict() for l in lessons], "coach": coach["label"] if coach else None,
                "coach_agreed": agreed, "net_worth_change": (self.world.net_worth(self.firm) - nw0),
                "events": [e for e in events if e.get("yours") or e["kind"] in ("BANKRUPTCY", "TECH_BREAKTHROUGH",
                                                                                  "MODEL_RELEASE", "FAB_SHORTAGE",
                                                                                  "FIRM_ENTRY")]}
        self.turns.append(turn)
        return turn

    def _autopilot(self) -> Action:
        for i, p in enumerate(self.firm.projects):
            if p.stage == Stage.PILOT_READY:
                return Action(Verb.LAUNCH, slot=i)
        for i, p in enumerate(self.firm.projects):
            if p.stage == Stage.PROTOTYPE_READY:
                a = Action(Verb.PILOT, slot=i)
                if check_action(self.world, self.firm, a)[0]:
                    return a
        return Action()

    def _step_month(self, a: Action, coach: dict | None, lessons) -> list[dict]:
        env, b = self.env, self.batch
        w = env.world
        seats = [s for s in range(env.S) if b.occupied[s] and b.controller[s] == "policy"]
        rival = self.rival_runner.act(b, seats) if (self.rival_runner and seats) else {}
        acts = {s: r["action"] for s, r in rival.items()}
        decisions = []
        if self.firm.alive and env.seat_of(self.fid) >= 0:
            acts[env.seat_of(self.fid)] = a
            decisions.append(decision_record(w, self.firm, env.candidates(0), a,
                                             value=coach["value_estimate"] if coach else None,
                                             controller="human",
                                             notes=[l.to_dict() for l in lessons] +
                                                   ([{"coach": coach["label"]}] if coach else [])))
        for s, r in rival.items():
            fid = int(b.firm_ids[s])
            decisions.append(decision_record(w, w.firms[fid], env.candidates(s), decode(r["action"], env.candidates(s)),
                                             value=r["value"], verb_probs=r["verb_probs"], controller="policy"))
        out = env.step(acts)
        self.rec.month(w, decisions, out["result"].events)
        if out["episode_done"] or not self.firm.alive:
            self._finish()
        else:
            self.batch = env.observe()
        lo = w.month
        return [{"month": e.month, "kind": e.kind.value, "yours": e.payload.get("firm") == self.fid,
                 **{k: v for k, v in e.payload.items() if not k.startswith("_")}}
                for e in out["result"].events if e.month >= lo]

    # ------------------------------------------------------------ end
    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.rec.final(self.world, {"team": self.team})

    def ghost(self) -> dict | None:
        """Same seed, same rivals, coach plays the team's seat for the whole game."""
        if self.net is None:
            return None
        from agents.runner import PolicyRunner
        env = self._make_env({**self.fixed, 0: "policy"})
        b = env.reset(self.seed)
        fid = int(env.seat_firm[0])
        self._set_profile(env.world, fid)
        runner = PolicyRunner(self.net, seed=self.seed)
        coach = PolicyRunner(self.net, deterministic=True, seed=self.seed)
        while True:
            seats = [s for s in range(env.S) if b.occupied[s] and b.controller[s] == "policy" and s != 0]
            acts = {s: r["action"] for s, r in runner.act(b, seats).items()} if seats else {}
            if b.occupied[0] and env.seat_firm[0] == fid:
                acts[0] = coach.act(b, [0])[0]["action"]
            out = env.step(acts)
            if out["episode_done"] or not env.world.firms[fid].alive:
                break
            b = env.observe()
        g = env.world.firms[fid]
        return {"value_created": g.cum_raw_reward, "survived": g.alive, "launches": g.launches,
                "net_worth": env.world.net_worth(g), "tech_failures": g.tech_failures,
                "commercial_failures": g.commercial_failures}

    def debrief(self) -> dict:
        if self.debrief_cache:
            return self.debrief_cache
        w, f = self.world, self.firm
        ranked = sorted(w.firms.values(), key=lambda g: -g.cum_raw_reward)
        ghost = self.ghost()
        top_issues = [{"principle": PRINCIPLES.get(k, k), "count": c} for k, c in self.lesson_counts.most_common(5)]
        strengths = [{"principle": PRINCIPLES.get(k, k), "count": c} for k, c in self.good_counts.most_common(3)]
        best_turn = max(self.turns, key=lambda t: t["net_worth_change"], default=None)
        worst_turn = min(self.turns, key=lambda t: t["net_worth_change"], default=None)
        d = {"team": self.team, "mission": self.mission_id, "seed": self.seed, "rivals": self.rivals,
             "value_created": f.cum_raw_reward, "risk_adjusted": f.cum_shaped_reward, "survived": f.alive,
             "net_worth": w.net_worth(f), "launches": f.launches, "tech_failures": f.tech_failures,
             "commercial_failures": f.commercial_failures, "deferrals": f.deferrals,
             "rank": ranked.index(f) + 1, "n_firms": len(ranked),
             "coach_agreement": self.agree / self.decisions if self.decisions and self.net is not None else None,
             "ghost": ghost, "coach_gap": (ghost["value_created"] - f.cum_raw_reward) if ghost else None,
             "top_issues": top_issues, "strengths": strengths, "best_turn": best_turn, "worst_turn": worst_turn,
             "replay": str(self.replay_path), "months": w.month}
        self.debrief_cache = d
        self._leaderboard(d)
        return d

    def _leaderboard(self, d: dict) -> None:
        p = Path("runs/leaderboard.json")
        rows = json.loads(p.read_text()) if p.exists() else []
        rows.append({k: d[k] for k in ("team", "mission", "seed", "rivals", "value_created", "risk_adjusted",
                                       "survived", "rank", "n_firms", "coach_gap", "coach_agreement", "months")}
                    | {"date": time.strftime("%Y-%m-%d %H:%M")})
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, default=float))


def index_action_for(session: GameSession, a: Action):
    return encode(a, session.env.candidates(0), session.firm.reserve_level)
