"""Autonomous live run for the viewer: every firm decides on its own.

RL firms are driven by the trained policy; rival firms by rule-based heuristics.
Nobody needs to play. Each `step()` advances one month and records, per firm,
what it decided, how confident the policy was, and what happened.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from agents.actions import decode
from agents.heuristics import HEURISTICS
from analysis.replay import ReplayRecorder, decision_record
from economy.actions import Action, Verb
from economy.config import load_config
from envs.multi_env import MultiFirmEnv

TIER = ["lean", "standard", "premium"]
RIVAL_ORDER = ["heuristic:best_npv", "heuristic:easy_first", "heuristic:cash_preserving", "heuristic:best_npv"]


def who(ctrl: str) -> str:
    if ctrl == "policy":
        return "RL agent"
    return "Rule: " + ctrl.split(":", 1)[-1].replace("_", " ")


def describe(world, firm, a: Action) -> str:
    v = a.verb
    if v == Verb.NOOP:
        return "Hold"
    if v == Verb.START:
        o = world.opps[a.opportunity_id]
        sec = world.cfg.demand.sectors[o.sector].name.replace("_", " ")
        return f"Start use case #{o.opportunity_id} ({sec}) · {world.models[a.model_id].name} · {TIER[a.tier]}"
    p = firm.projects[a.slot] if 0 <= a.slot < len(firm.projects) else None
    tgt = f"use case #{p.opportunity_id}" if p else f"slot {a.slot}"
    if v == Verb.RETRY:
        return f"Retry {tgt} · {world.models[a.model_id].name} · {TIER[a.tier]}"
    return f"{v.name.title()} {tgt}"


class LiveRun:
    def __init__(self, scenario: str = "config/base.yaml", seed: int = 42, rl_firms: int = 3,
                 months: int = 120, checkpoint: str = "runs/latest", replay_dir: str = "runs/replays"):
        self.cfg = load_config(scenario, overrides={"episode": {"months": months}})
        self.scenario, self.seed = scenario, seed
        from agents.runner import PolicyRunner
        from training.checkpoints import load_checkpoint
        net, _, _ = load_checkpoint(checkpoint)
        self.runner = PolicyRunner(net, deterministic=False, seed=seed)
        n0 = self.cfg.builders.initial_firms
        fixed = {s: ("policy" if s < rl_firms else RIVAL_ORDER[(s - rl_firms) % len(RIVAL_ORDER)]) for s in range(n0)}
        self.env = MultiFirmEnv(self.cfg, {"policy": 0.5, "heuristic:best_npv": 0.5}, fixed_controllers=fixed,
                                min_policy_seats=0)
        self.batch = self.env.reset(seed)
        self.rng = np.random.default_rng(seed)
        self.done = False
        self.log: list[dict] = []          # every decision, newest last
        self.news: list[dict] = []         # public events + gate results
        self.last: dict[int, dict] = {}    # firm_id -> latest decision record
        self.value_hist: list[dict] = []   # month, firm, value created
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.replay_path = Path(replay_dir) / f"live-{Path(scenario).stem}-{seed}-{stamp}.jsonl"
        self.rec = ReplayRecorder(self.replay_path, self.cfg, seed, fixed, {"mode": "live", "rl_firms": rl_firms})
        self._snapshot_values()

    @property
    def world(self):
        return self.env.world

    @property
    def month(self) -> int:
        return self.world.month

    def step(self) -> None:
        if self.done:
            return
        env, b, w = self.env, self.batch, self.world
        seats = [s for s in range(env.S) if b.occupied[s]]
        rl = [s for s in seats if b.controller[s] == "policy"]
        pol = self.runner.act(b, rl) if rl else {}
        actions, decisions = {}, []
        for s in seats:
            f = w.firms[int(b.firm_ids[s])]
            ctrl = b.controller[s]
            if s in pol:
                a = decode(pol[s]["action"], env.candidates(s))
                conf = float(max(pol[s]["verb_probs"]))
                value = pol[s]["value"]
            else:
                a = HEURISTICS[ctrl.split(":", 1)[1]](w, f, self.rng)
                conf, value = None, None
            actions[s] = a
            rec = {"month": w.month + 1, "firm": f.firm_id, "who": who(ctrl), "decision": describe(w, f, a),
                   "verb": a.verb.name, "ai_confidence": conf, "ai_value": value}
            self.last[f.firm_id] = rec
            if a.verb != Verb.NOOP:
                self.log.append(rec)
            decisions.append(decision_record(w, f, env.candidates(s), a, value,
                                             pol[s]["verb_probs"] if s in pol else None, ctrl))
        out = env.step(actions)
        res = out["result"]
        for e in res.events:
            p = {k: v for k, v in e.payload.items() if not k.startswith("_")}
            keep = e.payload.get("_public") or e.kind.value in ("PROJECT_GATE_RESULT", "BANKRUPTCY", "FIRM_ENTRY")
            if keep and e.kind.value not in ("PRICE_UPDATE", "CAPACITY_ONLINE"):
                fid = p.get("firm")
                tag = who(w.firms[fid].controller) if fid is not None and fid in w.firms else ""
                self.news.append({"month": e.month, "event": e.kind.value.replace("_", " ").title(),
                                  "firm": fid, "who": tag, "detail": _detail(e.kind.value, p)})
        self.rec.month(w, decisions, res.events)
        self._snapshot_values()
        if out["episode_done"]:
            self.done = True
            self.rec.final(w)
        else:
            self.batch = env.observe()

    def _snapshot_values(self) -> None:
        w = self.world
        for f in w.firms.values():
            if f.alive or f.death_month == w.month:
                self.value_hist.append({"month": w.month, "firm": f"{f.firm_id} · {who(f.controller)}",
                                        "rl": f.controller == "policy", "value": round(f.cum_raw_reward, 3)})

    def leaderboard(self) -> list[dict]:
        w = self.world
        rows = []
        for f in w.firms.values():
            live = [p for p in f.projects if p.is_live]
            rows.append({"firm": f.firm_id, "who": who(f.controller), "risk profile": f.profile.name,
                         "status": "active" if f.alive else f"bankrupt (m{f.death_month})",
                         "value created $M": round(f.cum_raw_reward, 2), "cash $k": round(w.cash(f)),
                         "customers": sum(p.customers for p in f.projects), "live products": len(live),
                         "projects": ", ".join(f"#{p.opportunity_id} {p.stage.name.lower()}" for p in f.projects) or "—",
                         "last decision": self.last.get(f.firm_id, {}).get("decision", "—")})
        rows.sort(key=lambda r: -r["value created $M"])
        return rows

    def group_scores(self) -> dict:
        w = self.world
        out = {}
        for key, test in (("RL agents", lambda c: c == "policy"), ("Rule-based", lambda c: c != "policy")):
            fs = [f for f in w.firms.values() if test(f.controller)]
            out[key] = {"firms": len(fs), "alive": sum(f.alive for f in fs),
                        "avg_value": float(np.mean([f.cum_raw_reward for f in fs])) if fs else 0.0}
        return out


def _detail(kind: str, p: dict) -> str:
    if kind == "PROJECT_GATE_RESULT":
        g = p.get("gate")
        if g == "launch":
            return f"launched use case #{p.get('opportunity')} ({p.get('converted', 0)} pilot customers converted)"
        if g == "pilot_recruit":
            return f"pilot for #{p.get('opportunity')} deferred: no customer willing"
        res = p.get("result")
        extra = ""
        if g == "commercial" and res == "fail":
            extra = " (reliability too low)" if not p.get("reliability_ok", True) else " (value below cost)"
        return f"{g} gate {res} on use case #{p.get('opportunity')}{extra}"
    if kind in ("MODEL_RELEASE", "TECH_BREAKTHROUGH"):
        return f"{p.get('model')} model: capability {p.get('capability')}, reliability {p.get('reliability')}"
    if kind == "FAB_SHORTAGE":
        return f"chip fab loses {round(100 * p.get('severity', 0))}% capacity for {p.get('months')} months"
    if kind == "FIRM_ENTRY":
        return f"new firm enters ({p.get('profile')}, ${p.get('cash')}k)"
    if kind == "BANKRUPTCY":
        return f"firm goes bankrupt (debt ${p.get('debt')}k, age {p.get('age')} months)"
    if kind == "OPPORTUNITY_DISCOVERED":
        return f"new use case #{p.get('opportunity')} discovered"
    return ", ".join(f"{k}={v}" for k, v in p.items())
