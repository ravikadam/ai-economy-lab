"""Episode replay logs (JSONL), shared by the headless simulator and the human game.

Line types:
  header  — scenario config, seed, controllers, schema version
  month   — lagged public market record, every firm's summary, each decision
            (observation summary, selected action, estimated value, verb
            probabilities, coach notes) and the month's events
  final   — final firm summaries, state hash
"""
from __future__ import annotations

import json
from pathlib import Path

from economy.config import SCHEMA_VERSION
from economy.events import MarketEvent


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


class ReplayRecorder:
    def __init__(self, path: str | Path, cfg, seed: int, controllers: dict, extra: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        self._write({"type": "header", "schema_version": SCHEMA_VERSION, "scenario": cfg.model_dump(),
                     "seed": seed, "controllers": controllers, **(extra or {})})

    def _write(self, rec: dict) -> None:
        self.f.write(json.dumps(_clean(rec), default=str) + "\n")
        self.f.flush()

    def month(self, world, decisions: list[dict], events: list[MarketEvent]) -> None:
        self._write({"type": "month", "month": world.month, "market": world.price_history[-1],
                     "firms": [world.firm_summary(f) for f in world.firms.values()
                               if f.alive or f.death_month == world.month],
                     "decisions": decisions,
                     "events": [e.to_dict() for e in events]})

    def final(self, world, extra: dict | None = None) -> None:
        self._write({"type": "final", "month": world.month, "state_hash": world.state_hash(),
                     "firms": [world.firm_summary(f) for f in world.firms.values()], **(extra or {})})
        self.f.close()


def load_replay(path: str | Path) -> dict:
    out = {"header": None, "months": [], "final": None}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            t = rec.get("type")
            if t == "header":
                out["header"] = rec
            elif t == "month":
                out["months"].append(rec)
            elif t == "final":
                out["final"] = rec
    return out


def decision_record(world, firm, cand: list[int], action, value: float | None = None,
                    verb_probs: list[float] | None = None, controller: str = "", notes: list | None = None) -> dict:
    """What a firm saw (summary), what it did, what it expected."""
    return {"firm": firm.firm_id, "controller": controller, "action": action.to_dict(),
            "value_estimate": value, "verb_probs": verb_probs,
            "observation": {"cash": round(world.cash(firm), 1), "debt": round(firm.debt, 1),
                            "runway": round(world.runway(firm), 1), "candidates": cand,
                            "projects": [(p.opportunity_id, p.stage.name) for p in firm.projects]},
            "notes": notes or []}
