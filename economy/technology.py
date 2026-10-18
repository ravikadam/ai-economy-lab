"""Model capability progress and technical success probability (spec §5)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from economy.events import EventType

if TYPE_CHECKING:
    from economy.world import World
    from entities.opportunities import Opportunity


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def success_probability(world: "World", opp: "Opportunity", model_id: int, skill: float, effort: float) -> float:
    """sigmoid(α(cap − d) + β·skill + γ·effort − δ·burden + bias), clamped."""
    t = world.cfg.technology
    cap = world.models[model_id].capability[opp.sector]
    x = t.alpha * (cap - opp.latent_difficulty) + t.beta * skill + t.gamma * effort \
        - t.delta * opp.integration_burden + t.bias
    if cap < opp.capability_threshold:
        x -= 3.0
    return min(t.p_max, max(t.p_min, sigmoid(x)))


def schedule_initial_releases(world: "World") -> None:
    lo, hi = world.cfg.technology.release_interval
    for m in world.models:
        m.next_release = int(world.rng["tech"].integers(lo, hi + 1))


def advance_models(world: "World") -> None:
    t = world.cfg.technology
    rng = world.rng["tech"]
    frontier = max(world.models, key=lambda m: sum(m.capability))
    for m in world.models:
        if world.month < m.next_release:
            continue
        step = float(rng.uniform(*t.capability_step))
        breakthrough = rng.random() < t.breakthrough_prob
        if breakthrough:
            step *= 2.5
        old = list(m.capability)
        for s in range(len(m.capability)):
            noise = float(rng.uniform(0.6, 1.4))
            inc = step * noise
            if m.open_weight and m is not frontier:
                gap = max(0.0, frontier.capability[s] - m.capability[s])
                inc = max(inc * 0.6, t.open_catchup * gap)
            m.capability[s] = min(1.5, m.capability[s] + inc)
        m.reliability = min(0.99, m.reliability + t.reliability_step * (2.0 if breakthrough else 1.0))
        m.releases += 1
        lo, hi = t.release_interval
        m.next_release = world.month + int(rng.integers(lo, hi + 1))
        payload = {"model": m.name, "capability": [round(c, 3) for c in m.capability],
                   "previous": [round(c, 3) for c in old], "reliability": round(m.reliability, 3)}
        world.emit(EventType.TECH_BREAKTHROUGH if breakthrough else EventType.MODEL_RELEASE, payload)


def best_capability(world: "World", sector: int) -> float:
    return max(m.capability[sector] for m in world.models)
