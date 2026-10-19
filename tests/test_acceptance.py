"""Validation and acceptance tests (spec §15)."""
import numpy as np
import pytest

from agents.heuristics import HEURISTICS
from agents.masks import build_masks, sample_from_masks
from agents.observation import build_observation, layout
from economy.accounting import Ledger, LedgerError
from economy.actions import Action, Verb
from economy.events import EventType
from economy.world import World
from envs.multi_env import MultiFirmEnv
from tests.conftest import cfg_with, run_world


def test_determinism_same_seed_same_hash(base_cfg):
    a = run_world(base_cfg, seed=7)
    b = run_world(base_cfg, seed=7)
    assert a.state_hash() == b.state_hash()
    c = run_world(base_cfg, seed=8)
    assert a.state_hash() != c.state_hash()


def test_no_demand_no_revenue_and_exit():
    sectors = [dict(s) for s in cfg_with().model_dump()["demand"]["sectors"]]
    for s in sectors:
        s["monthly_budget_mean"] = 0.0
    cfg = cfg_with(demand={"sectors": sectors}, builders={"entry_base_prob": 0.0, "entry_profit_sensitivity": 0.0})
    w = run_world(cfg, seed=3)
    revenue = sum(r["builder_revenue"] for r in w.price_history)
    assert revenue == 0.0
    assert all(not f.alive for f in w.firms.values()), "finite cash must burn down to exit"


def test_unlimited_compute_removes_shortage_but_not_tech_failure():
    cfg = cfg_with(ablation={"supply_bottlenecks": False})
    fails = 0
    for seed in range(1, 6):
        w = run_world(cfg, seed=seed, policy="easy_first")
        assert all(r["served_frac"] == 1.0 for r in w.price_history)
        fails += sum(r["tech_fail"] for r in w.price_history)
    assert fails > 0


def test_zero_innovation_success_means_no_adoption():
    cfg = cfg_with(technology={"p_min": 0.0, "p_max": 0.0})
    w = run_world(cfg, seed=2)
    assert sum(r["adoptions"] for r in w.price_history) == 0
    assert (w.ent.provider < 0).all()
    assert all(f.launches == 0 for f in w.firms.values())


def test_high_benefit_adequate_capacity_some_adoption():
    cfg = cfg_with(opportunities={"value_lognormal": [3.4, 0.3]},
                   datacenters={"initial_capacity": [300000.0, 300000.0]})
    total = sum(sum(r["adoptions"] for r in run_world(cfg, seed=s).price_history) for s in (1, 2))
    assert total > 0


def test_scarcity_raises_utilisation_and_price():
    cfg = cfg_with(datacenters={"initial_capacity": [1500.0, 1000.0], "order_trigger_utilization": 2.0},
                   opportunities={"value_lognormal": [3.2, 0.3]})
    w = run_world(cfg, seed=4)
    util = max(r["utilization"] for r in w.price_history)
    assert util > 0.9
    assert min(r["served_frac"] for r in w.price_history) < 1.0
    assert max(r["compute_price"] for r in w.price_history) > cfg.compute_price.initial


def test_delayed_capex_respects_lead_time():
    lead = 12
    cfg = cfg_with(fab={"wafer_capacity": 15.0, "capex_lead_time": lead},
                   datacenters={"initial_capacity": [3000.0, 2000.0]},
                   opportunities={"value_lognormal": [3.2, 0.3]})
    w = World(cfg, 5)
    rng = np.random.default_rng(5)
    invest_month, online_month = None, None
    cap0 = w.fab.wafer_capacity
    for _ in range(cfg.episode.months):
        w.step({f.firm_id: HEURISTICS["best_npv"](w, f, rng) for f in w.alive_firms()})
        if invest_month is None and w.fab.pending_capacity > 0:
            invest_month = w.month
        if online_month is None and w.fab.wafer_capacity > cap0 + 1e-9:
            online_month = w.month
            break
    assert invest_month is not None, "scenario should trigger fab investment"
    if online_month is not None:
        assert online_month - invest_month >= lead


def test_changing_agents_preserve_shapes():
    cfg = cfg_with(builders={"entry_start_month": 2, "entry_base_prob": 0.5, "entry_max_prob": 0.9})
    env = MultiFirmEnv(cfg, {"policy": 0.5, "heuristic:random": 0.5}, min_policy_seats=1)
    b = env.reset(11)
    D = layout(cfg).total
    rng = np.random.default_rng(0)
    births = deaths = 0
    for _ in range(cfg.episode.months):
        assert b.obs.shape == (env.S, D)
        acts = {}
        for s in range(env.S):
            if b.occupied[s] and b.controller[s] == "policy":
                f = env.world.firms[int(b.firm_ids[s])]
                cand, masks = build_masks(env.world, f)
                acts[s] = sample_from_masks(env.world, f, cand, masks, rng)
        out = env.step(acts)
        births += len(out["result"].born)
        deaths += len(out["result"].terminated)
        if out["episode_done"]:
            break
        b = env.observe()
    assert births > 0


def test_masked_random_actions_are_always_legal(base_cfg):
    w = World(base_cfg, 9)
    rng = np.random.default_rng(9)
    for _ in range(base_cfg.episode.months):
        acts = {}
        for f in w.alive_firms():
            cand, masks = build_masks(w, f)
            acts[f.firm_id] = sample_from_masks(w, f, cand, masks, rng)
        w.step(acts)
    assert sum(f.illegal_actions for f in w.firms.values()) == 0


def test_illegal_action_is_noop_with_diagnostic(base_cfg):
    w = World(base_cfg, 1)
    fid = next(iter(w.firms))
    r = w.step({fid: Action(Verb.LAUNCH, slot=2)})
    assert w.firms[fid].illegal_actions == 1
    assert fid in r.diagnostics
    assert any(e.kind == EventType.ILLEGAL_ACTION for e in r.events)
    assert r.shaped_rewards[fid] < r.raw_rewards[fid]


def test_ledger_conserves_and_blocks_overdraft():
    L = Ledger()
    L.transfer("ext:investors", "firm:0", 100.0, "seed")
    L.transfer("firm:0", "ext:labor", 40.0, "wage")
    L.check_invariants()
    assert abs(L.total()) < 1e-9
    with pytest.raises(LedgerError):
        L.transfer("firm:0", "ext:labor", 100.0, "wage")


def test_ledger_invariants_hold_through_episode(base_cfg):
    w = run_world(base_cfg, seed=12, policy="random")
    w.ledger.check_invariants()
    for f in w.firms.values():
        assert w.cash(f) >= -1e-9


def test_hidden_state_not_in_observation(base_cfg):
    w = World(base_cfg, 3)
    w.step({})
    f = next(iter(w.firms.values()))
    cand, _ = build_masks(w, f)
    o1 = build_observation(w, f, cand)
    w.ent.benefit *= 3.0            # hidden enterprise-specific value
    w.gate_audit.append({"p": 0.99})
    o2 = build_observation(w, f, cand)
    assert np.array_equal(o1, o2)


def test_observation_uses_lagged_market(base_cfg):
    w = World(base_cfg, 3)
    w.step({})
    f = next(iter(w.firms.values()))
    cand, _ = build_masks(w, f)
    before = build_observation(w, f, cand)
    w.compute_price *= 2.0          # current (unpublished) price change must not leak
    after = build_observation(w, f, cand)
    assert np.array_equal(before, after)


def test_bankrupt_firm_keeps_no_project_value(base_cfg):
    """A bankrupt firm's final value is −debt: unfinished projects and products count for nothing."""
    w = World(base_cfg, 5000)
    rng = np.random.default_rng(5000)
    seen = 0
    while seen == 0:
        r = w.step({f.firm_id: HEURISTICS["random"](w, f, rng) for f in w.alive_firms()})
        for fid in r.terminated:
            f = w.firms[fid]
            assert w.net_worth(f) == pytest.approx(-f.debt)
            assert f.cum_raw_reward == pytest.approx((-f.debt - base_cfg.builders.initial_cash) / 1000.0)
            seen += 1
        assert not r.done, "expected a random-policy firm to go bankrupt"
