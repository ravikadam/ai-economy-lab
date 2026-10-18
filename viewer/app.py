"""AI Economy Lab — workshop app.

  streamlit run viewer/app.py

Pages: Play (implementation-team game with AI coach) · Replay & market dashboard ·
Training & evaluation · Leaderboard · How the game works.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

os.chdir(ROOT)

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from analysis.metrics import decision_frame, event_frame, firm_frame, market_frame, training_frame  # noqa: E402
from analysis.replay import load_replay  # noqa: E402
from economy.actions import Action, Verb  # noqa: E402
from entities.firms import RESERVE_MONTHS  # noqa: E402
from game.coach import PRINCIPLES  # noqa: E402
from game.missions import MISSIONS  # noqa: E402

st.set_page_config(page_title="AI Economy Lab", page_icon="🧭", layout="wide")

VERB_LABEL = {
    "NOOP": "Hold (no new commitment)", "START": "Start a use case", "PILOT": "Run a paid pilot",
    "LAUNCH": "Launch to market", "SCALE": "Scale sales & support", "ABANDON": "Abandon a project",
    "RETRY": "Retry a failed project", "PAUSE": "Pause / resume development",
}
TIER = ["lean", "standard", "premium"]
SEV_ICON = {"critical": "🛑", "warn": "⚠️", "info": "ℹ️", "good": "✅"}


def money(k: float) -> str:
    return f"${k / 1000:,.2f}M" if abs(k) >= 1000 else f"${k:,.0f}k"


def checkpoint_available(path="runs/latest") -> bool:
    p = Path(path)
    if not p.exists() and Path("results/policy/best.pt").exists():
        return True        # fresh clone: the bundled trained policy is used
    return p.exists() and (any(p.glob("*.pt")) or any((p / "checkpoints").glob("*.pt")) if p.is_dir() else True)


# ======================================================================= PLAY
def page_play():
    g = st.session_state.get("game")
    if g is None:
        return play_setup()
    if g.finished:
        return play_debrief(g)
    play_turn(g)


def play_setup():
    st.title("🎮 Implementation Team Challenge")
    st.caption("Your team runs an AI implementation firm. Pick use cases, choose models and budgets, run pilots "
               "with enterprise customers, launch, scale and survive. Rival firms are rule-based or trained RL agents. "
               "An AI coach (the trained policy) critiques every decision.")
    has_ck = checkpoint_available()
    if not has_ck:
        st.info("No trained checkpoint found at `runs/latest`: rivals will be rule-based and the coach is off. "
                "Train one with `python -m training.train --config config/train.yaml`.")
    mcol, scol = st.columns([3, 2])
    with mcol:
        titles = {m["title"]: m for m in MISSIONS}
        m = titles[st.radio("Mission", list(titles), index=0)]
        st.markdown(f"**Brief.** {m['brief']}")
        st.markdown("**You will practise:** " + " · ".join(PRINCIPLES[o] for o in m["objectives"]))
    with scol:
        team = st.text_input("Team name", value="Team Alpha")
        rivals = st.selectbox("Rivals", ["heuristics", "rl", "mixed"], index=["heuristics", "rl", "mixed"].index(
            m["rivals"]) if has_ck else 0, help="rl/mixed need a trained checkpoint")
        profile = st.select_slider("Your risk appetite", ["cautious", "balanced", "aggressive"], value="balanced",
                                   help="Sets runway target and how the risk-adjusted score treats losses")
        mpt = st.radio("Turn length", [3, 1], format_func=lambda x: "Quarterly (3 months)" if x == 3 else "Monthly",
                       horizontal=True, help="Quarterly: your decision applies in month 1; ready projects are "
                                             "auto-piloted/launched in months 2–3")
        seed = st.number_input("World seed (same seed ⇒ same market for every team)", value=42, step=1)
        if st.button("Start mission ▶", type="primary", use_container_width=True):
            from game.session import GameSession
            with st.spinner("Building the economy…"):
                st.session_state.game = GameSession(team=team, scenario=m["scenario"], seed=int(seed), rivals=rivals,
                                                    checkpoint="runs/latest" if has_ck else None, months=m["months"],
                                                    months_per_turn=mpt, profile=profile, mission_id=m["id"])
                st.session_state.coach = None
                st.session_state.whatif = None
                st.session_state.last_turn = None
            st.rerun()


def play_turn(g):
    k = g.kpis()
    m = next((x for x in MISSIONS if x["id"] == g.mission_id), None)
    st.markdown(f"### {m['title'] if m else 'Custom game'} — {g.team}")
    st.progress(k["month"] / k["months_total"], text=f"Month {k['month']} of {k['months_total']}")
    c = st.columns(6)
    c[0].metric("Cash", money(k["cash"]), help=f"debt {money(k['debt'])}")
    c[1].metric("Runway", f"{k['runway']:.1f} mo", help=f"target {g.firm.profile.runway_target:.0f} months")
    c[2].metric("Net worth", money(k["net_worth"]))
    c[3].metric("Customers", k["customers"])
    c[4].metric("Value created", f"{k['value_created']:+.2f} $M")
    c[5].metric("Rank", f"{k['rank']} / {k['n_firms']}" if k["rank"] else "—")

    lt = st.session_state.get("last_turn")
    if lt:
        with st.expander(f"Last turn: {lt['label']} → net worth {money(lt['net_worth_change'])}", expanded=True):
            if lt.get("coach"):
                st.caption(f"Coach would have chosen: {lt['coach']} ({'agreed' if lt['coach_agreed'] else 'different'})")
            for e in lt["events"][:8]:
                st.write(f"• m{e['month']} **{e['kind']}** " +
                         ", ".join(f"{a}={b}" for a, b in e.items() if a not in ("month", "kind", "yours")))
    for w in g.warnings():
        (st.error if w["severity"] == "critical" else st.warning)(f"{SEV_ICON[w['severity']]} {w['message']}")

    left, right = st.columns([2, 1])
    with left:
        t = st.tabs(["Decide", "Portfolio", "Market", "Rivals", "News", "Turn log"])
        with t[0]:
            decide(g)
        with t[1]:
            pr = g.projects()
            if pr:
                st.dataframe(pd.DataFrame(pr), hide_index=True, use_container_width=True)
            else:
                st.caption("No projects yet — start a use case.")
        with t[2]:
            market_charts([r for r in g.world.price_history if r["month"] <= g.world.public["month"]], g.cfg)
        with t[3]:
            st.dataframe(pd.DataFrame(g.rivals_table()), hide_index=True, use_container_width=True)
        with t[4]:
            n = g.news()
            if n:
                st.dataframe(pd.DataFrame(n), hide_index=True, use_container_width=True)
            else:
                st.caption("Quiet market.")
        with t[5]:
            for tr in reversed(g.turns):
                st.write(f"**m{tr['month']}** {tr['label']} → {money(tr['net_worth_change'])}")
                for l in tr["lessons"]:
                    st.caption(f"{SEV_ICON[l['severity']]} {l['message']}")
    with right:
        coach_panel(g)


def decide(g):
    legal = g.legal()
    verbs = legal["verbs"]
    verb = st.selectbox("Decision", verbs, format_func=lambda v: VERB_LABEL[v])
    a = Action(Verb[verb])
    if verb == "START":
        cands = pd.DataFrame(g.candidates())
        st.dataframe(cands, hide_index=True, use_container_width=True, height=260)
        st.caption("p_success = your estimate of technical × commercial success (from noisy signals). "
                   "margin = monthly $k per customer after inference and support.")
        a.opportunity_id = st.selectbox("Use case", legal["candidates"],
                                        format_func=lambda o: f"#{o} · {g.cfg.demand.sectors[g.world.opps[o].sector].name}"
                                                              f" · value {g.world.opps[o].value_signal:.1f}")
        cols = st.columns(2)
        a.model_id = cols[0].radio("Model", list(range(g.cfg.n_models)), horizontal=True,
                                   format_func=lambda m: g.world.models[m].name)
        a.tier = cols[1].radio("Budget tier", legal["tiers"]["START"], horizontal=True, format_func=lambda t: TIER[t])
    elif verb in legal["slots"]:
        rows = {r["slot"]: r for r in g.projects()}
        a.slot = st.selectbox("Project", legal["slots"][verb],
                              format_func=lambda s: f"slot {s}: use case #{rows[s]['use_case']} · {rows[s]['stage']}")
        if verb == "RETRY":
            cols = st.columns(2)
            a.model_id = cols[0].radio("Model", list(range(g.cfg.n_models)), horizontal=True,
                                       format_func=lambda m: g.world.models[m].name)
            a.tier = cols[1].radio("Budget tier", legal["tiers"]["RETRY"], horizontal=True,
                                   format_func=lambda t: TIER[t])
    res = st.select_slider("Cash reserve floor (blocks spending below it from next month)", list(range(3)),
                           value=g.firm.reserve_level, format_func=lambda r: f"{RESERVE_MONTHS[r]:.0f} months burn")
    a.reserve = res
    cost = g.cost_of(a)
    if cost:
        st.caption(f"Up-front cost: {money(cost)}")
    for l in g.preview(a):
        st.markdown(f"{SEV_ICON.get(l['severity'], '•')} {l['message']}")
    b1, b2, b3 = st.columns(3)
    if b1.button("🤖 Ask the coach", use_container_width=True, disabled=g.net is None):
        st.session_state.coach = g.coach()
    if b2.button("🔮 What-if vs coach", use_container_width=True, disabled=g.net is None,
                 help="Replays the next 12 months 6× under your move and the coach's move (same random futures)"):
        with st.spinner("Simulating both futures…"):
            st.session_state.whatif = g.what_if(a)
    if b3.button("Commit decision ▶", type="primary", use_container_width=True):
        with st.spinner("Advancing the market…"):
            st.session_state.last_turn = g.submit(a)
            st.session_state.coach = None
            st.session_state.whatif = None
        st.rerun()


def coach_panel(g):
    st.markdown("#### 🤖 AI coach")
    if g.net is None:
        st.caption("Coach off (no trained checkpoint).")
        return
    c = st.session_state.get("coach")
    if c:
        st.success(c["label"])
        st.caption(f"Coach's value estimate of your position: {c['value_estimate']:+.2f} (normalised units)")
        vp = pd.DataFrame({"decision": [VERB_LABEL[k] for k in c["verb_probs"]], "p": list(c["verb_probs"].values())})
        st.plotly_chart(px.bar(vp, x="p", y="decision", orientation="h", height=220).update_layout(
            margin=dict(l=0, r=0, t=0, b=0), yaxis_title=None, xaxis_title="coach's probability"),
            use_container_width=True)
    else:
        st.caption("Ask the coach for its move before you commit, or play your own call and compare afterwards.")
    wi = st.session_state.get("whatif")
    if wi:
        st.markdown(f"**What-if, next {wi['horizon']} months** (Δ net worth, $M)")
        df = pd.DataFrame([{"move": "Your move", **wi["yours"]}, {"move": f"Coach: {wi['coach_label']}", **wi["coach"]}])
        fig = go.Figure()
        for _, r in df.iterrows():
            fig.add_trace(go.Bar(x=[r["mean"]], y=[r["move"][:40]], orientation="h", name=r["move"][:40],
                                 error_x=dict(type="data", symmetric=False, array=[r["p90"] - r["mean"]],
                                              arrayminus=[r["mean"] - r["p10"]])))
        fig.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bars: mean over 6 simulated futures; whiskers: 10th–90th percentile.")


def play_debrief(g):
    with st.spinner("Running the coach's ghost game on your seed…"):
        d = g.debrief()
    st.title(f"🏁 Debrief — {d['team']}")
    c = st.columns(5)
    c[0].metric("Value created", f"{d['value_created']:+.2f} $M")
    c[1].metric("Survived", "yes" if d["survived"] else "no")
    c[2].metric("Rank", f"{d['rank']} / {d['n_firms']}")
    c[3].metric("Launches", d["launches"])
    if d["coach_agreement"] is not None:
        c[4].metric("Agreement with coach", f"{d['coach_agreement']:.0%}")
    if d["ghost"]:
        gh = d["ghost"]
        st.markdown(f"**Ghost run** — the AI coach in your seat, same seed and rivals: value "
                    f"**{gh['value_created']:+.2f} $M**, {'survived' if gh['survived'] else 'went bankrupt'}, "
                    f"{gh['launches']} launches. Gap to you: **{d['coach_gap']:+.2f} $M**.")
        st.caption("The coach is a trained policy in a stylised simulator — a sparring partner, not ground truth.")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Where to improve")
        for i in d["top_issues"] or [{"principle": "No recurring issues flagged", "count": 0}]:
            st.write(f"• {i['principle']} — flagged {i['count']}×")
    with b:
        st.markdown("#### Strengths")
        for i in d["strengths"] or [{"principle": "—", "count": 0}]:
            st.write(f"• {i['principle']} — {i['count']}×")
    if d["best_turn"]:
        st.write(f"**Best turn:** m{d['best_turn']['month']} {d['best_turn']['label']} "
                 f"({money(d['best_turn']['net_worth_change'])})")
        st.write(f"**Worst turn:** m{d['worst_turn']['month']} {d['worst_turn']['label']} "
                 f"({money(d['worst_turn']['net_worth_change'])})")
    st.caption(f"Replay saved: `{d['replay']}` — open it on the Replay page.")
    if st.button("Play again", type="primary"):
        st.session_state.game = None
        st.session_state.last_turn = None
        st.rerun()


# ================================================================ CHARTS
def market_charts(hist: list[dict], cfg):
    if len(hist) < 2:
        st.caption("Market data appears after the first month (indicators are published with a one-month lag).")
        return
    df = pd.DataFrame(hist)
    sectors = [s.name for s in cfg.demand.sectors]
    models = [m.name for m in cfg.technology.models]
    c1, c2 = st.columns(2)
    conf = pd.DataFrame([dict(month=r["month"], **{s: r["sector_confidence"][i] for i, s in enumerate(sectors)})
                         for r in hist])
    c1.plotly_chart(px.line(conf, x="month", y=sectors, title="Enterprise confidence", height=260)
                    .update_layout(margin=dict(t=40, b=0), legend_title=None), use_container_width=True)
    adop = pd.DataFrame([dict(month=r["month"], **{s: r["sector_adoption"][i] for i, s in enumerate(sectors)})
                         for r in hist])
    c2.plotly_chart(px.area(adop, x="month", y=sectors, title="Adopted use cases", height=260)
                    .update_layout(margin=dict(t=40, b=0), legend_title=None), use_container_width=True)
    c3, c4 = st.columns(2)
    up = pd.DataFrame([dict(month=r["month"], **{m: r["model_unit_price"][j] * 1000 for j, m in enumerate(models)})
                       for r in hist])
    c3.plotly_chart(px.line(up, x="month", y=models, title="Inference price per unit ($)", height=260)
                    .update_layout(margin=dict(t=40, b=0), legend_title=None), use_container_width=True)
    c4.plotly_chart(px.line(df, x="month", y=["utilization", "served_frac"], title="Compute utilisation", height=260)
                    .update_layout(margin=dict(t=40, b=0), legend_title=None), use_container_width=True)


# ================================================================ REPLAY
def page_replay():
    st.title("📼 Replay & market dashboard")
    with st.expander("Run a new headless simulation"):
        scen = st.selectbox("Scenario", ["config/base.yaml"] + sorted(glob.glob("config/scenarios/*.yaml")))
        seed = st.number_input("Seed", value=42, step=1, key="simseed")
        opts = ["best_npv", "easy_first", "cash_preserving", "random"] + (["policy"] if checkpoint_available() else [])
        ctrls = [st.selectbox(f"Seat {i}", opts, index=[0, 1, 2, 0, 3][i] if i < 5 else 0, key=f"seat{i}")
                 for i in range(5)]
        ent = st.selectbox("Entrants", opts, index=0)
        if st.button("Run simulation"):
            from scripts.simulate import run
            with st.spinner("Simulating 120 months…"):
                p = run(scen, int(seed), ctrls, ent, "runs/latest" if checkpoint_available() else None, None, quiet=True)
            st.session_state.replay_pick = str(p)
            st.success(f"Wrote {p}")
    files = sorted(glob.glob("runs/replays/*.jsonl"), key=os.path.getmtime, reverse=True)
    if not files:
        st.info("No replays yet — run a simulation above or play a game.")
        return
    default = files.index(st.session_state["replay_pick"]) if st.session_state.get("replay_pick") in files else 0
    path = st.selectbox("Replay", files, index=default, format_func=lambda p: Path(p).name)
    rep = load_replay(path)
    if not rep["months"]:
        st.warning("Empty replay.")
        return
    h = rep["header"]
    st.caption(f"mode={h.get('mode')} · scenario={h['scenario']['name']} · seed={h['seed']} · "
               f"controllers={h['controllers']}" + (f" · state hash {rep['final']['state_hash'][:12]}" if rep["final"] else ""))
    m, f, ev, dec = market_frame(rep), firm_frame(rep), event_frame(rep), decision_frame(rep)
    tabs = st.tabs(["Market", "Firms", "Decisions", "Events"])
    with tabs[0]:
        sectors = [s["name"] for s in h["scenario"]["demand"]["sectors"]]
        models = [x["name"] for x in h["scenario"]["technology"]["models"]]
        grid = [
            ("Adopted use cases by sector", [f"adoption:{s}" for s in sectors]),
            ("Enterprise confidence", [f"confidence:{s}" for s in sectors]),
            ("Sector revenue ($k/month)", [f"revenue:{s}" for s in sectors]),
            ("Compute utilisation & served fraction", ["utilization", "served_frac"]),
            ("Compute price ($k/unit)", ["compute_price"]),
            ("Fab capacity, under construction, backlog", ["fab_nominal_capacity", "fab_pending", "fab_backlog"]),
            ("Model capability", [f"capability:{mn}:{s}" for mn in models for s in sectors]),
            ("Unit inference price ($k)", [f"unit_price:{mn}" for mn in models]),
            ("Pilots: pass / fail", ["pilot_pass", "pilot_fail"]),
            ("Market concentration (HHI)", ["hhi"]),
            ("Firms alive · entries · exits", ["n_firms", "entries_total", "exits_total"]),
            ("Welfare ($k/month)", ["welfare", "enterprise_benefit", "builder_profit"]),
        ]
        for i in range(0, len(grid), 2):
            cols = st.columns(2)
            for col, (title, ys) in zip(cols, grid[i:i + 2]):
                col.plotly_chart(px.line(m, x="month", y=ys, title=title, height=280)
                                 .update_layout(margin=dict(t=40, b=0), legend_title=None), use_container_width=True)
        comp = pd.DataFrame({lvl: [m[f"technical_pass:{lvl}"].sum(), m[f"technical_fail:{lvl}"].sum(),
                                   m[f"commercial_pass:{lvl}"].sum(), m[f"commercial_fail:{lvl}"].sum()]
                             for lvl in ("low", "mid", "high")},
                            index=["technical pass", "technical fail", "commercial pass", "commercial fail"])
        st.markdown("**Gate outcomes by use-case complexity**")
        st.dataframe(comp, use_container_width=True)
    with tabs[1]:
        f["firm"] = f["firm_id"].astype(str) + " · " + f["controller"].str.replace("heuristic:", "")
        st.plotly_chart(px.line(f, x="month", y="value_created", color="firm", title="Value created ($M)", height=360),
                        use_container_width=True)
        st.plotly_chart(px.line(f, x="month", y="cash", color="firm", title="Cash ($k)", height=320),
                        use_container_width=True)
        month = st.slider("Inspect month", 1, int(m.month.max()), int(m.month.max()))
        snap = next(r for r in rep["months"] if r["month"] == month)
        for fs in snap["firms"]:
            with st.expander(f"Firm {fs['firm_id']} · {fs['controller']} · {fs['profile']} · net worth "
                             f"{money(fs['net_worth'])} · {'alive' if fs['alive'] else 'bankrupt'}"):
                st.dataframe(pd.DataFrame(fs["projects"]), hide_index=True, use_container_width=True)
    with tabs[2]:
        if dec.empty:
            st.caption("No logged decisions (rule-based firms decide inside the simulator).")
        else:
            firm_pick = st.multiselect("Firms", sorted(dec.firm.unique()), default=sorted(dec.firm.unique())[:3])
            st.dataframe(dec[dec.firm.isin(firm_pick)], hide_index=True, use_container_width=True, height=420)
    with tabs[3]:
        kinds = st.multiselect("Event kinds", sorted(ev.kind.unique()), default=[k for k in sorted(ev.kind.unique())
                               if k not in ("ENTERPRISE_ADOPTION", "CHIP_DELIVERY")])
        st.dataframe(ev[ev.kind.isin(kinds)], hide_index=True, use_container_width=True, height=420)


# ============================================================== TRAINING
def page_training():
    st.title("🧠 Training & evaluation")
    runs = sorted([p for p in Path("runs").glob("*/metrics.jsonl")], key=os.path.getmtime, reverse=True)
    bench = Path("runs/benchmark.json")
    if bench.exists():
        b = json.loads(bench.read_text())
        c = st.columns(4)
        c[0].metric("Sim throughput", f"{b.get('vectorised_world_months_per_s', 0):,.0f} world-months/s")
        c[1].metric("100k transitions", f"{b.get('est_seconds_per_100k_world_months', 0):.0f} s")
        c[2].metric("PPO update · MPS", f"{b.get('update_mps_s_per_16384_samples_x4_epochs', float('nan')):.2f} s")
        c[3].metric("PPO update · CPU", f"{b.get('update_cpu_s_per_16384_samples_x4_epochs', float('nan')):.2f} s")
    if not runs:
        st.info("No training runs yet.")
        return
    run = st.selectbox("Run", runs, format_func=lambda p: p.parent.name).parent
    df = training_frame(run)
    if (run / "bc.json").exists():
        bc = json.loads((run / "bc.json").read_text())
        st.caption(f"Behaviour-cloning warm start from `{bc['teacher']}`: {bc['samples']:,} decisions, final verb "
                   f"accuracy {bc['history'][-1]['verb_acc']:.1%}. PPO then fine-tunes on the shaped reward.")
    c1, c2 = st.columns(2)
    d = df.dropna(subset=["policy_raw_return"]) if "policy_raw_return" in df else pd.DataFrame()
    if not d.empty:
        c1.plotly_chart(px.line(d, x="env_steps", y=["policy_raw_return", "heuristic_raw_return"],
                                title="Episode raw return in training worlds ($M)", height=320), use_container_width=True)
        c2.plotly_chart(px.line(d, x="env_steps", y=["policy_survival", "heuristic_survival"],
                                title="Survival rate", height=320), use_container_width=True)
    if "eval_raw_return" in df:
        e = df.dropna(subset=["eval_raw_return"])
        c1.plotly_chart(px.line(e, x="env_steps", y="eval_raw_return", markers=True,
                                title="Held-out dev-seed return (focal seat)", height=300), use_container_width=True)
    c2.plotly_chart(px.line(df, x="env_steps", y=["entropy", "approx_kl"], title="Entropy & KL", height=300),
                    use_container_width=True)
    rep = run / "eval" / "report.md"
    if rep.exists():
        st.markdown(rep.read_text())


def page_leaderboard():
    st.title("🏆 Leaderboard")
    p = Path("runs/leaderboard.json")
    if not p.exists():
        st.info("No games finished yet.")
        return
    df = pd.DataFrame(json.loads(p.read_text())).sort_values("value_created", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True)


def page_concept():
    p = Path("docs/concept.html")
    if p.exists():
        import streamlit.components.v1 as components
        components.html(p.read_text(), height=3200, scrolling=True)
    else:
        st.info("docs/concept.html not found.")


# ================================================================ WATCH
def _new_live_run(scenario, seed, rl_firms, months):
    from game.live import LiveRun
    st.session_state.live = LiveRun(scenario=scenario, seed=int(seed), rl_firms=int(rl_firms), months=int(months))
    st.session_state.live_playing = True


def page_watch():
    st.title("📺 Live run — AI firms compete on their own")
    st.caption("Nobody plays. Firms marked **RL agent** are driven by the trained reinforcement-learning policy; "
               "the others follow rule-based strategies. One step = one simulated month.")
    if not checkpoint_available():
        st.error("No trained RL checkpoint at `runs/latest`. Train one first: "
                 "`python -m training.train --config config/train.yaml`.")
        return
    with st.expander("Run settings", expanded="live" not in st.session_state):
        c = st.columns(5)
        scen = c[0].selectbox("Market scenario", ["config/base.yaml"] + sorted(glob.glob("config/scenarios/*.yaml")),
                              format_func=lambda p: Path(p).stem.replace("_", " "))
        seed = c[1].number_input("Seed", value=42, step=1)
        rl = c[2].slider("RL firms at start (of 5)", 1, 5, 3)
        months = c[3].select_slider("Length (months)", [60, 84, 120], value=120)
        if c[4].button("Start new run", type="primary", use_container_width=True) or "live" not in st.session_state:
            _new_live_run(scen, seed, rl, months)
    run = st.session_state.live
    b1, b2, b3 = st.columns([1, 1, 3])
    playing = st.session_state.get("live_playing", True) and not run.done
    if b1.button("⏸ Pause" if playing else "▶ Resume", use_container_width=True, disabled=run.done):
        st.session_state.live_playing = not playing
        st.rerun()
    speed = b3.select_slider("Speed", ["1 month/sec", "3 months/sec", "6 months/sec"], value="3 months/sec",
                             key="live_speed")
    if b2.button("⏭ Finish run", use_container_width=True, disabled=run.done):
        while not run.done:
            run.step()
        st.rerun()
    steps = int(speed.split()[0])

    @st.fragment(run_every=1.0 if playing else None)
    def live_view():
        if st.session_state.get("live_playing", True) and not run.done:
            for _ in range(steps):
                run.step()
        render_live(run)
        if run.done and playing:
            st.session_state.live_playing = False
            st.rerun(scope="app")    # full rerun drops the auto-refresh timer

    live_view()


def render_live(run):
    w = run.world
    total = run.cfg.episode.months
    st.progress(run.month / total, text=f"Month {run.month} of {total}" + ("  ·  finished" if run.done else ""))
    gs = run.group_scores()
    c = st.columns(4)
    c[0].metric("RL agents · avg value created", f"{gs['RL agents']['avg_value']:+.2f} $M",
                help=f"{gs['RL agents']['alive']} of {gs['RL agents']['firms']} RL firms still active")
    c[1].metric("Rule-based · avg value created", f"{gs['Rule-based']['avg_value']:+.2f} $M",
                help=f"{gs['Rule-based']['alive']} of {gs['Rule-based']['firms']} rule-based firms still active")
    pub = w.price_history[-1]
    c[2].metric("Firms in market", pub["n_firms"], help=f"{w.entries} entered · {w.exits} bankrupt so far")
    c[3].metric("Compute price", f"${pub['compute_price'] * 1000:.1f}/unit", help=f"utilisation {pub['utilization']:.0%}")
    st.markdown("#### Leaderboard")
    lb = pd.DataFrame(run.leaderboard())
    st.dataframe(lb, hide_index=True, use_container_width=True,
                 column_config={"value created $M": st.column_config.NumberColumn(format="%+.2f")})
    left, right = st.columns([3, 2])
    with left:
        vh = pd.DataFrame(run.value_hist)
        if len(vh):
            fig = px.line(vh, x="month", y="value", color="firm", height=340,
                          title="Value created per firm ($M) — solid = RL agent, dotted = rule-based")
            fig.for_each_trace(lambda t: t.update(line_dash="dot") if "Rule" in t.name else t.update(line_width=3))
            fig.update_layout(margin=dict(t=40, b=0), legend_title=None, xaxis_range=[0, total])
            st.plotly_chart(fig, use_container_width=True, key=f"vh-{id(run)}")
        hist = pd.DataFrame(w.price_history)
        sectors = [s.name for s in run.cfg.demand.sectors]
        ad = pd.DataFrame([dict(month=r["month"], **{s: r["sector_adoption"][i] for i, s in enumerate(sectors)})
                           for r in w.price_history])
        st.plotly_chart(px.area(ad, x="month", y=sectors, title="Enterprise adoption by sector", height=260)
                        .update_layout(margin=dict(t=40, b=0), legend_title=None, xaxis_range=[0, total]),
                        use_container_width=True, key=f"ad-{id(run)}")
    with right:
        st.markdown("#### Latest decisions")
        log = pd.DataFrame(run.log[-14:][::-1])
        if len(log):
            log["ai_confidence"] = log["ai_confidence"].map(lambda x: f"{x:.0%}" if x == x and x is not None else "rule")
            st.dataframe(log[["month", "firm", "who", "decision", "ai_confidence"]], hide_index=True,
                         use_container_width=True, height=300)
        else:
            st.caption("Waiting for the first moves…")
        st.markdown("#### Market news")
        news = [n for n in run.news if n["event"] != "Opportunity Discovered"][-12:][::-1]
        if news:
            st.dataframe(pd.DataFrame(news)[["month", "event", "who", "detail"]], hide_index=True,
                         use_container_width=True, height=300)
    if run.done:
        lb_sorted = run.leaderboard()
        win = lb_sorted[0]
        st.success(f"Run finished. Winner: firm {win['firm']} ({win['who']}) with {win['value created $M']:+.2f} $M. "
                   f"Replay saved to `{run.replay_path}` (open it on the Replay page).")


PAGES = {"📺 Watch live run": page_watch, "📼 Replay & dashboard": page_replay,
         "🧠 Training & evaluation": page_training, "🗺️ How the game works": page_concept,
         "🎮 Human play (optional)": page_play, "🏆 Leaderboard": page_leaderboard}
choice = st.sidebar.radio("AI Economy Lab", list(PAGES))
st.sidebar.caption("Stylised simulator · illustrative parameters · not a forecast of real AI markets.")
PAGES[choice]()
