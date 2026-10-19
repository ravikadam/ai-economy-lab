"""Regenerate every paper figure as vector PDF from the v2 run data in results/v2/.

Usage (from the repo root):  python paper/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results" / "v2"   # every figure uses only the v2 run

ACC = "#1f6fd1"      # accent: what each figure is about
GREY = "#9aa0a6"
DARK = "#202124"
REF = "#5f6368"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": REF,
    "axes.grid": True, "grid.color": "#e8eaed", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})
W = 6.3  # inches, text width


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_pure_rl():
    rows = [r for r in jl(RES / "control_pure/metrics.jsonl") if "policy_raw_return" in r]
    k = [r["env_steps"] / 1000 for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.4))
    a.plot(k, [r["policy_raw_return"] for r in rows], color=GREY, lw=1.6)
    a.axhline(0, color=REF, ls="--", lw=0.8)
    a.axhline(-1.9, color=REF, ls="--", lw=0.8)
    a.text(k[-1], 0.08, "break-even", ha="right", color=REF, fontsize=7.5)
    a.text(k[-1], -1.82, "bankrupt (-1.9)", ha="right", color=REF, fontsize=7.5)
    a.set_title("Value created per firm ($M)", loc="left")
    a.set_xlabel("training experience (thousand firm-months)")
    ent = [r["entropy"] for r in rows]
    b.plot(k, ent, color=ACC, lw=1.6)
    b.set_ylim(0, max(2.6, max(ent) * 1.05))
    b.set_title("Exploration (policy entropy)", loc="left")
    b.set_xlabel("training experience (thousand firm-months)")
    fig.tight_layout()
    save(fig, "fig_pure_rl")


def fig_training():
    fig, ax = plt.subplots(figsize=(W, 2.8))
    rule = None
    for s in range(5):
        rows = [r for r in jl(RES / f"seed{s}/metrics.jsonl") if "dev" in r]
        k = [r["env_steps"] / 1000 for r in rows]
        v = [r["dev"]["score"] for r in rows]
        rule = rows[0]["dev"]["rule"]
        ax.plot(k, v, color=ACC, lw=1.3, alpha=0.75)
        i = max(range(len(v)), key=lambda j: v[j])
        ax.scatter([k[i]], [v[i]], s=36, facecolors="none", edgecolors=DARK, lw=1.2, zorder=3)
        ax.text(k[-1] + 8, v[-1], f"run {s + 1}", fontsize=7, color=REF, va="center")
    ax.axhline(rule, color=REF, ls="--", lw=0.9)
    ax.text(5, rule + 0.2, f"best rule on the same 40 practice markets: {rule:.1f}", color=REF, fontsize=7.5)
    ax.set_ylim(rule - 1, None)
    ax.set_xlabel("training experience (thousand firm-months)")
    ax.set_ylabel("score on practice markets\n($M per firm)")
    fig.tight_layout()
    save(fig, "fig_training")


def fig_one_game():
    from game.live import LiveRun
    r = LiveRun(seed=47, rl_firms=3, months=120, checkpoint=str(RES / "seed0/best.pt"),
                replay_dir=str(ROOT / "paper" / ".replay_tmp"))
    while not r.done:
        r.step()
    w = r.world
    init = [f for f in w.firms.values() if f.birth_month == 0]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    shock = [h["month"] for h in w.price_history if h.get("fab_shock", 0) > 0]
    spans, start = [], None
    for m in range(0, 122):
        on = m in shock
        if on and start is None:
            start = m
        if not on and start is not None:
            spans.append((start, m)); start = None
    for a, b in spans:
        ax.axvspan(a, b, color="#eceff1", lw=0)
    top = max(v["value"] for v in r.value_hist)
    if spans:
        ax.text(spans[0][0], top * 1.04, "fab shortage", fontsize=7, color=REF)
    ends = []
    for f in init:
        pts = [(v["month"], v["value"]) for v in r.value_hist if v["firm"].startswith(f"{f.firm_id} ")]
        rl = f.controller == "policy"
        ax.plot(*zip(*pts), color=ACC if rl else GREY, lw=1.8, ls="-" if rl else "--")
        lab = (f"RL agent ({f.profile.name})" if rl else
               "Rule: " + {"heuristic:best_npv": "best expected value", "heuristic:easy_first": "easy-first",
                           "heuristic:cash_preserving": "cash-preserving"}.get(f.controller, f.controller))
        ends.append([pts[-1][1], f"{lab} {pts[-1][1]:.1f}", rl])
    ends.sort()
    for i in range(1, len(ends)):              # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + top * 0.06)
    for y, lab, rl in ends:
        ax.text(121.5, y, lab, va="center", fontsize=7.5, color=DARK if rl else REF)
    ax.set_xlim(0, 120)
    ax.set_ylim(min(-2, ax.get_ylim()[0]), top * 1.12)
    ax.set_xlabel("month")
    ax.set_ylabel("value created ($M)")
    fig.tight_layout()
    save(fig, "fig_one_game")
    return {"seed": 47, "spans": spans,
            "firms": {f.firm_id: {"controller": f.controller, "profile": f.profile.name, "launches": f.launches,
                                  "alive": f.alive} for f in init},
            "ends": [(e[1]) for e in ends], "entrants": sum(1 for f in w.firms.values() if f.birth_month > 0),
            "entrant_bankrupt": sum(1 for f in w.firms.values() if f.birth_month > 0 and not f.alive)}


REGIMES = [("base", "Base market (100)"), ("fast_tech", "Fast tech progress (50)"),
           ("slow_tech", "Slow tech progress (50)"), ("harsh_failures", "Harsher use cases (50)"),
           ("entrant_wave", "Wave of entrants (50)"), ("compute_crunch", "Compute crunch (50)")]


def fig_test():
    t = json.load(open(RES / "test.json"))["scenarios"]
    v2 = {sc: [p for p in t[sc]["runs"] if p["run"].startswith("v2-seed")] for sc, _ in REGIMES}
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for i, (sc, lab) in enumerate(REGIMES):
        y = len(REGIMES) - 1 - i
        for j, p in enumerate(v2[sc]):
            c = ACC if p["better"] else ("#c5221f" if p["worse"] else GREY)
            yy = y + (j - 2) * 0.11
            ax.plot([p["diff"] - p["ci95"], p["diff"] + p["ci95"]], [yy, yy], color=c, lw=1.2)
            ax.scatter([p["diff"]], [yy], color=c, s=14, zorder=3)
        m = sum(p["diff"] for p in v2[sc]) / len(v2[sc])
        ax.plot([m, m], [y - 0.33, y + 0.33], color=DARK, lw=2)
        nb = sum(p["better"] for p in v2[sc])
        ax.text(1.02, y, f"{m:+.1f}  ({nb} of 5 clearly better)", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(range(len(REGIMES)), [l for _, l in REGIMES][::-1])
    ax.set_xlabel("learning agent minus best rule ($M per company); bars show the likely range for each training run")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_test")


def fig_controls():
    t = json.load(open(RES / "test.json"))["scenarios"]
    arms = [("v2-seed", "all fixes (5 training runs)", ACC), ("v2nodr-seed", "without varied markets (3 runs)", GREY),
            ("v2pure-seed", "without imitation (1 run)", "#c5221f")]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    for i, (sc, lab) in enumerate(REGIMES):
        y = len(REGIMES) - 1 - i
        for j, (pre, name, c) in enumerate(arms):
            ds = [p["diff"] for p in t[sc]["runs"] if p["run"].startswith(pre)]
            if not ds:
                continue
            m = sum(ds) / len(ds)
            ax.barh(y + (1 - j) * 0.26, m, height=0.24, color=c, label=name if i == 0 else None)
            ax.scatter(ds, [y + (1 - j) * 0.26] * len(ds), color=DARK, s=5, zorder=3)
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(range(len(REGIMES)), [l for _, l in REGIMES][::-1])
    ax.set_xlabel("learning agent minus best rule ($M per company); bars: average, dots: individual training runs")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_controls")


def fig_ablation():
    a = json.load(open(RES / "ablations.json"))["ablations"]
    full = a["full"]["diff"]
    labels = {"technical_failure": "technical failure", "delayed_capex": "delayed capacity",
              "supply_bottlenecks": "supply bottlenecks", "confidence_feedback": "confidence feedback",
              "risk_heterogeneity": "different risk appetites", "competition": "competition"}
    rows = sorted([(f"without {lab}", a[k]["diff"], a[k]["ci95"]) for k, lab in labels.items()], key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for i, (lab, d, h) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, d, color=ACC if d < 0 else GREY, height=0.6)
        ax.plot([d - h, d + h], [y, y], color=DARK, lw=1)
        ax.text(1.01, y, f"{d:+.1f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
    ax.axvline(full, color=REF, ls="--", lw=0.8)
    ax.text(full, len(rows) - 0.35, f" full simulation: {full:+.1f}", fontsize=7.5, color=REF)
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("learning agents (average of 5) minus best rule ($M per company), 50 markets each")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_adoption():
    st = json.load(open(RES / "stats.json"))["base"]
    ad = st["adoption_by_sector_mean"]
    m = list(range(1, len(ad[0]) + 1))
    names = ["customer operations", "software engineering", "regulated"]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for i, n in enumerate(names):
        reg = i == 2
        ax.plot(m, ad[i], color=ACC if reg else GREY, lw=2.2 if reg else 1.6)
        ax.text(121.5, ad[i][-1] + (2.5 if i == 0 else 0), f"{n} {ad[i][-1]:.0f}", va="center", fontsize=8,
                color=DARK if reg else REF)
    ax.set_xlim(0, 120)
    ax.set_xlabel("month")
    ax.set_ylabel("active deployments\n(enterprise x use case)")
    fig.tight_layout()
    save(fig, "fig_adoption")


def fig_calibration():
    st = json.load(open(RES / "stats.json"))["base"]
    bins = st["tech_pass_by_difficulty"]
    names = ["easiest", "easy", "middle", "hard", "hardest"]
    ranges = ["<0.40", "0.40-0.55", "0.55-0.70", "0.70-0.85", ">0.85"]
    labs = [f"{n}\n{r}\nn = {b['n']}" for n, r, b in zip(names, ranges, bins)]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for i, b in enumerate(bins):
        ax.bar(i, b["pass"], width=0.55, color=ACC)
        ax.plot([i - 0.36, i + 0.36], [b["p_mean"]] * 2, color=DARK, lw=2)
        ax.text(i, max(b["pass"], b["p_mean"]) + 0.03, f"{b['pass']:.0%}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(bins)), labs)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("technical gates passed")
    ax.set_xlabel("true difficulty of the use case (bars: observed; dashes: assigned odds)")
    fig.tight_layout()
    save(fig, "fig_calibration")


def fig_price():
    st = json.load(open(RES / "stats.json"))
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for key, lab, c in (("base", "base market", GREY), ("crunch", "compute crunch", ACC)):
        p = [x * 1000 for x in st[key]["compute_price"]["mean"]]
        m = list(range(1, len(p) + 1))
        ax.plot(m, p, color=c, lw=2)
        ax.text(121.5, p[-1], f"{lab} ${p[-1]:.1f}", va="center", fontsize=8, color=DARK if c == ACC else REF)
    pc = [x * 1000 for x in st["crunch"]["compute_price"]["mean"]]
    i = max(range(len(pc)), key=lambda j: pc[j])
    ax.annotate(f"peak ${pc[i]:.1f}, month {i + 1}", (i + 1, pc[i]), xytext=(0, 8), textcoords="offset points",
                ha="center")
    ax.axhline(30, color=REF, ls="--", lw=0.8)
    ax.text(120, 30.6, "price ceiling $30", ha="right", fontsize=7.5, color=REF)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 33)
    ax.set_xlabel("month")
    ax.set_ylabel("compute price ($ per GPU-eq unit)")
    fig.tight_layout()
    save(fig, "fig_price")


def fig_survival():
    sv = json.load(open(RES / "stats.json"))["crunch"]["survival_curve_initial"]
    names = {"policy": "learning agent", "best_npv": "best expected value", "cash_preserving": "cash-preserving",
             "easy_first": "easy-first"}
    ends = sorted((sv[k][-1], k) for k in names)
    pos = {}
    for i, (y, k) in enumerate(ends):          # keep end labels at least 6 points apart
        pos[k] = y if i == 0 else max(y, pos[ends[i - 1][1]] + 0.06)
    off = {k: pos[k] - sv[k][-1] for k in names}
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for k, lab in names.items():
        y = [1.0] + sv[k]
        m = list(range(0, len(y)))
        c = ACC if k == "policy" else GREY
        ax.step(m, y, where="post", color=c, lw=2.2 if k == "policy" else 1.5)
        ax.text(121.5, y[-1] + off[k], f"{lab} {y[-1]:.0%}", va="center", fontsize=8,
                color=DARK if k == "policy" else REF)
    ax.set_xlim(0, 120)
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("month")
    ax.set_ylabel("founding companies still solvent")
    fig.tight_layout()
    save(fig, "fig_survival")


if __name__ == "__main__":
    for f in (fig_pure_rl, fig_training, fig_test, fig_controls, fig_ablation, fig_adoption, fig_calibration,
              fig_price, fig_survival):
        f()
    info = fig_one_game()
    (RES / "one_game.json").write_text(json.dumps(info, indent=2, default=str))
