"""Regenerate every paper figure as vector PDF from the data in results/.

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
RES = ROOT / "results"

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
    rows = [r for r in jl(RES / "policy/metrics_pure_ppo_failed.jsonl") if "policy_raw_return" in r]
    k = [r["env_steps"] / 1000 for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.4))
    a.plot(k, [r["policy_raw_return"] for r in rows], color=GREY, lw=2, marker="o", ms=3)
    a.axhline(0, color=REF, ls="--", lw=0.8)
    a.axhline(-1.9, color=REF, ls="--", lw=0.8)
    a.text(k[-1], 0.08, "break-even", ha="right", color=REF, fontsize=7.5)
    a.text(k[-1], -1.82, "bankrupt (-1.9)", ha="right", color=REF, fontsize=7.5)
    a.set_ylim(-2.4, 0.5)
    a.set_title("Value created per firm ($M)", loc="left")
    a.set_xlabel("training experience (thousand firm-months)")
    b.plot(k, [r["entropy"] for r in rows], color=ACC, lw=2, marker="o", ms=3)
    b.set_ylim(0, 2.6)
    b.set_title("Exploration (policy entropy)", loc="left")
    b.set_xlabel("training experience (thousand firm-months)")
    b.annotate(f"{rows[-1]['entropy']:.2f}", (k[-1], rows[-1]["entropy"]), xytext=(-6, 10),
               textcoords="offset points", ha="right", color=DARK)
    fig.tight_layout()
    save(fig, "fig_pure_rl")


def fig_training():
    rows = [r for r in jl(RES / "policy/metrics.jsonl") if "eval_raw_return" in r]
    k = [r["env_steps"] / 1000 for r in rows]
    v = [r["eval_raw_return"] for r in rows]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.plot(k, v, color=ACC, lw=2, marker="o", ms=3.5)
    ax.axhline(v[0], color=REF, ls="--", lw=0.8)
    ax.text(k[-1], v[0] - 1.1, f"level right after imitation: {v[0]:.1f}", ha="right", color=REF, fontsize=7.5)
    i = max(range(len(v)), key=lambda j: v[j])
    ax.scatter([k[i]], [v[i]], s=80, facecolors="none", edgecolors=ACC, lw=1.5, zorder=3)
    ax.annotate(f"best {v[i]:.1f} (saved)", (k[i], v[i]), xytext=(0, 9), textcoords="offset points",
                ha="center", color=DARK)
    ax.set_ylim(8, 24.5)
    ax.set_xlabel("training experience (thousand firm-months)")
    ax.set_ylabel("value created per firm ($M)")
    fig.tight_layout()
    save(fig, "fig_training")


def fig_one_game():
    from game.live import LiveRun
    r = LiveRun(seed=47, rl_firms=3, months=120, replay_dir=str(ROOT / "paper" / ".replay_tmp"))
    while not r.done:
        r.step()
    w = r.world
    init = [f for f in w.firms.values() if f.birth_month == 0]
    names = {0: "RL agent (aggressive)", 1: "RL agent (balanced)", 2: "RL agent (cautious)",
             3: "Rule: best expected value", 4: "Rule: easy-first"}
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for a, b in [(3, 12), (70, 77)]:
        ax.axvspan(a, b, color="#eceff1", lw=0)
        ax.text(a, 25.3, "fab shortage", fontsize=7, color=REF)
    for m in (17, 73):
        ax.axvline(m, color=REF, ls=":", lw=0.8)
    ax.text(18, 23.2, "frontier breakthrough", fontsize=7, color=REF)
    ax.text(74, 2.5, "frontier breakthrough", fontsize=7, color=REF)
    for f in init:
        pts = [(v["month"], v["value"]) for v in r.value_hist if v["firm"].startswith(f"{f.firm_id} ")]
        rl = f.controller == "policy"
        ax.plot(*zip(*pts), color=ACC if rl else GREY, lw=1.8, ls="-" if rl else "--")
        ax.text(121.5, pts[-1][1] + (0.6 if f.firm_id == 4 else (-0.6 if f.firm_id == 3 else 0)),
                f"{names[f.firm_id]} {pts[-1][1]:.1f}", va="center", fontsize=7.5,
                color=DARK if rl else REF)
    ax.set_xlim(0, 120)
    ax.set_ylim(-2, 26.5)
    ax.set_xlabel("month")
    ax.set_ylabel("value created ($M)")
    fig.tight_layout()
    save(fig, "fig_one_game")


def fig_forest():
    e30 = json.load(open(RES / "eval/report.json"))
    e100 = json.load(open(RES / "eval100/report.json"))
    rows = [("Base market", e100["base"], 100), ("Fast tech progress", e30["scenario:fast_tech"], 30),
            ("Slow tech progress", e30["scenario:slow_tech"], 30),
            ("Harsher use cases", e30["scenario:harsh_failures"], 30),
            ("Compute crunch", e100["scenario:compute_crunch"], 100),
            ("Wave of new entrants", e30["scenario:entrant_wave"], 30)]
    fig, ax = plt.subplots(figsize=(W, 2.8))
    for i, (name, rep, n) in enumerate(rows):
        b = rep["best_npv"]
        d, h = b["policy_minus_this"], b["policy_minus_this_ci95"]
        clear = d - h > 0
        c = ACC if clear else GREY
        y = len(rows) - 1 - i
        ax.plot([d - h, d + h], [y, y], color=c, lw=2)
        ax.scatter([d], [y], color=c, s=40, zorder=3)
        ax.text(12.6, y, f"{d:+.1f}", va="center", ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)), [f"{n} ({k} markets)" for n, _, k in rows][::-1])
    ax.axvline(0, color=REF, lw=1)
    ax.set_xlim(-6, 13)
    ax.set_xlabel("learning agent minus best-expected-value rule, value created per company ($M), mean and 95% CI")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_forest")


def fig_multiseed():
    ms = json.load(open(RES / "multiseed.json"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, key, title in ((axes[0], "base", "Base market"), (axes[1], "compute_crunch", "Compute crunch")):
        r = ms[key]
        runs = r["runs"]
        for i, run in enumerate(runs):
            d, h = run["diff"], run["ci95"]
            c = ACC if d - h > 0 else ("#c5221f" if d + h < 0 else GREY)
            y = len(runs) - 1 - i
            ax.plot([d - h, d + h], [y, y], color=c, lw=2)
            ax.scatter([d], [y], color=c, s=36, zorder=3)
        m = r["diff_mean_across_runs"]
        ax.axvline(0, color=REF, lw=1)
        ax.axvline(m, color=DARK, ls="--", lw=0.9)
        ax.text(m + 0.15, -0.95, f"mean of 5 runs {m:+.1f}", ha="left", fontsize=7.5)
        ax.set_ylim(-1.3, len(runs) - 0.5)
        ax.set_title(f"{title}: best rule {r['rule_mean']:.1f} $M", loc="left")
        ax.set_xlim(-5, 5.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(5), [f"training seed {i}" for i in range(5)][::-1])
    fig.supxlabel("learning agent minus best-expected-value rule ($M per company), mean and 95% CI over 100 markets",
                  fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_multiseed")


def fig_ablation():
    e30 = json.load(open(RES / "eval/report.json"))
    full = e30["base"]["best_npv"]["policy_minus_this"]
    labels = {"technical_failure": "technical failure", "delayed_capex": "delayed capacity",
              "supply_bottlenecks": "supply bottlenecks", "confidence_feedback": "confidence feedback",
              "risk_heterogeneity": "different risk appetites", "competition": "competition"}
    rows = []
    for k, lab in labels.items():
        s = e30[f"ablation:no_{k}"]
        rows.append((f"without {lab}", s["policy"]["raw_return_mean"] - s["best_npv"]["raw_return_mean"]))
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for i, (lab, d) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, d, color=ACC if d < 0 else GREY, height=0.6)
        ax.text(15.5, y, f"{d:+.1f}", va="center", ha="right", fontsize=8.5)
    ax.axvline(full, color=REF, ls="--", lw=0.8)
    ax.text(full + 0.3, len(rows) - 0.4, f"full simulation: {full:+.1f}", fontsize=7.5, color=REF)
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(-8, 16)
    ax.set_xlabel("learning agent minus best-expected-value rule ($M per company), 30 markets each")
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
    off = {"best_npv": 0.03, "cash_preserving": -0.03, "policy": 0, "easy_first": 0}
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
    for f in (fig_multiseed, fig_pure_rl, fig_training, fig_one_game, fig_forest, fig_ablation, fig_adoption, fig_calibration,
              fig_price, fig_survival):
        f()
