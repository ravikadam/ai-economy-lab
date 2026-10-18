"""Static dashboards (PNG) for a replay and for a training run.

Usage: python -m analysis.plots --replay runs/replays/<file>.jsonl
       python -m analysis.plots --run runs/latest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.metrics import firm_frame, market_frame, training_frame  # noqa: E402
from analysis.replay import load_replay  # noqa: E402

INK, MUTED = "#1f2430", "#6b7280"
SERIES = ["#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2"]


def _style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)


def save_dashboard(replay_path: str | Path) -> Path:
    rp = Path(replay_path)
    rep = load_replay(rp)
    m = market_frame(rep)
    f = firm_frame(rep)
    sectors = [s["name"] for s in rep["header"]["scenario"]["demand"]["sectors"]]
    models = [x["name"] for x in rep["header"]["scenario"]["technology"]["models"]]
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    ax = axes.ravel()
    for i, s in enumerate(sectors):
        ax[0].plot(m.month, m[f"adoption:{s}"], color=SERIES[i], label=s)
        ax[1].plot(m.month, m[f"confidence:{s}"], color=SERIES[i], label=s)
    _style(ax[0], "Adopted use cases by sector"); ax[0].legend(fontsize=7, frameon=False)
    _style(ax[1], "Enterprise confidence by sector")
    for k, lvl in enumerate(("low", "mid", "high")):
        ax[2].bar(k - 0.2, m[f"technical_pass:{lvl}"].sum(), 0.4, color=SERIES[2])
        ax[2].bar(k + 0.2, m[f"technical_fail:{lvl}"].sum(), 0.4, color=SERIES[3])
    ax[2].set_xticks([0, 1, 2], ["low", "mid", "high"]); _style(ax[2], "Technical gates by complexity (pass/fail)")
    ax[3].plot(m.month, m.utilization, color=SERIES[0], label="utilisation")
    ax[3].plot(m.month, m.served_frac, color=SERIES[1], label="served fraction")
    _style(ax[3], "Compute utilisation"); ax[3].legend(fontsize=7, frameon=False)
    ax[4].plot(m.month, m.compute_price * 1000, color=SERIES[0])
    _style(ax[4], "Compute price ($ per unit)")
    ax[5].plot(m.month, m.fab_nominal_capacity, color=SERIES[0], label="fab capacity")
    ax[5].plot(m.month, m.fab_pending, color=SERIES[1], label="under construction")
    ax[5].plot(m.month, m.fab_backlog, color=SERIES[3], label="backlog")
    _style(ax[5], "Fab capacity, capex lag, backlog (wafers)"); ax[5].legend(fontsize=7, frameon=False)
    for j, mn in enumerate(models):
        for i, s in enumerate(sectors):
            ax[6].plot(m.month, m[f"capability:{mn}:{s}"], color=SERIES[i], ls="-" if j == 0 else "--")
        ax[7].plot(m.month, m[f"unit_price:{mn}"] * 1000, color=SERIES[j], label=mn)
    _style(ax[6], "Model capability (solid frontier, dashed open)")
    _style(ax[7], "Unit inference price ($)"); ax[7].legend(fontsize=7, frameon=False)
    for k, (fid, g) in enumerate(f.groupby("firm_id")):
        ax[8].plot(g.month, g.cash, color=SERIES[k % len(SERIES)], lw=1)
        ax[9].plot(g.month, g.value_created, color=SERIES[k % len(SERIES)], lw=1)
    _style(ax[8], "Builder cash ($k)"); _style(ax[9], "Builder value created ($M, cumulative)")
    ax[10].plot(m.month, m.hhi, color=SERIES[4]); _style(ax[10], "Market concentration (HHI)")
    ax[11].plot(m.month, m.welfare.cumsum(), color=SERIES[2], label="cum. welfare")
    ax11b = ax[11].twinx()
    ax11b.step(m.month, m.n_firms, color=MUTED, where="post", lw=1)
    _style(ax[11], "Cumulative welfare ($k) · firms alive (grey)")
    fig.suptitle(f"AI Economy Lab — {rep['header']['scenario']['name']} · seed {rep['header']['seed']}",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    out = rp.with_suffix(".png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_training_curves(run_dir: str | Path) -> Path:
    df = training_frame(run_dir)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if "policy_raw_return" in df:
        d = df.dropna(subset=["policy_raw_return"])
        ax[0].plot(d.env_steps, d.policy_raw_return, color=SERIES[0], label="policy (raw)")
        ax[0].plot(d.env_steps, d.heuristic_raw_return, color=SERIES[1], label="rule-based rivals (raw)")
        ax[0].legend(fontsize=8, frameon=False)
        ax[1].plot(d.env_steps, d.policy_survival, color=SERIES[2])
    if "eval_raw_return" in df:
        e = df.dropna(subset=["eval_raw_return"])
        ax[0].plot(e.env_steps, e.eval_raw_return, color=SERIES[3], marker="o", ms=3, lw=1, label="held-out eval")
        ax[0].legend(fontsize=8, frameon=False)
    ax[2].plot(df.env_steps, df.entropy, color=SERIES[4])
    _style(ax[0], "Episode raw return ($M)"); _style(ax[1], "Policy firm survival"); _style(ax[2], "Policy entropy")
    fig.tight_layout()
    out = Path(run_dir) / "training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay")
    ap.add_argument("--run")
    a = ap.parse_args()
    if a.replay:
        print(save_dashboard(a.replay))
    if a.run:
        print(save_training_curves(a.run))


if __name__ == "__main__":
    main()
