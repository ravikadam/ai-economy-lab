"""PFAR hypotheses P1-P3 (paper/PROTOCOL_v2.md, plan R3) from results/v2/test_pfar.json (markets 700-799).

P1: PFAR's five-agent lead over the rule in normal markets exceeds v2's (two-level bootstrap of PFAR - v2 above 0).
P2: PFAR's average survival in the compute crunch exceeds v2's (bootstrap interval of the difference above 0).
P3: PFAR from scratch is clearly better than the rule in normal markets in at least 4 of 5 runs (v2pure: 3 of 5).

Usage: python -m scripts.pfar_eval  -> results/v2/pfar_eval.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B = 20_000
ARMS = {"v2": "v2-seed", "pfar": "pfar-seed", "v2pure": "v2pure-seed", "pfarpure": "pfarpure-seed"}


def arm(d, pre, key):
    rule = np.array(d["rule_per_market"])
    rows = [r for r in d["runs"] if r["run"].startswith(pre)]
    if key == "lead":
        return np.stack([np.array(r["per_market"]) - rule for r in rows])
    return np.array([r[key] for r in rows]), rows


def boot_diff(A, F, rng):
    """Two-level bootstrap (runs within each arm, markets shared) of mean(A) - mean(F)."""
    n = A.shape[1]
    out = np.empty(B)
    for b in range(B):
        m = rng.integers(0, n, n)
        out[b] = A[rng.integers(0, len(A), len(A))][:, m].mean() - F[rng.integers(0, len(F), len(F))][:, m].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    t = json.load(open("results/v2/test_pfar.json"))["scenarios"]
    rng = np.random.default_rng(20261022)
    res = {"markets": "700-799 base, 700-749 conditions", "conditions": {}}
    for sc, d in t.items():
        c = {}
        for name, pre in ARMS.items():
            L = arm(d, pre, "lead")
            surv, rows = arm(d, pre, "survival")
            c[name] = {"lead": float(L.mean()), "per_run_lead": L.mean(1).round(2).tolist(),
                       "clearly_better": int(sum(r["better"] for r in rows)),
                       "clearly_worse": int(sum(r["worse"] for r in rows)),
                       "survival": float(surv.mean()), "per_run_survival": surv.round(2).tolist()}
        c["pfar_minus_v2"] = {"lead": c["pfar"]["lead"] - c["v2"]["lead"],
                              "interval": boot_diff(arm(d, ARMS["pfar"], "lead"), arm(d, ARMS["v2"], "lead"), rng)}
        c["pfarpure_minus_v2pure"] = {"lead": c["pfarpure"]["lead"] - c["v2pure"]["lead"],
                                      "interval": boot_diff(arm(d, ARMS["pfarpure"], "lead"),
                                                            arm(d, ARMS["v2pure"], "lead"), rng)}
        res["conditions"][sc] = c
    base, crunch = res["conditions"]["base"], res["conditions"]["compute_crunch"]
    # P2: survival difference, bootstrapping runs (survival is per run, averaged over markets)
    sp, sv = np.array(crunch["pfar"]["per_run_survival"]), np.array(crunch["v2"]["per_run_survival"])
    sd = [sp[rng.integers(0, 5, 5)].mean() - sv[rng.integers(0, 5, 5)].mean() for _ in range(B)]
    res["P1"] = {"pfar_minus_v2_lead": base["pfar_minus_v2"]["lead"], "interval": base["pfar_minus_v2"]["interval"],
                 "supported": base["pfar_minus_v2"]["interval"][0] > 0}
    res["P2"] = {"pfar_minus_v2_crunch_survival": float(sp.mean() - sv.mean()),
                 "interval": [float(np.percentile(sd, 2.5)), float(np.percentile(sd, 97.5))],
                 "supported": float(np.percentile(sd, 2.5)) > 0}
    res["P3"] = {"pfarpure_clearly_better": base["pfarpure"]["clearly_better"],
                 "v2pure_clearly_better": base["v2pure"]["clearly_better"],
                 "supported": base["pfarpure"]["clearly_better"] >= 4}
    Path("results/v2/pfar_eval.json").write_text(json.dumps(res, indent=2))
    for p in ("P1", "P2", "P3"):
        print(p, res[p])
    for sc, c in res["conditions"].items():
        print(f"{sc:15s} " + "  ".join(f"{k} {c[k]['lead']:+.2f} ({c[k]['clearly_better']}/5, surv {c[k]['survival']:.2f})"
                                     for k in ARMS) + f" | pfar-v2 {c['pfar_minus_v2']['lead']:+.2f} "
              f"{np.round(c['pfar_minus_v2']['interval'], 2)}")


if __name__ == "__main__":
    main()
