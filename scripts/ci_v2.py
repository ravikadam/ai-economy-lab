"""Confidence intervals for the v2 test (results/v2/test.json -> results/v2/intervals.json).

Every number starts from paired per-market differences: in each test market, the agent's value created (the
average of its three sampled plays, or its single greedy play) minus the best-expected-value rule's value in
the same market.

* per_agent.normal   -- the pre-registered interval: mean +/- 1.96 * sd / sqrt(n) over the n paired differences.
* per_agent.boot     -- percentile bootstrap over markets (10,000 resamples), a check on the normal approximation.
* pooled             -- the five-agent average lead. The agents share the same test markets, so their results are
                        not independent. "markets" resamples markets only (agents fixed); "agents_and_markets"
                        resamples the five training runs and the markets (two-level bootstrap), which also counts
                        run-to-run luck.
* arms               -- comparison runs (no varied markets, no imitation, imitation only) minus the full-recipe runs
                        with the same training seeds, paired by market; resamples markets and runs within each arm.
                        min_detectable_diff_80pct_power = (1.96 + 0.84) x bootstrap standard error of that difference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B = 10_000


def pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    t = json.load(open("results/v2/test.json"))["scenarios"]
    rng = np.random.default_rng(20261019)
    out = {"method": __doc__, "resamples": B, "scenarios": {}}
    for sc, d in t.items():
        rule = np.array(d["rule_per_market"])
        runs = {r["run"]: np.array(r["per_market"]) - rule for r in d["runs"]}
        v2 = np.stack([runs[k] for k in sorted(runs) if k.startswith("v2-seed")])
        n = rule.size
        res = {"n_markets": n, "per_agent": {}, "pooled": {}, "arms": {}}
        for k in sorted(runs):
            x = runs[k]
            idx = rng.integers(0, n, (B, n))
            h = 1.96 * x.std(ddof=1) / np.sqrt(n)
            res["per_agent"][k.split("-2026")[0]] = {"mean": float(x.mean()),
                                                      "normal": [float(x.mean() - h), float(x.mean() + h)],
                                                      "boot": pct(x[idx].mean(1))}
        idx = rng.integers(0, n, (B, n))
        res["pooled"]["mean"] = float(v2.mean())
        res["pooled"]["markets"] = pct(v2[:, idx].mean(axis=(0, 2)))
        ag = rng.integers(0, len(v2), (B, len(v2)))
        res["pooled"]["agents_and_markets"] = pct(np.array([v2[ag[b]][:, idx[b]].mean() for b in range(B)]))
        c = np.corrcoef(v2)
        res["pooled"]["mean_pairwise_correlation"] = float(c[np.triu_indices(len(v2), 1)].mean())
        for arm, pre in (("no_varied_markets", "v2nodr-seed"), ("no_imitation", "v2pure-seed"),
                         ("imitation_only", "v2bc-seed")):
            keys = [k for k in sorted(runs) if k.startswith(pre)]
            if not keys:
                continue
            seeds = sorted({int(k[len(pre):].split("-")[0]) for k in keys})
            a = np.stack([runs[k] for k in keys])
            f = np.stack([runs[k] for k in sorted(runs) if any(k.startswith(f"v2-seed{s}-") for s in seeds)])
            diff, lead = [], []
            for b in range(B):
                ia = rng.integers(0, len(a), len(a))
                jf = rng.integers(0, len(f), len(f))
                diff.append(a[ia][:, idx[b]].mean() - f[jf][:, idx[b]].mean())
                lead.append(a[ia][:, idx[b]].mean())
            # smallest true difference detectable with 80% power at the 5% level (two-sided), from the bootstrap SE
            mde = float((1.959964 + 0.841621) * np.std(diff, ddof=1))
            res["arms"][arm] = {"mean_diff": float(a.mean() - f.mean()), "interval": pct(diff),
                                "lead_vs_rule": float(a.mean()), "lead_interval": pct(lead),
                                "min_detectable_diff_80pct_power": mde,
                                "arm_runs": len(a), "matched_full_runs": len(f),
                                "n_clearly_better": int(sum(res["per_agent"][k.split("-2026")[0]]["normal"][0] > 0
                                                           for k in keys))}
        out["scenarios"][sc] = res
        print(f"== {sc}: pooled {res['pooled']['mean']:+.2f} markets {np.round(res['pooled']['markets'], 2)} "
              f"agents+markets {np.round(res['pooled']['agents_and_markets'], 2)} "
              f"corr {res['pooled']['mean_pairwise_correlation']:.2f}")
        for k, v in res["per_agent"].items():
            print(f"   {k:14s} {v['mean']:+.2f} normal {np.round(v['normal'], 2)} boot {np.round(v['boot'], 2)}")
        for k, v in res["arms"].items():
            print(f"   arm {k:18s} lead {v['lead_vs_rule']:+.2f} {np.round(v['lead_interval'], 2)} | vs full "
                  f"{v['mean_diff']:+.2f} {np.round(v['interval'], 2)} MDE80 {v['min_detectable_diff_80pct_power']:.2f} "
                  f"({v['arm_runs']} runs, {v['n_clearly_better']} clearly better)")
    Path("results/v2/intervals.json").write_text(json.dumps(out, indent=2))
    print("-> results/v2/intervals.json")


if __name__ == "__main__":
    main()
