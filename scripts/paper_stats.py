"""Aggregate runs/paper/experiments.json into the statistics the paper reports (runs/paper/stats.json)."""
import json, numpy as np
from collections import defaultdict
from pathlib import Path

R = json.loads(Path("runs/paper/experiments.json").read_text())
out = {}
def sc(r): return "base" if "base" in r["scenario"] else "crunch"
SECT = ["customer_ops", "software_eng", "regulated"]
for name in ("base", "crunch"):
    rs = [r for r in R if sc(r) == name]
    H = lambda k: np.array([[m[k] for m in r["history"][1:]] for r in rs])
    months = list(range(1, len(rs[0]["history"])))
    d = {"n": len(rs)}
    for k in ("utilization", "served_frac", "compute_price", "compute_capacity", "compute_demand", "n_firms", "hhi", "welfare", "fab_backlog", "fab_nominal_capacity", "fab_pending", "builder_profit", "enterprise_benefit", "enterprise_payments", "churn", "adoptions"):
        a = H(k); d[k] = {"mean": a.mean(0).tolist(), "p10": np.percentile(a, 10, 0).tolist(), "p90": np.percentile(a, 90, 0).tolist()}
    ad = np.array([[m["sector_adoption"] for m in r["history"][1:]] for r in rs])
    cf = np.array([[m["sector_confidence"] for m in r["history"][1:]] for r in rs])
    d["adoption_by_sector_mean"] = ad.mean(0).T.tolist(); d["confidence_by_sector_mean"] = cf.mean(0).T.tolist()
    cap = np.array([[m["model_capability"] for m in r["history"][1:]] for r in rs])  # runs x months x models x sectors
    d["capability_mean"] = cap.mean(0).transpose(1, 2, 0).tolist()
    d["total_welfare"] = [float(sum(m["welfare"] for m in r["history"])) for r in rs]
    d["total_adoptions"] = [int(sum(m["adoptions"] for m in r["history"])) for r in rs]
    d["total_churn"] = [int(sum(m["churn"] for m in r["history"])) for r in rs]
    d["min_served"] = [float(min(m["served_frac"] for m in r["history"][1:])) for r in rs]
    d["entries"] = [r["history"][-1]["entries_total"] for r in rs]; d["exits"] = [r["history"][-1]["exits_total"] for r in rs]
    # bullwhip: CV of month-on-month changes, orders vs end demand, months 13-120
    bw = []
    for r in rs:
        dem = np.array([m["compute_demand"] for m in r["history"][13:]]); orde = np.array(r["orders"][12:])
        if dem.std() > 0 and orde.mean() > 0:
            bw.append(float((orde.std() / max(orde.mean(), 1e-9)) / (np.diff(dem).std() / max(dem.mean(), 1e-9) + 1e-9)))
    d["bullwhip_ratio"] = bw
    d["order_months_share"] = float(np.mean([np.mean(np.array(r["orders"]) > 1e-6) for r in rs]))
    # firms
    F = [f | {"seed": r["seed"]} for r in rs for f in r["firms"]]
    by = defaultdict(list)
    for f in F: by[f["controller"]].append(f)
    d["firms_by_controller"] = {c: {"n": len(v), "raw_mean": float(np.mean([x["raw"] for x in v])), "survival": float(np.mean([x["alive"] for x in v])),
                                    "launches": float(np.mean([x["launches"] for x in v])), "tech_fail": float(np.mean([x["tech_fail"] for x in v])),
                                    "comm_fail": float(np.mean([x["comm_fail"] for x in v])), "deferrals": float(np.mean([x["deferrals"] for x in v]))} for c, v in by.items()}
    prof = defaultdict(list)
    for f in F: prof[(f["controller"], f["profile"])].append(f)
    d["by_profile"] = {f"{c}|{p}": {"n": len(v), "raw_mean": float(np.mean([x["raw"] for x in v])), "survival": float(np.mean([x["alive"] for x in v])), "p10": float(np.percentile([x["raw"] for x in v], 10))} for (c, p), v in prof.items()}
    ent = defaultdict(list)
    for f in F: ent[("entrant" if f["entrant"] else "incumbent")].append(f)
    d["entrant_vs_incumbent"] = {k: {"n": len(v), "raw_mean": float(np.mean([x["raw"] for x in v])), "survival": float(np.mean([x["alive"] for x in v]))} for k, v in ent.items()}
    # survival curve of initial firms by month, by controller
    surv = {}
    for c in ("policy", "best_npv", "easy_first", "cash_preserving"):
        v = [f for f in F if f["controller"] == c and not f["entrant"]]
        surv[c] = [float(np.mean([(f["death"] is None or f["death"] > m) for f in v])) for m in months]
    d["survival_curve_initial"] = surv
    # starts / learned behaviour
    st = defaultdict(list)
    for r in rs:
        for s in r["starts"]: st[s[0]].append(s)
    beh = {}
    for c, v in st.items():
        v = np.array(v, dtype=object)
        n_eps_firms = len(by[c])
        beh[c] = {"starts": len(v), "starts_per_firm": len(v) / max(1, n_eps_firms),
                  "frontier_share": float(np.mean(v[:, 3] == 0)), "tier_share": [float(np.mean(v[:, 4] == t)) for t in range(3)],
                  "sector_share": [float(np.mean(v[:, 2] == s)) for s in range(3)],
                  "mean_true_difficulty": float(np.mean(v[:, 6].astype(float))), "mean_signal": float(np.mean(v[:, 5].astype(float))),
                  "first_start_month": float(np.mean([min(x[7] for x in v if True)]))}
    d["behaviour"] = beh
    vb = defaultdict(lambda: defaultdict(int))
    for r in rs:
        for c, vv in r["verbs"].items():
            for k, n in vv.items(): vb[c][k] += n
    d["verbs"] = {c: {k: n / sum(v.values()) for k, n in v.items()} for c, v in vb.items()}
    # gates by complexity
    g = [x for r in rs for x in r["gates"]]
    tech = [x for x in g if x["gate"] == "technical"]; com = [x for x in g if x["gate"] == "commercial"]
    bins = [0, 0.4, 0.55, 0.7, 0.85, 2]
    d["tech_pass_by_difficulty"] = [{"lo": bins[i], "hi": bins[i+1], "n": len([x for x in tech if bins[i] <= x["d"] < bins[i+1]]),
                                     "pass": float(np.mean([x["ok"] for x in tech if bins[i] <= x["d"] < bins[i+1]])) if [x for x in tech if bins[i] <= x["d"] < bins[i+1]] else None,
                                     "p_mean": float(np.mean([x["p"] for x in tech if bins[i] <= x["d"] < bins[i+1]])) if [x for x in tech if bins[i] <= x["d"] < bins[i+1]] else None} for i in range(len(bins)-1)]
    d["commercial_by_sector"] = [{"sector": SECT[s], "n": len([x for x in com if x["sector"] == s]),
                                  "pass": float(np.mean([x["ok"] for x in com if x["sector"] == s])) if [x for x in com if x["sector"] == s] else None,
                                  "rel_fail": int(sum(1 for x in com if x["sector"] == s and not x["rel_ok"])),
                                  "val_fail": int(sum(1 for x in com if x["sector"] == s and x["rel_ok"] and not x["val_ok"]))} for s in range(3)]
    # first mover
    fm = []
    for r in rs:
        cust = {(f["id"], int(o)): n for f in r["firms"] for o, n in f["customers_by_opp"].items()}
        byopp = defaultdict(list)
        for fid, opp, m in r["launches"]: byopp[opp].append((m, fid))
        for opp, L in byopp.items():
            L = sorted(set(L))
            if len(L) < 2: continue
            tot = sum(cust.get((fid, opp), 0) for _, fid in L)
            if tot == 0: continue
            fm.append(cust.get((L[0][1], opp), 0) / tot)
    d["first_mover_share"] = fm; d["first_mover_n_launchers_mean"] = None
    out[name] = d
Path("runs/paper/stats.json").write_text(json.dumps(out, default=float))
for name, d in out.items():
    print("==", name, "n", d["n"])
    print(" welfare", np.mean(d["total_welfare"]), "adopt", np.mean(d["total_adoptions"]), "churn", np.mean(d["total_churn"]), "min_served", np.mean(d["min_served"]), "entries", np.mean(d["entries"]), "exits", np.mean(d["exits"]))
    print(" bullwhip median", np.median(d["bullwhip_ratio"]) if d["bullwhip_ratio"] else None, "n", len(d["bullwhip_ratio"]), "order months share", d["order_months_share"])
    for c, v in d["firms_by_controller"].items(): print(" ", c, {k: round(x, 2) for k, x in v.items()})
    for k, v in sorted(d["by_profile"].items()): print("  prof", k, {a: round(b, 2) for a, b in v.items()})
    print(" entrant", d["entrant_vs_incumbent"])
    for c, v in d["behaviour"].items(): print("  beh", c, {k: (round(x, 2) if isinstance(x, float) else [round(y, 2) for y in x] if isinstance(x, list) else x) for k, x in v.items()})
    for c, v in d["verbs"].items(): print("  verbs", c, {k: round(x, 3) for k, x in sorted(v.items())})
    print(" tech by d", [(b["lo"], b["n"], b["pass"] and round(b["pass"], 2), b["p_mean"] and round(b["p_mean"], 2)) for b in d["tech_pass_by_difficulty"]])
    print(" comm by sector", d["commercial_by_sector"])
    print(" first mover share mean", np.mean(d["first_mover_share"]) if d["first_mover_share"] else None, "n", len(d["first_mover_share"]))
    print(" util peak mean", max(d["utilization"]["mean"]), "price max mean", max(d["compute_price"]["mean"]), "hhi end", d["hhi"]["mean"][-1], "firms end", d["n_firms"]["mean"][-1])
    print(" surv curves end", {c: round(v[-1], 2) for c, v in d["survival_curve_initial"].items()})
