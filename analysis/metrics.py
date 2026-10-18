"""Turn replay logs and training logs into tidy DataFrames (spec §14)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def market_frame(replay: dict) -> pd.DataFrame:
    rows = []
    sectors = [s["name"] for s in replay["header"]["scenario"]["demand"]["sectors"]]
    models = [m["name"] for m in replay["header"]["scenario"]["technology"]["models"]]
    for m in replay["months"]:
        r = dict(m["market"])
        r["month"] = m["month"]
        for i, s in enumerate(sectors):
            r[f"confidence:{s}"] = r["sector_confidence"][i]
            r[f"adoption:{s}"] = r["sector_adoption"][i]
            r[f"revenue:{s}"] = r["sector_revenue"][i]
        for j, mn in enumerate(models):
            for i, s in enumerate(sectors):
                r[f"capability:{mn}:{s}"] = r["model_capability"][j][i]
            r[f"reliability:{mn}"] = r["model_reliability"][j]
            r[f"unit_price:{mn}"] = r["model_unit_price"][j]
        for g in ("technical", "commercial"):
            for k, lvl in enumerate(("low", "mid", "high")):
                r[f"{g}_pass:{lvl}"] = r[f"{g}_pass_by_complexity"][k]
                r[f"{g}_fail:{lvl}"] = r[f"{g}_fail_by_complexity"][k]
        r["n_launches"] = len(r.pop("launches", []))
        for k in ("model_capability", "model_reliability", "model_unit_price", "sector_confidence", "sector_adoption",
                  "sector_revenue", "technical_pass_by_complexity", "technical_fail_by_complexity",
                  "commercial_pass_by_complexity", "commercial_fail_by_complexity"):
            r.pop(k, None)
        rows.append(r)
    return pd.DataFrame(rows)


def firm_frame(replay: dict) -> pd.DataFrame:
    rows = []
    for m in replay["months"]:
        for f in m["firms"]:
            rows.append({"month": m["month"], "firm_id": f["firm_id"], "controller": f["controller"],
                         "alive": f["alive"], "cash": f["cash"], "debt": f["debt"], "net_worth": f["net_worth"],
                         "runway": f["runway"], "customers": f["customers"], "profit": f["profit"],
                         "launches": f["launches"], "value_created": f["cum_raw_reward"],
                         "n_projects": len(f["projects"])})
    return pd.DataFrame(rows)


def event_frame(replay: dict) -> pd.DataFrame:
    rows = []
    for m in replay["months"]:
        for e in m["events"]:
            p = {k: v for k, v in e["payload"].items() if not k.startswith("_")}
            rows.append({"month": e["month"], "kind": e["kind"], "public": e["payload"].get("_public", False),
                         "firm": p.get("firm"), "detail": json.dumps(p, default=str)})
    return pd.DataFrame(rows)


def decision_frame(replay: dict) -> pd.DataFrame:
    rows = []
    for m in replay["months"]:
        for d in m["decisions"]:
            a = d["action"]
            rows.append({"month": m["month"], "firm": d["firm"], "controller": d["controller"], "verb": a["verb"],
                         "opportunity": a["opportunity_id"], "slot": a["slot"], "model": a["model_id"],
                         "tier": a["tier"], "value_estimate": d.get("value_estimate"),
                         "cash": d["observation"]["cash"], "runway": d["observation"]["runway"],
                         "notes": "; ".join(n.get("message", n.get("coach", "")) for n in d.get("notes", []))})
    return pd.DataFrame(rows)


def training_frame(run_dir: str | Path) -> pd.DataFrame:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
