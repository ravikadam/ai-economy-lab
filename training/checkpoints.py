"""Checkpoint save/load. A checkpoint is self-describing: weights + scenario + layout + train config."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from agents.observation import layout
from agents.policy import ActorCritic
from economy.config import ScenarioConfig


def build_net(cfg: ScenarioConfig, hidden: int = 256, emb: int = 64) -> ActorCritic:
    return ActorCritic(layout(cfg), cfg.n_models, cfg.n_tiers, hidden=hidden, emb=emb)


def save_checkpoint(path: Path, net: ActorCritic, cfg: ScenarioConfig, train_cfg: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "scenario": cfg.model_dump(), "train": train_cfg, "meta": meta}, path)


def resolve(path: str | Path) -> Path:
    """Accept a .pt file, a run directory, or runs/latest; return a .pt file.
    If runs/latest does not exist (fresh clone), fall back to the bundled results/policy."""
    p = Path(path)
    if not p.exists() and p.as_posix().rstrip("/") == "runs/latest" and Path("results/policy").is_dir():
        p = Path("results/policy")
    if p.is_dir():
        for name in ("best.pt", "final.pt"):
            if (p / name).exists():
                return p / name
        cks = sorted((p / "checkpoints").glob("ckpt_*.pt"))
        if cks:
            return cks[-1]
        raise FileNotFoundError(f"no checkpoint in {p}")
    return p


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ActorCritic, ScenarioConfig, dict]:
    ck = torch.load(resolve(path), map_location=device, weights_only=False)
    cfg = ScenarioConfig.model_validate(ck["scenario"])
    t = ck.get("train", {})
    net = build_net(cfg, t.get("hidden", 256), t.get("emb", 64))
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, cfg, ck


def link_latest(run_dir: Path) -> None:
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(run_dir.name, latest)
    except OSError:
        (run_dir.parent / "LATEST").write_text(run_dir.name)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
