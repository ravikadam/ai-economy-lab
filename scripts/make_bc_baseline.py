"""Imitation-only baseline (revision plan R1, item 2).

For each full-recipe run, take the network saved after imitation and before any PPO update
(checkpoints/bc_init.pt), choose its action mode (sampled or greedy) on the 40 practice markets with the same
DevEvaluator used during training, and save it as runs/v2bc-seedN/best.pt so scripts/test_v2.py can test it.

Usage: python -m scripts.make_bc_baseline [--workers 3]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from training.checkpoints import load_checkpoint, save_checkpoint
    from training.dev_eval import DevEvaluator
    dev = None
    for run in sorted(glob.glob("runs/v2-seed*")):
        seed = Path(run).name.split("-")[1].replace("seed", "")
        net, cfg, ck = load_checkpoint(Path(run) / "checkpoints" / "bc_init.pt")
        if dev is None:
            t = ck.get("train", {})
            dev = DevEvaluator(cfg, t.get("hidden", 256), t.get("emb", 64), a.workers)
        ev = dev(net)
        out = Path(f"runs/v2bc-seed{seed}")
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out / "best.pt", net, cfg, ck.get("train", {}),
                        {"iter": 0, "env_steps": 0, "eval": ev, "source": str(Path(run) / "checkpoints/bc_init.pt")})
        print(f"{out}: mode {ev['mode']} practice score {ev['score']:.2f} (sampled {ev['sampled']:.2f}, "
              f"greedy {ev['greedy']:.2f}, rule {ev['rule']:.2f})", flush=True)
    dev.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
