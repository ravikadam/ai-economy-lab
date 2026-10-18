"""Shared-policy masked PPO over vectorised, seeded worlds (spec §9 Phase B/C).

Usage:  python -m training.train --config config/train.yaml [--device mps|cpu|auto] [--steps N]

One policy controls every "policy" seat in every world (parameter sharing; this
is NOT a claim of genuine self-play equilibrium). Some seats are rule-based
baselines and some are frozen historical checkpoints, to reduce
non-stationarity. Trains on the shaped reward, logs the raw economic reward
separately, checkpoints every N env steps and early-stops on held-out dev seeds.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict

from economy.config import load_config
from training.checkpoints import build_net, link_latest, save_checkpoint, write_json
from training.vec import CTRL_CODES, SubprocWorlds


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str = "config/base.yaml"
    run_name: str = "ppo"
    total_env_steps: int = 200_000          # world-months
    num_workers: int = 8
    worlds_per_worker: int = 4
    rollout_len: int = 60
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.05
    hidden: int = 256
    emb: int = 64
    device: str = "auto"                    # training device: auto | mps | cpu
    rollout_device: str = "cpu"             # inference during rollouts (small batches: CPU is usually faster)
    controller_mix: dict[str, float] = {"policy": 0.6, "heuristic:best_npv": 0.15, "heuristic:easy_first": 0.1,
                                        "heuristic:cash_preserving": 0.1, "frozen": 0.05}
    frozen_snapshot_every: int = 10         # iterations
    frozen_pool_size: int = 5
    checkpoint_every: int = 50_000          # env steps
    eval_every: int = 5                     # iterations
    eval_seeds: list[int] = [1000, 1001, 1002, 1003, 1004]
    patience: int = 8                       # evals without improvement before early stop
    seed: int = 0
    train_seed_base: int = 1_000_000
    bc_seed_base: int = 3_000_000
    bc_teacher: str | None = "best_npv"     # behaviour-cloning warm start (None = pure PPO from scratch)
    bc_episodes: int = 64
    bc_epochs: int = 6
    value_warmup_iters: int = 3             # value-only updates after BC so early advantages are sane


def pick_device(name: str) -> str:
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        print("[train] MPS not available, falling back to CPU")
        return "cpu"
    return name


class RunningStd:
    """Running std of discounted returns, used to scale rewards (VecNormalize-style)."""

    def __init__(self, shape, gamma):
        self.ret = np.zeros(shape)
        self.gamma = gamma
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def update(self, r, done, valid):
        self.ret = self.ret * self.gamma * (~done) + r
        x = self.ret[valid]
        if x.size == 0:
            return
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


def ppo_update(net, opt, data, tc: TrainConfig, device: str, pg_coef: float = 1.0) -> dict:
    N = data["obs"].shape[0]
    t = {k: torch.as_tensor(v, device=device) for k, v in data.items()}
    adv = t["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    stop = False
    for _ in range(tc.epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, tc.minibatch):
            mb = perm[i:i + tc.minibatch]
            out = net(t["obs"][mb], t["verb_mask"][mb], t["cand_mask"][mb], t["slot_mask"][mb], t["tier_mask"][mb],
                      actions=t["actions"][mb])
            ratio = torch.exp(out.logp - t["logp"][mb])
            a = adv[mb]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - tc.clip, 1 + tc.clip) * a).mean()
            v_clip = t["values"][mb] + torch.clamp(out.value - t["values"][mb], -tc.clip * 5, tc.clip * 5)
            v_loss = 0.5 * torch.max((out.value - t["returns"][mb]) ** 2, (v_clip - t["returns"][mb]) ** 2).mean()
            ent = out.entropy.mean()
            loss = pg_coef * (pg - tc.ent_coef * ent) + tc.vf_coef * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if pg_coef == 0.0:      # value warm-up: leave the (BC) policy and shared trunk untouched
                for name, prm in net.named_parameters():
                    if not name.startswith("value_head"):
                        prm.grad = None
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            stats["pg_loss"].append(pg.item()); stats["v_loss"].append(v_loss.item())
            stats["entropy"].append(ent.item()); stats["approx_kl"].append(kl)
            stats["clip_frac"].append(((ratio - 1).abs() > tc.clip).float().mean().item())
            if kl > tc.target_kl:
                stop = True
                break
        if stop:
            break
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed; also shifts training and imitation market seeds so runs are independent")
    args = ap.parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    tc = TrainConfig.model_validate(raw or {})
    if args.device: tc.device = args.device
    if args.steps: tc.total_env_steps = args.steps
    if args.workers: tc.num_workers = args.workers
    if args.run_name: tc.run_name = args.run_name
    if args.seed is not None and args.seed != tc.seed:
        tc.seed = args.seed
        tc.train_seed_base += 10_000_000 * args.seed
        tc.bc_seed_base += 10_000_000 * args.seed

    random.seed(tc.seed); np.random.seed(tc.seed); torch.manual_seed(tc.seed)
    cfg = load_config(tc.scenario)
    device = pick_device(tc.device)
    rdev = pick_device(tc.rollout_device)
    run_dir = Path("runs") / f"{tc.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "train_config.json", tc.model_dump())
    write_json(run_dir / "scenario.json", cfg.model_dump())
    link_latest(run_dir)
    print(f"[train] run={run_dir} device={device} rollout_device={rdev} scenario={cfg.name}")

    net = build_net(cfg, tc.hidden, tc.emb).to(device)
    if tc.bc_teacher:
        from training.bc import behaviour_clone
        bc = behaviour_clone(net, cfg, device, teacher=tc.bc_teacher, episodes=tc.bc_episodes,
                             workers=tc.num_workers, epochs=tc.bc_epochs, seed_base=tc.bc_seed_base)
        write_json(run_dir / "bc.json", bc)
        save_checkpoint(run_dir / "checkpoints" / "bc_init.pt", net, cfg, tc.model_dump(), {"bc": bc})
    act_net = copy.deepcopy(net).to(rdev).eval()
    opt = torch.optim.Adam(net.parameters(), lr=tc.lr, eps=1e-5)
    frozen_pool: list[dict] = []
    frozen_net = copy.deepcopy(act_net)
    gen = torch.Generator(device=rdev); gen.manual_seed(tc.seed)

    mix = dict(tc.controller_mix)
    vec = SubprocWorlds(cfg, mix, tc.num_workers, tc.worlds_per_worker, tc.train_seed_base)
    W, S, T = vec.n, cfg.builders.max_firms, tc.rollout_len
    batch = vec.reset()
    D = batch.obs.shape[-1]
    V, K, P, Ti = batch.verb_mask.shape[-1], batch.cand_mask.shape[-1], batch.slot_mask.shape[-1], \
        batch.tier_mask.shape[-1]
    rstd = RunningStd((W, S), tc.gamma)

    from training.evaluate import evaluate_focal   # after spawn-safe imports
    metrics_f = open(run_dir / "metrics.jsonl", "w")
    env_steps, it, best_eval, bad_evals, next_ckpt = 0, 0, -1e9, 0, tc.checkpoint_every
    ep_window: list[dict] = []
    t_start = time.time()
    POL, FRZ = CTRL_CODES["policy"], CTRL_CODES["frozen"]

    try:
        while env_steps < tc.total_env_steps:
            it += 1
            t_it = time.time()
            buf = {"obs": np.zeros((T, W, S, D), np.float32), "verb_mask": np.zeros((T, W, S, V), bool),
                   "cand_mask": np.zeros((T, W, S, K), bool), "slot_mask": np.zeros((T, W, S, V, P), bool),
                   "tier_mask": np.zeros((T, W, S, V, Ti), bool), "actions": np.zeros((T, W, S, 6), np.int64),
                   "logp": np.zeros((T, W, S), np.float32), "values": np.zeros((T, W, S), np.float32),
                   "rew": np.zeros((T, W, S), np.float32), "done": np.zeros((T, W, S), bool),
                   "valid": np.zeros((T, W, S), bool)}
            raw_sum = 0.0
            t_inf = t_env = 0.0
            for step in range(T):
                valid = batch.ctrl == POL
                frozen = batch.ctrl == FRZ
                buf["obs"][step] = batch.obs; buf["verb_mask"][step] = batch.verb_mask
                buf["cand_mask"][step] = batch.cand_mask; buf["slot_mask"][step] = batch.slot_mask
                buf["tier_mask"][step] = batch.tier_mask; buf["valid"][step] = valid
                actions = [dict() for _ in range(W)]
                t0 = time.time()
                for sel, model in ((valid, act_net), (frozen, frozen_net)):
                    wi, si = np.nonzero(sel)
                    if wi.size == 0:
                        continue
                    with torch.no_grad():
                        out = model(torch.as_tensor(batch.obs[wi, si], device=rdev),
                                    torch.as_tensor(batch.verb_mask[wi, si], device=rdev),
                                    torch.as_tensor(batch.cand_mask[wi, si], device=rdev),
                                    torch.as_tensor(batch.slot_mask[wi, si], device=rdev),
                                    torch.as_tensor(batch.tier_mask[wi, si], device=rdev), generator=gen)
                    a = out.actions.cpu().numpy()
                    if model is act_net:
                        buf["actions"][step, wi, si] = a
                        buf["logp"][step, wi, si] = out.logp.cpu().numpy()
                        buf["values"][step, wi, si] = out.value.cpu().numpy()
                    for k in range(wi.size):
                        actions[wi[k]][int(si[k])] = tuple(int(x) for x in a[k])
                t1 = time.time()
                raw, shaped, done, episodes, batch = vec.step(actions)
                t_env += time.time() - t1
                t_inf += t1 - t0
                rstd.update(shaped, done, valid)
                buf["rew"][step] = shaped / rstd.std
                buf["done"][step] = done
                raw_sum += float(raw[valid].sum())
                ep_window += [e for ep in episodes for e in ep]
                env_steps += W
            # bootstrap values for the state after the rollout
            with torch.no_grad():
                last_v = act_net.value(torch.as_tensor(batch.obs.reshape(W * S, D), device=rdev)).cpu().numpy()
            last_v = last_v.reshape(W, S) * (batch.ctrl == POL)
            # GAE per seat trajectory
            adv = np.zeros((T, W, S), np.float32)
            gae = np.zeros((W, S), np.float32)
            for t in reversed(range(T)):
                if t == T - 1:
                    nv = last_v
                    nvalid = batch.ctrl == POL
                else:
                    nv = buf["values"][t + 1]
                    nvalid = buf["valid"][t + 1]
                nonterm = (~buf["done"][t]) & nvalid
                delta = buf["rew"][t] + tc.gamma * nv * nonterm - buf["values"][t]
                gae = delta + tc.gamma * tc.gae_lambda * nonterm * gae
                gae = np.where(buf["valid"][t], gae, 0.0)
                adv[t] = gae
            returns = adv + buf["values"]
            m = buf["valid"]
            data = {"obs": buf["obs"][m], "verb_mask": buf["verb_mask"][m], "cand_mask": buf["cand_mask"][m],
                    "slot_mask": buf["slot_mask"][m], "tier_mask": buf["tier_mask"][m], "actions": buf["actions"][m],
                    "logp": buf["logp"][m], "values": buf["values"][m], "adv": adv[m], "returns": returns[m]}
            t_up = time.time()
            net.train()
            stats = ppo_update(net, opt, data, tc, device, pg_coef=0.0 if it <= tc.value_warmup_iters else 1.0)
            t_up = time.time() - t_up
            act_net.load_state_dict({k: v.to(rdev) for k, v in net.state_dict().items()})

            if it % tc.frozen_snapshot_every == 0:
                frozen_pool.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                frozen_pool = frozen_pool[-tc.frozen_pool_size:]
            if frozen_pool:
                frozen_net.load_state_dict(random.choice(frozen_pool))

            pol_eps = [e for e in ep_window if e["controller"] == "policy"]
            heur_eps = [e for e in ep_window if e["controller"].startswith("heuristic")]
            rec = {"iter": it, "env_steps": env_steps, "samples": int(m.sum()), "time": time.time() - t_start,
                   "fps": W * T / (time.time() - t_it), "t_env": t_env, "t_inference": t_inf, "t_update": t_up,
                   "device": device, "rollout_raw_reward_sum": raw_sum, **stats}
            if pol_eps:
                rec["policy_raw_return"] = float(np.mean([e["raw_return"] for e in pol_eps]))
                rec["policy_shaped_return"] = float(np.mean([e["shaped_return"] for e in pol_eps]))
                rec["policy_survival"] = float(np.mean([e["survived"] for e in pol_eps]))
                rec["policy_launches"] = float(np.mean([e["launches"] for e in pol_eps]))
                rec["policy_tech_fail"] = float(np.mean([e["tech_failures"] for e in pol_eps]))
            if heur_eps:
                rec["heuristic_raw_return"] = float(np.mean([e["raw_return"] for e in heur_eps]))
                rec["heuristic_survival"] = float(np.mean([e["survived"] for e in heur_eps]))
            if pol_eps or heur_eps:
                ep_window = []
            if it % tc.eval_every == 0:
                net.eval()
                ev = evaluate_focal(act_net, cfg, tc.eval_seeds, focal="policy", device=rdev)
                rec["eval_raw_return"] = ev["raw_return_mean"]
                rec["eval_survival"] = ev["survival"]
                if ev["raw_return_mean"] > best_eval:
                    best_eval, bad_evals = ev["raw_return_mean"], 0
                    save_checkpoint(run_dir / "best.pt", net, cfg, tc.model_dump(),
                                    {"iter": it, "env_steps": env_steps, "eval": ev})
                else:
                    bad_evals += 1
            if env_steps >= next_ckpt:
                save_checkpoint(run_dir / "checkpoints" / f"ckpt_{env_steps:09d}.pt", net, cfg, tc.model_dump(),
                                {"iter": it, "env_steps": env_steps})
                next_ckpt += tc.checkpoint_every
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            msg = (f"[it {it:4d}] steps={env_steps:>8d} fps={rec['fps']:7.0f} "
                   f"pol_R={rec.get('policy_raw_return', float('nan')):7.2f} "
                   f"heur_R={rec.get('heuristic_raw_return', float('nan')):7.2f} "
                   f"surv={rec.get('policy_survival', float('nan')):.2f} ent={stats['entropy']:.2f} "
                   f"kl={stats['approx_kl']:.3f} env={t_env:.1f}s inf={t_inf:.1f}s upd={t_up:.1f}s")
            if "eval_raw_return" in rec:
                msg += f" | eval_R={rec['eval_raw_return']:.2f} best={best_eval:.2f}"
            print(msg, flush=True)
            if bad_evals >= tc.patience:
                print(f"[train] early stop: no held-out improvement in {tc.patience} evals")
                break
    finally:
        vec.close()
        metrics_f.close()
    save_checkpoint(run_dir / "final.pt", net, cfg, tc.model_dump(), {"iter": it, "env_steps": env_steps})
    if not (run_dir / "best.pt").exists():
        save_checkpoint(run_dir / "best.pt", net, cfg, tc.model_dump(), {"iter": it, "env_steps": env_steps})
    print(f"[train] done: {env_steps} env steps in {time.time() - t_start:.0f}s -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
