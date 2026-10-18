"""Run a trained policy for selected seats of a MultiFirmEnv (eval, replay, game rivals, coach)."""
from __future__ import annotations

import numpy as np
import torch

from agents.policy import ActorCritic


def to_t(x, device, dtype=None):
    return torch.as_tensor(x, device=device, dtype=dtype)


class PolicyRunner:
    def __init__(self, net: ActorCritic, device: str = "cpu", deterministic: bool = False, seed: int = 0):
        self.net = net.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def act(self, batch, seats: list[int]) -> dict[int, dict]:
        """Return {seat: {"action": idx tuple, "value": V(s), "verb_probs": [...], "logp": ...}}."""
        if not seats:
            return {}
        idx = np.array(seats)
        d = self.device
        out = self.net(to_t(batch.obs[idx], d), to_t(batch.verb_mask[idx], d), to_t(batch.cand_mask[idx], d),
                       to_t(batch.slot_mask[idx], d), to_t(batch.tier_mask[idx], d),
                       deterministic=self.deterministic, generator=self.gen)
        acts = out.actions.cpu().numpy()
        vals = out.value.cpu().numpy()
        vp = out.verb_probs.cpu().numpy()
        lp = out.logp.cpu().numpy()
        return {s: {"action": tuple(int(x) for x in acts[i]), "value": float(vals[i]),
                    "verb_probs": vp[i].tolist(), "logp": float(lp[i])} for i, s in enumerate(seats)}

    @torch.no_grad()
    def evaluate_action(self, batch, seat: int, action: tuple) -> dict:
        """Log-prob of an arbitrary (e.g. human) action under the policy, plus V(s)."""
        d = self.device
        sl = slice(seat, seat + 1)
        out = self.net(to_t(batch.obs[sl], d), to_t(batch.verb_mask[sl], d), to_t(batch.cand_mask[sl], d),
                       to_t(batch.slot_mask[sl], d), to_t(batch.tier_mask[sl], d),
                       actions=to_t(np.array([action]), d, torch.long))
        return {"logp": float(out.logp[0]), "value": float(out.value[0]), "verb_probs": out.verb_probs[0].tolist()}
