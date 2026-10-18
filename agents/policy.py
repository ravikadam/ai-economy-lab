"""Shared actor-critic with an autoregressive, masked, factorised action head.

Sampling order: verb -> (candidate | slot) -> model -> tier, plus an
independent reserve head. Candidate and slot logits come from per-item
encoders (permutation-equivariant), so the policy scores opportunities by their
features rather than by list position. Risk-profile parameters are part of the
observation, so one shared network can behave differently per firm profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.masks import N_RESERVE
from agents.observation import ObsLayout
from economy.actions import N_VERBS, Verb

NEG = -1e8
START = int(Verb.START)
SLOT_VERB_IDX = torch.tensor([int(v) for v in (Verb.PILOT, Verb.LAUNCH, Verb.SCALE, Verb.ABANDON,
                                               Verb.RETRY, Verb.PAUSE)])
MODEL_TIER_VERBS = (int(Verb.START), int(Verb.RETRY))


@dataclass
class PolicyOutput:
    actions: torch.Tensor     # [B, 6] (verb, cand, slot, model, tier, reserve)
    logp: torch.Tensor        # [B]
    entropy: torch.Tensor     # [B]
    value: torch.Tensor       # [B]
    verb_probs: torch.Tensor  # [B, V]


def _mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.Tanh()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, logits, torch.full_like(logits, NEG))


def _cat_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits, -1)
    p = lp.exp()
    return -(torch.where(mask, p * lp, torch.zeros_like(p))).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, lay: ObsLayout, n_models: int, n_tiers: int, hidden: int = 256, emb: int = 64):
        super().__init__()
        self.lay = lay
        self.n_models, self.n_tiers = n_models, n_tiers
        self.glob = _mlp(lay.own + lay.market, hidden, hidden)
        self.cand_enc = _mlp(lay.cand, emb, emb)
        self.slot_enc = _mlp(lay.slot, emb, emb)
        self.trunk = nn.Sequential(nn.Tanh(), nn.Linear(hidden + 2 * emb, hidden), nn.Tanh())
        self.verb_head = nn.Linear(hidden, N_VERBS)
        self.cand_head = _mlp(emb + hidden, emb, 1)
        self.slot_head = _mlp(emb + hidden + N_VERBS, emb, 1)
        self.mt_ctx = nn.Linear(emb + hidden + N_VERBS, emb)
        self.model_head = nn.Linear(emb, n_models)
        self.tier_head = nn.Linear(emb + n_models, n_tiers)
        self.reserve_head = nn.Linear(hidden, N_RESERVE)
        self.value_head = _mlp(hidden, hidden // 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
        for head in (self.verb_head, self.model_head, self.tier_head, self.reserve_head):
            nn.init.orthogonal_(head.weight, 0.01)

    # ------------------------------------------------------------------
    def _split(self, obs: torch.Tensor):
        L = self.lay
        i = 0
        own = obs[:, i:i + L.own]; i += L.own
        slots = obs[:, i:i + L.n_slots * L.slot].reshape(-1, L.n_slots, L.slot); i += L.n_slots * L.slot
        cands = obs[:, i:i + L.n_cand * L.cand].reshape(-1, L.n_cand, L.cand); i += L.n_cand * L.cand
        market = obs[:, i:i + L.market]
        return own, slots, cands, market

    def encode(self, obs: torch.Tensor):
        own, slots, cands, market = self._split(obs)
        g = self.glob(torch.cat([own, market], -1))
        ce = self.cand_enc(cands)                     # [B, K, E]
        se = self.slot_enc(slots)                     # [B, P, E]
        cpres = cands[:, :, :1]
        spres = slots[:, :, :1]
        cpool = (ce * cpres).sum(1) / cpres.sum(1).clamp(min=1.0)
        spool = (se * spres).sum(1) / spres.sum(1).clamp(min=1.0)
        h = self.trunk(torch.cat([g, cpool, spool], -1))
        return h, ce, se

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        h, _, _ = self.encode(obs)
        return self.value_head(h).squeeze(-1)

    def forward(self, obs, verb_mask, cand_mask, slot_mask, tier_mask, actions=None, deterministic=False,
                generator: torch.Generator | None = None) -> PolicyOutput:
        """If `actions` is given, evaluate their log-prob; otherwise sample."""
        B = obs.shape[0]
        dev = obs.device
        h, ce, se = self.encode(obs)
        value = self.value_head(h).squeeze(-1)
        ar = torch.arange(B, device=dev)

        def pick(logits, mask, given):
            ml = masked_logits(logits, mask)
            if given is not None:
                a = given
            elif deterministic:
                a = ml.argmax(-1)
            else:
                probs = F.softmax(ml, -1)
                a = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            lp = F.log_softmax(ml, -1)[ar, a]
            return a, lp, _cat_entropy(ml, mask), ml

        g = actions if actions is not None else None
        # verb
        verb, lp_v, ent_v, ml_v = pick(self.verb_head(h), verb_mask, None if g is None else g[:, 0])
        verb_oh = F.one_hot(verb, N_VERBS).float()
        is_start = verb == START
        is_slot = torch.isin(verb, SLOT_VERB_IDX.to(dev))
        is_mt = (verb == MODEL_TIER_VERBS[0]) | (verb == MODEL_TIER_VERBS[1])
        # candidate (only meaningful for START)
        hK = h.unsqueeze(1).expand(-1, ce.shape[1], -1)
        cl = self.cand_head(torch.cat([ce, hK], -1)).squeeze(-1)
        cmask = cand_mask.clone()
        cmask[~is_start] = True                     # placeholder head when unused
        cand, lp_c, ent_c, _ = pick(cl, cmask, None if g is None else g[:, 1])
        # slot (only for slot verbs); mask conditioned on chosen verb
        hP = torch.cat([h, verb_oh], -1).unsqueeze(1).expand(-1, se.shape[1], -1)
        sl = self.slot_head(torch.cat([se, hP], -1)).squeeze(-1)
        smask = slot_mask[ar, verb].clone()
        smask[~is_slot] = True
        slot, lp_s, ent_s, _ = pick(sl, smask, None if g is None else g[:, 2])
        # model + tier conditioned on chosen target
        tgt = torch.where(is_start.unsqueeze(-1), ce[ar, cand], se[ar, slot])
        ctx = torch.tanh(self.mt_ctx(torch.cat([tgt, h, verb_oh], -1)))
        mmask = torch.ones(B, self.n_models, dtype=torch.bool, device=dev)
        model, lp_m, ent_m, _ = pick(self.model_head(ctx), mmask, None if g is None else g[:, 3])
        tmask = tier_mask[ar, verb].clone()
        tmask[~is_mt] = True
        m_oh = F.one_hot(model, self.n_models).float()
        tier, lp_t, ent_t, _ = pick(self.tier_head(torch.cat([ctx, m_oh], -1)), tmask, None if g is None else g[:, 4])
        rmask = torch.ones(B, N_RESERVE, dtype=torch.bool, device=dev)
        reserve, lp_r, ent_r, _ = pick(self.reserve_head(h), rmask, None if g is None else g[:, 5])

        f_s, f_sl, f_mt = is_start.float(), is_slot.float(), is_mt.float()
        logp = lp_v + f_s * lp_c + f_sl * lp_s + f_mt * (lp_m + lp_t) + lp_r
        ent = ent_v + f_s * ent_c + f_sl * ent_s + f_mt * (ent_m + ent_t) + 0.2 * ent_r
        acts = torch.stack([verb, cand, slot, model, tier, reserve], -1)
        return PolicyOutput(acts, logp, ent, value, F.softmax(ml_v, -1))
