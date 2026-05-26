# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/goal.py — Cieľové správanie a orbitálne plánovanie v Z_13[eta]
====================================================================
This module implements goal-directed behavior purely algebraically.

Distance metric:
    d_goal(|Ψ⟩, |C⟩) = N(P1 · (|Ψ⟩ − |C⟩))  mod 13

Planning searches the T3 orbit {T3^n · |Ψ⟩} for the smallest n
and selects the channel:
    d == 0        → direct T3 jump
    d < 6         → P1 dominant shift
    d < 10        → P23 subdominant exploration
    d >= 10       → Witt lift required (Hensel catastrophe)

All operations are vectorized over the 168 orbit elements.
Zero Python loops over sequence positions.

Mathematical context:
    Section I.2  (Cieľové správanie ako spektrálny posun)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, T3_POW, P1_MAT
from .ring import Z13
from .spectral import proj


class ZetaGoalPlanner:
    """Vectorized goal planning over the T3 orbit."""

    @staticmethod
    def distance(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """d_goal = N(P1 · (state − goal)) mod 13.
        state, goal: (..., 3).  Returns scalar mod 13."""
        dev = state.device
        diff = Z13.sub(state, goal) % P
        p1_diff = proj(diff, P1_MAT.to(dev))
        return Z13.norm(p1_diff)  # (...) mod 13

    @staticmethod
    def plan(state: torch.Tensor, goal: torch.Tensor, max_n: int = 168) -> dict:
        """Search T3^n · state for best approach to goal.
        state: (K, 3), goal: (K, 3).  Fully vectorized over n."""
        dev = state.device
        K = state.shape[0]
        M = T3_POW[:max_n].to(dev)  # (max_n, 3, 3)
        # evolved[n, k, :] = T3^n · state[k]
        evolved = torch.einsum('nrc,kc->nkr', M, state.to(dev)) % P  # (max_n, K, 3)
        goal_exp = goal.to(dev).unsqueeze(0).expand(max_n, -1, -1)   # (max_n, K, 3)
        diff = (evolved - goal_exp) % P                               # (max_n, K, 3)
        # P1 projection on flattened diff
        p1_flat = proj(diff.reshape(-1, 3), P1_MAT.to(dev))         # (max_n*K, 3)
        p1_diff = p1_flat.reshape(max_n, K, 3)
        norms = Z13.norm(p1_diff)                                     # (max_n, K)
        d_vec = norms.sum(1) % P                                     # (max_n,)

        # Direct hit
        direct = (d_vec == 0).nonzero(as_tuple=True)[0]
        if direct.numel() > 0:
            n = int(direct[0].item())
            return {'n': n, 'd': 0, 'channel': 'direct', 'lift': False, 'state': evolved[n]}

        # P1 candidates: d < 6
        p1_mask = d_vec < 6
        if p1_mask.any():
            masked = torch.where(p1_mask, d_vec, torch.full_like(d_vec, 999))
            n = int(masked.argmin().item())
            d_val = int(d_vec[n].item())
            return {'n': n, 'd': d_val, 'channel': 'P1', 'lift': False, 'state': evolved[n]}

        # P23 candidates: 6 <= d < 10
        p23_mask = (d_vec >= 6) & (d_vec < 10)
        if p23_mask.any():
            masked = torch.where(p23_mask, d_vec, torch.full_like(d_vec, 999))
            n = int(masked.argmin().item())
            d_val = int(d_vec[n].item())
            return {'n': n, 'd': d_val, 'channel': 'P23', 'lift': False, 'state': evolved[n]}

        # Fallback: closest, but recommend Witt lift
        n = int(d_vec.argmin().item())
        d_val = int(d_vec[n].item())
        return {'n': n, 'd': d_val, 'channel': 'none', 'lift': True, 'state': evolved[n]}
