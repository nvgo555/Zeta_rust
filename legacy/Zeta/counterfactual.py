# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/counterfactual.py  —  Counterfactual Trajectory Exploration
=================================================================
This module explores parallel trajectories in the T3 orbit space.  Each
branch applies a different T3^k to the same initial state.  The best
branch is selected via ultrametric distance in Z_13[eta].

Mathematical context:
    Section 9  (Learning via Buchberger and Nullstellensatz)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, T3_POW, _VAL_LUT, _VAL_MAX
from .gqm import GQMState


class CounterfactualBranch:

    @staticmethod
    def branch(state: torch.Tensor, n_branches: int = 6) -> torch.Tensor:
        k = min(n_branches, ORD)
        M = T3_POW[:k]
        return (M @ state.unsqueeze(-1)).squeeze(-1) % P

    @staticmethod
    def best_branch(branches: torch.Tensor, goal: torch.Tensor) -> int:
        dist = (branches[:, 0] - goal[0].item()) % P
        val = _VAL_LUT[dist.clamp(0, _VAL_MAX-1)]
        return int(val.argmax().item())

    @staticmethod
    def gqm_superposition(branches: torch.Tensor) -> GQMState:
        K = branches.shape[0]
        coeffs = torch.zeros(max(K, 8), 3, dtype=torch.long)
        coeffs[:K] = branches % P
        return GQMState.from_coeffs(coeffs)
