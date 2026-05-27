# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/orbit.py  —  T3 Orbit Planner
====================================
This module plans trajectories in the T3 orbit space.  Given a current
state and a goal state, it finds the smallest N ∈ {0..167} such that
T3^N * state = goal.  All 168 T3 powers are tested simultaneously via
vectorized broadcast — O(168 * 3) = O(1) per query.

Mathematical context:
    Section 2.2  (The Sole Matrix T3 ∈ SL(3,Z))

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, T3_POW
from .ntt import NTT


class OrbitPlanner:

    @staticmethod
    def plan_orbit_vec(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        dev = state.device
        all_s = (T3_POW.to(dev) @ state.unsqueeze(-1)).squeeze(-1) % P
        match = (all_s == goal.to(dev).unsqueeze(0)).all(-1)
        idx = torch.where(match)[0]
        return idx[0] if idx.numel() > 0 else torch.tensor(0, dtype=torch.long)

    @staticmethod
    def plan_orbit(state: torch.Tensor, goal: torch.Tensor) -> int:
        return int(OrbitPlanner.plan_orbit_vec(state, goal).item())

    @staticmethod
    def orbit_spectrum(state: torch.Tensor) -> torch.Tensor:
        dev = state.device
        orbit = (T3_POW.to(dev) @ state.unsqueeze(-1)).squeeze(-1) % P
        N = NTT.best_size(ORD)
        return NTT.ntt(orbit[:N], N)

    @staticmethod
    def closure_check(state: torch.Tensor, N: int) -> bool:
        from .spectral import t3n
        return bool(torch.all(t3n(t3n(state, N), ORD-N) == state))
