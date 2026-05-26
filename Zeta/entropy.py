# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/entropy.py — Entropic Monitoring and Hensel Catastrophe Trigger
=====================================================================
When the entropy of the self-state hits the barrier S_max,
the system triggers a Witt lift — not retraining, but a dimensional
jump to higher Hensel precision.

Mathematical context:
    Section I.3  (Dlhodobá pamäť ako T3 orbita)
    Section IV.3  (Witt generovanie nových konceptov)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P
from .ring import Z13
from .witt import WittVector


class EntropyMonitor:
    """Monitors GQM entropy and triggers Witt lift on catastrophe."""

    def __init__(self, threshold: int = 10):
        self.S_max = int(threshold) % P

    def check(self, entropy: int) -> bool:
        """S_G >= S_max ? → Hensel catastrophe triggered."""
        return (int(entropy) % P) >= self.S_max

    def lift_if_needed(self, state, prec: int) -> torch.Tensor:
        """Return Witt vector at precision prec or prec+1.
        state: GQMState or (..., 3) tensor."""
        from .gqm import GQMState
        if isinstance(state, GQMState):
            S = state.entropy()
            rep = state.coeffs[0]
        else:
            flat = state.reshape(-1, 3)
            S = int((Z13.norm(flat) == 0).sum().item())
            rep = flat[0]
        if self.check(S):
            return WittVector.from_ring(rep, prec + 1)
        return WittVector.from_ring(rep, prec)
