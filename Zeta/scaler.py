# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/scaler.py — Autonomous Scaling Engine
============================================
This module manages automatic scaling of the Zeta engine based on
internal state metrics: entropy, error rate, orbit saturation.

Scaling mechanisms (all purely Z_13[eta]):
    1. Multi-layer Witt head — all PREC layers active in forward
    2. S3 conjugate orbits — 6 independent T3^n · S3_g evolution paths
    3. Dynamic MERA depth — log_2(L) levels instead of fixed 3
    4. Hensel precision lift — triggered by entropy barrier

The scaler autonomously selects which mechanisms to activate.
No external hyperparameters.  No Euclidean geometry.

Mathematical context:
    Section VI  (Autonómia ako štrukturálna konštanta)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ZetaConfig
from .ring import Z13


class ZetaScaler:
    """Autonomous scaling state machine.  Tracks engine health and
    activates scaling mechanisms without external intervention."""

    def __init__(self, model) -> None:
        self.model = model
        self.s3_orbit = -1          # -1 = disabled, 0..5 = active S3 conjugate
        self.mera_dynamic = False   # auto-depth MERA
        self.witt_full = False      # use all PREC layers in forward
        self.error_history: list = []
        self.entropy_history: list = []
        self.step = 0

    def observe(self, hall: int, entropy: int, delta_norm: int) -> None:
        """Record metrics and decide scaling actions."""
        self.error_history.append(hall)
        self.entropy_history.append(entropy)
        if len(self.error_history) > 24:
            self.error_history.pop(0)
        if len(self.entropy_history) > 24:
            self.entropy_history.pop(0)
        self.step += 1

    def decide(self) -> dict:
        """Autonomous scaling decisions based on 24-step history."""
        actions = {}
        avg_err = sum(self.error_history) // max(1, len(self.error_history))
        avg_ent = sum(self.entropy_history) // max(1, len(self.entropy_history))

        # 1. Witt full-layer: activate when entropy > 6
        if avg_ent > 6 and not self.witt_full:
            self.witt_full = True
            actions['witt_full'] = True

        # 2. S3 orbit: cycle through conjugates when error rate > 50%
        if avg_err > 12 and len(self.error_history) >= 12:  # >50% of 24 steps
            self.s3_orbit = (self.s3_orbit + 1) % 6
            actions['s3_orbit'] = self.s3_orbit

        # 3. Dynamic MERA: activate when sequence length > 64
        if hasattr(self.model, 'ctx') and self.model.ctx > 64:
            self.mera_dynamic = True
            actions['mera_dynamic'] = True

        # 4. Hensel lift: model handles via EntropyMonitor
        #    (scaler only reports recommendation)
        if avg_ent >= 10:
            actions['recommend_lift'] = True

        return actions

    def reset_orbit(self) -> None:
        """Reset S3 orbit when goal is reached."""
        self.s3_orbit = -1
