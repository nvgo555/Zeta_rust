# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/crystal.py  —  Crystal Lattice in Z_13[eta]
==================================================
This module treats Z_13[eta] as a 3D lattice {1, eta, eta^2} over Z_13.
The structure factor is computed via NTT, and the Voronoi decomposition
uses the p-adic valuation rather than Euclidean distance.

Mathematical context:
    Section 12  (Physical Connections)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW
from .spectral import proj
from .ntt import NTT
from .kernel import PAdicKernel


class CrystalLattice:

    @staticmethod
    def unit_cell(dev: torch.device) -> torch.Tensor:
        return ETA_POW[:ORD].to(dev)

    @staticmethod
    def dominant_sublattice(dev: torch.device) -> torch.Tensor:
        from .constants import P1_MAT
        return proj(ETA_POW[:ORD].to(dev), P1_MAT.to(dev))

    @staticmethod
    def structure_factor(f: torch.Tensor, N: int) -> torch.Tensor:
        return NTT.ntt(f, N)

    @staticmethod
    def reciprocal(N: int, dev: torch.device) -> torch.Tensor:
        return NTT.ntt(ETA_POW[:N].to(dev), N)

    @staticmethod
    def voronoi(L: int, dev: torch.device) -> torch.Tensor:
        pos = torch.arange(L, device=dev)
        orbit0 = ETA_POW[:ORD, 0].to(dev)
        dist = (pos.unsqueeze(1) - orbit0.unsqueeze(0)).abs()
        return PAdicKernel._val(dist).argmax(1)
