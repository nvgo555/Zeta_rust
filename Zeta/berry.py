# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/berry.py  —  Berry Phase and Algebraic SVD Proxy
=======================================================
This module computes the Berry connection via Sylvester P1 projectors
and provides an algebraic SVD proxy through the NTT power spectrum.
No Euclidean singular value decomposition is used.

Mathematical context:
    Section 11.2  (Physical and Geometric Modules)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ZetaConfig
from .ring import Z13
from .spectral import proj
from .ntt import NTT


class BerrySVD:

    @staticmethod
    def connection(row: torch.Tensor) -> torch.Tensor:
        from .constants import P1_MAT
        dA = torch.zeros_like(row)
        if row.shape[0] > 1:
            dA[:-1] = Z13.sub(row[1:], row[:-1])
        return Z13.smul(proj(dA, P1_MAT), ZetaConfig.I_P)

    @staticmethod
    def phase(row: torch.Tensor) -> int:
        return int(BerrySVD.connection(row)[:, 0].sum().item()) % P

    @staticmethod
    def svd_proxy(x: torch.Tensor) -> torch.Tensor:
        N = NTT.best_size(x.shape[0])
        return NTT.spectrum(x[:N], N)

    @staticmethod
    def dominant_mode(x: torch.Tensor) -> int:
        return int(BerrySVD.svd_proxy(x).argmax().item())
