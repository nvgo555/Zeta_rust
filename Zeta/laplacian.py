# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/laplacian.py  —  Vladimirov p-adic Laplacian
==================================================
This module implements the p-adic Laplacian Delta_p over the discrete
set {0, ..., L-1} with weights given by the ultrametric kernel.

The weight matrix W[i,j] = eta^{v_13(|i-j|)} and the Laplacian matrix
L = W - diag(rowsum) define a graph Laplacian on the Bruhat-Tits tree.

Mathematical context:
    Section 5  (Ultrametric Geometry)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW, ZERO_R
from .ring import Z13
from .kernel import PAdicKernel
from .ntt import NTT


class PAdicLaplacian:

    @staticmethod
    def weight_matrix(L: int, dev: torch.device) -> torch.Tensor:
        """W[i,j] = eta^{v_13(|i-j|)}.  (L,L,3).  Diagonal = 0."""
        idx = torch.arange(L, device=dev)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        val = PAdicKernel._val(dist)
        W = ETA_POW.to(dev)[val.clamp(0, ORD - 1)]
        W[torch.eye(L, dtype=torch.bool, device=dev)] = ZERO_R.to(dev)
        return W

    @staticmethod
    def matrix(L: int, dev: torch.device) -> torch.Tensor:
        W = PAdicLaplacian.weight_matrix(L, dev)
        rowsum = W.sum(1) % P
        Lm = W.clone()
        Lm[torch.eye(L, dtype=torch.bool, device=dev)] = Z13.neg(rowsum)
        return Lm

    @staticmethod
    def apply(Lm: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        L = Lm.shape[0]
        return Z13.mul(Lm, f.unsqueeze(0).expand(L, L, 3)).sum(1) % P

    @staticmethod
    def ntt_eigendecomp(L: int, dev: torch.device) -> torch.Tensor:
        Lm = PAdicLaplacian.matrix(L, dev)
        N = NTT.best_size(L)
        return NTT.ntt(Lm[:, :N], N)
