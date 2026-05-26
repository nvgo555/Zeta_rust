# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/mera.py  —  MERA Tensor Network in Z_13[eta]
===================================================
This module implements Multi-scale Entanglement Renormalization Ansatz
coarse-graining over the p-adic ring.  Each level groups pairs of tokens
via ring addition followed by T3^{2^k} evolution.

The explicit MERA chunker connects to the implicit MERA decimation
performed by the antisymmetric Dirac operator (Section 7.2).

Mathematical context:
    Section 7  (MERA and Hierarchical Renormalization)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import List
import torch

from .constants import P, ORD, T3_POW
from .ring import Z13
from .spectral import t3n
from .ntt import NTT


class MERAChunker:

    @staticmethod
    def coarse_grain(x: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
        result = [x];  curr = x
        if levels >= 1:
            L = curr.shape[1]
            if L >= 2:
                even = curr[:, 0::2, :, :]
                odd  = curr[:, 1::2, :, :]
                mlen = min(even.shape[1], odd.shape[1])
                curr = Z13.add(even[:, :mlen], odd[:, :mlen])
            curr = t3n(curr, 1);  result.append(curr)
        if levels >= 2:
            L = curr.shape[1]
            if L >= 2:
                even = curr[:, 0::2, :, :]
                odd  = curr[:, 1::2, :, :]
                mlen = min(even.shape[1], odd.shape[1])
                curr = Z13.add(even[:, :mlen], odd[:, :mlen])
            curr = t3n(curr, 2);  result.append(curr)
        if levels >= 3:
            L = curr.shape[1]
            if L >= 2:
                even = curr[:, 0::2, :, :]
                odd  = curr[:, 1::2, :, :]
                mlen = min(even.shape[1], odd.shape[1])
                curr = Z13.add(even[:, :mlen], odd[:, :mlen])
            curr = t3n(curr, 4);  result.append(curr)
        return result

    @staticmethod
    def ntt_per_level(x: torch.Tensor) -> torch.Tensor:
        B, L, D, _ = x.shape;  N = NTT.best_size(L)
        xp = x[:, :N, :, :]
        return NTT.ntt(xp.permute(0,2,1,3).reshape(B*D, N, 3), N).reshape(B, D, N, 3).permute(0,2,1,3)

    @staticmethod
    def renormalize(levels: List[torch.Tensor]) -> torch.Tensor:
        result = levels[-1]
        if len(levels) >= 3:
            L_target = levels[-2].shape[1]
            factor = max(1, (L_target + result.shape[1]-1)//result.shape[1])
            up = result.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(
                result.shape[0], result.shape[1]*factor, result.shape[2], result.shape[3]
            )[:, :L_target]
            result = Z13.add(levels[-2], up)
        if len(levels) >= 2:
            L_target = levels[-3].shape[1] if len(levels)>=3 else levels[0].shape[1]
            factor = max(1, (L_target + result.shape[1]-1)//result.shape[1])
            up = result.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(
                result.shape[0], result.shape[1]*factor, result.shape[2], result.shape[3]
            )[:, :L_target]
            result = Z13.add(levels[0] if len(levels)==2 else levels[-3], up)
        return result


    @staticmethod
    def coarse_grain_dynamic(x: torch.Tensor) -> List[torch.Tensor]:
        """Auto-depth MERA: log_2(L) levels, capped at 10.
        Each level groups pairs with T3^{2^k} evolution."""
        L = x.shape[1]
        levels = min(int(torch.log2(torch.tensor(float(L))).ceil().item()), 10)
        return MERAChunker.coarse_grain(x, levels=max(levels, 1))
