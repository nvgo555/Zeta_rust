# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/mahler.py  —  Mahler Expansion in Z_13[eta]
=================================================
This module computes Mahler coefficients Delta^k f(0) / k! for functions
over the p-adic ring.  Finite differences are vectorized via binomial
matrix construction.  The NTT bridge connects Mahler expansion to
frequency-domain analysis.

Mathematical context:
    Section 8  (Arithmetic and Transform Modules)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P
from .ring import Z13
from .ntt import NTT


class MahlerExpansion:
    _INV_FACT = [1, 1, 7, 9, 3, 5, 11, 9, 3, 7, 9, 1, 1]   # 1/k! mod 13

    @staticmethod
    def _binom_matrix(K: int, dev: torch.device) -> torch.Tensor:
        k_idx = torch.arange(K, device=dev)
        j_idx = torch.arange(K, device=dev)
        kj = k_idx.unsqueeze(1) - j_idx.unsqueeze(0)          # (K,K)
        mask = (j_idx.unsqueeze(0) <= k_idx.unsqueeze(1)).long()
        inv_f = torch.tensor(MahlerExpansion._INV_FACT[:K], dtype=torch.long, device=dev)
        sign = torch.tensor([[pow(P-1, max(0, int(kj[i,j])), P) for j in range(K)]
                              for i in range(K)], dtype=torch.long, device=dev)
        factors = torch.where(j_idx.unsqueeze(0).expand(K,K) > 0,
                              (k_idx.unsqueeze(1).expand(K,K) - j_idx.unsqueeze(0).expand(K,K) + 1) % P,
                              torch.ones(K, K, dtype=torch.long, device=dev))
        num = torch.cumprod(factors, dim=1) % P
        num[:, 0] = 1
        return (num * inv_f.unsqueeze(0) * sign * mask) % P

    @staticmethod
    def finite_differences(f: torch.Tensor) -> torch.Tensor:
        N = f.shape[0];  dev = f.device
        B = MahlerExpansion._binom_matrix(N, dev)
        return (B.unsqueeze(-1) * f.unsqueeze(0)).sum(1) % P

    @staticmethod
    def coeffs(f: torch.Tensor) -> torch.Tensor:
        N = f.shape[0];  dev = f.device
        diffs = MahlerExpansion.finite_differences(f)
        inv_f = torch.tensor(MahlerExpansion._INV_FACT[:N], dtype=torch.long, device=dev)
        return (diffs * inv_f.unsqueeze(-1)) % P

    @staticmethod
    def reconstruct(ak: torch.Tensor, n: int) -> torch.Tensor:
        K = ak.shape[0];  dev = ak.device;  nm = int(n) % P
        k_idx = torch.arange(K, device=dev)
        factors = torch.where(k_idx > 0, (nm - k_idx + 1) % P,
                              torch.ones(K, dtype=torch.long, device=dev))
        num = torch.cumprod(factors, 0) % P;  num[0] = 1
        inv_f = torch.tensor(MahlerExpansion._INV_FACT[:K], dtype=torch.long, device=dev)
        cn = (num * inv_f) % P
        return (cn.unsqueeze(-1) * ak).sum(0) % P

    @staticmethod
    def ntt_bridge(f: torch.Tensor, N: int) -> torch.Tensor:
        return NTT.ntt(Z13.sub(torch.roll(f, -1, 0), f), N)
