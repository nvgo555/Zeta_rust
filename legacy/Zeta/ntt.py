# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/ntt.py  —  Number Theoretic Transform over Z_13[eta]
===========================================================
This module implements the NTT over the cubic p-adic ring with twiddle
factors omega_N = eta^{168/N}.  Valid sizes are all divisors N of 168
with N mod 13 != 0.

Two paths:
    Scalar (N | 12) : twiddles in F_13, einsum with (N,N) matrix
    Ring   (N ∤ 12) : twiddles in Z_13[eta], broadcast ring_mul

All transforms are O(N log N) and fully vectorized.  Twiddle matrices are
precomputed at import time in constants.py.

Mathematical context:
    Section 8.1  (Number Theoretic Transform)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import List
import torch

from .constants import P, ORD, _NTT_VALID, _NTT_FWD, _NTT_INV
from .ring import Z13


class NTT:
    VALID: List[int] = _NTT_VALID

    @staticmethod
    def ntt(x: torch.Tensor, N: int) -> torch.Tensor:
        """Forward NTT.  x:(...,N,3) → (...,N,3)."""
        assert N in _NTT_FWD, f"NTT size {N} invalid"
        T = _NTT_FWD[N].to(x.device)
        if T.dim() == 2:          # scalar twiddle
            return torch.einsum('kn,...nc->...kc', T, x) % P
        # ring twiddle
        # T:(N,N,3), x:(...,N,3) -> output:(...,N,3)
        # X[k] = sum_n x[n] * T[n,k]  (element-wise mod 13)
        return Z13.mul(x.unsqueeze(-3), T).sum(-2) % P

    @staticmethod
    def intt(X: torch.Tensor, N: int) -> torch.Tensor:
        """Inverse NTT.  X:(...,N,3) → (...,N,3)."""
        assert N in _NTT_INV
        T = _NTT_INV[N].to(X.device)
        if T.dim() == 2:
            return torch.einsum('nk,...kc->...nc', T, X) % P
        return Z13.mul(X.unsqueeze(-3), T).sum(-2) % P

    @staticmethod
    def conv(a: torch.Tensor, b: torch.Tensor, N: int) -> torch.Tensor:
        """Cyclic convolution via convolution theorem."""
        return NTT.intt(Z13.mul(NTT.ntt(a, N), NTT.ntt(b, N)), N)

    @staticmethod
    def spectrum(x: torch.Tensor, N: int) -> torch.Tensor:
        """Algebraic power spectrum N(NTT(x)[k]).  (...,N) Long."""
        return Z13.norm(NTT.ntt(x, N))

    @staticmethod
    def cross(a: torch.Tensor, b: torch.Tensor, N: int) -> torch.Tensor:
        """NTT(a)·conj(NTT(b)).  Galois conj replaces complex conj."""
        return Z13.mul(NTT.ntt(a, N), Z13.conj(NTT.ntt(b, N)))

    @staticmethod
    def autocorr(a: torch.Tensor, N: int) -> torch.Tensor:
        """R[m] = Σ_n a[n]·ā[n+m].  (...,N,3)."""
        return NTT.intt(NTT.cross(a, a, N), N)

    @staticmethod
    def best_size(L: int) -> int:
        """Largest valid NTT size ≤ L."""
        return max((n for n in _NTT_VALID if n <= L), default=1)
