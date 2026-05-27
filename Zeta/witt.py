# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/witt.py  —  Witt Vectors W_k(Z_13[eta])
=============================================
This module implements Witt vector arithmetic of precision k over the
cubic p-adic ring.  Witt vectors provide a canonical lift from the
residue field to characteristic zero without floating-point arithmetic.

Operations:
    wadd  : Witt addition via carry propagation (tensor shift)
    wmul  : Witt multiplication via anti-diagonal scatter-add
    winv  : Newton inversion (k-1 explicit steps)
    ghost : Witt ghost map Phi_k(w) = sum 13^i * w_i
    frobenius : Galois action on Witt components

Mathematical context:
    Section 8.2  (Witt Vectors)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW, ONE_R
from .ring import Z13, CRT, F169


class WittVector:

    @staticmethod
    def from_ring(a: torch.Tensor, prec: int) -> torch.Tensor:
        """τ(a) = (a,0,...,0).  a:(...,3) → (...,prec,3)."""
        out = torch.zeros(*a.shape[:-1], prec, 3, dtype=torch.long, device=a.device)
        out[..., 0, :] = a % P
        return out

    @staticmethod
    def to_ring(w: torch.Tensor) -> torch.Tensor:
        return w[..., 0, :] % P

    @staticmethod
    def wadd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Witt addition.  Carry via tensor shift along prec axis."""
        if x.shape[-2] == 1:
            return (x + y) % P
        raw = x + y
        carry = raw // P
        carry_in = torch.zeros_like(carry)
        carry_in[..., 1:, :] = carry[..., :-1, :]
        return (raw + carry_in) % P

    @staticmethod
    def wneg(x: torch.Tensor) -> torch.Tensor:
        return (P - x) % P

    @staticmethod
    def wsub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return WittVector.wadd(x, WittVector.wneg(y))

    @staticmethod
    def wmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Witt multiply.  Outer exact products → anti-diagonal scatter_add."""
        prec = x.shape[-2];  dev = x.device;  batch = x.shape[:-2]
        P_ex = Z13.mul_exact(x.unsqueeze(-2), y.unsqueeze(-3))     # (...,prec,prec,3)
        i_idx = torch.arange(prec, device=dev)
        diag = (i_idx.unsqueeze(1) + i_idx.unsqueeze(0)).reshape(-1)   # (prec²,)
        Pf = P_ex.reshape(*batch, prec*prec, 3)
        L = torch.zeros(*batch, 2*prec - 1, 3, dtype=torch.long, device=dev)
        L.scatter_add_(-2, diag.view(*([1]*len(batch)), prec*prec, 1).expand_as(Pf), Pf)
        raw = L[..., :prec, :]
        carry = raw // P;  ci = torch.zeros_like(carry);  ci[..., 1:, :] = carry[..., :-1, :]
        return (raw + ci) % P

    @staticmethod
    def wpow(x: torch.Tensor, n: int) -> torch.Tensor:
        """x^n.  8-step unrolled (n < 168)."""
        prec = x.shape[-2]
        r = WittVector.from_ring(ONE_R.expand(*x.shape[:-2], 3), prec).to(x.device)
        b = x.clone()
        if n & 1:   r = WittVector.wmul(r, b)
        b = WittVector.wmul(b, b);  n >>= 1
        if n & 1:   r = WittVector.wmul(r, b)
        b = WittVector.wmul(b, b);  n >>= 1
        if n & 1:   r = WittVector.wmul(r, b)
        b = WittVector.wmul(b, b);  n >>= 1
        if n & 1:   r = WittVector.wmul(r, b)
        b = WittVector.wmul(b, b);  n >>= 1
        if n & 1:   r = WittVector.wmul(r, b)
        b = WittVector.wmul(b, b);  n >>= 1
        if n & 1:   r = WittVector.wmul(r, b)
        b = WittVector.wmul(b, b);  n >>= 1
        if n & 1:   r = WittVector.wmul(r, b)
        b = WittVector.wmul(b, b);  n >>= 1
        if n & 1:   r = WittVector.wmul(r, b)
        return r

    @staticmethod
    def winv(x: torch.Tensor) -> torch.Tensor:
        """Newton inversion.  prec−1 ≤ 7 explicit steps."""
        prec = x.shape[-2];  batch = x.shape[:-2];  dev = x.device
        r = WittVector.from_ring(Z13.inv(x[..., 0, :]), prec)
        two = WittVector.from_ring(ONE_R.expand(*batch, 3).to(dev), prec)
        two[..., 0, 0] = 2
        for _ in range(prec - 1):
            r = WittVector.wmul(r, WittVector.wsub(two, WittVector.wmul(x, r)))
        return r

    @staticmethod
    def ghost(w: torch.Tensor) -> torch.Tensor:
        """Φ_k(w) = Σ_{i≤k} 13^i·w_i.  cumsum × powers-of-13 vector."""
        prec = w.shape[-2];  dev = w.device
        pk = torch.tensor([P**i for i in range(prec)], dtype=torch.long, device=dev)
        return (w * pk.view(*([1]*(w.dim()-2)), prec, 1)).cumsum(-2) % P

    @staticmethod
    def frobenius(w: torch.Tensor) -> torch.Tensor:
        return CRT.compose(Z13.phi1(w), F169.frobenius(Z13.phi2(w)))

    @staticmethod
    def teichmuller(a: torch.Tensor, prec: int) -> torch.Tensor:
        return WittVector.from_ring(a, prec)
