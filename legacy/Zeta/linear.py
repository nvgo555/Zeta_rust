# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/linear.py  -  Shared Ring Linear Algebra Operations
==========================================================
This module provides common linear operations used across all model
components: ring-linear maps, Born normalization (replacing softmax),
ring attention, discrete Laplacian smoothing, deterministic seed
generation via eta powers, and algebraic dropout.

All operations are O(1) or O(N log N) per element.  No Python loops.

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW, ZetaConfig
from .ring import Z13
from .hilbert import HilbertEta


def ring_lin(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """f(x) = Σ_d x[d]*W[d,:].  x:(...,D_in,3)  W:(D_in,D_out,3) → (...,D_out,3)."""
    return Z13.mul(x.unsqueeze(-2), W.to(x.device)).sum(-3) % P


def born_norm(S: torch.Tensor) -> torch.Tensor:
    """Row-normalize score matrix.  Replaces softmax.  S:(B,Lq,Lk,3) → same."""
    B, Lq, Lk, _ = S.shape;  flat = S.reshape(B*Lq, Lk, 3)
    ns = HilbertEta.norm_sq(flat)
    inv_ns = Z13.inv(ns).unsqueeze(1).expand(-1, Lk, 3)
    return Z13.mul(flat, inv_ns).reshape(B, Lq, Lk, 3) % P


def ring_attend(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """O[b,i,d] = Σ_j A[b,i,j]*V[b,j,d].  A:(B,L,L,3)  V:(B,L,D,3) → (B,L,D,3)."""
    return Z13.mul(A.unsqueeze(-2), V.unsqueeze(1)).sum(2) % P


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """Δx[i] = x[i+1] + x[i-1] − 2x[i].  −2 ≡ 11.  x:(B,L,D,3) → same."""
    return (torch.roll(x, 1, 1) + torch.roll(x, -1, 1) + 11*x) % P


def w_seed(d_in: int, d_out: int) -> torch.Tensor:
    """W[i,j] = η^{(i*d_out+j) mod 168}.  (d_in,d_out,3).  Zero nn.Parameter."""
    i = torch.arange(d_in).unsqueeze(1);  j = torch.arange(d_out).unsqueeze(0)
    return ETA_POW[(i * d_out + j) % ORD]


def alg_dropout(x: torch.Tensor, rate: int, step: int, training: bool) -> torch.Tensor:
    if not training:
        return x
    D = x.shape[-2];  dev = x.device
    k = torch.arange(D, device=dev)
    keep = (ETA_POW.to(dev)[(rate * k * (step + 1)) % ORD, 0] != 0).long()
    return x * keep.view(*([1]*(x.dim()-2)), D, 1)
