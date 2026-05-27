# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/attention.py  -  Spectral Attention via Bruhat-Tits Tree
=============================================================
This module implements the sole attention mechanism of the Zeta engine.
It combines:

    1. Sylvester split into dominant (P1, 1D) and subdominant (P23, 2D) channels
    2. THE ONLY KERNEL - Bruhat-Tits tree mixing via PAdicKernel.apply(x)
    3. Local Q/K/V projections (pointwise along D, never L×L)
    4. Born normalization replacing softmax

The tree kernel performs global information mixing in O(L*log_13(L))
without ever allocating an L×L matrix.  Token communication occurs
exclusively through common ancestors in the Bruhat-Tits tree.

Mathematical context:
    Section 5.2  (The Sole Kernel)
    Section 7.1  (Explicit MERA Chunker)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

from .constants import P, ORD, ZetaConfig, P1_MAT, P23_MAT
from .ring import Z13
from .spectral import proj
from .kernel import PAdicKernel
from .hilbert import HilbertEta
from .linear import ring_lin, laplacian, w_seed, alg_dropout


class SpectralAttention(nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__();  assert D % 6 == 0;  hd = D // 3
        self._step = 0
        self.register_buffer('P1',  P1_MAT.clone())
        self.register_buffer('P23', P23_MAT.clone())
        # Q/K/V seeds for DOM (P1) and SUB (P23) channels
        self.register_buffer('WQ_dom', w_seed(D, hd))
        self.register_buffer('WK_dom', w_seed(D, hd))
        self.register_buffer('WV_dom', w_seed(D, hd))
        self.register_buffer('WQ_sub', w_seed(D, hd))
        self.register_buffer('WK_sub', w_seed(D, hd))
        self.register_buffer('WV_sub', w_seed(D, hd))
        self.register_buffer('WO', w_seed(2*hd, D))

    def forward(self, x: torch.Tensor, dm: int, li: int = 0) -> Tuple[torch.Tensor, dict]:
        B, L, D, _ = x.shape;  dev = x.device

        # ── Sylvester split ──────────────────────────────────────────────────
        x_p1  = proj(x, self.P1.to(dev))    # dominant 1D channel
        x_p23 = proj(x, self.P23.to(dev))   # subdominant 2D channel

        # ── THE ONLY KERNEL: Bruhat-Tits tree mixing ─────────────────────────
        # This replaces the L×L attention matrix.  O(L * log_13(L)).
        # Each token receives information from all others via tree ancestors.
        # Fast path for L ≤ 169, standard for longer sequences
        if L <= 169:
            x_p1_k  = PAdicKernel.apply_fast(x_p1)
            x_p23_k = PAdicKernel.apply_fast(x_p23)
        else:
            x_p1_k  = PAdicKernel.apply(x_p1)
            x_p23_k = PAdicKernel.apply(x_p23)

        # ── Local Q/K/V projections (no L×L operations) ─────────────────────
        # Q and K are used for local channel alignment.
        # V carries the kernel-mixed information.
        Q1 = ring_lin(x_p1_k,  self.WQ_dom.to(dev))   # (B, L, hd, 3)
        K1 = ring_lin(x_p1_k,  self.WK_dom.to(dev))
        V1 = ring_lin(x_p1_k,  self.WV_dom.to(dev))
        Q2 = ring_lin(x_p23_k, self.WQ_sub.to(dev))
        K2 = ring_lin(x_p23_k, self.WK_sub.to(dev))
        V2 = ring_lin(x_p23_k, self.WV_sub.to(dev))

        # ── Local attention score: channel-wise ring inner product ──────────
        # S[b,l,d] = trace(Q[b,l,d] * K[b,l,d]) - pointwise along D, NOT L×L.
        # This is O(B·L·D), never O(B·L²).
        S1 = Z13.trace(Z13.mul(Q1, K1)) % P            # (B, L, hd)
        S2 = Z13.trace(Z13.mul(Q2, K2)) % P            # (B, L, hd)

        # ── Weight V by local scores ────────────────────────────────────────
        # V is weighted pointwise along D dimension, then summed.
        # O1[b,l,d] = S1[b,l,d] * V1[b,l,d]  -  still O(B·L·D)
        O1 = Z13.mul(V1, S1.unsqueeze(-1).expand_as(V1))   # (B, L, hd, 3)
        O2 = Z13.mul(V2, S2.unsqueeze(-1).expand_as(V2))

        # ── Combine channels and project ────────────────────────────────────
        out = ring_lin(torch.cat([O1, O2], dim=2) % P, self.WO.to(dev))

        # ── Residual + Laplacian + HilbertEta layer norm ────────────────────
        out = HilbertEta.layer_norm(laplacian(Z13.add(x, out)))
        out = alg_dropout(out, ZetaConfig.DROP, self._step, self.training)
        return out, {'li': li, 'dm': dm, 'step': self._step}
