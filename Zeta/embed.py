# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/embed.py  —  Token Embedding and Positional Encoding
==========================================================
This module provides deterministic embeddings without learned parameters:

    ZRingEmbed : E(v,d) = eta^{(v*D + d) mod 168}  —  (V,D,3) table
    TatePE     : PE[n,2k] = eta^{n*k}, PE[n,2k+1] = eta^{-n*k}

Both are constructed from precomputed ETA_POW tables.  No nn.Parameter.
The Tate positional encoding supports arbitrary sequence lengths at
runtime through dynamic computation.

Mathematical context:
    Section 2.1  (The Base Ring Z_13[eta])

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .constants import P, ORD, ETA_POW, ETA_IPOW, ZetaConfig
from .ring import Z13


class ZRingEmbed(nn.Module):
    """E(v,d) = η^{(v·D+d) mod 168}.  (V,D,3).  Zero nn.Parameter."""
    def __init__(self, V: int, D: int):
        super().__init__()
        i = torch.arange(V).unsqueeze(1) * D + torch.arange(D).unsqueeze(0)
        self.register_buffer('W', ETA_POW[i % ORD])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.W[t % self.W.shape[0]]


class TatePE(nn.Module):
    """PE[n,2k] = η^{nk},  PE[n,2k+1] = η^{−nk}.  Dynamic any L."""
    def __init__(self, D: int):
        super().__init__();  assert D % 2 == 0;  self.D = D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D, _ = x.shape;  dev = x.device
        n = torch.arange(L, device=dev);  k = torch.arange(D//2, device=dev)
        nk = (n.unsqueeze(1) * k.unsqueeze(0)) % ORD
        PE = torch.zeros(L, D, 3, dtype=torch.long, device=dev)
        PE[:, 0::2] = ETA_POW.to(dev)[nk]
        PE[:, 1::2] = ETA_IPOW.to(dev)[nk]
        return Z13.add(x, PE.unsqueeze(0).expand(B, -1, -1, -1))
