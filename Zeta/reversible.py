# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/reversible.py  —  Time-Reversible Sequence Transform
==========================================================
This module implements invertible encoding and decoding via the T3 orbit:

    encode : x_t → T3^t * x_t        (forward evolution)
    decode : y_t → T3^{168-t} * y_t  (inverse evolution)

The roundtrip property decode(encode(x)) = x holds exactly, providing
time reversibility for the learning mechanism.

Mathematical context:
    Section 6.3  (Density Matrix and Quantum Cache)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, T3_POW


class ReversibleGenerator:

    @staticmethod
    def encode(x: torch.Tensor) -> torch.Tensor:
        B, L, D, _ = x.shape;  dev = x.device
        t_idx = torch.arange(L, device=dev) % ORD
        M = T3_POW.to(dev)[t_idx]
        return torch.einsum('bldc,lcr->bldr', x.long(), M.long()) % P

    @staticmethod
    def decode(y: torch.Tensor) -> torch.Tensor:
        B, L, D, _ = y.shape;  dev = y.device
        t_idx = (ORD - torch.arange(L, device=dev)) % ORD
        M = T3_POW.to(dev)[t_idx]
        return torch.einsum('bldc,lcr->bldr', y.long(), M.long()) % P

    @staticmethod
    def roundtrip_check(x: torch.Tensor) -> bool:
        return bool(torch.all(ReversibleGenerator.decode(
                               ReversibleGenerator.encode(x)) == x))
