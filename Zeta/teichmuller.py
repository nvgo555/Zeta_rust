# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/teichmuller.py  —  Teichmüller Lifts
==========================================
This module implements the multiplicative Teichmüller section tau(a) =
(a, 0, ..., 0) in Witt vectors, satisfying tau(a*b) = tau(a)*tau(b).

Mathematical context:
    Section 8  (NTT, Witt Vectors, and Hensel Lifting)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import ORD, ETA_POW
from .witt import WittVector
from .ntt import NTT


class Teichmuller:
    lift = staticmethod(WittVector.teichmuller)

    @staticmethod
    def orbit(prec: int) -> torch.Tensor:
        return WittVector.from_ring(ETA_POW[:ORD], prec)

    @staticmethod
    def mul_table(K: int) -> torch.Tensor:
        i = torch.arange(K);  j = torch.arange(K)
        return ETA_POW[(i.unsqueeze(1) + j.unsqueeze(0)) % ORD]

    @staticmethod
    def ntt_spectrum(N: int, dev: torch.device) -> torch.Tensor:
        return NTT.ntt(ETA_POW[:N].to(dev), N)
