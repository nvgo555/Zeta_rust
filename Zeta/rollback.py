# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/rollback.py  —  Quantum Rollback and Error Correction
===========================================================
This module implements algebraic error detection and correction.

    Detection : pred ≠ label (mod 13)  →  hallucination detected
    Correction  : head += eta^{step}     —  single orbital step
    Multi-precision : Witt vector addition of tau(eta^{step})

All corrections are deterministic and time-reversible.

Mathematical context:
    Section 9.2  (Orbital Correction)
    Section 9.3  (Precision Escalation)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW
from .ring import Z13
from .witt import WittVector
from .ntt import NTT


class QuantumRollback:

    @staticmethod
    def detect(pred: int, label: int) -> bool:
        return (pred % P) != (label % P)

    @staticmethod
    def rollback_ring(head: torch.Tensor, step: int, dev: torch.device) -> torch.Tensor:
        ep = ETA_POW[step % ORD].to(dev)
        return Z13.add(head, ep.view(*([1]*(head.dim()-1)), 3).expand_as(head))

    @staticmethod
    def rollback_witt(w: torch.Tensor, step: int, prec: int) -> torch.Tensor:
        correction = WittVector.teichmuller(ETA_POW[step % ORD], prec)
        return WittVector.wadd(w, correction.expand_as(w))

    @staticmethod
    def correction_spectrum(step: int) -> torch.Tensor:
        ep = ETA_POW[step % ORD].unsqueeze(0).expand(12, 3)
        return NTT.ntt(ep, 12)
