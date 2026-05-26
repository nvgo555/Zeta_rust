# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/hensel.py  —  Hensel Lifting of the Tribonacci Root
=========================================================
This module constructs the exact root eta of x^3 - x^2 - x - 1 modulo
13^k via Newton iteration.  The derivative f'(7) = 7^{-1} mod 13
provides the Newton correction factor.

Digits of eta in base 13: [7, 2, 1, 12, 11, ...]
Each digit corresponds to one Witt vector level, linking Hensel lifting
to Witt vector arithmetic.

Mathematical context:
    Section 8.3  (Hensel Lifting)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ZetaConfig


class HenselLifter:
    _f  = staticmethod(lambda x: x**3 - x**2 - x - 1)
    _Nw = 7                      # f'(7)^{−1} mod 13
    _C: dict = {1: 7, 2: 33, 3: 202, 4: 26566, 5: 340737}

    @staticmethod
    def eta_int(prec: int) -> int:
        if prec in HenselLifter._C:
            return HenselLifter._C[prec]
        best = max(k for k in HenselLifter._C if k <= prec)
        a = HenselLifter._C[best];  pk = P**best
        for k in range(best, prec):
            t = (-(HenselLifter._f(a) // pk) * HenselLifter._Nw) % P
            a += t * pk;  pk *= P;  HenselLifter._C[k+1] = a
        return a

    @staticmethod
    def digits(prec: int) -> torch.Tensor:
        """Base-13 digits of η.  (prec,) Long."""
        ds = [ZetaConfig.ETA0];  a = ZetaConfig.ETA0;  pk = P
        for _ in range(1, prec):
            t = (-(HenselLifter._f(a) // pk) * HenselLifter._Nw) % P
            a += t * pk;  ds.append(int(t));  pk *= P
        return torch.tensor(ds[:prec], dtype=torch.long)

    @staticmethod
    def witt_eta(prec: int) -> torch.Tensor:
        """η as Witt vector.  (prec,3) Long."""
        d = HenselLifter.digits(prec)
        w = torch.zeros(prec, 3, dtype=torch.long);  w[:, 0] = d
        return w

    @staticmethod
    def verify(k: int) -> bool:
        return HenselLifter._f(HenselLifter.eta_int(k)) % (P**k) == 0


class HenselIO:
    """Empirický uzáver: senzory 0..12 ↔ Z_13[η], aktuátory ↔ inverzný Hensel."""

    @staticmethod
    def sensor_embed(s: int) -> torch.Tensor:
        """s ∈ {0, ..., 12} → (s, 0, 0) ∈ Z_13[η]."""
        return torch.tensor([int(s) % P, 0, 0], dtype=torch.long)

    @staticmethod
    def actuator_decode(a: torch.Tensor) -> tuple:
        """(a0, a1, a2) → (motor, smer, intenzita) v Z_13."""
        a = a % P
        return (int(a[0].item()), int(a[1].item()), int(a[2].item()))

    @staticmethod
    def feedback_check(measured: torch.Tensor, predicted: torch.Tensor) -> bool:
        """Konflikt? → Buchberger detekuje."""
        return not torch.all(measured == predicted)
