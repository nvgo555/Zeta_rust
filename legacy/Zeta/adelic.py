# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/adelic.py  —  Adelic Product and Strong Approximation
============================================================
This module computes the adelic norm across primes ell ≠ 13 and
implements the strong approximation theorem for simultaneous congruences.

Mathematical context:
    Section 14  (Moonshine and the Leech Lattice)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import Dict
import torch

from .constants import P
from .ring import Z13


class AdelicProduct:
    _PRIMES = [2, 3, 5, 7, 11]

    @staticmethod
    def residue(a: torch.Tensor, ell: int) -> torch.Tensor:
        return a % ell

    @staticmethod
    def adelic_norm(a: torch.Tensor) -> torch.Tensor:
        result = Z13.norm(a)
        def _local_norm(ae, ell):
            a0, a1, a2 = ae[...,0], ae[...,1], ae[...,2]
            return (a0*(a0*a0 - a1*a1 - a2*a2)
                  + a1*(2*a0*a2 - a1*a2)
                  + a2*(a0*a1 - a2*a2)) % ell % P
        for pr in AdelicProduct._PRIMES:
            result = (result * _local_norm(a % pr, pr)) % P
        return result

    @staticmethod
    def strong_approx(targets: Dict[int, int]) -> torch.Tensor:
        M = 30030
        x = (targets.get(2,  0) * 15015 * 1)  % M
        x = (x + targets.get(3,  0) * 10010 * 2) % M
        x = (x + targets.get(5,  0) * 6006  * 1) % M
        x = (x + targets.get(7,  0) * 4290  * 2) % M
        x = (x + targets.get(11, 0) * 2730  * 1) % M
        x = (x + targets.get(P,  0) * 2310  * 6) % M
        return torch.tensor([x % P, 0, 0], dtype=torch.long)
