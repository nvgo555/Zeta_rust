# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/entanglement.py  —  Discrete Holographic Entanglement in SL(3,Z)
======================================================================
This module implements holographic entropy formulas over the p-adic ring.
The central charge Delta = 2 mod 13 connects to the Ryu-Takayanagi
formula, Page curve, and wormhole entropy.

All formulas are algebraic consequences of det(T3) = 1.  No parameters
are fitted.

Mathematical context:
    Section 13  (ER=EPR and Holography)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, T3_POW, P23_MAT
from .ring import Z13, F169
from .spectral import proj


class EntanglementGeometry:
    DELTA = 2

    @staticmethod
    def ryu_takayanagi(nA: int, nB: int) -> int:
        return (EntanglementGeometry.DELTA * nA % P * nB) % P

    @staticmethod
    def wormhole_entropy(n: int) -> int:
        return (EntanglementGeometry.DELTA * (n % P)) % P

    @staticmethod
    def page_curve(t: int, N: int) -> int:
        t_mod = t % P;  n_mod = N % P
        nt = (n_mod - t_mod) % P
        return (EntanglementGeometry.DELTA * min(t_mod, nt)) % P

    @staticmethod
    def state_entropy(psi: torch.Tensor, n_strata: int = 12) -> int:
        dev = psi.device
        states = (T3_POW.to(dev)[:n_strata] @ psi.unsqueeze(-1)).squeeze(-1) % P
        p23 = proj(states, P23_MAT.to(dev))
        return (int((F169.norm(Z13.phi2(p23)) == 0).sum().item())
                * EntanglementGeometry.DELTA) % P

    @staticmethod
    def info_conservation(n: int) -> bool:
        M = T3_POW[n % ORD]
        det = ((M[0,0]*(M[1,1]*M[2,2]-M[1,2]*M[2,1])
              - M[0,1]*(M[1,0]*M[2,2]-M[1,2]*M[2,0])
              + M[0,2]*(M[1,0]*M[2,1]-M[1,1]*M[2,0])) % P)
        return det == 1
