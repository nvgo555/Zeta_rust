# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/spectral.py  -  T3 SL(3,Z) Orbit, Sylvester Projectors, and S3 Galois Symmetry
===================================================================================
This module contains the spectral decomposition of the sole evolution matrix
T3 ∈ SL(3,Z), the Sylvester projectors P1 and P23, and the Galois group
S3 = Gal(Q(eta)/Q) acting on the Tribonacci polynomial.

Key identities:
    T3^n = 7^n * P1 + P23 * T3^n          (spectral decomposition)
    P1^2 = P1,  P23^2 = P23,  P1*P23 = 0  (idempotence and orthogonality)
    P1 + P23 = I                           (completeness)
    T3 * P1 = 7 * P1                       (dominant eigenvalue)

The S3 group has 6 elements represented as (3,3) permutation matrices
over Z_13.  Composition is precomputed as a (6,6) multiplication table.

Mathematical context:
    Section 2.2  (The Sole Matrix T3 ∈ SL(3,Z))
    Section 3.1  (Sylvester Projectors)
    Section 4    (Galois Symmetry)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import Tuple
import torch

from .constants import P, ORD, ETA_POW, T3_POW, P1_MAT, P23_MAT, S3_MATS, _S3_MUL
from .ring import Z13


def proj(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply (3,3) Z_13-linear map M to (...,3):  M @ x.  Single implementation."""
    M = M.to(x.device)
    return torch.stack([
        (x[...,0]*M[0,0] + x[...,1]*M[0,1] + x[...,2]*M[0,2]) % P,
        (x[...,0]*M[1,0] + x[...,1]*M[1,1] + x[...,2]*M[1,2]) % P,
        (x[...,0]*M[2,0] + x[...,1]*M[2,1] + x[...,2]*M[2,2]) % P,
    ], dim=-1)


def t3n(x: torch.Tensor, n: int) -> torch.Tensor:
    """T3^n · x.  O(1) table lookup.  x:(...,3) → (...,3)."""
    return proj(x, T3_POW[n % ORD])




def t3n_s3(x: torch.Tensor, n: int, g: int) -> torch.Tensor:
    """T3^n conjugated by S3 element g.  O(1) lookup + proj.
    Provides 6 independent evolution paths (one per Galois element).
    x:(...,3) → (...,3)."""
    return S3Galois.apply(t3n(x, n), g)

class SylvesterProjectors:
    """P1 + P23 = I,  P1² = P1,  P23² = P23,  P1·P23 = 0,  T3·P1 = 7·P1."""

    @staticmethod
    def p1(x: torch.Tensor) -> torch.Tensor:
        return proj(x, P1_MAT)

    @staticmethod
    def p23(x: torch.Tensor) -> torch.Tensor:
        return proj(x, P23_MAT)

    @staticmethod
    def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p1x = proj(x, P1_MAT)
        return p1x, (x - p1x) % P


class SpectralDecomposition:
    """T3^n = 7^n·P1 + P23·T3^n.  All via t3n() and proj()."""

    @staticmethod
    def evolve(x: torch.Tensor, n: int) -> torch.Tensor:
        """7^n·P1·x + P23·T3^n·x.  O(1)."""
        p1x = proj(x, P1_MAT)
        return (Z13.smul(p1x, pow(7, n, P)) + t3n((x - p1x) % P, n)) % P

    @staticmethod
    def eigenvalue_dominant(n: int) -> int:
        """λ₁^n = 7^n mod 13."""
        return int(Z13.phi1(ETA_POW[n % ORD]).item())


class S3Galois:
    """S₃ Galois group.  6 elements.  Uses proj() for application."""
    MATS  = S3_MATS          # (6,3,3)
    NAMES = ['e','(12)','(01)','(02)','(012)','(021)']
    _MUL  = _S3_MUL          # (6,6) precomputed multiplication

    @staticmethod
    def apply(x: torch.Tensor, g: int) -> torch.Tensor:
        return proj(x, S3_MATS[g % 6])

    @staticmethod
    def compose_indices(g: int, h: int) -> int:
        return S3Galois._MUL[g % 6, h % 6].item()

    @staticmethod
    def orbit(x: torch.Tensor) -> torch.Tensor:
        """All 6 images.  x:(3,) → (6,3)."""
        return torch.stack([proj(x, S3_MATS[k]) for k in range(6)])

    @staticmethod
    def casimir(x: torch.Tensor) -> torch.Tensor:
        """Σ_{g∈S3} g·x - S3-invariant projection."""
        return S3Galois.orbit(x).sum(0) % P
