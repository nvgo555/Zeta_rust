# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/ring.py  -  Z_13[eta] Ring Arithmetic, CRT, and F_169 Field Operations
=============================================================================
This module implements all arithmetic in the cubic p-adic integer ring
R = Z_13[eta] / (eta^3 - eta^2 - eta - 1), together with its CRT
decomposition R ≅ F_13 × F_169 and the quadratic extension F_169.

Classes:
    Z13  : ring multiplication, addition, inversion, trace, norm, Galois conjugation
    CRT  : Chinese Remainder Theorem decomposition and composition
    F169 : arithmetic in F_13[alpha] / (alpha^2 + 6*alpha + 2)

All methods operate on (...,3) torch.long tensors with modulo 13 reduction.
Zero Python loops in data paths.  Ring inversion is O(1) table lookup.

Mathematical context:
    Section 2.1  (The Base Ring Z_13[eta])
    Section 2.3  (Ring Arithmetic)
    Section 3.2  (Chinese Remainder Theorem)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import Tuple
import torch

from .constants import (
    P, ORD, ETA_POW, ETA_IPOW, INV_TBL,
    ONE_R, ZERO_R, S3_MATS, _S3_MUL,
)


class Z13:
    """All arithmetic in Z_13[η] / (η³ − η² − η − 1)."""

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a0,a1,a2 = a[...,0], a[...,1], a[...,2]
        b0,b1,b2 = b[...,0], b[...,1], b[...,2]
        c0 = a0*b0;  c1 = a0*b1 + a1*b0;  c2 = a0*b2 + a1*b1 + a2*b0
        c3 = a1*b2 + a2*b1;  c4 = a2*b2
        return torch.stack([(c0+c3+c4)%P,
                            (c1+c3+2*c4)%P,
                            (c2+c3+2*c4)%P], dim=-1)

    @staticmethod
    def mul_exact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Without mod P.  Values ≤ 864.  Used by Witt wmul."""
        a0,a1,a2 = a[...,0], a[...,1], a[...,2]
        b0,b1,b2 = b[...,0], b[...,1], b[...,2]
        c0 = a0*b0;  c1 = a0*b1 + a1*b0;  c2 = a0*b2 + a1*b1 + a2*b0
        c3 = a1*b2 + a2*b1;  c4 = a2*b2
        return torch.stack([c0+c3+c4, c1+c3+2*c4, c2+c3+2*c4], dim=-1)

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b) % P

    @staticmethod
    def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b) % P

    @staticmethod
    def neg(a: torch.Tensor) -> torch.Tensor:
        return (-a) % P

    @staticmethod
    def smul(a: torch.Tensor, k: int) -> torch.Tensor:
        return (a * (int(k) % P)) % P

    @staticmethod
    def inv(a: torch.Tensor) -> torch.Tensor:
        """O(1) table lookup.  Zero-div → (0,0,0)."""
        t  = INV_TBL.to(a.device)
        a0 = a[...,0] % P;  a1 = a[...,1] % P;  a2 = a[...,2] % P
        return t[a0, a1, a2]

    @staticmethod
    def pow(a: torch.Tensor, n: int) -> torch.Tensor:
        """Binary exponentiation, 8 unrolled steps (n < 168 < 256)."""
        n = int(n) % ORD
        r = torch.zeros_like(a);  r[...,0] = 1;  b = a.clone() % P
        if n & 1:   r = Z13.mul(r, b) % P
        b = Z13.mul(b, b) % P;  n >>= 1
        if n & 1:   r = Z13.mul(r, b) % P
        b = Z13.mul(b, b) % P;  n >>= 1
        if n & 1:   r = Z13.mul(r, b) % P
        b = Z13.mul(b, b) % P;  n >>= 1
        if n & 1:   r = Z13.mul(r, b) % P
        b = Z13.mul(b, b) % P;  n >>= 1
        if n & 1:   r = Z13.mul(r, b) % P
        b = Z13.mul(b, b) % P;  n >>= 1
        if n & 1:   r = Z13.mul(r, b) % P
        b = Z13.mul(b, b) % P;  n >>= 1
        if n & 1:   r = Z13.mul(r, b) % P
        b = Z13.mul(b, b) % P;  n >>= 1
        if n & 1:   r = Z13.mul(r, b) % P
        return r

    @staticmethod
    def trace(a: torch.Tensor) -> torch.Tensor:
        """Tr_{Q(η)/Q}(α) = 3a₀ + a₁ + 3a₂  mod 13."""
        return (3*a[...,0] + a[...,1] + 3*a[...,2]) % P

    @staticmethod
    def norm(a: torch.Tensor) -> torch.Tensor:
        """N_{Q(η)/Q}(α) = det(M_α) mod 13."""
        a0,a1,a2 = a[...,0], a[...,1], a[...,2]
        r0 = torch.stack([a0, a2, (a1+a2)%P], dim=-1)
        r1 = torch.stack([a1, (a0+a2)%P, (a1+2*a2)%P], dim=-1)
        r2 = torch.stack([a2, (a1+a2)%P, (a0+a1+2*a2)%P], dim=-1)
        return (r0[...,0]*(r1[...,1]*r2[...,2] - r1[...,2]*r2[...,1])
              - r0[...,1]*(r1[...,0]*r2[...,2] - r1[...,2]*r2[...,0])
              + r0[...,2]*(r1[...,0]*r2[...,1] - r1[...,1]*r2[...,0])) % P

    @staticmethod
    def conj(a: torch.Tensor) -> torch.Tensor:
        """Galois conjugation J: a ↦ CRT⁻¹(φ₁(a), Frob(φ₂(a))).  J² = id.
        Verified: φ₁(J(a)) = φ₁(a)  and  φ₂(J(a)) = Frob(φ₂(a))."""
        phi1 = (a[...,0] + 7*a[...,1] + 10*a[...,2]) % P
        c0   = (a[...,0] + 11*a[...,2]) % P
        c1   = (a[...,1] +  7*a[...,2]) % P
        # Frobenius on F_169: (c0, c1) ↦ (c0 + 7*c1, 12*c1)
        return torch.stack([(phi1 + 7*c1) % P,
                            (3*phi1 + 10*c0 + 12*c1) % P,
                            (7*phi1 + 6*c0) % P], dim=-1)

    @staticmethod
    def phi1(a: torch.Tensor) -> torch.Tensor:
        """CRT Z_13 projection: a₀ + 7a₁ + 10a₂.  → scalar."""
        return (a[...,0] + 7*a[...,1] + 10*a[...,2]) % P

    @staticmethod
    def phi2(a: torch.Tensor) -> torch.Tensor:
        """CRT F_169 projection.  → (...,2)."""
        return torch.stack([(a[...,0] + 11*a[...,2]) % P,
                            (a[...,1] +  7*a[...,2]) % P], dim=-1)


class CRT:
    """Z_13[η] ≅ Z_13 × F_169.  Delegates to Z13 (zero duplication)."""
    phi1        = staticmethod(Z13.phi1)
    phi2        = staticmethod(Z13.phi2)
    galois_conj = staticmethod(Z13.conj)

    @staticmethod
    def compose(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """φ⁻¹(α, β).  alpha:(...,)  beta:(...,2) → (...,3)."""
        dev = alpha.device
        e1  = torch.tensor([1,3,7], dtype=torch.long, device=dev)
        e2  = torch.tensor([0,10,6], dtype=torch.long, device=dev)
        ee2 = torch.tensor([6,6,3], dtype=torch.long, device=dev)
        return (alpha.unsqueeze(-1)*e1
              + beta[...,0:1]*e2
              + beta[...,1:2]*ee2) % P

    @staticmethod
    def decompose(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return Z13.phi1(a), Z13.phi2(a)

    @staticmethod
    def is_unit(a: torch.Tensor) -> torch.Tensor:
        p2 = Z13.phi2(a)
        return (Z13.phi1(a) != 0) & ((p2[...,0] != 0) | (p2[...,1] != 0))


class F169:
    """F_13[α] / (α² + 6α + 2),  α² = 7α + 11.
    λ₂ = α = (0,1),  λ₃ = ᾱ = (7,12).
    """
    lam2 = torch.tensor([0, 1], dtype=torch.long)
    lam3 = torch.tensor([7, 12], dtype=torch.long)

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        c0 = (a[...,0]*b[...,0] + 11*a[...,1]*b[...,1]) % P
        c1 = (a[...,0]*b[...,1] + a[...,1]*b[...,0] + 7*a[...,1]*b[...,1]) % P
        return torch.stack([c0, c1], dim=-1)

    @staticmethod
    def norm(a: torch.Tensor) -> torch.Tensor:
        return (a[...,0]**2 - 6*a[...,0]*a[...,1] + 2*a[...,1]**2) % P

    @staticmethod
    def inv(a: torch.Tensor) -> torch.Tensor:
        N = F169.norm(a);  Ni = N.pow(P-2) % P
        return torch.stack([((a[...,0] - 6*a[...,1])*Ni) % P,
                            ((-a[...,1]) % P * Ni) % P], dim=-1)

    @staticmethod
    def frobenius(a: torch.Tensor) -> torch.Tensor:
        """σ: a₀ + a₁α ↦ a₀ + a₁ᾱ."""
        return torch.stack([(a[...,0] - 6*a[...,1]) % P,
                            (-a[...,1]) % P], dim=-1)

    @staticmethod
    def pow(a: torch.Tensor, n: int) -> torch.Tensor:
        n = int(n) % 168
        r = torch.zeros(*a.shape[:-1], 2, dtype=torch.long, device=a.device)
        r[...,0] = 1;  b = a.clone()
        if n & 1:   r = F169.mul(r, b)
        b = F169.mul(b, b);  n >>= 1
        if n & 1:   r = F169.mul(r, b)
        b = F169.mul(b, b);  n >>= 1
        if n & 1:   r = F169.mul(r, b)
        b = F169.mul(b, b);  n >>= 1
        if n & 1:   r = F169.mul(r, b)
        b = F169.mul(b, b);  n >>= 1
        if n & 1:   r = F169.mul(r, b)
        b = F169.mul(b, b);  n >>= 1
        if n & 1:   r = F169.mul(r, b)
        b = F169.mul(b, b);  n >>= 1
        if n & 1:   r = F169.mul(r, b)
        b = F169.mul(b, b);  n >>= 1
        if n & 1:   r = F169.mul(r, b)
        return r
