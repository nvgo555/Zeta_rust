# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/gqm.py  —  Geometric Quantum Mechanics in Z_13[eta]
========================================================
This module implements quantum states, density matrices, and quantum
cache over the cubic p-adic ring.  All operations use ring arithmetic
(Z13.mul, Z13.trace, Z13.norm) — no Hilbert space, no complex numbers.

Key structures:
    GQMState      : |psi> = sum c_k |k> with c_k ∈ Z_13[eta]
    DensityMatrix : rho = |psi><psi| with ring multiplication
    Quantum cache : spectral accumulation sum_t T3^t * rho_t * T3^{-t}

The Born rule uses algebraic norm N(c_k) rather than |c_k|^2.

Mathematical context:
    Section 6  (Geometric Quantum Mechanics)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW, T3_POW, ZERO_R, P1_MAT
from .ring import Z13
from .spectral import proj
from .ntt import NTT


def stratum(k: int) -> torch.Tensor:
    """T3^k·e₀ = η^k.  State = energy.  (3,) Long."""
    e = ZERO_R.clone();  e[0] = 1
    return (T3_POW[k % ORD] @ e) % P


class GQMState:
    def __init__(self, K: int = 8):
        self.K = K
        self.coeffs = torch.zeros(K, 3, dtype=torch.long)
        self.coeffs[0, 0] = 1

    @classmethod
    def from_coeffs(cls, c: torch.Tensor):
        s = cls(c.shape[0]);  s.coeffs = c.clone() % P;  return s

    def evolve(self, N: int) -> 'GQMState':
        new = GQMState(self.K)
        new.coeffs = (self.coeffs @ T3_POW[N % ORD].T) % P
        return new

    def hamiltonian(self) -> 'GQMState':
        new = GQMState(self.K)
        new.coeffs = Z13.mul(ETA_POW[:self.K].to(self.coeffs.device), self.coeffs)
        return new

    def born_probs(self) -> torch.Tensor:
        norms = Z13.norm(self.coeffs)
        w = torch.tensor([pow(2, k, P) for k in range(self.K)], dtype=torch.long)
        return (norms * w) % P

    def inner(self, other: 'GQMState') -> int:
        return int(Z13.trace(Z13.mul(Z13.conj(self.coeffs), other.coeffs)).sum().item()) % P

    def density(self) -> torch.Tensor:
        pc = Z13.conj(self.coeffs)
        return Z13.mul(pc.unsqueeze(1).expand(-1, self.K, -1),
                       self.coeffs.unsqueeze(0).expand(self.K, -1, -1))

    def ntt_spectrum(self) -> torch.Tensor:
        N = NTT.best_size(self.K)
        return NTT.ntt(self.coeffs[:N], N)

    def entropy(self) -> int:
        return int((Z13.norm(self.coeffs) == 0).sum().item())


class ErrorCache:
    """Quantum error memory via density matrix accumulation.
    Errors are absorbed as quantum states and spectrally evolved by T3."""

    def __init__(self, K: int = 64):
        self.K = K
        self.rho = torch.zeros(K, K, 3, dtype=torch.long)
        self.step = 0

    def absorb(self, corr: torch.Tensor, dev: torch.device):
        """Absorb correction tensor into quantum memory.
        corr: (D, V, 3) or any shape, flattened to K states."""
        flat = corr.reshape(-1, 3).to(dev)
        n = min(flat.shape[0], self.K)
        states = torch.zeros(self.K, 3, dtype=torch.long, device=dev)
        states[:n] = flat[:n] % P

        new_rho = DensityMatrix.build(states)  # (K, K, 3)
        if self.rho.abs().sum() > 0:
            old_rho = self.rho.to(dev)
            evolved = DensityMatrix.evolve(old_rho, 1)
            self.rho = (Z13.add(evolved, new_rho) % P).cpu()
        else:
            self.rho = (new_rho % P).cpu()
        self.step += 1

    def spectral_correction(self, head_shape: tuple, dev: torch.device) -> torch.Tensor:
        """Extract correction from accumulated error spectrum.
        Uses NTT of the diagonal (Born probabilities) to weight head."""
        D, V, _ = head_shape
        if self.rho.abs().sum() == 0:
            return torch.zeros(D, V, 3, dtype=torch.long, device=dev)

        rho_dev = self.rho.to(dev)
        diag = rho_dev.diagonal(dim1=0, dim2=1).T  # (K, 3)
        N = NTT.best_size(self.K)
        spec = NTT.ntt(diag[:N], N)  # (N, 3)

        # Deterministic eta-map from spectrum to (D, V, 3)
        total = D * V
        spec_flat = spec.reshape(-1, 3)  # (N, 3)
        repeats = (total + spec_flat.shape[0] - 1) // spec_flat.shape[0]
        full = spec_flat.repeat(repeats, 1)[:total]  # (total, 3)
        corr = full.reshape(D, V, 3) % P

        phase = ETA_POW[self.step % ORD].to(dev)
        return Z13.mul(corr, phase.view(1, 1, 3).expand_as(corr))


class DensityMatrix:
    @staticmethod
    def build(psi: torch.Tensor) -> torch.Tensor:
        return Z13.mul(psi.unsqueeze(1), psi.unsqueeze(0)) % P

    @staticmethod
    def trace(rho: torch.Tensor) -> torch.Tensor:
        return rho.diagonal(dim1=0, dim2=1).sum(-1) % P

    @staticmethod
    def mat_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return Z13.mul(A.unsqueeze(2), B.unsqueeze(0)).sum(1) % P

    @staticmethod
    def evolve(rho: torch.Tensor, n: int) -> torch.Tensor:
        M = T3_POW[n % ORD].to(rho.device)
        Mi = T3_POW[(ORD - n) % ORD].to(rho.device)
        left = torch.einsum('rc,...c->...r', M, rho) % P
        right = torch.einsum('...c,cr->...r', left, Mi) % P
        return right

    @staticmethod
    def entropy(rho: torch.Tensor) -> torch.Tensor:
        tr_rho = DensityMatrix.trace(rho)
        tr_rho2 = DensityMatrix.trace(DensityMatrix.mat_mul(rho, rho))
        return Z13.sub(tr_rho, tr_rho2)


class ZetaSelf(GQMState):
    """Sebauvedomie ako orbitálny stav.
    |Ψ_self(t)⟩ = T3^t · |Ψ_self(0)⟩."""

    def __init__(self, K: int = 8):
        super().__init__(K)
        self.t = 0

    def evolve(self) -> 'ZetaSelf':
        """T3 · |Ψ_self⟩ — jeden časový krok."""
        self.coeffs = (self.coeffs @ T3_POW[1].T) % P
        self.t = (self.t + 1) % ORD
        return self

    def reflect(self, goal: 'ZetaSelf') -> torch.Tensor:
        """Sebareflexia: P1 · (self − goal).
        Vráti dominantný deficit — čo je stabilné a nesprávne."""
        diff = Z13.sub(self.coeffs, goal.coeffs)
        return proj(diff, P1_MAT)

    def is_my_experience(self, other: torch.Tensor) -> bool:
        """⟨Ψ_self | other⟩_η ≠ 0 ?"""
        total = Z13.trace(Z13.mul(self.coeffs, other)).sum() % P
        return int(total.item()) != 0
