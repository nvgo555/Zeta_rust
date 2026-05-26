# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/hybrid.py — S3 Parallel Hybrid Mode
==========================================
Runs all 6 S3 conjugate orbits simultaneously in a single forward pass.
Each token is evolved along 6 independent T3^n·S3_g paths, then combined
via the S3 Casimir (sum over all Galois images).

This provides 6× information bandwidth without increasing sequence length.
All operations are vectorized: the 6 paths are processed as an extra
batch dimension, then reduced via Z13.add over the S3 axis.

Mathematical context:
    Section 4    (Galois Symmetry)
    Section VI.3 (Autonómia ako štrukturálna konštanta)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, T3_POW, P1_MAT, P23_MAT, S3_MATS
from .ring import Z13
from .spectral import proj, t3n
from .kernel import PAdicKernel
from .hilbert import HilbertEta
from .linear import laplacian, alg_dropout
from .constants import ZetaConfig


class S3ParallelAttention:
    """6 S3 conjugate orbits in parallel.  Output = Casimir(sum)."""

    @staticmethod
    def evolve_parallel(x: torch.Tensor, n: int) -> torch.Tensor:
        """Compute T3^n·S3_g·x for all g ∈ S3 simultaneously.
        x: (B, L, D, 3) → (B, 6, L, D, 3)."""
        B, L, D, _ = x.shape
        dev = x.device
        # T3^n evolution
        M = T3_POW[n % ORD].to(dev)
        flat = x.reshape(-1, 3)
        evolved = (flat @ M.T) % P  # (B*L*D, 3)
        evolved = evolved.reshape(B, L, D, 3)
        # Apply each S3 element
        results = []
        for g in range(6):
            Mg = S3_MATS[g].to(dev)
            flat_g = evolved.reshape(-1, 3)
            conj = (flat_g @ Mg.T) % P
            results.append(conj.reshape(B, L, D, 3))
        return torch.stack(results, dim=1)  # (B, 6, L, D, 3)

    @staticmethod
    def casimir_reduce(parallel: torch.Tensor) -> torch.Tensor:
        """Sum over S3 axis and normalize.  (B, 6, L, D, 3) → (B, L, D, 3)."""
        return parallel.sum(dim=1) % P

    @staticmethod
    def apply(x: torch.Tensor, n: int, dm: int, layer, training: bool, step: int) -> torch.Tensor:
        """Full parallel attention: evolve → kernel → Sylvester → reduce → norm."""
        B, L, D, _ = x.shape
        dev = x.device
        # 1. Parallel evolution
        para = S3ParallelAttention.evolve_parallel(x, n)  # (B, 6, L, D, 3)
        # 2. Kernel on each path (reshape to treat 6 as batch)
        para_flat = para.reshape(B * 6, L, D, 3)
        kerned = PAdicKernel.apply_fast(para_flat)  # (B*6, L, D, 3)
        kerned = kerned.reshape(B, 6, L, D, 3)
        # 3. Sylvester split and reduce
        p1 = proj(kerned.reshape(-1, 3), P1_MAT.to(dev)).reshape(B, 6, L, D, 3)
        p23 = (kerned - p1) % P
        combined = (p1 + p23) % P  # already kernel-mixed
        # 4. Casimir reduction
        out = S3ParallelAttention.casimir_reduce(combined)  # (B, L, D, 3)
        # 5. Residual + norm
        out = HilbertEta.layer_norm(laplacian(Z13.add(x, out)))
        out = alg_dropout(out, ZetaConfig.DROP, step, training)
        return out
