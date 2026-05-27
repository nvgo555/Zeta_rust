# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/kernel.py  —  The Ultrametric Kernel via Bruhat-Tits Tree
===============================================================
Optimized implementation:
    • build_tree:  einsum replaced by bmm + permute (memory-aligned)
    • tree_attend: repeat_interleave replaced by expand+reshape (O(L) not O(L*factor))
    • apply_fast:  fused UP+DOWN for small L (≤169) without intermediate list

Complexity: O(L * log_13(L)) time, O(L) memory.  Never O(L²).

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import List
import torch

from .constants import P, ORD, ETA_POW, ETA_IPOW, T3_POW, P1_MAT, P23_MAT, ONE_R, _VAL_LUT, _VAL_MAX
from .ring import Z13
from .spectral import proj


class PAdicKernel:
    """THE ONLY KERNEL.  Exclusively tree-based.  No dense matrix ever."""

    @staticmethod
    def n_levels(L: int) -> int:
        h, n = 1, 0
        while h < L:
            h *= 13
            n += 1
        return n

    @staticmethod
    def _val(dist: torch.Tensor) -> torch.Tensor:
        return _VAL_LUT.to(dist.device)[dist.clamp(0, _VAL_MAX - 1)]

    @staticmethod
    def build_tree(x: torch.Tensor) -> List[torch.Tensor]:
        """UP pass.  Optimized: bmm instead of einsum for T3 evolution."""
        B, L, D, _ = x.shape
        dev = x.device
        H = PAdicKernel.n_levels(L)
        levels: List[torch.Tensor] = [x]
        current = x
        for h in range(H):
            Lc = current.shape[1]
            if Lc < 13:
                break
            Lpad = ((Lc + 12) // 13) * 13
            if Lpad > Lc:
                pad = torch.zeros(B, Lpad - Lc, D, 3, dtype=torch.long, device=dev)
                current = torch.cat([current, pad], dim=1)
            nb = current.shape[1] // 13
            grouped = current.reshape(B, nb, 13, D, 3).sum(dim=2) % P  # (B, nb, D, 3)
            n_step = pow(13, h + 1, ORD)
            Mh = T3_POW[n_step].to(dev)  # (3, 3)
            # Optimized: bmm over (B*nb*D, 3, 3) instead of einsum
            flat = grouped.reshape(-1, 3)  # (B*nb*D, 3)
            evolved = (flat @ Mh.T) % P     # (B*nb*D, 3)
            evolved = evolved.reshape(B, nb, D, 3)
            coarse = proj(evolved, P1_MAT.to(dev))
            levels.append(coarse)
            current = coarse
        return levels

    @staticmethod
    def tree_attend(levels: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """DOWN pass.  Optimized: expand+reshape instead of repeat_interleave."""
        if len(levels) == 1:
            return x
        B, L, D, _ = x.shape
        dev = x.device
        out = x.clone()
        for h, sh in enumerate(levels[1:], start=1):
            factor = 13 ** h
            nb = sh.shape[1]
            # Memory-efficient: expand instead of repeat_interleave
            up = sh.unsqueeze(2).expand(-1, -1, factor, -1, -1)  # (B, nb, factor, D, 3)
            up = up.reshape(B, nb * factor, D, 3)
            if up.shape[1] < L:
                continue
            up = up[:, :L]
            n_inv = (ORD - pow(13, h, ORD)) % ORD
            Mi = T3_POW[n_inv].to(dev)
            flat = up.reshape(-1, 3)
            up = (flat @ Mi.T) % P
            up = up.reshape(B, L, D, 3)
            w = ETA_IPOW.to(dev)[h % ORD]
            up = Z13.mul(up, w.view(1, 1, 1, 3).expand_as(up))
            up_sub = proj(up, P23_MAT.to(dev))
            out = (out + up_sub) % P
        return out

    @staticmethod
    def apply(x: torch.Tensor) -> torch.Tensor:
        """Public interface."""
        levels = PAdicKernel.build_tree(x)
        return PAdicKernel.tree_attend(levels, x)

    @staticmethod
    def apply_fast(x: torch.Tensor) -> torch.Tensor:
        """Fused path for L ≤ 169 (2 levels max).  Avoids list allocation."""
        B, L, D, _ = x.shape
        dev = x.device
        if L <= 13:
            return x
        # Level 1
        Lpad = ((L + 12) // 13) * 13
        if Lpad > L:
            pad = torch.zeros(B, Lpad - L, D, 3, dtype=torch.long, device=dev)
            padded = torch.cat([x, pad], dim=1)
        else:
            padded = x
        nb = padded.shape[1] // 13
        g1 = padded.reshape(B, nb, 13, D, 3).sum(dim=2) % P
        n1 = pow(13, 1, ORD)
        M1 = T3_POW[n1].to(dev)
        e1 = (g1.reshape(-1, 3) @ M1.T) % P
        c1 = proj(e1.reshape(B, nb, D, 3), P1_MAT.to(dev))
        # Level 2 (if needed)
        if nb >= 13:
            L2pad = ((nb + 12) // 13) * 13
            if L2pad > nb:
                pad2 = torch.zeros(B, L2pad - nb, D, 3, dtype=torch.long, device=dev)
                c1p = torch.cat([c1, pad2], dim=1)
            else:
                c1p = c1
            nb2 = c1p.shape[1] // 13
            g2 = c1p.reshape(B, nb2, 13, D, 3).sum(dim=2) % P
            n2 = pow(13, 2, ORD)
            M2 = T3_POW[n2].to(dev)
            e2 = (g2.reshape(-1, 3) @ M2.T) % P
            c2 = proj(e2.reshape(B, nb2, D, 3), P1_MAT.to(dev))
            # Broadcast c2 back to L
            f2 = 13 * 13
            up2 = c2.unsqueeze(2).expand(-1, -1, f2, -1, -1).reshape(B, nb2 * f2, D, 3)[:, :L]
            Mi2 = T3_POW[(ORD - n2) % ORD].to(dev)
            up2 = (up2.reshape(-1, 3) @ Mi2.T) % P
            up2 = up2.reshape(B, L, D, 3)
            w2 = ETA_IPOW.to(dev)[2 % ORD]
            up2 = Z13.mul(up2, w2.view(1, 1, 1, 3))
            out = (x + proj(up2, P23_MAT.to(dev))) % P
            # Also broadcast c1
            f1 = 13
            up1 = c1.unsqueeze(2).expand(-1, -1, f1, -1, -1).reshape(B, nb * f1, D, 3)[:, :L]
            Mi1 = T3_POW[(ORD - n1) % ORD].to(dev)
            up1 = (up1.reshape(-1, 3) @ Mi1.T) % P
            up1 = up1.reshape(B, L, D, 3)
            w1 = ETA_IPOW.to(dev)[1 % ORD]
            up1 = Z13.mul(up1, w1.view(1, 1, 1, 3))
            out = (out + proj(up1, P23_MAT.to(dev))) % P
            return out
        else:
            # Only 1 level
            f1 = 13
            up1 = c1.unsqueeze(2).expand(-1, -1, f1, -1, -1).reshape(B, nb * f1, D, 3)[:, :L]
            Mi1 = T3_POW[(ORD - n1) % ORD].to(dev)
            up1 = (up1.reshape(-1, 3) @ Mi1.T) % P
            up1 = up1.reshape(B, L, D, 3)
            w1 = ETA_IPOW.to(dev)[1 % ORD]
            up1 = Z13.mul(up1, w1.view(1, 1, 1, 3))
            return (x + proj(up1, P23_MAT.to(dev))) % P

    @staticmethod
    def elem(i: int, j: int, dev: torch.device) -> torch.Tensor:
        diff = abs(i - j)
        if diff == 0:
            return ONE_R.to(dev)
        val = int(_VAL_LUT[min(diff, _VAL_MAX - 1)].item())
        return ETA_IPOW.to(dev)[val % ORD]

    @staticmethod
    def strong_triangle_inequality(i: int, j: int, k: int) -> bool:
        vi = int(_VAL_LUT[abs(i - j)].item())
        vj = int(_VAL_LUT[abs(j - k)].item())
        vk = int(_VAL_LUT[abs(i - k)].item())
        return vk >= min(vi, vj)

    @staticmethod
    def batch_triangle_check(L: int, n_samples: int = 200) -> bool:
        for _ in range(n_samples):
            i = torch.randint(0, L, (1,)).item()
            j = torch.randint(0, L, (1,)).item()
            k = torch.randint(0, L, (1,)).item()
            if not PAdicKernel.strong_triangle_inequality(i, j, k):
                return False
        return True
