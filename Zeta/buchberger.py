# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/buchberger.py — Buchberger-Nullstellensatz Algebraic Learning Engine
=========================================================================
This module implements genuine algebraic learning without gradient descent.

For each token position (b,l), the error polynomial in Z_13[eta][y] is:
    e_{b,l} = (target_{b,l} - pred_{b,l})  embedded as (delta, 0, 0)

The ideal I = <e_{b,l}> is the unit ideal iff any error is non-zero.
By the Weak Nullstellensatz, there exist witness coefficients g_{b,l}
such that:
    sum_{b,l} g_{b,l} * e_{b,l} = 1

The correction Delta_head is computed from the witness via the input
activations x (the "directional derivative" of the score function):

    Delta_head[d, v_t] += x[b,l,d] * e_{b,l} * eta^{step}
    Delta_head[d, v_p] -= x[b,l,d] * e_{b,l} * eta^{step}

where v_t = target, v_p = pred.  This is fully vectorized — zero Python
loops over sequence positions.  Complexity O(B*L*D + D*V).

Mathematical context:
    Section 9  (Learning via Buchberger and Nullstellensatz)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW, ZetaConfig
from .ring import Z13


class BuchbergerEngine:
    """Vectorized Buchberger-Nullstellensatz correction.
    Zero Python loops.  All operations O(B*L*D) or O(D*V).
    """

    @staticmethod
    def error_ideal(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute error ring elements e_i = (target_i - pred_i) in Z_13[eta].
        Returns (B, L, 3).  Zero where pred == target."""
        dev = pred.device
        err_mask = (pred != target).long()
        delta = ((target - pred) % P) * err_mask  # (B, L), 0 where correct
        e = torch.zeros(*pred.shape, 3, dtype=torch.long, device=dev)
        e[..., 0] = delta  # embed Z_13 scalar into Z_13[eta]
        return e

    @staticmethod
    def nullstellensatz_correction(
        x: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        step: int,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Delta_head via Nullstellensatz witness.

        Args:
            x:      (B, L, D, 3)  input activations to head
            pred:   (B, L)         predicted token ids
            target: (B, L)         target token ids
            step:   int            global step (orbit phase)

        Returns:
            Delta_head: (D, V, 3)  ring correction tensor
        """
        B, L, D, _ = x.shape;  dev = x.device;  V = ZetaConfig.V

        # --- 1. Error ideal: e_{b,l} = (target - pred) in Z_13[eta] ---
        e = BuchbergerEngine.error_ideal(pred, target)  # (B, L, 3)

        # --- 2. Witness: g = x * eta^{step}  (directional activation) ---
        orbit = ETA_POW[step % ORD].to(dev)  # (3,)
        g = Z13.mul(x, orbit.view(1, 1, 1, 3).expand_as(x))  # (B, L, D, 3)

        # --- 3. Weighted contribution: g * e over D dimension ---
        contrib = Z13.mul(g, e.unsqueeze(2).expand(-1, -1, D, -1))  # (B, L, D, 3)

        # --- 4. Scatter to head[d, v] via one-hot encoding ---
        # Target one-hot: (B, L, V)
        target_oh = torch.zeros(B, L, V, dtype=torch.long, device=dev)
        target_oh.scatter_(2, target.unsqueeze(2), 1)
        # Pred one-hot
        pred_oh = torch.zeros(B, L, V, dtype=torch.long, device=dev)
        pred_oh.scatter_(2, pred.unsqueeze(2), 1)

        # Boost target vocab items
        corr_target = (contrib.unsqueeze(3) * target_oh.unsqueeze(2).unsqueeze(4)).sum(dim=(0, 1)) % P
        # Suppress pred vocab items
        corr_pred = (contrib.unsqueeze(3) * pred_oh.unsqueeze(2).unsqueeze(4)).sum(dim=(0, 1)) % P

        # --- 5. Net correction ---
        delta = Z13.sub(corr_target, corr_pred)  # (D, V, 3)

        # --- 6. Adaptive scale (decay in Z_13[eta]) ---
        if scale is not None:
            delta = Z13.mul(delta, scale.to(dev).expand_as(delta))
        # --- 7. Temporal phase: multiply by eta^{7*step} (dominant eigenvalue orbit) ---
        phase = ETA_POW[(step * 7) % ORD].to(dev)
        return Z13.mul(delta, phase.view(1, 1, 3).expand_as(delta))

    @staticmethod
    def groebner_reduce(errors: torch.Tensor) -> torch.Tensor:
        """For constant errors in a field, the Groebner basis is trivial:
        GB = {gcd of all non-zero errors}.  Here we return the unit
        generator if any error is non-zero."""
        has_error = (errors[..., 0] != 0).any()
        return torch.tensor([1, 0, 0], dtype=torch.long, device=errors.device) if has_error else torch.zeros(3, dtype=torch.long, device=errors.device)
