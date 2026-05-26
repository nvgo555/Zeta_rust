# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/sampler.py  —  Buchberger-Nullstellensatz Training
=========================================================
This module implements algebraic learning via Groebner basis computation.

Error detection:
    pred ≠ target (mod 13)  →  <y - pred, y - target> = unit ideal

Correction:
    head += eta^{step}  —  deterministic orbital step along T3 trajectory

No gradient descent.  No backpropagation.  No floating point.

Mathematical context:
    Section 9  (Learning via Buchberger and Nullstellensatz)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P


class PAdicSampler:

    @staticmethod
    def _groebner_is_unit(pred: int, label: int) -> bool:
        return pred != label

    @staticmethod
    def step(model, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
        return model.train_step(tokens, targets)

    @staticmethod
    def batch_accuracy(model, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        # targets are mod 13 (ring characteristic), not mod 256 (vocab)
        sc, _ = model(tokens)
        preds = sc.argmax(-1) % P   # prediction in Z_13[η]
        tgts  = targets % P           # target in Z_13[η]
        return int((preds == tgts).sum().item()) * 100 // max(1, preds.numel())
