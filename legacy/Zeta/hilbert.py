# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/hilbert.py  -  Hilbert-eta Inner Product and Algebraic Layer Norm
=======================================================================
This module defines the inner product <u,v>_eta = sum_k Z13.mul(u_k,v_k)
* eta^{-k} and the algebraic layer normalization x * ||x||_eta^{-1},
which replaces Euclidean LayerNorm in the Zeta engine.

Mathematical context:
    Section 2.3  (Ring Arithmetic)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_IPOW
from .ring import Z13


class HilbertEta:

    @staticmethod
    def inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        D = u.shape[-2];  dev = u.device
        wts = ETA_IPOW[:D].to(dev)
        return Z13.mul(Z13.mul(u, v), wts.expand_as(u)).sum(-2) % P

    @staticmethod
    def norm_sq(u: torch.Tensor) -> torch.Tensor:
        D = u.shape[-2];  dev = u.device
        wts = ETA_IPOW[:D].to(dev)
        return Z13.mul(Z13.mul(u, u), wts.expand_as(u)).sum(-2) % P

    @staticmethod
    def layer_norm(x: torch.Tensor) -> torch.Tensor:
        B, L, D, _ = x.shape;  flat = x.reshape(B*L, D, 3)
        ns = HilbertEta.norm_sq(flat)
        inv_ns = Z13.inv(ns).unsqueeze(1).expand(-1, D, 3)
        return Z13.mul(flat, inv_ns).reshape(B, L, D, 3) % P
