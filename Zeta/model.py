# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/model.py  —  Zeta p-Adic Language Model
=============================================
Architecture:
    embed → TatePE → N × (t3n/S3 evolution + SpectralAttention) → multi-Witt head

Scaling features:
    • Multi-layer Witt forward — all PREC layers combined with weights 1,2,3,4
    • S3 conjugate orbits — 6 independent evolution paths, autonomously cycled
    • Convergent decay — half-step correction (7 ≡ 2^{-1} mod 13) when norm > 6
    • Hensel lift — precision escalation via witt_head

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

from .constants import P, ORD, ETA_POW, ZetaConfig
from .spectral import t3n, t3n_s3
from .embed import ZRingEmbed, TatePE
from .attention import SpectralAttention
from .constants import delta_max
from .ring import Z13
from .buchberger import BuchbergerEngine
from .gqm import ErrorCache
from .witt import WittVector


class ZetaModel(nn.Module):
    def __init__(self,
                 V:   int = ZetaConfig.V,
                 D:   int = ZetaConfig.D,
                 N:   int = ZetaConfig.N_LAYERS,
                 ctx: int = ZetaConfig.CTX):
        super().__init__();  assert D % 6 == 0
        self.V = V;  self.D = D;  self.N = N;  self.ctx = ctx;  self.step = 0
        self.prec = 1
        PREC_MAX = ZetaConfig.PREC
        self.embed = ZRingEmbed(V, D);  self.pe = TatePE(D)
        self.layers = nn.ModuleList([SpectralAttention(D) for _ in range(N)])
        d = torch.arange(D).unsqueeze(1) * V + torch.arange(V).unsqueeze(0)
        self.register_buffer('head', ETA_POW[d % ORD])
        w = WittVector.from_ring(self.head, PREC_MAX)
        self.register_buffer('witt_head', w)
        self.error_cache: ErrorCache | None = None
        self._last_x: torch.Tensor | None = None
        self.s3_orbit = -1  # -1 = T3 only, 0..5 = S3 conjugate active

    def _sync_head(self) -> None:
        """Synchronize composite head from all active Witt layers."""
        dev = self.head.device
        active = self.witt_head[:, :, :self.prec, :].to(dev)
        # Combine layers with weights 1,2,3,... mod 13
        w = torch.arange(1, self.prec + 1, device=dev).view(1, 1, -1, 1)
        combo = (active * w).sum(2) % P  # (D, V, 3)
        self.head.data = combo

    def _lift(self) -> None:
        if self.prec < ZetaConfig.PREC:
            self.prec += 1
            self._sync_head()

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, L = tokens.shape;  dev = tokens.device;  assert L <= self.ctx
        dm = delta_max(L)
        aud = {}
        x = self.pe(self.embed(tokens.to(dev)))

        # Evolution: T3 or S3 conjugate or parallel hybrid
        def evolve(x_in, n):
            if self.s3_orbit >= 0:
                return t3n_s3(x_in, n, self.s3_orbit)
            return t3n(x_in, n)

        # Kernel: fast fused path for L ≤ 169, standard for longer
        def kernel(x_in):
            _, L, _, _ = x_in.shape
            if L <= 169:
                return PAdicKernel.apply_fast(x_in)
            return PAdicKernel.apply(x_in)

        if self.N > 0:
            x = evolve(x, 1);  self.layers[0]._step = self.step;  x, la = self.layers[0](x, dm, 0);  aud['L0'] = la
        if self.N > 1:
            x = evolve(x, 2);  self.layers[1]._step = self.step;  x, la = self.layers[1](x, dm, 1);  aud['L1'] = la
        if self.N > 2:
            x = evolve(x, 3);  self.layers[2]._step = self.step;  x, la = self.layers[2](x, dm, 2);  aud['L2'] = la
        if self.N > 3:
            x = evolve(x, 4);  self.layers[3]._step = self.step;  x, la = self.layers[3](x, dm, 3);  aud['L3'] = la
        if self.N > 4:
            x = evolve(x, 5);  self.layers[4]._step = self.step;  x, la = self.layers[4](x, dm, 4);  aud['L4'] = la
        if self.N > 5:
            x = evolve(x, 6);  self.layers[5]._step = self.step;  x, la = self.layers[5](x, dm, 5);  aud['L5'] = la
        if self.N > 6:
            x = evolve(x, 7);  self.layers[6]._step = self.step;  x, la = self.layers[6](x, dm, 6);  aud['L6'] = la
        if self.N > 7:
            x = evolve(x, 8);  self.layers[7]._step = self.step;  x, la = self.layers[7](x, dm, 7);  aud['L7'] = la
        if self.N > 8:
            x = evolve(x, 9);  self.layers[8]._step = self.step;  x, la = self.layers[8](x, dm, 8);  aud['L8'] = la
        if self.N > 9:
            x = evolve(x, 10); self.layers[9]._step = self.step;  x, la = self.layers[9](x, dm, 9);  aud['L9'] = la
        if self.N > 10:
            x = evolve(x, 11); self.layers[10]._step = self.step; x, la = self.layers[10](x, dm, 10); aud['L10'] = la

        # Multi-layer Witt head: combine all active precision layers
        self._sync_head()
        H = self.head.to(dev)
        sc = Z13.trace(Z13.mul(x.unsqueeze(-2), H.unsqueeze(0).unsqueeze(0))).sum(2) % P
        self._last_x = x.detach().clone()
        self.step += 1;  aud['dm'] = dm
        return sc, aud

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
        B, L = tokens.shape
        sc, _ = self(tokens)
        preds = sc.argmax(-1)
        tgts = targets.to(tokens.device)
        hall = int((preds != tgts).sum().item())
        delta_norm = 0

        if hall > 0 and self._last_x is not None:
            delta = BuchbergerEngine.nullstellensatz_correction(
                self._last_x, preds, tgts, self.step
            )
            delta_norm = int(Z13.norm(delta).sum().item())
            if delta_norm > 6:
                delta = Z13.smul(delta, 7)  # 7 ≡ 2^{-1} mod 13
                delta_norm = int(Z13.norm(delta).sum().item())

            delta_witt = WittVector.from_ring(delta, ZetaConfig.PREC)
            active = self.witt_head[:, :, :self.prec, :]
            new_active = WittVector.wadd(active, delta_witt[:, :, :self.prec, :])
            self.witt_head[:, :, :self.prec, :].data = new_active
            self._sync_head()

            if self.error_cache is None:
                self.error_cache = ErrorCache(K=min(self.D * self.V, 64))
            self.error_cache.absorb(delta, self.head.device)

            spec_corr = self.error_cache.spectral_correction(self.head.shape, self.head.device)
            spec_corr = Z13.smul(spec_corr, 2)
            spec_witt = WittVector.from_ring(spec_corr, ZetaConfig.PREC)
            active2 = self.witt_head[:, :, :self.prec, :]
            new_active2 = WittVector.wadd(active2, spec_witt[:, :, :self.prec, :])
            self.witt_head[:, :, :self.prec, :].data = new_active2
            self._sync_head()

            if self.error_cache.step > 0:
                from .entropy import EntropyMonitor
                em = EntropyMonitor(threshold=10)
                if em.check(delta_norm) and self.prec < ZetaConfig.PREC:
                    self._lift()

        return {
            'hall': hall, 'step': self.step, 'delta_norm': delta_norm,
            'cache_steps': self.error_cache.step if self.error_cache else 0,
            'prec': self.prec, 's3_orbit': self.s3_orbit,
        }
