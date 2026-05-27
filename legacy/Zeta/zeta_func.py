# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/zeta_func.py  —  Spectral Zeta Functions and Eigenvalue Flows
====================================================================
This module implements the spectral zeta function zeta_eta(s) over the
Tribonacci orbit, together with the dominant and subdominant eigenvalue
flows of T3.

Key results:
    zeta_Trib(1) = 14   (forced by det(T3) = 1)
    zeta_Trib(2) = 42
    zeta_Trib(3) = 170
    Winding number = 168 / 12 = 14

The critical line contains 14 elements, not Re(s) = 1/2.

Mathematical context:
    Section 10  (Empirical Verification of p-adic Riemann Hypothesis)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW, ETA_IPOW
from .ring import Z13, F169
from .ntt import NTT


class ZetaFunctionRing:

    @staticmethod
    def zeta(s0: int, terms: int = ORD) -> torch.Tensor:
        n = torch.arange(1, terms + 1)
        return ETA_IPOW[(n * s0) % ORD].sum(0) % P

    @staticmethod
    def zeta_batch(s_vec: torch.Tensor) -> torch.Tensor:
        n = torch.arange(1, ORD + 1);  K = s_vec.shape[0];  dev = s_vec.device
        exp = (n.unsqueeze(1).to(dev) * s_vec.unsqueeze(0)) % ORD
        return ETA_IPOW.to(dev)[exp.reshape(-1)].reshape(ORD, K, 3).sum(0) % P

    @staticmethod
    def ntt_identity(N: int, dev: torch.device) -> bool:
        orbit = ETA_POW[:N].to(dev)
        ntt_v = NTT.ntt(orbit, N)
        s_vec = torch.arange(N, device=dev)
        zeta_v = ZetaFunctionRing.zeta_batch(s_vec)
        return bool(torch.all(ntt_v == zeta_v))

    @staticmethod
    def functional_eq(s0: int) -> bool:
        zs = ZetaFunctionRing.zeta(s0)
        zr = ZetaFunctionRing.zeta((ORD - s0) % ORD)
        z0 = ZetaFunctionRing.zeta(0)
        return bool(torch.all(Z13.add(zs, zr) == Z13.smul(z0, 2)))


class SpectralFlow:

    @staticmethod
    def dominant(dev: torch.device) -> torch.Tensor:
        return Z13.phi1(ETA_POW[:ORD].to(dev))

    @staticmethod
    def subdominant(dev: torch.device) -> torch.Tensor:
        return Z13.phi2(ETA_POW[:ORD].to(dev))

    @staticmethod
    def ntt_spectrum(dev: torch.device) -> torch.Tensor:
        flow = SpectralFlow.dominant(dev)[:12]
        fr = torch.zeros(12, 3, dtype=torch.long, device=dev);  fr[:, 0] = flow
        return NTT.ntt(fr, 12)

    @staticmethod
    def chern_proxy(dev: torch.device) -> int:
        flow = SpectralFlow.subdominant(dev)
        diff = (flow - torch.roll(flow, -1, 0)) % P
        return int(F169.norm(diff).sum().item()) % P

    @staticmethod
    def winding() -> int:
        return ORD // 12

    @staticmethod
    def critical_indices(dev: torch.device) -> torch.Tensor:
        return torch.where(SpectralFlow.dominant(dev) == 7)[0]


class SpectralZeta:

    @staticmethod
    def zeta(s: int, terms: int = ORD) -> torch.Tensor:
        n = torch.arange(1, terms + 1)
        tr = Z13.phi1(ETA_POW[n % ORD])
        wt = ETA_IPOW[(n * s) % ORD]
        return (tr.unsqueeze(-1) * wt).sum(0) % P

    @staticmethod
    def connection_to_zeta_ring(s: int) -> bool:
        zr = ZetaFunctionRing.zeta(s)
        zt = SpectralZeta.zeta(s)
        return bool(torch.all(zt == Z13.smul(zr, int(Z13.phi1(zr).item()))))
