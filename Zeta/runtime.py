# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/runtime.py  —  Device Manager and System Runtime
=======================================================
This module provides the single entry point for the Zeta engine.

ZetaDevice : all precomputed tables as nn.Buffers, moved to device
             via a single .to(device) call.
ZetaRuntime: singleton runtime with forward pass, Buchberger training,
             Riemann Hypothesis check, and comprehensive benchmarking.

Mathematical context:
    Section 16  (Implementation Architecture)
    Section 17  (Runtime Axioms)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import Optional, Dict
import time
import torch
import torch.nn as nn

from .constants import (
    P, ORD, ZetaConfig,
    ETA_POW, ETA_IPOW, T3_POW, INV_TBL,
    P1_MAT, P23_MAT, V_EIG, W_EIG,
    S3_MATS, _VAL_LUT, _NTT_VALID, _NTT_FWD, _NTT_INV,
    ONE_R, ZERO_R, delta_max,
)
from .model import ZetaModel
from .dirac import AdelicDiracOperator
from .tokenizer import tokenise, detokenise


class ZetaDevice(nn.Module):
    """Device-local copy of all precomputed algebraic tables."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('eta_pow',  ETA_POW.clone())
        self.register_buffer('eta_ipow', ETA_IPOW.clone())
        self.register_buffer('t3_pow',   T3_POW.clone())
        self.register_buffer('inv_tbl',  INV_TBL.clone())
        self.register_buffer('p1_mat',   P1_MAT.clone())
        self.register_buffer('p23_mat',  P23_MAT.clone())
        self.register_buffer('v_eig',    V_EIG.clone())
        self.register_buffer('w_eig',    W_EIG.clone())
        self.register_buffer('s3_mats',  S3_MATS.clone())
        self.register_buffer('val_lut',  _VAL_LUT.clone())
        self.register_buffer('one_r',    ONE_R.clone())
        self.register_buffer('zero_r',   ZERO_R.clone())
        for n in _NTT_VALID:
            self.register_buffer(f'ntt_fwd_{n}', _NTT_FWD[n].clone())
            self.register_buffer(f'ntt_inv_{n}', _NTT_INV[n].clone())

    def mem_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())

    def mem_str(self) -> str:
        return f"{self.mem_bytes() / 1024**2:.2f} MB"


class ZetaRuntime:
    _instance: Optional['ZetaRuntime'] = None

    def __init__(self, device: torch.device,
                 V: int, D: int, N: int, ctx: int) -> None:
        self.device = device
        self.config = ZetaConfig
        self.tables = ZetaDevice()
        if str(device) != 'cpu':
            self.tables = self.tables.to(device)
        else:
            if torch.cuda.is_available():
                for name, buf in self.tables.named_buffers():
                    try:
                        pinned = buf.pin_memory()
                        setattr(self.tables, name.replace('.', '_'), pinned)
                    except Exception:
                        pass
        self.model = ZetaModel(V=V, D=D, N=N, ctx=ctx).to(device)
        self._V = V;  self._D = D;  self._N = N;  self._ctx = ctx

    @classmethod
    def init(cls, device: str = 'cpu',
             V:   int = ZetaConfig.V,
             D:   int = ZetaConfig.D,
             N:   int = ZetaConfig.N_LAYERS,
             ctx: int = ZetaConfig.CTX,
             force_reinit: bool = False) -> 'ZetaRuntime':
        dev = torch.device(device)
        if cls._instance is None or force_reinit or cls._instance.device != dev:
            t0 = time.perf_counter()
            cls._instance = cls(dev, V, D, N, ctx)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[ZetaRuntime] device={dev}  tables={cls._instance.tables.mem_str()}  "
                  f"init={elapsed:.1f}ms  model=V{V}·D{D}·N{N}·ctx{ctx}  "
                  f"params={sum(p.numel() for p in cls._instance.model.parameters())}")
        return cls._instance

    def forward(self, tokens: torch.Tensor):
        with torch.no_grad():
            return self.model(tokens.to(self.device))

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
        with torch.no_grad():
            return self.model.train_step(tokens.to(self.device), targets.to(self.device))

    def encode(self, text: str) -> torch.Tensor:
        ids = tokenise(text)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    def decode(self, logits: torch.Tensor) -> str:
        ids = logits.argmax(-1)[0].tolist()
        return detokenise(ids)

    def rh_check(self, L: int = 12) -> dict:
        return AdelicDiracOperator.rh_check(L=L, dev=self.device)

    def benchmark(self, B: int = 4, L: int = 64) -> Dict[str, float]:
        dev = self.device
        from .ring import Z13
        from .kernel import PAdicKernel
        from .ntt import NTT
        from .witt import WittVector

        tokens = torch.randint(0, self._V, (B, L), dtype=torch.long, device=dev)
        results: Dict[str, float] = {}
        for _ in range(3):
            self.forward(tokens)
        N_REPS = 20
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            self.forward(tokens)
        results['forward_ms'] = (time.perf_counter()-t0)/N_REPS*1000
        # ring mul
        a = ETA_POW[1].to(dev);  b = ETA_POW[3].to(dev)
        t0 = time.perf_counter()
        for _ in range(100_000):  Z13.mul(a, b)
        results['ring_mul_us'] = (time.perf_counter()-t0)/100000*1e6
        # ring inv
        t0 = time.perf_counter()
        for _ in range(100_000):  Z13.inv(a)
        results['ring_inv_us'] = (time.perf_counter()-t0)/100000*1e6
        # T3 lookup
        t0 = time.perf_counter()
        for _ in range(100_000):  _ = T3_POW[42]
        results['t3_lookup_us'] = (time.perf_counter()-t0)/100000*1e6
        # kernel (tree-based, O(L*log_13(L)))
        t0 = time.perf_counter()
        x_test = torch.randint(0, P, (1, L, 8, 3), dtype=torch.long, device=dev)
        for _ in range(100):  PAdicKernel.apply_fast(x_test)
        results['kernel_tree_ms'] = (time.perf_counter()-t0)/100*1000
        # NTT
        for n in [4, 7, 12, 14, 28]:
            if n in _NTT_VALID:
                x = ETA_POW[:n].to(dev)
                t0 = time.perf_counter()
                for _ in range(10_000):  NTT.ntt(x, n)
                results[f'ntt_{n}_us'] = (time.perf_counter()-t0)/10000*1e6
        # Witt
        xw = WittVector.from_ring(a, ZetaConfig.PREC)
        yw = WittVector.from_ring(b, ZetaConfig.PREC)
        t0 = time.perf_counter()
        for _ in range(50_000):  WittVector.wadd(xw, yw)
        results['witt_add_us'] = (time.perf_counter()-t0)/50000*1e6
        t0 = time.perf_counter()
        for _ in range(20_000):  WittVector.wmul(xw, yw)
        results['witt_mul_us'] = (time.perf_counter()-t0)/20000*1e6
        # delta_max
        t0 = time.perf_counter()
        for _ in range(1_000_000):  delta_max(L)
        results['delta_max_us'] = (time.perf_counter()-t0)/1000000*1e6
        return results

    def print_benchmark(self, B: int = 4, L: int = 64) -> None:
        r = self.benchmark(B, L)
        w = 50
        print(f"\n{'═'*w}")
        print(f" ZetaRuntime Benchmark  device={self.device}  B={B}  L={L}")
        print(f"{'─'*w}")
        for k, v in r.items():
            unit = 'ms' if k.endswith('_ms') else 'µs'
            print(f"  {k:<28} {v:>8.3f} {unit}")
        print(f"{'═'*w}\n")

    def summary(self) -> str:
        lines = [
            "═"*60,
            " Zeta v7.0.0  —  Z_13[η] Cubic p-Adic Integer AI",
            "─"*60,
            f"  device        : {self.device}",
            f"  tables        : {self.tables.mem_str()} (all torch.long)",
            f"  model         : V={self._V} D={self._D} N={self._N} ctx={self._ctx}",
            f"  nn.Parameter  : {sum(p.numel() for p in self.model.parameters())}",
            f"  NTT sizes     : {_NTT_VALID}",
            f"  ring char     : {P}   orbit: {ORD}",
            f"  inv_tbl shape : {INV_TBL.shape}  (O(1) ring inversion)",
            f"  val_lut shape : {_VAL_LUT.shape}  (O(1) p-adic valuation)",
            "─"*60,
            " Kernel: G(i,j)=eta^{-v_13(|i-j|)}  —  THE ONLY KERNEL",
            " Strong triangle inequality enforced",
            "─"*60,
            " Modules: Z13 · CRT · F169 · S3Galois · Spectral · PAdicKernel",
            "          NTT · WittVector · HenselLifter · GQMState",
            "          MahlerExpansion · PAdicLaplacian · ZetaFunctionRing",
            "          SpectralFlow · EntanglementGeometry · AdelicDiracOperator",
            "          Teichmuller · CrystalLattice · AdelicProduct",
            "          SpectralAttention · ZRingEmbed · TatePE · ZetaModel",
            "          MERAChunker · OrbitPlanner · ReversibleGenerator",
            "          SpectralZeta · BerrySVD · QuantumRollback",
            "          CounterfactualBranch · FullAutonomyCycle · ARDTArchitecture",
            "═"*60,
        ]
        return "\n".join(lines)
