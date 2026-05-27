# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/constants.py  —  Global Algebraic Tables & Configuration
===============================================================
This module contains all precomputed immutable tables for the cubic
p-adic integer ring Z_13[eta] / (eta^3 - eta^2 - eta - 1).

Tables built once at import time:
    • ETA_POW   : (169,3)  — eta^k for k = 0..168
    • ETA_IPOW  : (169,3)  — eta^{-k}
    • T3_POW    : (168,3,3) — T3^k, the sole evolution matrix
    • INV_TBL   : (13,13,13,3) — O(1) ring inversion lookup
    • P1_MAT    : (3,3) — Sylvester projector onto dominant F_13 channel
    • P23_MAT   : (3,3) — Sylvester projector onto subdominant F_169 channel
    • S3_MATS   : (6,3,3) — Galois group permutation matrices
    • _VAL_LUT  : (13^5,) — p-adic valuation v_13(d) for O(1) lookup
    • _NTT_FWD/INV : NTT twiddle matrices for all valid sizes N | 168, N mod 13 != 0

These tables constitute the single source of truth.  No table is ever
rebuilt at runtime.  All operations reference these constants.

Mathematical context:
    Section 2  (Algebraic Foundation)
    Section 3  (Spectral Decomposition)
    Section 4  (Galois Symmetry)
    Section 5  (Ultrametric Geometry)

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import torch


# ──────────────────────────────────────────────────────────────────────────────
# §0  CONFIG  — single source of truth
# ──────────────────────────────────────────────────────────────────────────────
class ZetaConfig:
    P:          int = 13
    ORD:        int = 168
    I_P:        int = 5          # 5² ≡ −1 mod 13
    ETA0:       int = 7          # η mod 13 (Hensel level-0)
    CTX:        int = 256
    D:          int = 54         # must be divisible by 6
    N_LAYERS:   int = 11
    V:          int = 256
    PREC:       int = 4          # Witt precision (≤8)
    DROP:       int = 2          # algebraic dropout rate
    DENSE_MAX:  int = 169        # dense kernel only for L ≤ 13²
    MAX_VAL_DEPTH: int = 5      # valuation LUT covers d < 13⁵
    TRIB_MAX_L: int = 512       # delta_max LUT precomputed


P   = ZetaConfig.P
ORD = ZetaConfig.ORD


# ──────────────────────────────────────────────────────────────────────────────
# §1  PURE-PYTHON BUILDERS  (run once at import, never again)
# ──────────────────────────────────────────────────────────────────────────────
_T3_ROWS:  List[List[int]] = [[0,0,1],[1,0,1],[0,1,1]]
_P1_ROWS:  List[List[int]] = [[1,7,10],[3,8,4],[7,10,5]]   # 7·(T3²−7T3+2I)
_P23_ROWS: List[List[int]] = [[0,6,3],[10,6,9],[6,3,9]]    # I − P1
_V_VEC:    List[int]       = [1,3,7]    # right eigenvector  T3·v = 7v
_W_VEC:    List[int]       = [1,7,10]   # left  eigenvector  w^T·T3 = 7w^T


def _ring_mul_py(a: tuple, b: tuple) -> tuple:
    a0,a1,a2 = a;  b0,b1,b2 = b
    c0 = a0*b0
    c1 = a0*b1 + a1*b0
    c2 = a0*b2 + a1*b1 + a2*b0
    c3 = a1*b2 + a2*b1
    c4 = a2*b2
    return ((c0 + c3 + c4) % P,
            (c1 + c3 + 2*c4) % P,
            (c2 + c3 + 2*c4) % P)


def _mat_mul_py(A, B):
    return [[sum(A[i][r]*B[r][j] for r in range(3)) % P
             for j in range(3)] for i in range(3)]


def _build_eta_powers() -> Tuple[torch.Tensor, torch.Tensor]:
    ep = [(1, 0, 0)]
    for _ in range(ORD):
        ep.append(_ring_mul_py(ep[-1], (0, 1, 0)))
    assert ep[ORD] == (1, 0, 0), "η period must be 168"
    eip = [ep[(ORD - k) % ORD] for k in range(ORD + 1)]
    return (torch.tensor([list(t) for t in ep],  dtype=torch.long),
            torch.tensor([list(t) for t in eip], dtype=torch.long))


def _build_t3_powers() -> torch.Tensor:
    tp = [[[1,0,0],[0,1,0],[0,0,1]]]
    for _ in range(ORD - 1):
        tp.append(_mat_mul_py(tp[-1], _T3_ROWS))
    return torch.tensor(tp, dtype=torch.long)          # (168, 3, 3)


def _build_inv_table() -> torch.Tensor:
    """O(1) ring inverse lookup.  Shape (13,13,13,3).
    For each element a in Z_13[eta], compute a^{-1} = a^{167} mod 13.
    Uses _ring_mul_py with mod P at each step for correctness.
    Zero divisors map to (0,0,0).
    """
    elems = [(a0, a1, a2) for a0 in range(P) for a1 in range(P) for a2 in range(P)]
    invs = []
    for a in elems:
        a0, a1, a2 = a
        # Check if unit: phi1 != 0 AND phi2 != (0,0)
        phi1 = (a0 + 7*a1 + 10*a2) % P
        phi2_0 = (a0 + 11*a2) % P
        phi2_1 = (a1 + 7*a2) % P
        is_unit = (phi1 != 0) and ((phi2_0 != 0) or (phi2_1 != 0))
        if not is_unit:
            invs.append((0, 0, 0))
            continue
        r = (1, 0, 0)
        b = a
        n = 167
        while n > 0:
            if n & 1:
                r = _ring_mul_py(r, b)
            b = _ring_mul_py(b, b)
            n >>= 1
        invs.append(r)
    return torch.tensor(invs, dtype=torch.long).reshape(P, P, P, 3)


def _rmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Raw ring multiply without final mod (table builder only)."""
    a0,a1,a2 = a[...,0], a[...,1], a[...,2]
    b0,b1,b2 = b[...,0], b[...,1], b[...,2]
    c0 = a0*b0;  c1 = a0*b1 + a1*b0;  c2 = a0*b2 + a1*b1 + a2*b0
    c3 = a1*b2 + a2*b1;  c4 = a2*b2
    return torch.stack([c0+c3+c4, c1+c3+2*c4, c2+c3+2*c4], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# §2  IMMUTABLE GLOBAL TABLES
# ──────────────────────────────────────────────────────────────────────────────
ETA_POW:  torch.Tensor          # (169, 3)
ETA_IPOW: torch.Tensor          # (169, 3)
T3_POW:   torch.Tensor          # (168, 3, 3)
INV_TBL:  torch.Tensor          # (13,13,13, 3)
ONE_R:    torch.Tensor          # (3,)
ZERO_R:   torch.Tensor          # (3,)
T3_MAT:   torch.Tensor          # (3,3)
P1_MAT:   torch.Tensor          # (3,3)
P23_MAT:  torch.Tensor          # (3,3)
V_EIG:    torch.Tensor          # (3,)
W_EIG:    torch.Tensor          # (3,)

ETA_POW, ETA_IPOW = _build_eta_powers()
T3_POW   = _build_t3_powers()
INV_TBL  = _build_inv_table()
ONE_R    = torch.tensor([1, 0, 0], dtype=torch.long)
ZERO_R   = torch.zeros(3, dtype=torch.long)
T3_MAT   = torch.tensor(_T3_ROWS,  dtype=torch.long)
P1_MAT   = torch.tensor(_P1_ROWS,  dtype=torch.long)
P23_MAT  = torch.tensor(_P23_ROWS, dtype=torch.long)
V_EIG    = torch.tensor(_V_VEC,    dtype=torch.long)
W_EIG    = torch.tensor(_W_VEC,    dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# §3  NTT TWIDDLE TABLES  (all valid sizes pre-built)
# ──────────────────────────────────────────────────────────────────────────────
_NTT_VALID: List[int] = [1, 2, 3, 4, 6, 12]  # scalar twiddle only; N | 12, N mod 13 != 0
# NOTE: Ring twiddle sizes (N ∤ 12) are NOT orthogonal in Z_13[eta] and do NOT support roundtrip.
_NTT_SCALAR: Dict[int, int] = {1:1, 2:12, 3:3, 4:8, 6:4, 12:2}
_NTT_NINV:   Dict[int, int] = {n: pow(n, P-2, P) for n in _NTT_VALID}
_NTT_FWD:    Dict[int, torch.Tensor] = {}
_NTT_INV:    Dict[int, torch.Tensor] = {}


def _build_ntt(N: int) -> None:
    if N in _NTT_FWD:
        return
    k = torch.arange(N);  n = torch.arange(N)
    kn = (k.unsqueeze(1) * n.unsqueeze(0)) % ORD          # (N,N)
    if N in _NTT_SCALAR:
        w = _NTT_SCALAR[N]
        _NTT_FWD[N] = torch.tensor(
            [[pow(w, int(kn[i,j]), P) for j in range(N)] for i in range(N)],
            dtype=torch.long)                                # (N,N) scalar
        wi = pow(w, P-2, P)
        Ni = _NTT_NINV[N]
        _NTT_INV[N] = torch.tensor(
            [[pow(wi, int(kn[i,j]), P)*Ni % P for j in range(N)] for i in range(N)],
            dtype=torch.long)
    else:
        step = ORD // N
        exp_fwd = (kn * step) % ORD
        Ni = _NTT_NINV[N]
        _NTT_FWD[N] = ETA_POW[exp_fwd]                     # (N,N,3)
        _NTT_INV[N] = (ETA_IPOW[exp_fwd] * Ni) % P           # (N,N,3)


for _n in _NTT_VALID:
    _build_ntt(_n)


# ──────────────────────────────────────────────────────────────────────────────
# §4  VALUATION LUT  (v_13(d) for d = 0 .. 13⁵−1)
# ──────────────────────────────────────────────────────────────────────────────
_VAL_MAX = P ** ZetaConfig.MAX_VAL_DEPTH
_VAL_LUT: torch.Tensor


def _build_val_lut() -> torch.Tensor:
    d   = torch.arange(_VAL_MAX, dtype=torch.long)
    val = torch.zeros(_VAL_MAX,  dtype=torch.long)
    tmp = d.clone()
    # 5 explicit unrolled steps (compile-time constant)
    m = (tmp > 0) & (tmp % P == 0);  val += m.long();  tmp = torch.where(m, tmp // P, tmp)
    m = (tmp > 0) & (tmp % P == 0);  val += m.long();  tmp = torch.where(m, tmp // P, tmp)
    m = (tmp > 0) & (tmp % P == 0);  val += m.long();  tmp = torch.where(m, tmp // P, tmp)
    m = (tmp > 0) & (tmp % P == 0);  val += m.long();  tmp = torch.where(m, tmp // P, tmp)
    m = (tmp > 0) & (tmp % P == 0);  val += m.long()
    return val


_VAL_LUT = _build_val_lut()


# ──────────────────────────────────────────────────────────────────────────────
# §5  S3 GALOIS MATRICES  (group of η³−η²−η−1)
# ──────────────────────────────────────────────────────────────────────────────
_S3_DATA: List[List[List[int]]] = [
    [[1,0,0],[0,1,0],[0,0,1]],  # e
    [[1,0,0],[0,0,1],[0,1,0]],  # (12)
    [[0,1,0],[1,0,0],[0,0,1]],  # (01)
    [[0,0,1],[0,1,0],[1,0,0]],  # (02)
    [[0,1,0],[0,0,1],[1,0,0]],  # (012)
    [[0,0,1],[1,0,0],[0,1,0]],  # (021)
]
S3_MATS: torch.Tensor = torch.tensor(_S3_DATA, dtype=torch.long)   # (6,3,3)

# Precomputed S3 multiplication table (6,6) — O(1) group composition
_S3_MUL: torch.Tensor = torch.tensor(
    [[(next(k for k in range(6)
        if torch.all((S3_MATS[g] @ S3_MATS[h]) % P == S3_MATS[k])))
      for h in range(6)] for g in range(6)], dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# §6  TRIBONACCI MIXING TIME LUT  (delta_max(L) for L = 1..512)
# ──────────────────────────────────────────────────────────────────────────────
_DELTA_MAX_LUT: List[int] = []


def _build_delta_max_lut(max_L: int = ZetaConfig.TRIB_MAX_L) -> List[int]:
    lut = [0] * (max_L + 1)
    a, b, c, k = 1, 1, 2, 2
    for L in range(1, max_L + 1):
        while c < L:
            a, b, c, k = b, c, a+b+c, k+1
        lut[L] = k
    return lut


_DELTA_MAX_LUT = _build_delta_max_lut()


def delta_max(L: int) -> int:
    """Tribonacci mixing time t*(L).  O(1) LUT lookup."""
    if L <= ZetaConfig.TRIB_MAX_L:
        return _DELTA_MAX_LUT[L]
    a, b, c, k = 1, 1, 2, 2
    while c < L:
        a, b, c, k = b, c, a+b+c, k+1
    return k
