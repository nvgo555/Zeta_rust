# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
Zeta p-Adic Integer AI  v7.0.0 (final)
========================================
Algebraic AI built exclusively on:
    • Z_13[eta] / (eta^3 - eta^2 - eta - 1)
    • T3 ∈ SL(3,Z)  —  sole evolution matrix
    • Sylvester P1 (1D) + P23 (2D)
    • CRT  Z_13 × F_169
    • G(i,j) = eta^{-v_13(|i-j|)}  —  THE ONLY ultrametric kernel
    • Buchberger-Nullstellensatz  —  THE ONLY learning mechanism
    • Witt head  —  Hensel-precision weights with convergent decay
    • ZetaGoalPlanner  —  goal-directed orbit planning
    • ZetaSelf  —  persistent self-state with orbital evolution
    • EntropyMonitor  —  Hensel catastrophe trigger
    • HenselIO  —  empirical closure sensor/actuator mapping
    • AutonomousLoop  —  closed-loop persistent autonomy
    • ZetaScaler  —  autonomous scaling (multi-layer Witt, S3 orbits, dynamic MERA)

Zero float.  Zero gradient descent.  Zero Euclidean geometry.

Author : Dávid Navrátil <david.navratil2016@gmail.com>
License: CC-BY-NC-4.0
"""
__version__ = "7.0.0"
__author__  = "Dávid Navrátil"
__email__   = "david.navratil2016@gmail.com"
__license__ = "CC-BY-NC-4.0"

from .constants import ZetaConfig, P, ORD
from .runtime import ZetaRuntime, ZetaDevice
from .model import ZetaModel
from .axioms import AxiomVerifier
from .buchberger import BuchbergerEngine
from .gqm import ErrorCache, ZetaSelf
from .goal import ZetaGoalPlanner
from .entropy import EntropyMonitor
from .hensel import HenselIO
from .autonomy import AutonomousLoop, FullAutonomyCycle
from .scaler import ZetaScaler
from .hybrid import S3ParallelAttention
from .tokenizer import trigram_tokenise, detrigram_tokenise, pad_batch_trigram
