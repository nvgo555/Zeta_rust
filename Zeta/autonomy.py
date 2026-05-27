# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/autonomy.py  —  Full Autonomy Cycle with Autonomous Scaling
=================================================================
Persistent self, goal planning, entropy monitoring, and automatic
scaling via ZetaScaler (multi-layer Witt, S3 orbits, dynamic MERA).

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
import torch

from .constants import P, ORD, ETA_POW
from .mera import MERAChunker
from .gqm import GQMState, ZetaSelf
from .orbit import OrbitPlanner
from .ring import Z13
from .rollback import QuantumRollback
from .dirac import AdelicDiracOperator
from .goal import ZetaGoalPlanner
from .entropy import EntropyMonitor
from .scaler import ZetaScaler


class FullAutonomyCycle:

    @staticmethod
    def cycle(model, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
        dev = tokens.device
        sc, aud = model(tokens)
        preds = sc.argmax(-1) % P
        feat = model.embed(tokens.to(dev))
        levels = MERAChunker.coarse_grain(feat, levels=2)
        top = levels[-1][0, 0, :8]
        state = GQMState.from_coeffs(top)
        goal = ETA_POW[preds[0, -1] % ORD]
        curr = state.coeffs[0]
        n_steps = OrbitPlanner.plan_orbit(curr, goal)
        hall = QuantumRollback.detect(int(preds[0,-1].item()), int(targets[0,-1].item()))
        if hall:
            model.head.data = QuantumRollback.rollback_ring(model.head.data, model.step, dev)
        rh = AdelicDiracOperator.rh_check(L=8, dev=dev)
        return {
            'preds': preds, 'hall': hall, 'n_steps': n_steps,
            'gqm_norm': int(Z13.norm(state.coeffs).sum().item()),
            'rh': rh['RH'], 'mera_lvls': len(levels),
            **{f'L{i}': la for i, la in enumerate(aud.values()) if isinstance(la, dict)},
        }


class AutonomousLoop:
    """Autonómny cyklus s autonómnym škálovaním."""

    def __init__(self, runtime) -> None:
        self.rt = runtime
        self.self_state = ZetaSelf(K=8)
        self.goal_state: ZetaSelf | None = None
        self.entropy_monitor = EntropyMonitor(threshold=10)
        self.scaler = ZetaScaler(runtime.model)
        self.history: list = []

    def cycle(self, tokens: torch.Tensor, target_goal: torch.Tensor | None = None) -> dict:
        sc, aud = self.rt.forward(tokens)
        preds = sc.argmax(-1) % P
        feat = self.rt.model.embed(tokens)
        top = feat[0, 0, :8, :]

        # Persistent self-evolution
        self.self_state.evolve()
        self.self_state.coeffs = Z13.add(self.self_state.coeffs, top) % P

        # Goal planning
        plan = None
        if target_goal is not None:
            self.goal_state = ZetaSelf.from_coeffs(target_goal)
        if self.goal_state is not None:
            plan = ZetaGoalPlanner.plan(self.self_state.coeffs, self.goal_state.coeffs)

        # Entropy
        S = self.self_state.entropy()
        lift = self.entropy_monitor.check(S)
        if lift and hasattr(self.rt.model, '_lift'):
            self.rt.model._lift()

        # Action / Buchberger
        hall = int((preds != tokens).sum().item())
        metrics = {}
        if hall > 0:
            metrics = self.rt.train_step(tokens, tokens)

        # Autonomous scaling observation
        dn = metrics.get('delta_norm', 0)
        self.scaler.observe(hall, S, dn)
        scale_actions = self.scaler.decide()

        # Apply scaling actions to model
        if 's3_orbit' in scale_actions:
            self.rt.model.s3_orbit = scale_actions['s3_orbit']
        if 'witt_full' in scale_actions and not hasattr(self.rt.model, '_witt_full_flag'):
            self.rt.model._witt_full_flag = True  # forward already uses all layers via _sync_head

        entry = {
            'self_t': self.self_state.t,
            'entropy': S,
            'plan': plan,
            'lift_triggered': lift,
            'hall': hall,
            'scale_actions': scale_actions,
            **metrics,
        }
        self.history.append(entry)
        if len(self.history) > 168:
            self.history.pop(0)
        return entry

    def reflect(self) -> dict:
        if len(self.history) < 2 or self.goal_state is None:
            return {'status': 'insufficient_history'}
        recent = self.history[-12:]
        entropies = [h['entropy'] for h in recent]
        halls = [h['hall'] for h in recent]
        deficit = self.self_state.reflect(self.goal_state)
        return {
            'avg_entropy': sum(entropies) // len(entropies),
            'total_hall': sum(halls),
            'p1_deficit_norm': int(Z13.norm(deficit).sum().item()),
            'status': 'reflected',
        }
