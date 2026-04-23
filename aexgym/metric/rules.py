from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from aexgym.metric.state import GaussianMetricState


def normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))


class ActiveSetRule:
    """Armwise guardrail pruning with a single shrinking active set."""

    def __init__(
        self,
        target_idx: int,
        control_arm: Optional[int] = None,
        guardrail_indices: Optional[Sequence[int]] = None,
        guardrail_floors: Optional[Sequence[float] | Tensor] = None,
        violation_prob_threshold: float = 1.0,
        stop_on_singleton: bool = True,
    ) -> None:
        self.target_idx = int(target_idx)
        self.control_arm = control_arm
        self.guardrail_indices = None if guardrail_indices is None else tuple(int(i) for i in guardrail_indices)
        self.guardrail_floors = None if guardrail_floors is None else torch.as_tensor(guardrail_floors)
        self.violation_prob_threshold = float(violation_prob_threshold)
        self.stop_on_singleton = bool(stop_on_singleton)

    def apply(self, state: GaussianMetricState) -> GaussianMetricState:
        if state.active_count == 0:
            raise ValueError("active set cannot be empty")
        if state.stopped:
            return state.clone()
        if state.active_count == 1:
            return self._maybe_stop(state, "singleton_active")

        next_active = state.active.clone()
        if self.guardrail_floors is not None:
            next_active = self._apply_guardrail_pruning(state, next_active)

        next_active = self._ensure_nonempty(state, next_active)
        next_state = state.replace(active=next_active)
        if next_state.active_count == 1:
            return self._maybe_stop(next_state, "singleton_active")
        return next_state

    def violation_probabilities(self, state: GaussianMetricState) -> Tensor:
        if self.guardrail_floors is None:
            return torch.zeros(state.n_arms, 0, dtype=state.mean.dtype, device=state.mean.device)
        guardrail_indices = self._guardrail_indices(state)
        floors = self.guardrail_floors.to(dtype=state.mean.dtype, device=state.mean.device)
        if floors.numel() == 1 and len(guardrail_indices) > 1:
            floors = floors.repeat(len(guardrail_indices))
        if floors.shape != (len(guardrail_indices),):
            raise ValueError("guardrail_floors must be scalar or match guardrail_indices")

        probs = []
        for floor, metric_idx in zip(floors, guardrail_indices):
            var = torch.clamp(state.cov[:, metric_idx, metric_idx], min=1e-16)
            z = (floor - state.mean[:, metric_idx]) / torch.sqrt(var)
            probs.append(normal_cdf(z))
        return torch.stack(probs, dim=1)

    def unsafe_arms(self, true_theta: Tensor) -> Tensor:
        if self.guardrail_floors is None:
            return torch.zeros(true_theta.shape[0], dtype=torch.bool, device=true_theta.device)
        guardrail_indices = self.guardrail_indices
        if guardrail_indices is None:
            guardrail_indices = tuple(range(1, true_theta.shape[1]))
        floors = self.guardrail_floors.to(dtype=true_theta.dtype, device=true_theta.device)
        if floors.numel() == 1 and len(guardrail_indices) > 1:
            floors = floors.repeat(len(guardrail_indices))
        unsafe = torch.zeros(true_theta.shape[0], dtype=torch.bool, device=true_theta.device)
        for floor, metric_idx in zip(floors, guardrail_indices):
            unsafe = unsafe | (true_theta[:, metric_idx] < floor)
        if self.control_arm is not None:
            unsafe[self.control_arm] = False
        return unsafe

    def selected_violates(self, true_theta: Tensor, selected_arm: int) -> bool:
        return bool(self.unsafe_arms(true_theta)[selected_arm].item())

    def _guardrail_indices(self, state: GaussianMetricState) -> tuple[int, ...]:
        if self.guardrail_indices is None:
            return tuple(i for i in range(state.n_metrics) if i != self.target_idx)
        return self.guardrail_indices

    def _apply_guardrail_pruning(self, state: GaussianMetricState, active: Tensor) -> Tensor:
        violation_probs = self.violation_probabilities(state)
        if violation_probs.numel() == 0:
            return active
        prune = torch.any(violation_probs >= self.violation_prob_threshold, dim=1)
        prune = prune & state.active
        if self.control_arm is not None:
            prune[self.control_arm] = False
        return active & ~prune

    def _ensure_nonempty(self, state: GaussianMetricState, active: Tensor) -> Tensor:
        if bool(active.any()):
            return active
        recovered = torch.zeros_like(active)
        if self.control_arm is not None:
            recovered[self.control_arm] = True
            return recovered
        old_active_idx = state.active_indices()
        target_values = state.mean[old_active_idx, self.target_idx]
        recovered[int(old_active_idx[torch.argmax(target_values)].item())] = True
        return recovered

    def _maybe_stop(self, state: GaussianMetricState, reason: str) -> GaussianMetricState:
        if not self.stop_on_singleton:
            return state
        return state.replace(stopped=True, stop_reason=reason)


class NoActiveSetRule(ActiveSetRule):
    """Rule for scalar parity runs: active set never shrinks during the horizon."""

    def __init__(self, target_idx: int = 0) -> None:
        super().__init__(target_idx=target_idx, stop_on_singleton=True)

    def apply(self, state: GaussianMetricState) -> GaussianMetricState:
        if state.active_count == 1:
            return self._maybe_stop(state, "singleton_active")
        return state
