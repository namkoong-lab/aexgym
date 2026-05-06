from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from aexgym.core.state import GaussianMetricState


@dataclass(frozen=True)
class SmoothingConfig:
    """Finite positive beta gives smooth decisions; beta <= 0 or inf gives hard decisions."""

    beta: float = math.inf

    @property
    def hard(self) -> bool:
        return self.beta <= 0 or math.isinf(self.beta)


@dataclass(frozen=True)
class ActiveSetDecision:
    active: Tensor
    diagnostics: dict = field(default_factory=dict)


def normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))


def _threshold_ge(score: Tensor, threshold: float, smoothing: SmoothingConfig) -> Tensor:
    threshold_tensor = torch.as_tensor(threshold, dtype=score.dtype, device=score.device)
    if smoothing.hard:
        return (score >= threshold_tensor).to(score.dtype)
    return torch.sigmoid(float(smoothing.beta) * (score - threshold_tensor))


def _threshold_le(score: Tensor, threshold: float, smoothing: SmoothingConfig) -> Tensor:
    threshold_tensor = torch.as_tensor(threshold, dtype=score.dtype, device=score.device)
    if smoothing.hard:
        return (score <= threshold_tensor).to(score.dtype)
    return torch.sigmoid(float(smoothing.beta) * (threshold_tensor - score))


class ActiveSetRule:
    """Single active-set rule implementation for hard and smooth rollouts."""

    def __init__(
        self,
        target_metric_idx: int,
        control_arm_idx: Optional[int] = None,
        guardrail_metric_indices: Optional[Sequence[int]] = None,
        guardrail_floors: Optional[Sequence[float] | Tensor] = None,
        violation_prob_threshold: float = 1.0,
        target_futility_floor: Optional[float] = None,
        target_futility_prob_threshold: Optional[float] = None,
        control_margin: float = 0.0,
        control_prob_threshold: Optional[float] = None,
        shutdown_prob_threshold: Optional[float] = None,
        guardrail_shutdown_threshold: Optional[float] = None,
        early_success_prob_threshold: Optional[float] = None,
        min_success_epoch: int = 0,
        best_arm_num_samples: int = 256,
        best_arm_seed: int = 12345,
    ) -> None:
        self.target_metric_idx = int(target_metric_idx)
        self.control_arm_idx = control_arm_idx
        self.guardrail_metric_indices = None if guardrail_metric_indices is None else tuple(int(i) for i in guardrail_metric_indices)
        self.guardrail_floors = None if guardrail_floors is None else torch.as_tensor(guardrail_floors)
        self.violation_prob_threshold = float(violation_prob_threshold)
        self.target_futility_floor = target_futility_floor
        self.target_futility_prob_threshold = target_futility_prob_threshold
        self.control_margin = float(control_margin)
        self.control_prob_threshold = control_prob_threshold
        self.shutdown_prob_threshold = shutdown_prob_threshold
        self.guardrail_shutdown_threshold = guardrail_shutdown_threshold
        self.early_success_prob_threshold = early_success_prob_threshold
        self.min_success_epoch = int(min_success_epoch)
        self.best_arm_num_samples = int(best_arm_num_samples)
        self.best_arm_seed = int(best_arm_seed)

    def evaluate(self, state: GaussianMetricState, model, *, smoothing: SmoothingConfig | None = None) -> ActiveSetDecision:
        smoothing = smoothing or SmoothingConfig()
        if int((state.active > 0).sum().item()) == 0:
            raise ValueError("active set cannot be empty")

        active = torch.clamp(state.active, min=0.0, max=1.0)
        diagnostics: dict[str, object] = {}

        arm_survival = torch.ones_like(active)
        guardrail_violation_probs = self.violation_probabilities(state, model)
        if guardrail_violation_probs.numel() > 0:
            guardrail_survival = _threshold_le(guardrail_violation_probs, self.violation_prob_threshold, smoothing).prod(dim=1)
            arm_survival = arm_survival * guardrail_survival
            diagnostics["guardrail_violation_probabilities"] = guardrail_violation_probs.detach().cpu().tolist()

        if self.target_futility_floor is not None and self.target_futility_prob_threshold is not None:
            p_target_ok = self.target_threshold_probabilities(state, model, self.target_futility_floor)
            arm_survival = arm_survival * _threshold_ge(p_target_ok, self.target_futility_prob_threshold, smoothing)
            diagnostics["target_ok_probabilities"] = p_target_ok.detach().cpu().tolist()

        if self.control_prob_threshold is not None and self.control_arm_idx is not None:
            p_better = self.control_contrast_probabilities(state, model)
            arm_survival = arm_survival * _threshold_ge(p_better, self.control_prob_threshold, smoothing)
            diagnostics["control_contrast_probabilities"] = p_better.detach().cpu().tolist()

        if self.control_arm_idx is not None and 0 <= self.control_arm_idx < state.n_arms and active[self.control_arm_idx] > 0:
            arm_survival = arm_survival.clone()
            arm_survival[self.control_arm_idx] = 1.0

        new_active = active * arm_survival
        new_active = self._ensure_nonempty(state, model, new_active, smoothing)

        shutdown_weight = self._shutdown_weight(state, model, new_active, guardrail_violation_probs, smoothing)
        if shutdown_weight is not None and self.control_arm_idx is not None:
            control_active = torch.zeros_like(new_active)
            control_active[self.control_arm_idx] = 1.0
            new_active = (1.0 - shutdown_weight) * new_active + shutdown_weight * control_active
            diagnostics["control_collapse_weight"] = float(shutdown_weight.detach().cpu().item())

        success_weight, winner = self._early_success_weight(state, model, new_active, smoothing)
        if success_weight is not None and winner is not None:
            winner_active = torch.zeros_like(new_active)
            winner_active[winner] = 1.0
            new_active = (1.0 - success_weight) * new_active + success_weight * winner_active
            diagnostics["winner_collapse_weight"] = float(success_weight.detach().cpu().item())
            diagnostics["winner"] = int(winner)

        new_active = self._ensure_nonempty(state, model, new_active, smoothing)
        if smoothing.hard:
            new_active = (new_active > 0).to(dtype=state.dtype, device=state.device)

        return ActiveSetDecision(active=new_active, diagnostics=diagnostics)

    def apply(self, state: GaussianMetricState, model=None) -> GaussianMetricState:
        if model is None:
            raise TypeError("ActiveSetRule.apply now requires model; use evaluate(state, model, smoothing=...)")
        decision = self.evaluate(state, model, smoothing=SmoothingConfig())
        return state.replace(active=decision.active)

    def violation_probabilities(self, state: GaussianMetricState, model) -> Tensor:
        if self.guardrail_floors is None:
            return torch.zeros(state.n_arms, 0, dtype=state.dtype, device=state.device)
        guardrail_metric_indices = self._guardrail_metric_indices(state)
        floors = self.guardrail_floors.to(dtype=state.dtype, device=state.device)
        if floors.numel() == 1 and len(guardrail_metric_indices) > 1:
            floors = floors.repeat(len(guardrail_metric_indices))
        if floors.shape != (len(guardrail_metric_indices),):
            raise ValueError("guardrail_floors must be scalar or match guardrail_metric_indices")

        mean, cov = model.posterior_moments(state)
        probs = []
        for floor, guardrail_metric_idx in zip(floors, guardrail_metric_indices):
            var = torch.clamp(cov[:, guardrail_metric_idx, guardrail_metric_idx], min=1e-16)
            z = (floor - mean[:, guardrail_metric_idx]) / torch.sqrt(var)
            probs.append(normal_cdf(z))
        return torch.stack(probs, dim=1)

    def target_threshold_probabilities(self, state: GaussianMetricState, model, floor: float) -> Tensor:
        mean = model.target_mean(state)
        var = torch.clamp(model.target_variance(state), min=1e-16)
        z = (mean - float(floor)) / torch.sqrt(var)
        return normal_cdf(z)

    def control_contrast_probabilities(self, state: GaussianMetricState, model) -> Tensor:
        probs = torch.ones(state.n_arms, dtype=state.dtype, device=state.device)
        if self.control_arm_idx is None:
            return probs
        for arm_idx in range(state.n_arms):
            if arm_idx == self.control_arm_idx:
                probs[arm_idx] = 1.0
            else:
                probs[arm_idx] = model.target_control_probability_ge(state, arm_idx, margin=self.control_margin)
        return probs

    def best_arm_probabilities(self, state: GaussianMetricState, model, active: Tensor, smoothing: SmoothingConfig) -> Tensor:
        mean = model.target_mean(state)
        var = torch.clamp(model.target_variance(state), min=1e-16)
        if self.best_arm_num_samples <= 0:
            logits = mean + torch.log(torch.clamp(active, min=torch.finfo(active.dtype).tiny))
            if smoothing.hard:
                probs = torch.zeros_like(mean)
                probs[int(torch.argmax(torch.where(active > 0, mean, torch.full_like(mean, -torch.inf))).item())] = 1.0
                return probs
            return F.softmax(float(smoothing.beta) * logits, dim=0)

        sobol = torch.quasirandom.SobolEngine(state.n_arms, scramble=True, seed=self.best_arm_seed)
        u = sobol.draw(self.best_arm_num_samples).to(dtype=state.dtype, device=state.device)
        u = torch.clamp(u, min=torch.finfo(u.dtype).eps, max=1.0 - torch.finfo(u.dtype).eps)
        z = torch.sqrt(torch.tensor(2.0, dtype=state.dtype, device=state.device)) * torch.erfinv(2.0 * u - 1.0)
        samples = mean.unsqueeze(0) + torch.sqrt(var).unsqueeze(0) * z
        samples = samples + torch.log(torch.clamp(active, min=torch.finfo(active.dtype).tiny)).unsqueeze(0)
        if smoothing.hard:
            winners = torch.argmax(samples, dim=1)
            return torch.bincount(winners, minlength=state.n_arms).to(dtype=state.dtype) / self.best_arm_num_samples
        return F.softmax(float(smoothing.beta) * samples, dim=1).mean(dim=0)

    def unsafe_arms(self, true_theta: Tensor) -> Tensor:
        if self.guardrail_floors is None:
            return torch.zeros(true_theta.shape[0], dtype=torch.bool, device=true_theta.device)
        guardrail_metric_indices = self.guardrail_metric_indices
        if guardrail_metric_indices is None:
            guardrail_metric_indices = tuple(range(1, true_theta.shape[1]))
        floors = self.guardrail_floors.to(dtype=true_theta.dtype, device=true_theta.device)
        if floors.numel() == 1 and len(guardrail_metric_indices) > 1:
            floors = floors.repeat(len(guardrail_metric_indices))
        unsafe = torch.zeros(true_theta.shape[0], dtype=torch.bool, device=true_theta.device)
        for floor, guardrail_metric_idx in zip(floors, guardrail_metric_indices):
            unsafe = unsafe | (true_theta[:, guardrail_metric_idx] < floor)
        if self.control_arm_idx is not None:
            unsafe[self.control_arm_idx] = False
        return unsafe

    def selected_violates(self, true_theta: Tensor, selected_arm_idx: int) -> bool:
        return bool(self.unsafe_arms(true_theta)[selected_arm_idx].item())

    def _guardrail_metric_indices(self, state: GaussianMetricState) -> tuple[int, ...]:
        if self.guardrail_metric_indices is None:
            return tuple(i for i in range(state.n_metrics) if i != self.target_metric_idx)
        return self.guardrail_metric_indices

    def _ensure_nonempty(self, state: GaussianMetricState, model, active: Tensor, smoothing: SmoothingConfig) -> Tensor:
        if torch.sum(active) > 0:
            return active
        recovered = torch.zeros_like(active)
        if self.control_arm_idx is not None:
            recovered[self.control_arm_idx] = 1.0
            return recovered
        old_active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if old_active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        target_values = model.target_mean(state)[old_active_arm_idx]
        recovered[int(old_active_arm_idx[torch.argmax(target_values)].item())] = 1.0
        if smoothing.hard:
            return recovered
        return torch.clamp(recovered + torch.finfo(active.dtype).eps, max=1.0)

    def _shutdown_weight(
        self,
        state: GaussianMetricState,
        model,
        active: Tensor,
        guardrail_violation_probs: Tensor,
        smoothing: SmoothingConfig,
    ) -> Optional[Tensor]:
        if self.control_arm_idx is None:
            return None
        weights = []
        if self.guardrail_shutdown_threshold is not None and guardrail_violation_probs.numel() > 0:
            alarm = torch.max(guardrail_violation_probs * state.active[:, None])
            weights.append(_threshold_ge(alarm, self.guardrail_shutdown_threshold, smoothing))
        if self.shutdown_prob_threshold is not None:
            if self.control_arm_idx is None:
                return None
            p_better = self.control_contrast_probabilities(state, model)
            p_safe = torch.ones_like(p_better)
            if guardrail_violation_probs.numel() > 0:
                p_safe = torch.prod(1.0 - guardrail_violation_probs, dim=1)
            treatment = torch.ones_like(active, dtype=torch.bool)
            treatment[self.control_arm_idx] = False
            p_candidate = torch.where(treatment, active * p_better * p_safe, torch.zeros_like(p_better))
            p_any = 1.0 - torch.prod(torch.clamp(1.0 - p_candidate, min=0.0, max=1.0))
            weights.append(_threshold_le(p_any, self.shutdown_prob_threshold, smoothing))
        if not weights:
            return None
        return torch.clamp(torch.stack(weights).max(), min=0.0, max=1.0)

    def _early_success_weight(
        self,
        state: GaussianMetricState,
        model,
        active: Tensor,
        smoothing: SmoothingConfig,
    ) -> tuple[Optional[Tensor], Optional[int]]:
        if self.early_success_prob_threshold is None or state.t < self.min_success_epoch:
            return None, None
        best_probs = self.best_arm_probabilities(state, model, active, smoothing)
        if self.control_arm_idx is not None:
            eligible = torch.ones_like(active, dtype=torch.bool)
            eligible[self.control_arm_idx] = False
        else:
            eligible = active > 0
        eligible = eligible & (active > 0)
        if int(eligible.sum().item()) == 0:
            return None, None
        scores = torch.where(eligible, best_probs, torch.full_like(best_probs, -torch.inf))
        winner = int(torch.argmax(scores).item())
        weight = _threshold_ge(best_probs[winner], self.early_success_prob_threshold, smoothing)
        return torch.clamp(weight, min=0.0, max=1.0), winner


class NoActiveSetRule(ActiveSetRule):
    """Rule for scalar parity runs: active weights do not shrink during the horizon."""

    def __init__(self, target_metric_idx: int = 0) -> None:
        super().__init__(target_metric_idx=target_metric_idx)

    def evaluate(self, state: GaussianMetricState, model, *, smoothing: SmoothingConfig | None = None) -> ActiveSetDecision:
        return ActiveSetDecision(active=state.active.clone(), diagnostics={})
