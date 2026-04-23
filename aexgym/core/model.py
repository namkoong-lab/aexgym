from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from aexgym.core.state import GaussianMetricState


def _as_float_tensor(value, *, dtype=None, device=None) -> Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if not tensor.dtype.is_floating_point:
        tensor = tensor.to(torch.float64)
    return tensor


def project_allocation(raw: Tensor, active: Tensor, eps: float = 0.0) -> Tensor:
    """Project nonnegative weights onto the active-arm simplex."""

    weights = torch.clamp(raw.to(dtype=torch.float64), min=0.0)
    active = active.to(dtype=torch.bool, device=weights.device)
    weights = torch.where(active, weights, torch.zeros_like(weights))
    active_count = int(active.sum().item())
    if active_count == 0:
        raise ValueError("cannot project onto an empty active set")

    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        weights = active.to(dtype=weights.dtype) / active_count
    else:
        weights = weights / total

    if eps > 0 and active_count > 1:
        floor = active.to(dtype=weights.dtype) * eps
        weights = torch.where(active, torch.clamp(weights, min=eps), torch.zeros_like(weights))
        weights = weights / weights.sum()
        weights = torch.maximum(weights, floor)
        weights = weights / weights.sum()

    return weights


class GaussianMetricModel:
    """Gaussian conjugate model for arm-level vector metrics.

    The effective observation for arm ``a`` at epoch ``t`` is
    ``Y[t, a] ~ N(theta[a], obs_cov[a] / (batch_sizes[t] * allocation[a]))``.
    Arms with zero allocation, inactive arms, and stopped states are frozen.
    """

    def __init__(
        self,
        prior_mean,
        prior_cov,
        obs_cov,
        target_idx: int,
        batch_sizes,
        control_arm: Optional[int] = None,
        allocation_floor: float = 1e-10,
    ) -> None:
        prior_mean = _as_float_tensor(prior_mean)
        dtype = prior_mean.dtype
        device = prior_mean.device
        prior_cov = _as_float_tensor(prior_cov, dtype=dtype, device=device)
        obs_cov = _as_float_tensor(obs_cov, dtype=dtype, device=device)
        batch_sizes = _as_float_tensor(batch_sizes, dtype=dtype, device=device)

        if prior_mean.ndim != 2:
            raise ValueError("prior_mean must have shape (n_arms, n_metrics)")
        n_arms, n_metrics = prior_mean.shape
        if prior_cov.shape != (n_arms, n_metrics, n_metrics):
            raise ValueError("prior_cov must have shape (n_arms, n_metrics, n_metrics)")
        if obs_cov.shape == (n_metrics, n_metrics):
            obs_cov = obs_cov.unsqueeze(0).repeat(n_arms, 1, 1)
        if obs_cov.shape != (n_arms, n_metrics, n_metrics):
            raise ValueError("obs_cov must have shape (n_arms, n_metrics, n_metrics) or (n_metrics, n_metrics)")
        if not 0 <= target_idx < n_metrics:
            raise ValueError("target_idx is out of range")
        if batch_sizes.ndim != 1 or batch_sizes.numel() == 0:
            raise ValueError("batch_sizes must be a nonempty 1D tensor")
        if torch.any(batch_sizes <= 0):
            raise ValueError("batch_sizes must be positive")
        if control_arm is not None and not 0 <= control_arm < n_arms:
            raise ValueError("control_arm is out of range")

        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.obs_cov = obs_cov
        self.target_idx = int(target_idx)
        self.batch_sizes = batch_sizes
        self.control_arm = control_arm
        self.allocation_floor = float(allocation_floor)
        self.n_arms = int(n_arms)
        self.n_metrics = int(n_metrics)

        self._obs_precision = torch.linalg.inv(obs_cov)

    @property
    def horizon(self) -> int:
        return int(self.batch_sizes.numel())

    def initial_state(self, active: Optional[Tensor] = None) -> GaussianMetricState:
        if active is None:
            active = torch.ones(self.n_arms, dtype=torch.bool, device=self.prior_mean.device)
        return GaussianMetricState(
            mean=self.prior_mean.clone(),
            cov=self.prior_cov.clone(),
            active=active.to(dtype=torch.bool, device=self.prior_mean.device),
            t=0,
            stopped=False,
        )

    def batch_size(self, t: int) -> Tensor:
        if t >= self.horizon:
            raise IndexError("state epoch is beyond model horizon")
        return self.batch_sizes[t]

    def validate_allocation(self, state: GaussianMetricState, allocation: Tensor) -> Tensor:
        allocation = project_allocation(allocation.to(device=state.mean.device), state.active)
        inactive_mass = torch.sum(torch.abs(allocation[~state.active]))
        if inactive_mass > 1e-8:
            raise ValueError("allocation has nonzero mass on inactive arms")
        return allocation.to(dtype=state.mean.dtype, device=state.mean.device)

    def sample_observation(
        self,
        true_theta: Tensor,
        allocation: Tensor,
        state: GaussianMetricState,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        true_theta = _as_float_tensor(true_theta, dtype=state.mean.dtype, device=state.mean.device)
        if true_theta.shape != state.mean.shape:
            raise ValueError("true_theta must match state mean shape")
        allocation = self.validate_allocation(state, allocation)
        y = torch.zeros_like(state.mean)
        b_t = self.batch_size(state.t)

        for arm in range(self.n_arms):
            if (not bool(state.active[arm])) or allocation[arm] <= self.allocation_floor:
                continue
            obs_cov = self.obs_cov[arm] / (b_t * allocation[arm])
            chol = torch.linalg.cholesky(obs_cov)
            z = torch.randn(self.n_metrics, dtype=state.mean.dtype, device=state.mean.device, generator=generator)
            y = y.clone()
            y[arm] = true_theta[arm] + chol @ z
        return y

    def update(self, state: GaussianMetricState, allocation: Tensor, observation: Tensor) -> GaussianMetricState:
        if state.stopped:
            return state.clone()
        if state.t >= self.horizon:
            return state.replace(stopped=True, stop_reason=state.stop_reason or "horizon")

        observation = _as_float_tensor(observation, dtype=state.mean.dtype, device=state.mean.device)
        if observation.shape != state.mean.shape:
            raise ValueError("observation must match state mean shape")

        allocation = self.validate_allocation(state, allocation)
        next_mean = state.mean.clone()
        next_cov = state.cov.clone()
        b_t = self.batch_size(state.t)

        for arm in range(self.n_arms):
            if (not bool(state.active[arm])) or allocation[arm] <= self.allocation_floor:
                continue
            prior_cov = state.cov[arm]
            prior_precision = torch.linalg.inv(prior_cov)
            obs_precision = self._obs_precision[arm]
            weighted_obs_precision = b_t * allocation[arm] * obs_precision
            post_precision = prior_precision + weighted_obs_precision
            post_cov = torch.linalg.inv(post_precision)
            natural_mean = prior_precision @ state.mean[arm] + weighted_obs_precision @ observation[arm]
            next_cov[arm] = post_cov
            next_mean[arm] = post_cov @ natural_mean

        return GaussianMetricState(
            mean=next_mean,
            cov=next_cov,
            active=state.active.clone(),
            t=state.t + 1,
            stopped=False,
        )

    def posterior_random_walk(
        self,
        mean: Tensor,
        cov: Tensor,
        active: Tensor,
        allocation: Tensor,
        t: int,
        z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Simulate posterior mean/cov evolution without explicit observations."""

        allocation = project_allocation(allocation.to(device=mean.device), active).to(dtype=mean.dtype)
        next_mean = mean.clone()
        next_cov = cov.clone()
        b_t = self.batch_size(t)
        eye = torch.eye(self.n_metrics, dtype=mean.dtype, device=mean.device)

        for arm in range(self.n_arms):
            if (not bool(active[arm])) or allocation[arm] <= self.allocation_floor:
                continue
            prior_cov = cov[arm]
            prior_precision = torch.linalg.inv(prior_cov)
            post_precision = prior_precision + b_t * allocation[arm] * self._obs_precision[arm]
            post_cov = torch.linalg.inv(post_precision)
            increment_cov = (prior_cov - post_cov + (1e-10 * eye)).clone()
            chol = torch.linalg.cholesky(increment_cov)
            next_cov[arm] = post_cov
            next_mean[arm] = mean[arm] + chol @ z[arm]
        return next_mean, next_cov
