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


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _cholesky_inverse(matrix: Tensor) -> Tensor:
    chol = torch.linalg.cholesky(_symmetrize(matrix))
    return _symmetrize(torch.cholesky_inverse(chol))


def project_allocation(raw: Tensor, active: Tensor, eps: float = 0.0) -> Tensor:
    """Project nonnegative proposal weights through hard or smooth active weights."""

    weights = torch.clamp(raw.to(dtype=torch.float64, device=active.device), min=0.0)
    active = torch.clamp(active.to(dtype=weights.dtype, device=weights.device), min=0.0, max=1.0)
    weights = weights * active
    available = active > 0
    if int(available.sum().item()) == 0:
        raise ValueError("cannot project onto an empty active set")

    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        weights = active / active.sum()
    else:
        weights = weights / total

    if eps > 0 and int(available.sum().item()) > 1:
        floor = torch.where(available, torch.full_like(weights, eps), torch.zeros_like(weights))
        weights = torch.where(available, torch.clamp(weights, min=eps), torch.zeros_like(weights))
        weights = torch.maximum(weights / weights.sum(), floor)
        weights = weights / weights.sum()

    return weights


class GaussianMetricModel:
    """Gaussian conjugate model using additive information-form state."""

    def __init__(
        self,
        prior_mean,
        prior_cov,
        obs_cov,
        target_metric_idx: int,
        batch_sizes,
        control_arm_idx: Optional[int] = None,
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
        if not 0 <= target_metric_idx < n_metrics:
            raise ValueError("target_metric_idx is out of range")
        if batch_sizes.ndim != 1 or batch_sizes.numel() == 0:
            raise ValueError("batch_sizes must be a nonempty 1D tensor")
        if torch.any(batch_sizes <= 0):
            raise ValueError("batch_sizes must be positive")
        if control_arm_idx is not None and not 0 <= control_arm_idx < n_arms:
            raise ValueError("control_arm_idx is out of range")

        self.prior_mean = prior_mean
        self.prior_cov = _symmetrize(prior_cov)
        self.obs_cov = _symmetrize(obs_cov)
        self.target_metric_idx = int(target_metric_idx)
        self.batch_sizes = batch_sizes
        self.control_arm_idx = control_arm_idx
        self.allocation_floor = float(allocation_floor)
        self.n_arms = int(n_arms)
        self.n_metrics = int(n_metrics)

        self.prior_precision = _cholesky_inverse(self.prior_cov)
        self.prior_information = torch.einsum("kij,kj->ki", self.prior_precision, self.prior_mean)
        self.obs_precision = _cholesky_inverse(self.obs_cov)

    @property
    def horizon(self) -> int:
        return int(self.batch_sizes.numel())

    def initial_state(self, active: Optional[Tensor] = None) -> GaussianMetricState:
        if active is None:
            active = torch.ones(self.n_arms, dtype=self.prior_mean.dtype, device=self.prior_mean.device)
        return GaussianMetricState(
            n_eff=torch.zeros(self.n_arms, dtype=self.prior_mean.dtype, device=self.prior_mean.device),
            sum_g=torch.zeros(self.n_arms, self.n_metrics, dtype=self.prior_mean.dtype, device=self.prior_mean.device),
            active=active.to(dtype=self.prior_mean.dtype, device=self.prior_mean.device),
            t=0,
        )

    def batch_size(self, t: int) -> Tensor:
        if t >= self.horizon:
            raise IndexError("state stage is beyond model horizon")
        return self.batch_sizes[t]

    def posterior_precision(self, state: GaussianMetricState) -> Tensor:
        return _symmetrize(self.prior_precision + state.n_eff[:, None, None] * self.obs_precision)

    def posterior_cholesky(self, state: GaussianMetricState) -> Tensor:
        return torch.linalg.cholesky(self.posterior_precision(state))

    def information_vector(self, state: GaussianMetricState) -> Tensor:
        return self.prior_information + torch.einsum("kij,kj->ki", self.obs_precision, state.sum_g)

    def posterior_mean(self, state: GaussianMetricState) -> Tensor:
        chol = self.posterior_cholesky(state)
        h = self.information_vector(state).unsqueeze(-1)
        return torch.cholesky_solve(h, chol).squeeze(-1)

    def posterior_cov(self, state: GaussianMetricState) -> Tensor:
        return _symmetrize(torch.cholesky_inverse(self.posterior_cholesky(state)))

    def posterior_moments(self, state: GaussianMetricState) -> tuple[Tensor, Tensor]:
        chol = self.posterior_cholesky(state)
        h = self.information_vector(state).unsqueeze(-1)
        mean = torch.cholesky_solve(h, chol).squeeze(-1)
        cov = _symmetrize(torch.cholesky_inverse(chol))
        return mean, cov

    def target_mean(self, state: GaussianMetricState) -> Tensor:
        return self.posterior_mean(state)[:, self.target_metric_idx]

    def target_variance(self, state: GaussianMetricState) -> Tensor:
        return self.posterior_cov(state)[:, self.target_metric_idx, self.target_metric_idx]

    def empirical_mean(self, state: GaussianMetricState) -> Tensor:
        safe_n = torch.clamp(state.n_eff, min=torch.finfo(state.dtype).eps)
        empirical = state.sum_g / safe_n[:, None]
        return torch.where((state.n_eff > 0)[:, None], empirical, self.prior_mean)

    def selected_arm(self, state: GaussianMetricState) -> int:
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("cannot select an arm from an empty active set")
        target_values = self.target_mean(state)[active_arm_idx]
        return int(active_arm_idx[torch.argmax(target_values)].item())

    def terminal_value(self, state: GaussianMetricState, terminal_beta: float = float("inf")) -> Tensor:
        target = self.target_mean(state)
        active = torch.clamp(state.active, min=0.0)
        active_arm_idx = torch.nonzero(active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("cannot evaluate terminal value for an empty active set")
        if terminal_beta == float("inf"):
            return torch.max(target[active_arm_idx])
        beta = float(terminal_beta)
        if beta <= 0:
            return torch.sum((active / active.sum()) * target)
        log_weights = torch.log(torch.clamp(active, min=torch.finfo(active.dtype).tiny))
        return torch.logsumexp(log_weights + beta * target, dim=0) / beta

    def linear_probability_leq(self, state: GaussianMetricState, arm_idx: int, contrast: Tensor, threshold: float | Tensor) -> Tensor:
        mean = self.posterior_mean(state)[arm_idx]
        chol = self.posterior_cholesky(state)[arm_idx]
        contrast = contrast.to(dtype=state.dtype, device=state.device)
        y = torch.linalg.solve_triangular(chol, contrast.unsqueeze(-1), upper=False).squeeze(-1)
        var = torch.clamp(torch.sum(y * y), min=1e-16)
        z = (torch.as_tensor(threshold, dtype=state.dtype, device=state.device) - contrast.dot(mean)) / torch.sqrt(var)
        return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=state.dtype, device=state.device))))

    def target_control_probability_ge(self, state: GaussianMetricState, arm_idx: int, margin: float = 0.0) -> Tensor:
        if self.control_arm_idx is None:
            raise ValueError("control_arm_idx is required for control contrasts")
        mean, cov = self.posterior_moments(state)
        diff_mean = mean[arm_idx, self.target_metric_idx] - mean[self.control_arm_idx, self.target_metric_idx]
        diff_var = torch.clamp(
            cov[arm_idx, self.target_metric_idx, self.target_metric_idx]
            + cov[self.control_arm_idx, self.target_metric_idx, self.target_metric_idx],
            min=1e-16,
        )
        z = (diff_mean - margin) / torch.sqrt(diff_var)
        return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=state.dtype, device=state.device))))

    def validate_allocation(self, state: GaussianMetricState, allocation: Tensor) -> Tensor:
        return project_allocation(allocation.to(device=state.device), state.active).to(dtype=state.dtype, device=state.device)

    def sample_observation(
        self,
        true_theta: Tensor,
        allocation: Tensor,
        state: GaussianMetricState,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        true_theta = _as_float_tensor(true_theta, dtype=state.dtype, device=state.device)
        if true_theta.shape != (self.n_arms, self.n_metrics):
            raise ValueError("true_theta must match model arm/metric shape")
        allocation = self.validate_allocation(state, allocation)
        y = torch.zeros(self.n_arms, self.n_metrics, dtype=state.dtype, device=state.device)
        b_t = self.batch_size(state.t)

        for arm_idx in range(self.n_arms):
            if state.active[arm_idx] <= 0 or allocation[arm_idx] <= self.allocation_floor:
                continue
            obs_cov = self.obs_cov[arm_idx] / (b_t * allocation[arm_idx])
            chol = torch.linalg.cholesky(obs_cov)
            z = torch.randn(self.n_metrics, dtype=state.dtype, device=state.device, generator=generator)
            y[arm_idx] = true_theta[arm_idx] + chol @ z
        return y

    def update(self, state: GaussianMetricState, allocation: Tensor, observation: Tensor) -> GaussianMetricState:
        observation = _as_float_tensor(observation, dtype=state.dtype, device=state.device)
        if observation.shape != (self.n_arms, self.n_metrics):
            raise ValueError("observation must match model arm/metric shape")

        allocation = self.validate_allocation(state, allocation)
        info_increment = self.batch_size(state.t) * allocation
        update_mask = (state.active > 0) & (allocation > self.allocation_floor)
        info_increment = torch.where(update_mask, info_increment, torch.zeros_like(info_increment))

        return GaussianMetricState(
            n_eff=state.n_eff + info_increment,
            sum_g=state.sum_g + info_increment[:, None] * observation,
            active=state.active.clone(),
            t=state.t + 1,
        )

    def posterior_random_walk(
        self,
        state: GaussianMetricState,
        allocation: Tensor,
        z: Tensor,
        t: Optional[int] = None,
    ) -> GaussianMetricState:
        """Simulate posterior evolution in information form without explicit observations."""

        if t is None:
            t = state.t
        allocation = self.validate_allocation(state, allocation)
        info_increment = self.batch_size(t) * allocation
        update_mask = (state.active > 0) & (allocation > self.allocation_floor)
        info_increment = torch.where(update_mask, info_increment, torch.zeros_like(info_increment))

        mean, cov = self.posterior_moments(state)
        next_n_eff = state.n_eff + info_increment
        next_precision = _symmetrize(self.prior_precision + next_n_eff[:, None, None] * self.obs_precision)
        next_cov = _symmetrize(torch.cholesky_inverse(torch.linalg.cholesky(next_precision)))
        increment_cov = _symmetrize(cov - next_cov)
        eye = torch.eye(self.n_metrics, dtype=state.dtype, device=state.device)

        next_mean = mean.clone()
        for arm_idx in range(self.n_arms):
            if not bool(update_mask[arm_idx]):
                continue
            chol = torch.linalg.cholesky(_symmetrize(increment_cov[arm_idx] + 1e-10 * eye))
            next_mean[arm_idx] = mean[arm_idx] + chol @ z[arm_idx]

        next_h = torch.einsum("kij,kj->ki", next_precision, next_mean)
        next_sum_g = torch.einsum("kij,kj->ki", self.obs_cov, next_h - self.prior_information)
        next_sum_g = torch.where(update_mask[:, None], next_sum_g, state.sum_g)

        return GaussianMetricState(
            n_eff=next_n_eff,
            sum_g=next_sum_g,
            active=state.active.clone(),
            t=state.t + 1,
        )
