from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from aexgym.metric.model import GaussianMetricModel, project_allocation
from aexgym.metric.state import GaussianMetricState


class MetricPolicy:
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self.generator: Optional[torch.Generator] = None

    def reset(self, seed: int) -> None:
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        raise NotImplementedError


class UniformActivePolicy(MetricPolicy):
    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        return project_allocation(torch.ones(state.n_arms, dtype=state.mean.dtype, device=state.mean.device), state.active).to(state.mean.dtype)


class GaussianThompsonPolicy(MetricPolicy):
    def __init__(self, n_samples: int = 1000, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n_samples = int(n_samples)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        if state.active_count <= 1:
            return project_allocation(torch.ones(state.n_arms, dtype=state.mean.dtype, device=state.mean.device), state.active).to(state.mean.dtype)
        target_idx = model.target_idx
        mean = state.mean[:, target_idx]
        std = torch.sqrt(torch.clamp(state.cov[:, target_idx, target_idx], min=1e-16))
        z = torch.randn(self.n_samples, state.n_arms, dtype=state.mean.dtype, device=state.mean.device, generator=self.generator)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * z
        samples[:, ~state.active] = -torch.inf
        winners = torch.argmax(samples, dim=1)
        counts = torch.bincount(winners, minlength=state.n_arms).to(dtype=state.mean.dtype)
        return project_allocation(counts, state.active).to(state.mean.dtype)


class GaussianTopTwoThompsonPolicy(GaussianThompsonPolicy):
    def __init__(self, n_samples: int = 1000, coin: float = 0.5, name: Optional[str] = None) -> None:
        super().__init__(n_samples=n_samples, name=name)
        self.coin = float(coin)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        p_best = super().allocate(state, model)
        if state.active_count <= 1:
            return p_best
        p_best = torch.clamp(p_best, min=1e-8, max=1.0 - 1e-8)
        active = state.active.to(dtype=p_best.dtype)
        odds = torch.where(state.active, p_best / (1.0 - p_best), torch.zeros_like(p_best))
        sum_other_odds = torch.sum(odds) - odds
        probs = p_best * (self.coin + (1.0 - self.coin) * sum_other_odds)
        probs = probs * active
        return project_allocation(probs, state.active).to(state.mean.dtype)


def one_step_target_value(
    state: GaussianMetricState,
    model: GaussianMetricModel,
    allocation: Tensor,
    z: Tensor,
    residual_batch: Optional[Tensor] = None,
) -> Tensor:
    """Monte Carlo value of a target-only Gaussian update."""

    target_idx = model.target_idx
    active = state.active
    allocation = project_allocation(allocation.to(device=state.mean.device), active).to(dtype=state.mean.dtype)
    if residual_batch is None:
        residual_batch = model.batch_size(state.t)
    active_idx = state.active_indices()
    target_mean = state.mean[active_idx, target_idx]
    prior_var = torch.clamp(state.cov[active_idx, target_idx, target_idx], min=1e-16)
    obs_var = torch.clamp(model.obs_cov[active_idx, target_idx, target_idx], min=1e-16)
    active_allocation = allocation[active_idx]
    post_var = 1.0 / (1.0 / prior_var + residual_batch * active_allocation / obs_var)
    phi = torch.sqrt(torch.clamp(prior_var - post_var, min=0.0))
    values = target_mean.unsqueeze(0) + z[:, active_idx] * phi.unsqueeze(0)
    return torch.max(values, dim=1).values.mean()


class MyopicLookaheadPolicy(MetricPolicy):
    def __init__(
        self,
        epochs: int = 80,
        lr: float = 0.05,
        num_zs: int = 256,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.num_zs = int(num_zs)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        return _optimize_target_value(
            state=state,
            model=model,
            epochs=self.epochs,
            lr=self.lr,
            num_zs=self.num_zs,
            generator=self.generator,
            residual_batch=model.batch_size(state.t),
        )


def _optimize_target_value(
    state: GaussianMetricState,
    model: GaussianMetricModel,
    epochs: int,
    lr: float,
    num_zs: int,
    generator: Optional[torch.Generator],
    residual_batch: Tensor,
) -> Tensor:
    active_idx = state.active_indices()
    if active_idx.numel() <= 1:
        return project_allocation(torch.ones(state.n_arms, dtype=state.mean.dtype, device=state.mean.device), state.active).to(state.mean.dtype)

    z = torch.randn(num_zs, state.n_arms, dtype=state.mean.dtype, device=state.mean.device, generator=generator)
    logits = torch.zeros(active_idx.numel(), dtype=state.mean.dtype, device=state.mean.device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        probs = _active_logits_to_full_probs(logits, active_idx, state.n_arms)
        loss = -one_step_target_value(state, model, probs, z, residual_batch=residual_batch)
        loss.backward()
        optimizer.step()

    probs = _active_logits_to_full_probs(logits.detach(), active_idx, state.n_arms)
    return project_allocation(probs, state.active).to(state.mean.dtype)


def _active_logits_to_full_probs(logits: Tensor, active_idx: Tensor, n_arms: int) -> Tensor:
    active_probs = F.softmax(logits, dim=0)
    selector = F.one_hot(active_idx, num_classes=n_arms).to(dtype=active_probs.dtype, device=active_probs.device).T
    return selector @ active_probs


class FixedSequencePolicy(MetricPolicy):
    """Small utility policy for deterministic tests and ablations."""

    def __init__(self, sequence: Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.sequence = sequence

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        idx = min(state.t, self.sequence.shape[0] - 1)
        return project_allocation(self.sequence[idx].to(dtype=state.mean.dtype, device=state.mean.device), state.active).to(state.mean.dtype)
