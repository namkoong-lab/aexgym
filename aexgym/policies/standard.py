from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from aexgym.core.model import GaussianMetricModel, project_allocation
from aexgym.core.state import GaussianMetricState


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
        raw = torch.ones(state.n_arms, dtype=state.dtype, device=state.device)
        return project_allocation(raw, state.active).to(state.dtype)


class GaussianThompsonPolicy(MetricPolicy):
    def __init__(self, n_samples: int = 1000, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n_samples = int(n_samples)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        mean = model.target_mean(state)
        std = torch.sqrt(torch.clamp(model.target_variance(state), min=1e-16))
        z = torch.randn(self.n_samples, state.n_arms, dtype=state.dtype, device=state.device, generator=self.generator)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * z
        samples[:, state.active <= 0] = -torch.inf
        winners = torch.argmax(samples, dim=1)
        counts = torch.bincount(winners, minlength=state.n_arms).to(dtype=state.dtype, device=state.device)
        return project_allocation(counts, state.active).to(state.dtype)


class GaussianTopTwoThompsonPolicy(GaussianThompsonPolicy):
    def __init__(self, n_samples: int = 1000, coin: float = 0.5, name: Optional[str] = None) -> None:
        super().__init__(n_samples=n_samples, name=name)
        self.coin = float(coin)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        p_best = super().allocate(state, model)
        p_best = torch.clamp(p_best, min=1e-8, max=1.0 - 1e-8)
        active = (state.active > 0).to(dtype=p_best.dtype)
        odds = torch.where(state.active > 0, p_best / (1.0 - p_best), torch.zeros_like(p_best))
        sum_other_odds = torch.sum(odds) - odds
        probs = p_best * (self.coin + (1.0 - self.coin) * sum_other_odds)
        return project_allocation(probs * active, state.active).to(state.dtype)


def one_step_target_value(
    state: GaussianMetricState,
    model: GaussianMetricModel,
    allocation: Tensor,
    z: Tensor,
    residual_batch: Optional[Tensor] = None,
) -> Tensor:
    """Monte Carlo value of a target-only Gaussian update."""

    allocation = project_allocation(allocation.to(device=state.device), state.active).to(dtype=state.dtype)
    if residual_batch is None:
        residual_batch = model.batch_size(state.t)
    active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
    if active_arm_idx.numel() == 0:
        raise ValueError("active set cannot be empty")
    target_mean = model.target_mean(state)[active_arm_idx]
    prior_var = torch.clamp(model.target_variance(state)[active_arm_idx], min=1e-16)
    obs_var = torch.clamp(model.obs_cov[active_arm_idx, model.target_metric_idx, model.target_metric_idx], min=1e-16)
    active_allocation = allocation[active_arm_idx]
    post_var = 1.0 / (1.0 / prior_var + residual_batch * active_allocation / obs_var)
    phi = torch.sqrt(torch.clamp(prior_var - post_var, min=0.0))
    values = target_mean.unsqueeze(0) + z[:, active_arm_idx] * phi.unsqueeze(0)
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
    active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
    if active_arm_idx.numel() == 0:
        raise ValueError("active set cannot be empty")
    z = torch.randn(num_zs, state.n_arms, dtype=state.dtype, device=state.device, generator=generator)
    logits = torch.zeros(active_arm_idx.numel(), dtype=state.dtype, device=state.device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        probs = _active_logits_to_full_probs(logits, active_arm_idx, state.n_arms)
        loss = -one_step_target_value(state, model, probs, z, residual_batch=residual_batch)
        loss.backward()
        optimizer.step()

    probs = _active_logits_to_full_probs(logits.detach(), active_arm_idx, state.n_arms)
    return project_allocation(probs, state.active).to(state.dtype)


def _active_logits_to_full_probs(logits: Tensor, active_arm_idx: Tensor, n_arms: int) -> Tensor:
    active_probs = F.softmax(logits, dim=0)
    selector = F.one_hot(active_arm_idx, num_classes=n_arms).to(dtype=active_probs.dtype, device=active_probs.device).T
    return selector @ active_probs


class FixedSequencePolicy(MetricPolicy):
    """Small utility policy for deterministic tests and ablations."""

    def __init__(self, sequence: Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.sequence = sequence

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        idx = min(state.t, self.sequence.shape[0] - 1)
        return project_allocation(self.sequence[idx].to(dtype=state.dtype, device=state.device), state.active).to(state.dtype)
