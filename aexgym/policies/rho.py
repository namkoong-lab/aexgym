from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from aexgym.core.model import GaussianMetricModel, project_allocation
from aexgym.core.rules import ActiveSetRule
from aexgym.core.state import GaussianMetricState
from aexgym.policies.standard import MetricPolicy


@dataclass(frozen=True)
class RhoPlan:
    sequence: Tensor
    first_allocation: Tensor
    base_allocation: Tensor | None
    residual_sequence: Tensor | None
    simulation_value: float
    regularization_penalty: float
    objective_value: float
    objective_scale: float


@dataclass(frozen=True)
class _RealizedRhoSequence:
    sequence: Tensor
    base_allocation: Tensor | None = None
    residual_sequence: Tensor | None = None


class RhoSimulation(Protocol):
    def prepare(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        horizon: int,
        num_samples: int,
        generator: Optional[torch.Generator],
    ) -> Any: ...

    def evaluate(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        sequence: Tensor,
        prepared: Any,
    ) -> Tensor: ...


class RhoParameterization(Protocol):
    def initialize(self, state: GaussianMetricState, model: GaussianMetricModel, horizon: int) -> dict[str, Tensor]: ...

    def realize(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        parameters: dict[str, Tensor],
    ) -> _RealizedRhoSequence: ...


class SequenceRegularizer(Protocol):
    def penalty(self, sequence: Tensor) -> Tensor: ...


def _active_logits_to_full_probs(logits: Tensor, active_idx: Tensor, n_arms: int) -> Tensor:
    active_probs = F.softmax(logits, dim=0)
    selector = F.one_hot(active_idx, num_classes=n_arms).to(dtype=active_probs.dtype, device=active_probs.device).T
    return selector @ active_probs


def _project_sequence(sequence: Tensor, state: GaussianMetricState) -> Tensor:
    return torch.stack([project_allocation(sequence[h], state.active).to(dtype=state.mean.dtype) for h in range(sequence.shape[0])])


class ReducedTerminalRhoSimulation:
    def prepare(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        horizon: int,
        num_samples: int,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        del horizon
        return torch.randn(num_samples, state.n_arms, dtype=state.mean.dtype, device=state.mean.device, generator=generator)

    def evaluate(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        sequence: Tensor,
        prepared: Tensor,
    ) -> Tensor:
        target_idx = model.target_idx
        budgets = model.batch_sizes[state.t : state.t + sequence.shape[0]].to(dtype=state.mean.dtype, device=state.mean.device)
        cumulative_info = torch.sum(budgets.unsqueeze(1) * sequence, dim=0)
        active_idx = state.active_indices()
        target_mean = state.mean[active_idx, target_idx]
        prior_var = torch.clamp(state.cov[active_idx, target_idx, target_idx], min=1e-16)
        obs_var = torch.clamp(model.obs_cov[active_idx, target_idx, target_idx], min=1e-16)
        active_info = cumulative_info[active_idx]
        post_var = 1.0 / (1.0 / prior_var + active_info / obs_var)
        phi = torch.sqrt(torch.clamp(prior_var - post_var, min=0.0))
        values = target_mean.unsqueeze(0) + prepared[:, active_idx] * phi.unsqueeze(0)
        return torch.max(values, dim=1).values.mean()


class PathwiseStoppedRhoSimulation:
    def __init__(self, active_rule: ActiveSetRule) -> None:
        self.active_rule = active_rule

    def prepare(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        horizon: int,
        num_samples: int,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        return torch.randn(
            num_samples,
            horizon,
            state.n_arms,
            state.n_metrics,
            dtype=state.mean.dtype,
            device=state.mean.device,
            generator=generator,
        )

    def evaluate(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        sequence: Tensor,
        prepared: Tensor,
    ) -> Tensor:
        values = []
        for rollout_idx in range(prepared.shape[0]):
            path_state = state.clone()
            for h in range(sequence.shape[0]):
                if path_state.stopped or path_state.active_count == 1:
                    path_state = self.active_rule.apply(path_state)
                    continue
                allocation = project_allocation(sequence[h], path_state.active).to(dtype=path_state.mean.dtype, device=path_state.mean.device)
                next_mean, next_cov = model.posterior_random_walk(
                    mean=path_state.mean,
                    cov=path_state.cov,
                    active=path_state.active,
                    allocation=allocation,
                    t=path_state.t,
                    z=prepared[rollout_idx, h],
                )
                path_state = GaussianMetricState(
                    mean=next_mean,
                    cov=next_cov,
                    active=path_state.active.clone(),
                    t=path_state.t + 1,
                )
                path_state = self.active_rule.apply(path_state)
            values.append(path_state.terminal_value(model.target_idx))
        return torch.stack(values).mean()


class ConstantAllocationParameterization:
    def initialize(self, state: GaussianMetricState, model: GaussianMetricModel, horizon: int) -> dict[str, Tensor]:
        del model
        active_idx = state.active_indices()
        logits = torch.zeros(active_idx.numel(), dtype=state.mean.dtype, device=state.mean.device, requires_grad=True)
        return {
            "logits": logits,
            "_horizon": torch.tensor(horizon, device=state.mean.device),
        }

    def realize(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        parameters: dict[str, Tensor],
    ) -> _RealizedRhoSequence:
        del model
        active_idx = state.active_indices()
        allocation = project_allocation(_active_logits_to_full_probs(parameters["logits"], active_idx, state.n_arms), state.active).to(state.mean.dtype)
        sequence = allocation.unsqueeze(0).repeat(int(parameters["_horizon"].item()), 1)
        residual = torch.zeros_like(sequence)
        return _RealizedRhoSequence(sequence=sequence, base_allocation=allocation, residual_sequence=residual)


class BasePlusResidualLogitParameterization:
    def initialize(self, state: GaussianMetricState, model: GaussianMetricModel, horizon: int) -> dict[str, Tensor]:
        del model
        active_idx = state.active_indices()
        base_logits = torch.zeros(active_idx.numel(), dtype=state.mean.dtype, device=state.mean.device, requires_grad=True)
        residual_logits = torch.zeros(horizon, active_idx.numel(), dtype=state.mean.dtype, device=state.mean.device, requires_grad=True)
        return {
            "base_logits": base_logits,
            "residual_logits": residual_logits,
        }

    def realize(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        parameters: dict[str, Tensor],
    ) -> _RealizedRhoSequence:
        del model
        active_idx = state.active_indices()
        base_logits = parameters["base_logits"]
        residual_logits = parameters["residual_logits"]
        centered_residuals = residual_logits - residual_logits.mean(dim=0, keepdim=True)
        base_allocation = project_allocation(_active_logits_to_full_probs(base_logits, active_idx, state.n_arms), state.active).to(state.mean.dtype)
        sequence_active = F.softmax(base_logits.unsqueeze(0) + centered_residuals, dim=1)
        selector = F.one_hot(active_idx, num_classes=state.n_arms).to(dtype=state.mean.dtype, device=state.mean.device)
        sequence = _project_sequence(sequence_active @ selector, state)
        residual = sequence - base_allocation.unsqueeze(0)
        return _RealizedRhoSequence(sequence=sequence, base_allocation=base_allocation, residual_sequence=residual)


class FreeSequenceParameterization:
    def initialize(self, state: GaussianMetricState, model: GaussianMetricModel, horizon: int) -> dict[str, Tensor]:
        del model
        active_idx = state.active_indices()
        logits = torch.zeros(horizon, active_idx.numel(), dtype=state.mean.dtype, device=state.mean.device, requires_grad=True)
        return {"logits": logits}

    def realize(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        parameters: dict[str, Tensor],
    ) -> _RealizedRhoSequence:
        del model
        active_idx = state.active_indices()
        logits = parameters["logits"]
        selector = F.one_hot(active_idx, num_classes=state.n_arms).to(dtype=state.mean.dtype, device=state.mean.device)
        sequence_active = F.softmax(logits, dim=1)
        sequence = _project_sequence(sequence_active @ selector, state)
        return _RealizedRhoSequence(sequence=sequence)


class NoSequenceRegularizer:
    def penalty(self, sequence: Tensor) -> Tensor:
        return torch.zeros((), dtype=sequence.dtype, device=sequence.device)


class TemporalUniformityRegularizer:
    def __init__(self, weight: float) -> None:
        self.weight = float(weight)

    def penalty(self, sequence: Tensor) -> Tensor:
        if sequence.shape[0] <= 1:
            return torch.zeros((), dtype=sequence.dtype, device=sequence.device)
        centered = sequence - sequence.mean(dim=0, keepdim=True)
        return sequence.new_tensor(self.weight) * torch.mean(centered**2)


class RhoPolicy(MetricPolicy):
    def __init__(
        self,
        simulator: RhoSimulation,
        parameterization: RhoParameterization,
        regularizer: SequenceRegularizer | None = None,
        epochs: int = 80,
        lr: float = 0.05,
        num_samples: int = 128,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.simulator = simulator
        self.parameterization = parameterization
        self.regularizer = regularizer or NoSequenceRegularizer()
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.num_samples = int(num_samples)
        self.last_plan: RhoPlan | None = None

    @property
    def last_sequence_allocation(self) -> Tensor | None:
        return None if self.last_plan is None else self.last_plan.sequence

    @property
    def last_base_allocation(self) -> Tensor | None:
        return None if self.last_plan is None else self.last_plan.base_allocation

    @property
    def last_sequence_residual(self) -> Tensor | None:
        return None if self.last_plan is None else self.last_plan.residual_sequence

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        horizon = model.horizon - state.t
        objective_scale = torch.sum(model.batch_sizes[state.t :]).to(dtype=state.mean.dtype, device=state.mean.device)
        allocation = project_allocation(torch.ones(state.n_arms, dtype=state.mean.dtype, device=state.mean.device), state.active).to(state.mean.dtype)
        if state.active_count <= 1:
            sequence = allocation.unsqueeze(0).repeat(horizon, 1)
            penalty = self.regularizer.penalty(sequence)
            simulation_value = float(state.terminal_value(model.target_idx).item())
            objective_value = simulation_value - float(penalty.item())
            self.last_plan = RhoPlan(
                sequence=sequence.detach().clone(),
                first_allocation=allocation.detach().clone(),
                base_allocation=allocation.detach().clone(),
                residual_sequence=torch.zeros_like(sequence),
                simulation_value=simulation_value,
                regularization_penalty=float(penalty.item()),
                objective_value=objective_value,
                objective_scale=float(objective_scale.item()),
            )
            return allocation

        parameters = self.parameterization.initialize(state, model, horizon)
        trainable = [value for value in parameters.values() if isinstance(value, Tensor) and value.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        prepared = self.simulator.prepare(state, model, horizon, self.num_samples, self.generator)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            realized = self.parameterization.realize(state, model, parameters)
            simulation_value = self.simulator.evaluate(state, model, realized.sequence, prepared)
            regularization_penalty = self.regularizer.penalty(realized.sequence)
            objective_value = simulation_value - regularization_penalty
            loss = objective_scale * (-objective_value)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            realized = self.parameterization.realize(state, model, parameters)
            simulation_value = self.simulator.evaluate(state, model, realized.sequence, prepared)
            regularization_penalty = self.regularizer.penalty(realized.sequence)
            objective_value = simulation_value - regularization_penalty
            first_allocation = project_allocation(realized.sequence[0], state.active).to(state.mean.dtype)
            self.last_plan = RhoPlan(
                sequence=realized.sequence.detach().clone(),
                first_allocation=first_allocation.detach().clone(),
                base_allocation=None if realized.base_allocation is None else realized.base_allocation.detach().clone(),
                residual_sequence=None if realized.residual_sequence is None else realized.residual_sequence.detach().clone(),
                simulation_value=float(simulation_value.item()),
                regularization_penalty=float(regularization_penalty.item()),
                objective_value=float(objective_value.item()),
                objective_scale=float(objective_scale.item()),
            )
        return first_allocation.to(state.mean.dtype)
