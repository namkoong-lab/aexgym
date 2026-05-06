from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from aexgym.core.model import GaussianMetricModel, project_allocation
from aexgym.core.rules import ActiveSetRule, SmoothingConfig
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
    active_beta: float
    terminal_beta: float
    hard_simulation_value: float | None = None
    optimization_seed: int | None = None
    optimization_trace: tuple[dict[str, float], ...] = ()


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
        *,
        smoothing: SmoothingConfig,
        terminal_beta: float,
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


def _active_logits_to_full_probs(logits: Tensor, active_arm_idx: Tensor, n_arms: int) -> Tensor:
    active_probs = F.softmax(logits, dim=0)
    selector = F.one_hot(active_arm_idx, num_classes=n_arms).to(dtype=active_probs.dtype, device=active_probs.device).T
    return selector @ active_probs


def _project_sequence(sequence: Tensor, state: GaussianMetricState) -> Tensor:
    return torch.stack([project_allocation(sequence[h], state.active).to(dtype=state.dtype) for h in range(sequence.shape[0])])


class ReducedTerminalRhoSimulation:
    """Reduced terminal objective retained for scalar parity experiments."""

    def __init__(self, sample_method: str = "normal", sobol_seed: int = 42) -> None:
        if sample_method not in {"normal", "sobol"}:
            raise ValueError("sample_method must be 'normal' or 'sobol'")
        self.sample_method = sample_method
        self.sobol_seed = int(sobol_seed)

    def prepare(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        horizon: int,
        num_samples: int,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        del horizon, model
        if self.sample_method == "normal":
            return torch.randn(num_samples, state.n_arms, dtype=state.dtype, device=state.device, generator=generator)
        sobol = torch.quasirandom.SobolEngine(state.n_arms, scramble=True, seed=self.sobol_seed)
        u = sobol.draw(num_samples).to(dtype=state.dtype, device=state.device)
        u = torch.clamp(u, min=torch.finfo(u.dtype).eps, max=1.0 - torch.finfo(u.dtype).eps)
        return torch.sqrt(torch.tensor(2.0, dtype=u.dtype, device=u.device)) * torch.erfinv(2.0 * u - 1.0)

    def evaluate(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        sequence: Tensor,
        prepared: Tensor,
        *,
        smoothing: SmoothingConfig,
        terminal_beta: float,
    ) -> Tensor:
        del smoothing, terminal_beta
        budgets = model.batch_sizes[state.t : state.t + sequence.shape[0]].to(dtype=state.dtype, device=state.device)
        cumulative_info = torch.sum(budgets.unsqueeze(1) * sequence, dim=0)
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        target_mean = model.target_mean(state)[active_arm_idx]
        prior_var = torch.clamp(model.target_variance(state)[active_arm_idx], min=1e-16)
        obs_var = torch.clamp(model.obs_cov[active_arm_idx, model.target_metric_idx, model.target_metric_idx], min=1e-16)
        active_info = cumulative_info[active_arm_idx]
        post_var = 1.0 / (1.0 / prior_var + active_info / obs_var)
        phi = torch.sqrt(torch.clamp(prior_var - post_var, min=0.0))
        values = target_mean.unsqueeze(0) + prepared[:, active_arm_idx] * phi.unsqueeze(0)
        return torch.max(values, dim=1).values.mean()


class PathwiseActiveSetRhoSimulation:
    """Revision RHO simulator using active-weight projection in hard and smooth modes."""

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
            dtype=state.dtype,
            device=state.device,
            generator=generator,
        )

    def evaluate(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        sequence: Tensor,
        prepared: Tensor,
        *,
        smoothing: SmoothingConfig,
        terminal_beta: float,
    ) -> Tensor:
        values = []
        for rollout_idx in range(prepared.shape[0]):
            path_state = state.clone()
            for h in range(sequence.shape[0]):
                allocation = project_allocation(sequence[h], path_state.active).to(dtype=path_state.dtype, device=path_state.device)
                path_state = model.posterior_random_walk(
                    state=path_state,
                    allocation=allocation,
                    t=path_state.t,
                    z=prepared[rollout_idx, h],
                )
                decision = self.active_rule.evaluate(path_state, model, smoothing=smoothing)
                path_state = path_state.replace(active=decision.active)
            values.append(model.terminal_value(path_state, terminal_beta=math.inf if smoothing.hard else terminal_beta))
        return torch.stack(values).mean()


class ConstantAllocationParameterization:
    def initialize(self, state: GaussianMetricState, model: GaussianMetricModel, horizon: int) -> dict[str, Tensor]:
        del model
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        logits = torch.zeros(active_arm_idx.numel(), dtype=state.dtype, device=state.device, requires_grad=True)
        return {
            "logits": logits,
            "_horizon": torch.tensor(horizon, device=state.device),
        }

    def realize(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        parameters: dict[str, Tensor],
    ) -> _RealizedRhoSequence:
        del model
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        allocation = project_allocation(_active_logits_to_full_probs(parameters["logits"], active_arm_idx, state.n_arms), state.active).to(state.dtype)
        sequence = allocation.unsqueeze(0).repeat(int(parameters["_horizon"].item()), 1)
        residual = torch.zeros_like(sequence)
        return _RealizedRhoSequence(sequence=sequence, base_allocation=allocation, residual_sequence=residual)


class BasePlusResidualLogitParameterization:
    def initialize(self, state: GaussianMetricState, model: GaussianMetricModel, horizon: int) -> dict[str, Tensor]:
        del model
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        base_logits = torch.zeros(active_arm_idx.numel(), dtype=state.dtype, device=state.device, requires_grad=True)
        residual_logits = torch.zeros(horizon, active_arm_idx.numel(), dtype=state.dtype, device=state.device, requires_grad=True)
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
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        base_logits = parameters["base_logits"]
        residual_logits = parameters["residual_logits"]
        centered_residuals = residual_logits - residual_logits.mean(dim=0, keepdim=True)
        base_allocation = project_allocation(_active_logits_to_full_probs(base_logits, active_arm_idx, state.n_arms), state.active).to(state.dtype)
        sequence_active = F.softmax(base_logits.unsqueeze(0) + centered_residuals, dim=1)
        selector = F.one_hot(active_arm_idx, num_classes=state.n_arms).to(dtype=state.dtype, device=state.device)
        sequence = _project_sequence(sequence_active @ selector, state)
        residual = sequence - base_allocation.unsqueeze(0)
        return _RealizedRhoSequence(sequence=sequence, base_allocation=base_allocation, residual_sequence=residual)


class FreeSequenceParameterization:
    def initialize(self, state: GaussianMetricState, model: GaussianMetricModel, horizon: int) -> dict[str, Tensor]:
        del model
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        logits = torch.zeros(horizon, active_arm_idx.numel(), dtype=state.dtype, device=state.device, requires_grad=True)
        return {"logits": logits}

    def realize(
        self,
        state: GaussianMetricState,
        model: GaussianMetricModel,
        parameters: dict[str, Tensor],
    ) -> _RealizedRhoSequence:
        del model
        active_arm_idx = torch.nonzero(state.active > 0, as_tuple=False).flatten()
        if active_arm_idx.numel() == 0:
            raise ValueError("active set cannot be empty")
        logits = parameters["logits"]
        selector = F.one_hot(active_arm_idx, num_classes=state.n_arms).to(dtype=state.dtype, device=state.device)
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
        active_beta: float = math.inf,
        terminal_beta: float = 25.0,
        optimization_seed: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.simulator = simulator
        self.parameterization = parameterization
        self.regularizer = regularizer or NoSequenceRegularizer()
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.num_samples = int(num_samples)
        self.active_beta = float(active_beta)
        self.terminal_beta = float(terminal_beta)
        self.optimization_seed = None if optimization_seed is None else int(optimization_seed)
        self._optimization_call_count = 0
        self.last_plan: RhoPlan | None = None
        self.last_optimization_trace: list[dict[str, float]] = []

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._optimization_call_count = 0

    def _next_optimization_generator(self, state: GaussianMetricState) -> tuple[Optional[torch.Generator], Optional[int]]:
        if self.optimization_seed is None:
            return self.generator, None
        seed = self.optimization_seed + self._optimization_call_count
        self._optimization_call_count += 1
        generator = torch.Generator(device=state.device)
        generator.manual_seed(seed)
        return generator, seed

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        horizon = model.horizon - state.t
        objective_scale = torch.sum(model.batch_sizes[state.t :]).to(dtype=state.dtype, device=state.device)
        allocation = project_allocation(torch.ones(state.n_arms, dtype=state.dtype, device=state.device), state.active).to(state.dtype)
        if horizon <= 0:
            self.last_optimization_trace = []
            sequence = allocation.unsqueeze(0).repeat(max(horizon, 1), 1)
            penalty = self.regularizer.penalty(sequence)
            simulation_value = float(model.terminal_value(state).item())
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
                active_beta=self.active_beta,
                terminal_beta=self.terminal_beta,
                optimization_seed=None,
                optimization_trace=(),
            )
            return allocation

        parameters = self.parameterization.initialize(state, model, horizon)
        trainable = [value for value in parameters.values() if isinstance(value, Tensor) and value.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        optimization_generator, optimization_seed = self._next_optimization_generator(state)
        prepared = self.simulator.prepare(state, model, horizon, self.num_samples, optimization_generator)
        optimization_trace: list[dict[str, float]] = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            realized = self.parameterization.realize(state, model, parameters)
            smoothing = SmoothingConfig(beta=self.active_beta)
            simulation_value = self.simulator.evaluate(
                state,
                model,
                realized.sequence,
                prepared,
                smoothing=smoothing,
                terminal_beta=self.terminal_beta,
            )
            regularization_penalty = self.regularizer.penalty(realized.sequence)
            objective_value = simulation_value - regularization_penalty
            loss = objective_scale * (-objective_value)
            optimization_trace.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(loss.detach().cpu().item()),
                    "objective_value": float(objective_value.detach().cpu().item()),
                    "simulation_value": float(simulation_value.detach().cpu().item()),
                    "regularization_penalty": float(regularization_penalty.detach().cpu().item()),
                    "objective_scale": float(objective_scale.detach().cpu().item()),
                }
            )
            loss.backward()
            optimizer.step()

        self.last_optimization_trace = optimization_trace

        with torch.no_grad():
            realized = self.parameterization.realize(state, model, parameters)
            smoothing = SmoothingConfig(beta=self.active_beta)
            simulation_value = self.simulator.evaluate(
                state,
                model,
                realized.sequence,
                prepared,
                smoothing=smoothing,
                terminal_beta=self.terminal_beta,
            )
            hard_value = self.simulator.evaluate(
                state,
                model,
                realized.sequence,
                prepared,
                smoothing=SmoothingConfig(beta=math.inf),
                terminal_beta=math.inf,
            )
            regularization_penalty = self.regularizer.penalty(realized.sequence)
            objective_value = simulation_value - regularization_penalty
            first_allocation = project_allocation(realized.sequence[0], state.active).to(state.dtype)
            self.last_plan = RhoPlan(
                sequence=realized.sequence.detach().clone(),
                first_allocation=first_allocation.detach().clone(),
                base_allocation=None if realized.base_allocation is None else realized.base_allocation.detach().clone(),
                residual_sequence=None if realized.residual_sequence is None else realized.residual_sequence.detach().clone(),
                simulation_value=float(simulation_value.item()),
                regularization_penalty=float(regularization_penalty.item()),
                objective_value=float(objective_value.item()),
                objective_scale=float(objective_scale.item()),
                active_beta=self.active_beta,
                terminal_beta=self.terminal_beta,
                hard_simulation_value=float(hard_value.item()),
                optimization_seed=optimization_seed,
                optimization_trace=tuple(optimization_trace),
            )
        return first_allocation.to(state.dtype)
