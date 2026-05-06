from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from aexgym.core.model import GaussianMetricModel
from aexgym.core.rules import ActiveSetRule, NoActiveSetRule, SmoothingConfig


@dataclass(frozen=True)
class ExperimentInstance:
    true_theta: Tensor
    name: str = "instance"
    metadata: dict = field(default_factory=dict)


@dataclass
class RunResult:
    policy_name: str
    seed: int
    selected_arm: int
    true_best_arm: int
    completed_stages: int
    simple_regret: float
    guardrail_violation: bool
    unsafe_exposure: float
    active_path: list[list[float]]
    allocation_path: list[list[float]]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_results(results: Sequence[RunResult]) -> dict:
    if len(results) == 0:
        return {}
    regrets = torch.tensor([r.simple_regret for r in results], dtype=torch.float64)
    selected_true_best = torch.tensor([r.selected_arm == r.true_best_arm for r in results], dtype=torch.float64)
    completed_stages = torch.tensor([r.completed_stages for r in results], dtype=torch.float64)
    violations = torch.tensor([r.guardrail_violation for r in results], dtype=torch.float64)
    exposures = torch.tensor([r.unsafe_exposure for r in results], dtype=torch.float64)
    active_mass = torch.tensor([sum(path) for r in results for path in r.active_path], dtype=torch.float64)
    return {
        "n_runs": len(results),
        "mean_simple_regret": float(regrets.mean().item()),
        "se_simple_regret": float((regrets.std(unbiased=False) / max(len(results), 1) ** 0.5).item()),
        "selected_true_best_rate": float(selected_true_best.mean().item()),
        "average_completed_stages": float(completed_stages.mean().item()),
        "violation_rate": float(violations.mean().item()),
        "average_unsafe_exposure": float(exposures.mean().item()),
        "average_active_mass_logged": float(active_mass.mean().item()),
    }


class ExperimentRunner:
    """Shared simulation loop for Gaussian metric experiments."""

    def __init__(
        self,
        model: GaussianMetricModel,
        active_rule: Optional[ActiveSetRule] = None,
        smoothing: Optional[SmoothingConfig] = None,
    ) -> None:
        self.model = model
        self.active_rule = active_rule or NoActiveSetRule(target_metric_idx=model.target_metric_idx)
        self.smoothing = smoothing or SmoothingConfig()

    def run_one(self, instance: ExperimentInstance, policy, seed: int) -> RunResult:
        true_theta = torch.as_tensor(instance.true_theta, dtype=self.model.prior_mean.dtype, device=self.model.prior_mean.device)
        if true_theta.shape != self.model.prior_mean.shape:
            raise ValueError("instance true_theta must match model prior mean shape")

        if hasattr(policy, "reset"):
            policy.reset(seed)
        generator = torch.Generator(device=true_theta.device)
        generator.manual_seed(int(seed))

        state = self.model.initial_state()
        active_path = [[float(x) for x in state.active.detach().cpu().tolist()]]
        allocation_path: list[list[float]] = []
        unsafe_exposure = 0.0

        while state.t < self.model.horizon:
            allocation = policy.allocate(state, self.model)
            allocation = self.model.validate_allocation(state, allocation)
            allocation_path.append([float(x) for x in allocation.detach().cpu().tolist()])

            unsafe_arms = self.active_rule.unsafe_arms(true_theta)
            batch_size = self.model.batch_size(state.t)
            unsafe_exposure += float((batch_size * allocation * unsafe_arms.to(allocation.dtype)).sum().item())

            observation = self.model.sample_observation(true_theta, allocation, state, generator=generator)
            state = self.model.update(state, allocation, observation)
            decision = self.active_rule.evaluate(state, self.model, smoothing=self.smoothing)
            state = state.replace(active=decision.active)
            active_path.append([float(x) for x in state.active.detach().cpu().tolist()])

        selected_arm = self.model.selected_arm(state)
        target_values = true_theta[:, self.model.target_metric_idx]
        true_best_arm = int(torch.argmax(target_values).item())
        simple_regret = float((torch.max(target_values) - target_values[selected_arm]).item())
        guardrail_violation = self.active_rule.selected_violates(true_theta, selected_arm)

        return RunResult(
            policy_name=getattr(policy, "name", policy.__class__.__name__),
            seed=int(seed),
            selected_arm=selected_arm,
            true_best_arm=true_best_arm,
            completed_stages=int(state.t),
            simple_regret=simple_regret,
            guardrail_violation=guardrail_violation,
            unsafe_exposure=unsafe_exposure,
            active_path=active_path,
            allocation_path=allocation_path,
            metadata={"instance": instance.name, **instance.metadata},
        )

    def run(self, instance: ExperimentInstance, policies, seeds: Sequence[int]):
        if isinstance(policies, Mapping):
            return {
                name: [self.run_one(instance, policy, seed) for seed in seeds]
                for name, policy in policies.items()
            }
        return [self.run_one(instance, policies, seed) for seed in seeds]
