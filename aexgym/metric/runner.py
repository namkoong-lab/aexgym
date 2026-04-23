from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from aexgym.metric.model import GaussianMetricModel
from aexgym.metric.rules import ActiveSetRule, NoActiveSetRule
from aexgym.metric.state import active_indices_to_list


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
    stop_time: int
    stopped: bool
    stop_reason: str
    simple_regret: float
    guardrail_violation: bool
    unsafe_exposure: float
    active_path: list[list[int]]
    allocation_path: list[list[float]]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_results(results: Sequence[RunResult]) -> dict:
    if len(results) == 0:
        return {}
    regrets = torch.tensor([r.simple_regret for r in results], dtype=torch.float64)
    correct = torch.tensor([r.selected_arm == r.true_best_arm for r in results], dtype=torch.float64)
    stops = torch.tensor([r.stop_time for r in results], dtype=torch.float64)
    violations = torch.tensor([r.guardrail_violation for r in results], dtype=torch.float64)
    exposures = torch.tensor([r.unsafe_exposure for r in results], dtype=torch.float64)
    active_lengths = [len(path) for r in results for path in r.active_path]
    return {
        "n_runs": len(results),
        "mean_simple_regret": float(regrets.mean().item()),
        "se_simple_regret": float((regrets.std(unbiased=False) / max(len(results), 1) ** 0.5).item()),
        "percent_correct": float(correct.mean().item()),
        "average_stop_time": float(stops.mean().item()),
        "violation_rate": float(violations.mean().item()),
        "average_unsafe_exposure": float(exposures.mean().item()),
        "average_active_arms_logged": float(torch.tensor(active_lengths, dtype=torch.float64).mean().item()),
    }


class ExperimentRunner:
    """Shared simulation loop for Gaussian metric experiments."""

    def __init__(
        self,
        model: GaussianMetricModel,
        active_rule: Optional[ActiveSetRule] = None,
    ) -> None:
        self.model = model
        self.active_rule = active_rule or NoActiveSetRule(target_idx=model.target_idx)

    def run_one(self, instance: ExperimentInstance, policy, seed: int) -> RunResult:
        true_theta = torch.as_tensor(instance.true_theta, dtype=self.model.prior_mean.dtype, device=self.model.prior_mean.device)
        if true_theta.shape != self.model.prior_mean.shape:
            raise ValueError("instance true_theta must match model prior mean shape")

        if hasattr(policy, "reset"):
            policy.reset(seed)
        generator = torch.Generator(device=true_theta.device)
        generator.manual_seed(int(seed))

        state = self.model.initial_state()
        active_path = [active_indices_to_list(state.active)]
        allocation_path: list[list[float]] = []
        unsafe_exposure = 0.0

        while state.t < self.model.horizon and not state.stopped:
            allocation = policy.allocate(state, self.model)
            allocation = self.model.validate_allocation(state, allocation)
            allocation_path.append([float(x) for x in allocation.detach().cpu().tolist()])

            unsafe_arms = self.active_rule.unsafe_arms(true_theta)
            batch_size = self.model.batch_size(state.t)
            unsafe_exposure += float((batch_size * allocation * unsafe_arms.to(allocation.dtype)).sum().item())

            observation = self.model.sample_observation(true_theta, allocation, state, generator=generator)
            state = self.model.update(state, allocation, observation)
            state = self.active_rule.apply(state)
            active_path.append(active_indices_to_list(state.active))

        if not state.stopped:
            stop_reason = "horizon"
        else:
            stop_reason = state.stop_reason or "stopped"

        selected_arm = state.selected_arm(self.model.target_idx)
        target_values = true_theta[:, self.model.target_idx]
        true_best_arm = int(torch.argmax(target_values).item())
        simple_regret = float((torch.max(target_values) - target_values[selected_arm]).item())
        guardrail_violation = self.active_rule.selected_violates(true_theta, selected_arm)

        return RunResult(
            policy_name=getattr(policy, "name", policy.__class__.__name__),
            seed=int(seed),
            selected_arm=selected_arm,
            true_best_arm=true_best_arm,
            stop_time=int(state.t),
            stopped=bool(state.stopped),
            stop_reason=stop_reason,
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
