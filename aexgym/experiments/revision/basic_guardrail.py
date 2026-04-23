from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from aexgym.core import (
    ActiveSetRule,
    ExperimentInstance,
    ExperimentRunner,
    GaussianMetricModel,
    aggregate_results,
)
from aexgym.policies import (
    BasePlusResidualLogitParameterization,
    ConstantAllocationParameterization,
    GaussianThompsonPolicy,
    GaussianTopTwoThompsonPolicy,
    MyopicLookaheadPolicy,
    NoSequenceRegularizer,
    PathwiseStoppedRhoSimulation,
    ReducedTerminalRhoSimulation,
    RhoPolicy,
    TemporalUniformityRegularizer,
    UniformActivePolicy,
)


@dataclass
class GuardrailBasicConfig:
    n_arms: int = 5
    horizon: int = 5
    batch_sizes: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.0, 1.0])
    n_runs: int = 20
    seed: int = 11
    control_arm: int = 0
    target_idx: int = 0
    guardrail_idx: int = 1
    guardrail_floor: float = -0.25
    violation_prob_threshold: float = 0.8
    open_loop_temporal_regularization: float = 300.0
    policies: list[str] = field(default_factory=lambda: ["uniform", "ts", "ttts", "myopic", "constant_rho", "openloop_stopped_rho"])
    output: str | None = None


def load_config(path: str | None) -> GuardrailBasicConfig:
    if path is None:
        return GuardrailBasicConfig()
    return GuardrailBasicConfig(**json.loads(Path(path).read_text()))


def make_model(config: GuardrailBasicConfig) -> GaussianMetricModel:
    prior_mean = torch.zeros(config.n_arms, 2, dtype=torch.float64)
    prior_cov = torch.eye(2, dtype=torch.float64).repeat(config.n_arms, 1, 1)
    prior_cov[:, 0, 0] = 0.08
    prior_cov[:, 1, 1] = 0.12
    prior_cov[:, 0, 1] = 0.04
    prior_cov[:, 1, 0] = 0.04

    obs_cov = torch.eye(2, dtype=torch.float64).repeat(config.n_arms, 1, 1)
    obs_cov[:, 0, 0] = 1.0
    obs_cov[:, 1, 1] = 1.0
    obs_cov[:, 0, 1] = 0.35
    obs_cov[:, 1, 0] = 0.35

    batch_sizes = torch.as_tensor(config.batch_sizes[: config.horizon], dtype=torch.float64)
    if batch_sizes.numel() < config.horizon:
        tail = torch.ones(config.horizon - batch_sizes.numel(), dtype=torch.float64)
        batch_sizes = torch.cat([batch_sizes, tail])

    return GaussianMetricModel(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        obs_cov=obs_cov,
        target_idx=config.target_idx,
        batch_sizes=batch_sizes,
        control_arm=config.control_arm,
    )


def make_instance(config: GuardrailBasicConfig) -> ExperimentInstance:
    if config.n_arms != 5:
        raise ValueError("guardrail_basic currently fixes the synthetic truth at n_arms=5")
    true_theta = torch.tensor(
        [
            [0.00, 0.00],
            [0.10, -0.10],
            [0.16, -0.42],
            [0.07, -0.32],
            [0.04, 0.05],
        ],
        dtype=torch.float64,
    )
    return ExperimentInstance(
        true_theta=true_theta,
        name="guardrail_basic",
        metadata={"scenario": "guardrail_basic"},
    )


def make_rule(config: GuardrailBasicConfig) -> ActiveSetRule:
    return ActiveSetRule(
        target_idx=config.target_idx,
        control_arm=config.control_arm,
        guardrail_indices=[config.guardrail_idx],
        guardrail_floors=[config.guardrail_floor],
        violation_prob_threshold=config.violation_prob_threshold,
    )


def make_policies(config: GuardrailBasicConfig, rule: ActiveSetRule) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for name in config.policies:
        if name == "uniform":
            policies[name] = UniformActivePolicy(name="uniform")
        elif name == "ts":
            policies[name] = GaussianThompsonPolicy(n_samples=512, name="ts")
        elif name == "ttts":
            policies[name] = GaussianTopTwoThompsonPolicy(n_samples=512, coin=0.5, name="ttts")
        elif name == "myopic":
            policies[name] = MyopicLookaheadPolicy(epochs=40, num_zs=128, lr=0.06, name="myopic")
        elif name == "constant_rho":
            policies[name] = RhoPolicy(
                simulator=ReducedTerminalRhoSimulation(),
                parameterization=ConstantAllocationParameterization(),
                regularizer=NoSequenceRegularizer(),
                epochs=60,
                num_samples=256,
                lr=0.06,
                name="constant_rho",
            )
        elif name == "openloop_stopped_rho":
            policies[name] = RhoPolicy(
                simulator=PathwiseStoppedRhoSimulation(rule),
                parameterization=BasePlusResidualLogitParameterization(),
                regularizer=TemporalUniformityRegularizer(weight=config.open_loop_temporal_regularization),
                epochs=40,
                num_samples=64,
                lr=0.05,
                name="openloop_stopped_rho",
            )
        else:
            raise ValueError(f"unknown policy: {name}")
    return policies


def run_config(config: GuardrailBasicConfig) -> dict:
    model = make_model(config)
    rule = make_rule(config)
    runner = ExperimentRunner(model, active_rule=rule)
    instance = make_instance(config)
    policies = make_policies(config, rule)
    seeds = [config.seed + i for i in range(config.n_runs)]

    per_policy = {}
    aggregates = {}
    for policy_name, policy in policies.items():
        results = [runner.run_one(instance, policy, seed) for seed in seeds]
        per_policy[policy_name] = [result.to_dict() for result in results]
        aggregates[policy_name] = aggregate_results(results)

    return {
        "config": asdict(config),
        "aggregates": aggregates,
        "runs": per_policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the basic two-metric guardrail experiment.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output is not None:
        config.output = args.output
    payload = run_config(config)
    output_text = json.dumps(payload, indent=2)
    if config.output:
        Path(config.output).write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
