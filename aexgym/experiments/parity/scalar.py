from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from aexgym.core import (
    ExperimentInstance,
    ExperimentRunner,
    GaussianMetricModel,
    NoActiveSetRule,
    aggregate_results,
)
from aexgym.policies import (
    GaussianThompsonPolicy,
    GaussianTopTwoThompsonPolicy,
    MyopicLookaheadPolicy,
    UniformActivePolicy,
)
from aexgym.experiments.parity.rho_variants import DEFAULT_PARITY_RHO_VARIANTS, build_rho_policy


PARITY_SCENARIOS = {
    "exact_gaussian",
    "beta_bernoulli",
    "gamma_gumbel",
    "variance_noise",
    "arm_count",
    "batch_size",
    "nonuniform_prior",
    "unknown_variance",
    "horizon_misspecification",
}

BASELINE_POLICIES: tuple[str, ...] = ("uniform", "ts", "ttts", "myopic")


@dataclass
class ScalarParityConfig:
    scenario: str = "exact_gaussian"
    n_arms: int = 10
    horizon: int = 5
    batch_size: float = 1.0
    n_runs: int = 20
    seed: int = 1
    prior_mean: float = 0.0
    prior_var: float = 0.1
    obs_var: float = 1.0
    mean_scale: float = 0.1
    prior_pattern: str = "flat"
    baseline_policies: list[str] = field(default_factory=lambda: list(BASELINE_POLICIES))
    rho_variants: list[str] = field(default_factory=lambda: ["reduced_constant"])
    rho_epochs: int = 60
    rho_num_samples: int = 256
    rho_lr: float = 0.08
    rho_temporal_regularization: float = 300.0
    output: str | None = None


def load_config(path: str | None) -> ScalarParityConfig:
    if path is None:
        return ScalarParityConfig()
    data = json.loads(Path(path).read_text())
    return ScalarParityConfig(**data)


def prior_mean_vector(config: ScalarParityConfig) -> torch.Tensor:
    mean = torch.full((config.n_arms, 1), float(config.prior_mean), dtype=torch.float64)
    perturb = config.mean_scale
    if config.prior_pattern == "top_one":
        mean[0, 0] += perturb
    elif config.prior_pattern == "top_half":
        mean[: max(1, config.n_arms // 2), 0] += perturb
    elif config.prior_pattern == "descending":
        mean[:, 0] += torch.linspace(perturb, 0.0, config.n_arms, dtype=torch.float64)
    elif config.prior_pattern != "flat":
        raise ValueError(f"unknown prior_pattern: {config.prior_pattern}")
    return mean


def make_model(config: ScalarParityConfig) -> GaussianMetricModel:
    prior_mean = prior_mean_vector(config)
    prior_cov = torch.eye(1, dtype=torch.float64).repeat(config.n_arms, 1, 1) * config.prior_var
    obs_var = config.obs_var
    if config.scenario == "variance_noise":
        obs_cov = torch.linspace(0.2, 5.0, config.n_arms, dtype=torch.float64).reshape(config.n_arms, 1, 1)
    elif config.scenario == "unknown_variance":
        obs_cov = torch.full((config.n_arms, 1, 1), obs_var * 1.5, dtype=torch.float64)
    else:
        obs_cov = torch.full((config.n_arms, 1, 1), obs_var, dtype=torch.float64)

    horizon = config.horizon
    if config.scenario == "horizon_misspecification":
        horizon = max(1, config.horizon * 2)
    batch_sizes = torch.full((horizon,), config.batch_size, dtype=torch.float64)
    return GaussianMetricModel(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        obs_cov=obs_cov,
        target_idx=0,
        batch_sizes=batch_sizes,
    )


def make_instance(config: ScalarParityConfig, model: GaussianMetricModel, seed: int) -> ExperimentInstance:
    rng = np.random.default_rng(seed)
    if config.scenario == "beta_bernoulli":
        alpha = np.full(config.n_arms, 100.0)
        beta = np.full(config.n_arms, 100.0)
        probs = rng.beta(alpha, beta)
        true_values = (probs - 0.5) / max(config.mean_scale, 1e-12)
    elif config.scenario == "gamma_gumbel":
        shape = np.full(config.n_arms, 100.0)
        scale = np.full(config.n_arms, 0.01)
        true_values = rng.gamma(shape, scale)
    else:
        prior_mean = model.prior_mean[:, 0].cpu().numpy()
        prior_std = np.sqrt(model.prior_cov[:, 0, 0].cpu().numpy())
        true_values = rng.normal(prior_mean, prior_std)

    true_theta = torch.as_tensor(true_values, dtype=torch.float64).reshape(config.n_arms, 1)
    return ExperimentInstance(true_theta=true_theta, name=config.scenario, metadata={"scenario": config.scenario})


def make_policies(config: ScalarParityConfig) -> dict[str, object]:
    policies: dict[str, object] = {}
    for name in config.baseline_policies:
        if name == "uniform":
            policies[name] = UniformActivePolicy(name="uniform")
        elif name == "ts":
            policies[name] = GaussianThompsonPolicy(n_samples=512, name="ts")
        elif name == "ttts":
            policies[name] = GaussianTopTwoThompsonPolicy(n_samples=512, coin=0.5, name="ttts")
        elif name == "myopic":
            policies[name] = MyopicLookaheadPolicy(epochs=40, num_zs=128, lr=0.08, name="myopic")
        else:
            raise ValueError(f"unknown baseline policy: {name}")
    for variant in config.rho_variants:
        policies[variant] = build_rho_policy(
            variant,
            target_idx=0,
            epochs=config.rho_epochs,
            num_samples=config.rho_num_samples,
            lr=config.rho_lr,
            temporal_regularization=config.rho_temporal_regularization,
        )
    return policies


def run_config(config: ScalarParityConfig) -> dict:
    if config.scenario not in PARITY_SCENARIOS:
        raise ValueError(f"unknown scenario {config.scenario}; choose one of {sorted(PARITY_SCENARIOS)}")
    model = make_model(config)
    runner = ExperimentRunner(model, active_rule=NoActiveSetRule(target_idx=0))
    policies = make_policies(config)
    if not policies:
        raise ValueError("at least one baseline policy or rho variant must be configured")
    seeds = [config.seed + i for i in range(config.n_runs)]
    per_policy = {}
    aggregates = {}
    for policy_name, policy in policies.items():
        results = []
        for seed in seeds:
            instance = make_instance(config, model, seed)
            results.append(runner.run_one(instance, policy, seed))
        per_policy[policy_name] = [result.to_dict() for result in results]
        aggregates[policy_name] = aggregate_results(results)
    return {
        "config": asdict(config),
        "aggregates": aggregates,
        "runs": per_policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalar J=1 parity experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print available scenario names.")
    parser.add_argument("--list-baselines", action="store_true", help="Print available non-RHO baseline policy names.")
    parser.add_argument("--list-rho-variants", action="store_true", help="Print available parity RHO variant names.")
    args = parser.parse_args()

    if args.list_scenarios:
        print(json.dumps(sorted(PARITY_SCENARIOS), indent=2))
        return
    if args.list_baselines:
        print(json.dumps(list(BASELINE_POLICIES), indent=2))
        return
    if args.list_rho_variants:
        print(json.dumps(list(DEFAULT_PARITY_RHO_VARIANTS), indent=2))
        return

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
