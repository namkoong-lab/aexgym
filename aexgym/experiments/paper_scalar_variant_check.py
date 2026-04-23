from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from aexgym.experiments.paper_scalar_replication import (
    LegacyRho,
    PaperBandit,
    PaperScalarConfig,
    paper_priors,
    sample_theta,
)
from aexgym.metric import (
    ActiveSetRule,
    BasePlusResidualLogitParameterization,
    ConstantAllocationParameterization,
    GaussianMetricModel,
    NoSequenceRegularizer,
    PathwiseStoppedRhoSimulation,
    ReducedTerminalRhoSimulation,
    RhoPolicy,
    TemporalUniformityRegularizer,
)


@dataclass
class VariantCheckConfig(PaperScalarConfig):
    variants: list[str] = field(default_factory=lambda: ["legacy_j1", "reduced_j2_ignored", "pathwise_j2_ignored"])
    open_loop_epochs: int | None = None
    open_loop_rollouts: int | None = None
    open_loop_lr: float | None = None
    open_loop_temporal_regularization: float = 300.0
    secondary_obs_var: float = 1.0


def load_variant_config(path: str | None) -> VariantCheckConfig:
    if path is None:
        return VariantCheckConfig()
    return VariantCheckConfig(**json.loads(Path(path).read_text()))


def make_metric_model(
    mu_0: np.ndarray,
    sigma2_0: np.ndarray,
    s2: np.ndarray,
    horizon: int,
    n_metrics: int,
    secondary_obs_var: float,
) -> GaussianMetricModel:
    prior_mean = torch.zeros((len(mu_0), n_metrics), dtype=torch.float64)
    prior_mean[:, 0] = torch.as_tensor(mu_0, dtype=torch.float64)
    prior_cov = torch.eye(n_metrics, dtype=torch.float64).repeat(len(mu_0), 1, 1)
    prior_cov[:, 0, 0] = torch.as_tensor(sigma2_0, dtype=torch.float64)
    obs_cov = torch.eye(n_metrics, dtype=torch.float64).repeat(len(mu_0), 1, 1) * float(secondary_obs_var)
    obs_cov[:, 0, 0] = torch.as_tensor(s2, dtype=torch.float64)
    return GaussianMetricModel(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        obs_cov=obs_cov,
        target_idx=0,
        batch_sizes=torch.ones(horizon, dtype=torch.float64),
    )


def make_policy(config: VariantCheckConfig, variant: str):
    if variant == "legacy_j1":
        return LegacyRho(
            eps=config.rho_eps,
            n_max=config.rho_epochs,
            lr=config.rho_lr,
            num_zs=config.rho_num_zs,
        )
    if variant == "reduced_j1":
        return RhoPolicy(
            simulator=ReducedTerminalRhoSimulation(),
            parameterization=ConstantAllocationParameterization(),
            regularizer=NoSequenceRegularizer(),
            epochs=config.rho_epochs,
            num_samples=config.rho_num_zs,
            lr=config.rho_lr,
            name=variant,
        )
    if variant == "reduced_j2_ignored":
        return RhoPolicy(
            simulator=ReducedTerminalRhoSimulation(),
            parameterization=ConstantAllocationParameterization(),
            regularizer=NoSequenceRegularizer(),
            epochs=config.rho_epochs,
            num_samples=config.rho_num_zs,
            lr=config.rho_lr,
            name=variant,
        )
    if variant == "pathwise_j2_ignored":
        rule = ActiveSetRule(target_idx=0, stop_on_singleton=False)
        return RhoPolicy(
            simulator=PathwiseStoppedRhoSimulation(rule),
            parameterization=BasePlusResidualLogitParameterization(),
            regularizer=TemporalUniformityRegularizer(weight=config.open_loop_temporal_regularization),
            epochs=config.rho_epochs if config.open_loop_epochs is None else config.open_loop_epochs,
            num_samples=config.rho_num_zs if config.open_loop_rollouts is None else config.open_loop_rollouts,
            lr=config.rho_lr if config.open_loop_lr is None else config.open_loop_lr,
            name=variant,
        )
    raise ValueError(f"unknown variant: {variant}")


def run_variant_trial(config: VariantCheckConfig, seed: int, variant: str) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    alpha, beta, mu_0, sigma2_0 = paper_priors(config)
    s2 = (
        np.array([0.25] * config.n_arms, dtype=float)
        if config.reward.lower() == "bernoulli"
        else np.array([config.s2] * config.n_arms, dtype=float)
    )
    theta = sample_theta(config, alpha, beta)
    n_metrics = 1 if variant in {"legacy_j1", "reduced_j1"} else 2
    model = make_metric_model(mu_0, sigma2_0, s2, config.horizon, n_metrics, config.secondary_obs_var)
    bandit = PaperBandit(theta, config)
    policy = make_policy(config, variant)
    if hasattr(policy, "reset"):
        policy.reset(seed)

    state = model.initial_state()
    allocations = []
    for _ in range(model.horizon):
        allocation = policy.allocate(state, model)
        allocation = model.validate_allocation(state, allocation)
        p = allocation.detach().cpu().numpy().astype(float)
        rewards, arm_draws = bandit.sample_arms(p)
        demeaned_rewards = rewards - arm_draws * bandit.base_reward
        aggregate_g = bandit.mean_scale * demeaned_rewards
        observation = torch.zeros_like(state.mean)
        observation[:, 0] = torch.as_tensor(aggregate_g, dtype=torch.float64)
        state = model.update(state, allocation, observation)
        allocations.append([float(x) for x in p.tolist()])

    selected_arm = state.selected_arm(model.target_idx)
    true_best_arm = int(np.argmax(theta))
    return {
        "seed": seed,
        "variant": variant,
        "selected_arm": selected_arm,
        "true_best_arm": true_best_arm,
        "simple_regret": float(np.max(theta) - theta[selected_arm]),
        "selected_true_best": bool(selected_arm == true_best_arm),
        "allocation_path": allocations,
    }


def summarize_trials(trials: list[dict]) -> dict:
    regrets = np.array([trial["simple_regret"] for trial in trials], dtype=float)
    selected_true_best = np.array([trial["selected_true_best"] for trial in trials], dtype=float)
    return {
        "n_runs": int(len(trials)),
        "mean_simple_regret": float(regrets.mean()),
        "se_simple_regret": float(regrets.std(ddof=0) / max(len(regrets), 1) ** 0.5),
        "selected_true_best_rate": float(selected_true_best.mean()),
    }


def compare_variants(config: VariantCheckConfig) -> dict:
    seeds = list(range(config.seed, config.seed + config.n_runs))
    by_variant = {}
    for variant in config.variants:
        trials = [run_variant_trial(config, seed, variant) for seed in seeds]
        by_variant[variant] = {"summary": summarize_trials(trials), "trials": trials}

    baseline = by_variant["legacy_j1"]["summary"]
    comparisons = {}
    for variant, payload in by_variant.items():
        summary = payload["summary"]
        comparisons[variant] = {
            "mean_regret_diff_vs_legacy_j1": summary["mean_simple_regret"] - baseline["mean_simple_regret"],
            "selected_true_best_rate_diff_vs_legacy_j1": summary["selected_true_best_rate"] - baseline["selected_true_best_rate"],
        }
    return {
        "config": asdict(config),
        "summaries": {variant: payload["summary"] for variant, payload in by_variant.items()},
        "comparisons": comparisons,
        "trials": {variant: payload["trials"] for variant, payload in by_variant.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerically compare scalar paper RHO variants under the metric framework.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_variant_config(args.config)
    payload = compare_variants(config)
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
