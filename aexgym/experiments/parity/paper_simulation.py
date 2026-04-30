from __future__ import annotations

import csv
import functools
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np
import torch

from aexgym.core import ExperimentInstance, ExperimentRunner, GaussianMetricModel, NoActiveSetRule, project_allocation
from aexgym.core.state import GaussianMetricState
from aexgym.experiments.parity.rho_variants import build_rho_policy
from aexgym.policies import (
    GaussianThompsonPolicy,
    GaussianTopTwoThompsonPolicy,
    MetricPolicy,
    MyopicLookaheadPolicy,
    UniformActivePolicy,
)


POLICY_LABELS: dict[str, str] = {
    "uniform": "Uniform",
    "ts": "TS",
    "ttts": "TTTS",
    "myopic": "Myopic",
    "reduced_constant": "RHO",
    "pathwise_constant": "Pathwise RHO",
    "pathwise_base_residual_regularized": "Base+Residual RHO",
    "pathwise_free_sequence_regularized": "Free-Sequence RHO",
    "oracle_beta_ts": "Beta TS",
    "oracle_beta_ttts": "Beta TTTS",
    "successive_elimination": "SE",
    "ts_plus": "TS+",
}


@dataclass(frozen=True)
class PaperProfile:
    name: str
    n_runs: int
    horizons: tuple[int, ...]
    rho_epochs: int
    rho_num_samples: int
    rho_lr: float
    ts_samples: int
    myopic_epochs: int
    max_asos_settings: Optional[int] = None


PAPER_PROFILES: dict[str, PaperProfile] = {
    "smoke": PaperProfile(
        name="smoke",
        n_runs=2,
        horizons=(1, 2),
        rho_epochs=2,
        rho_num_samples=8,
        rho_lr=0.08,
        ts_samples=64,
        myopic_epochs=2,
        max_asos_settings=2,
    ),
    "paper": PaperProfile(
        name="paper",
        n_runs=10000,
        horizons=tuple(range(1, 11)),
        rho_epochs=30,
        rho_num_samples=1000,
        rho_lr=1.0,
        ts_samples=1000,
        myopic_epochs=20,
        max_asos_settings=None,
    ),
}


@dataclass(frozen=True)
class PaperScenario:
    scenario_id: str
    figure_id: str
    reward_model: str
    n_arms: int
    horizon: int
    batch_size: float
    n_runs: int
    obs_var: float = 1.0
    prior_pattern: str = "flat"
    prior_var: float = 1.0
    mean_scale: Optional[float] = None
    variance_perturbation: float = 0.0
    planning_horizon: Optional[int] = None
    batch_schedule: str = "fixed"
    estimate_variance_first_batch: bool = False
    asos_setting_index: Optional[int] = None
    nonstationary: bool = False
    policies: tuple[str, ...] = ("uniform", "ts", "ttts", "myopic")
    metadata: dict = field(default_factory=dict)

    @property
    def effective_mean_scale(self) -> float:
        if self.mean_scale is not None:
            return float(self.mean_scale)
        if self.batch_size <= 0:
            return 1.0
        return 1.0 / math.sqrt(float(self.batch_size))


@dataclass
class PaperBanditInstance:
    true_objective: np.ndarray
    gaussian_theta: np.ndarray
    true_obs_var: np.ndarray
    believed_obs_var: np.ndarray
    prior_mean: np.ndarray
    prior_var: np.ndarray
    batch_sizes: np.ndarray
    reward_model: str
    base_reward: float = 0.0
    mean_scale: float = 1.0
    finite_batch_legacy_observation: bool = False
    bernoulli_probs: Optional[np.ndarray] = None
    gumbel_location: Optional[np.ndarray] = None
    asos_time_means: Optional[np.ndarray] = None
    asos_time_vars: Optional[np.ndarray] = None
    nonstationary: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def n_arms(self) -> int:
        return int(self.true_objective.shape[0])


@dataclass
class PaperRunRecord:
    figure_id: str
    scenario_id: str
    policy_name: str
    policy_label: str
    seed: int
    selected_arm: int
    true_best_arm: int
    selected_true_best: bool
    simple_regret: float
    allocation_path: list[list[float]]
    model_n_metrics: int
    target_idx: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PaperPolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> torch.Tensor: ...


class PlanningHorizonPolicyAdapter(MetricPolicy):
    """Allocate with a longer planning model while the runner updates the actual model."""

    def __init__(self, policy: MetricPolicy, planning_horizon: int, batch_size: float, name: Optional[str] = None) -> None:
        super().__init__(name=name or policy.name)
        self.policy = policy
        self.planning_horizon = int(planning_horizon)
        self.batch_size = float(batch_size)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        if hasattr(self.policy, "reset"):
            self.policy.reset(seed)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> torch.Tensor:
        horizon = max(state.t + 1, self.planning_horizon)
        batch_sizes = torch.full((horizon,), self.batch_size, dtype=model.batch_sizes.dtype, device=model.batch_sizes.device)
        planning_model = GaussianMetricModel(
            prior_mean=model.prior_mean,
            prior_cov=model.prior_cov,
            obs_cov=model.obs_cov,
            target_idx=model.target_idx,
            batch_sizes=batch_sizes,
            control_arm=model.control_arm,
            allocation_floor=model.allocation_floor,
        )
        return self.policy.allocate(state, planning_model)


class OracleBetaThompsonPolicy(MetricPolicy):
    def __init__(self, alpha0: np.ndarray, beta0: np.ndarray, top_two: bool = False, n_samples: int = 1000, name: Optional[str] = None) -> None:
        super().__init__(name=name or ("oracle_beta_ttts" if top_two else "oracle_beta_ts"))
        self.alpha0 = np.asarray(alpha0, dtype=float)
        self.beta0 = np.asarray(beta0, dtype=float)
        self.top_two = bool(top_two)
        self.n_samples = int(n_samples)
        self.rng = np.random.default_rng(0)
        self.alpha = self.alpha0.copy()
        self.beta = self.beta0.copy()

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.alpha = self.alpha0.copy()
        self.beta = self.beta0.copy()

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> torch.Tensor:
        samples = self.rng.beta(self.alpha, self.beta, size=(self.n_samples, self.alpha.shape[0]))
        winners = np.argmax(samples, axis=1)
        probs = np.bincount(winners, minlength=self.alpha.shape[0]).astype(float)
        probs = probs / max(float(probs.sum()), 1.0)
        if self.top_two and self.alpha.shape[0] > 1:
            probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
            odds = probs / (1.0 - probs)
            probs = probs * (0.5 + 0.5 * (odds.sum() - odds))
        return project_allocation(torch.as_tensor(probs, dtype=state.mean.dtype), state.active).to(state.mean.dtype)

    def observe(self, rewards: np.ndarray, draws: np.ndarray, **kwargs) -> None:
        self.alpha = self.alpha + rewards
        self.beta = self.beta + draws - rewards

    def selected_arm(self) -> int:
        return int(np.argmax(self.alpha / (self.alpha + self.beta)))


class SuccessiveEliminationPolicy(MetricPolicy):
    def __init__(self, n_arms: int, const: float = 1.0, delta: float = 0.01, name: str = "successive_elimination") -> None:
        super().__init__(name=name)
        self.n_arms = int(n_arms)
        self.const = float(const)
        self.delta = float(delta)
        self.active = np.ones(self.n_arms, dtype=bool)
        self.total_rewards = np.zeros(self.n_arms, dtype=float)
        self.total_draws = np.zeros(self.n_arms, dtype=float)

    def reset(self, seed: int) -> None:
        del seed
        self.active = np.ones(self.n_arms, dtype=bool)
        self.total_rewards = np.zeros(self.n_arms, dtype=float)
        self.total_draws = np.zeros(self.n_arms, dtype=float)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> torch.Tensor:
        del model
        active = torch.as_tensor(self.active, dtype=torch.bool, device=state.mean.device) & state.active
        return project_allocation(torch.ones(state.n_arms, dtype=state.mean.dtype, device=state.mean.device), active).to(state.mean.dtype)

    def observe(self, rewards: np.ndarray, draws: np.ndarray, **kwargs) -> None:
        self.total_rewards += rewards
        self.total_draws += draws
        means = np.divide(self.total_rewards, self.total_draws, out=np.zeros_like(self.total_rewards), where=self.total_draws > 0)
        ci = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            if self.total_draws[arm] > 0:
                ci[arm] = self.const * math.sqrt(math.log((self.total_draws[arm] ** 2) * (self.n_arms / self.delta)) / (2.0 * self.total_draws[arm]))
        lcb = np.where(self.total_draws > 0, means - ci, -np.inf)
        ucb = np.where(self.total_draws > 0, means + ci, np.inf)
        max_lcb = np.max(lcb[self.active]) if np.any(self.active) else -np.inf
        self.active = self.active & (ucb >= max_lcb)
        if not np.any(self.active):
            self.active[int(np.argmax(means))] = True

    def selected_arm(self) -> int:
        means = np.divide(self.total_rewards, self.total_draws, out=np.zeros_like(self.total_rewards), where=self.total_draws > 0)
        return int(np.argmax(means))


class TsPlusPolicy(MetricPolicy):
    """Small maintained approximation of the legacy TS+ appendix baseline."""

    def __init__(self, epochs: int = 10, lr: float = 0.04, n_rollouts: int = 64, temperature: float = 1.0, name: str = "ts_plus") -> None:
        super().__init__(name=name)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.n_rollouts = int(n_rollouts)
        self.temperature = float(temperature)

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> torch.Tensor:
        active_idx = state.active_indices()
        if active_idx.numel() <= 1:
            return project_allocation(torch.ones(state.n_arms, dtype=state.mean.dtype, device=state.mean.device), state.active).to(state.mean.dtype)
        logits = torch.zeros(active_idx.numel(), dtype=state.mean.dtype, device=state.mean.device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=self.lr)
        horizon = model.horizon - state.t
        z = torch.randn(horizon, self.n_rollouts, state.n_arms, dtype=state.mean.dtype, device=state.mean.device, generator=self.generator)
        selector = torch.nn.functional.one_hot(active_idx, num_classes=state.n_arms).to(dtype=state.mean.dtype, device=state.mean.device)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            action = torch.softmax(logits, dim=0) @ selector
            value = _rollout_ts_plus_value(state, model, action, z, temperature=self.temperature)
            (-value).backward()
            optimizer.step()
        action = torch.softmax(logits.detach(), dim=0) @ selector
        return project_allocation(action, state.active).to(state.mean.dtype)


def _rollout_ts_plus_value(state: GaussianMetricState, model: GaussianMetricModel, first_action: torch.Tensor, z: torch.Tensor, temperature: float) -> torch.Tensor:
    means = state.mean[:, model.target_idx].unsqueeze(0).repeat(z.shape[1], 1)
    vars_ = state.cov[:, model.target_idx, model.target_idx].unsqueeze(0).repeat(z.shape[1], 1)
    obs_var = model.obs_cov[:, model.target_idx, model.target_idx].unsqueeze(0)
    for h in range(z.shape[0]):
        if h == 0:
            allocation = first_action.unsqueeze(0).repeat(z.shape[1], 1)
        else:
            posterior = means + torch.sqrt(torch.clamp(vars_, min=1e-16)) * z[h]
            allocation = torch.softmax(posterior / temperature, dim=1)
        b_t = model.batch_sizes[state.t + h]
        post_vars = 1.0 / (1.0 / torch.clamp(vars_, min=1e-16) + b_t * allocation / torch.clamp(obs_var, min=1e-16))
        phi = torch.sqrt(torch.clamp(vars_ - post_vars, min=0.0))
        means = means + phi * z[h]
        vars_ = post_vars
    return torch.max(means, dim=1).values.mean()


def prior_alpha_beta(n_arms: int, batch_size: float, prior_pattern: str, reward_model: str) -> tuple[np.ndarray, np.ndarray]:
    mean_scale = 1.0 / math.sqrt(float(batch_size))
    perturb = 1.0 / mean_scale
    if reward_model == "bernoulli":
        alpha = np.full(n_arms, float(batch_size))
        beta = np.full(n_arms, float(batch_size))
    else:
        alpha = np.full(n_arms, float(batch_size))
        beta = np.full(n_arms, 1.0 / float(batch_size))
    if prior_pattern == "top_one":
        alpha[0] += perturb
    elif prior_pattern == "top_half":
        alpha[: max(1, n_arms // 2)] += perturb
    elif prior_pattern == "descending":
        alpha -= np.linspace(0.0, perturb * (n_arms - 1) / n_arms, n_arms)
        alpha = np.clip(alpha, 1e-6, None)
    elif prior_pattern != "flat":
        raise ValueError(f"unknown prior_pattern: {prior_pattern}")
    return alpha, beta


def prior_mean_from_hyperparameters(alpha: np.ndarray, beta: np.ndarray, mean_scale: float, reward_model: str) -> np.ndarray:
    if reward_model == "bernoulli":
        return (alpha / (alpha + beta) - 0.5) / mean_scale
    return (alpha * beta - 1.0) / mean_scale


def make_batch_sizes(scenario: PaperScenario, rng: np.random.Generator) -> np.ndarray:
    if scenario.batch_schedule == "fixed":
        return np.full(scenario.horizon, float(scenario.batch_size))
    if scenario.batch_schedule == "poisson":
        return np.maximum(1.0, rng.poisson(float(scenario.batch_size), size=scenario.horizon).astype(float))
    if scenario.batch_schedule == "decreasing":
        means = np.array([max(1.0, float(scenario.batch_size) * (scenario.horizon - t) / scenario.horizon) for t in range(scenario.horizon)])
        return np.maximum(1.0, rng.poisson(means).astype(float))
    raise ValueError(f"unknown batch_schedule: {scenario.batch_schedule}")


def make_instance(scenario: PaperScenario, seed: int, data_path: Optional[Path] = None) -> PaperBanditInstance:
    rng = np.random.default_rng(int(seed))
    mean_scale = scenario.effective_mean_scale
    batch_sizes = make_batch_sizes(scenario, rng)
    if scenario.reward_model in {"bernoulli", "bernoulli_gaussian"}:
        alpha, beta = prior_alpha_beta(scenario.n_arms, scenario.batch_size, scenario.prior_pattern, "bernoulli")
        probs = rng.beta(alpha, beta)
        gaussian_theta = (probs - 0.5) / mean_scale
        believed_obs_var = np.full(scenario.n_arms, 0.25)
        true_obs_var = believed_obs_var.copy()
        return PaperBanditInstance(
            true_objective=probs,
            gaussian_theta=gaussian_theta,
            true_obs_var=true_obs_var,
            believed_obs_var=believed_obs_var,
            prior_mean=prior_mean_from_hyperparameters(alpha, beta, mean_scale, "bernoulli"),
            prior_var=np.full(scenario.n_arms, 0.1),
            batch_sizes=batch_sizes if scenario.reward_model == "bernoulli_gaussian" else np.ones(scenario.horizon),
            reward_model=scenario.reward_model,
            base_reward=0.5,
            mean_scale=mean_scale,
            finite_batch_legacy_observation=scenario.reward_model == "bernoulli",
            bernoulli_probs=probs,
            metadata={"alpha": alpha.tolist(), "beta": beta.tolist()},
        )
    if scenario.reward_model in {"gumbel", "gumbel_gaussian"}:
        alpha, beta = prior_alpha_beta(scenario.n_arms, scenario.batch_size, scenario.prior_pattern, "gumbel")
        locations = rng.gamma(shape=alpha, scale=beta)
        gaussian_theta = (locations - 1.0) / mean_scale
        believed_obs_var = np.full(scenario.n_arms, scenario.obs_var)
        true_obs_var = believed_obs_var.copy()
        if scenario.variance_perturbation > 0:
            perturbation = rng.lognormal(mean=0.0, sigma=scenario.variance_perturbation, size=scenario.n_arms)
            true_obs_var = believed_obs_var * perturbation
        gumbel_beta = (math.sqrt(6.0) / math.pi) * np.sqrt(true_obs_var)
        return PaperBanditInstance(
            true_objective=locations,
            gaussian_theta=gaussian_theta,
            true_obs_var=true_obs_var,
            believed_obs_var=believed_obs_var,
            prior_mean=prior_mean_from_hyperparameters(alpha, beta, mean_scale, "gumbel"),
            prior_var=np.full(scenario.n_arms, scenario.prior_var),
            batch_sizes=batch_sizes if scenario.reward_model == "gumbel_gaussian" else np.ones(scenario.horizon),
            reward_model=scenario.reward_model,
            base_reward=float(1.0 + np.euler_gamma * np.mean(gumbel_beta)),
            mean_scale=mean_scale,
            finite_batch_legacy_observation=scenario.reward_model == "gumbel",
            gumbel_location=locations,
            metadata={"alpha": alpha.tolist(), "beta": beta.tolist(), "gumbel_beta": gumbel_beta.tolist()},
        )
    if scenario.reward_model == "gaussian":
        prior_mean = _patterned_gaussian_prior_mean(scenario)
        prior_var = np.full(scenario.n_arms, scenario.prior_var)
        theta = rng.normal(prior_mean, np.sqrt(prior_var))
        obs_var = np.full(scenario.n_arms, scenario.obs_var)
        return PaperBanditInstance(
            true_objective=theta,
            gaussian_theta=theta,
            true_obs_var=obs_var,
            believed_obs_var=obs_var.copy(),
            prior_mean=prior_mean,
            prior_var=prior_var,
            batch_sizes=batch_sizes,
            reward_model="gaussian",
        )
    if scenario.reward_model == "asos":
        setting = load_asos_settings(data_path=data_path, limit=None)[scenario.asos_setting_index or 0]
        return make_asos_instance(setting, scenario, rng, batch_sizes=batch_sizes)
    raise ValueError(f"unknown reward_model: {scenario.reward_model}")


def _patterned_gaussian_prior_mean(scenario: PaperScenario) -> np.ndarray:
    mean = np.zeros(scenario.n_arms, dtype=float)
    perturb = scenario.metadata.get("prior_mean_perturb", 0.1)
    if scenario.prior_pattern == "top_one":
        mean[0] += perturb
    elif scenario.prior_pattern == "top_half":
        mean[: max(1, scenario.n_arms // 2)] += perturb
    elif scenario.prior_pattern == "descending":
        mean += np.linspace(perturb, 0.0, scenario.n_arms)
    elif scenario.prior_pattern != "flat":
        raise ValueError(f"unknown prior_pattern: {scenario.prior_pattern}")
    return mean


def make_model(instance: PaperBanditInstance) -> GaussianMetricModel:
    prior_mean = torch.as_tensor(instance.prior_mean, dtype=torch.float64).reshape(instance.n_arms, 1)
    prior_cov = torch.as_tensor(instance.prior_var, dtype=torch.float64).reshape(instance.n_arms, 1, 1)
    obs_cov = torch.as_tensor(instance.believed_obs_var, dtype=torch.float64).reshape(instance.n_arms, 1, 1)
    batch_sizes = torch.as_tensor(instance.batch_sizes, dtype=torch.float64)
    return GaussianMetricModel(prior_mean=prior_mean, prior_cov=prior_cov, obs_cov=obs_cov, target_idx=0, batch_sizes=batch_sizes)


def sample_observation(
    instance: PaperBanditInstance,
    allocation: np.ndarray,
    t: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    allocation = np.asarray(allocation, dtype=float)
    allocation = allocation / allocation.sum()
    if instance.finite_batch_legacy_observation:
        draws = rng.multinomial(int(round(1.0 / (instance.mean_scale**2))), allocation)
        if instance.reward_model == "bernoulli":
            rewards = rng.binomial(draws, instance.bernoulli_probs)
            gaussian_obs = instance.mean_scale * (rewards - draws * instance.base_reward)
            return rewards.astype(float), draws.astype(float), gaussian_obs.astype(float), None
        if instance.reward_model == "gumbel":
            rewards = np.zeros(instance.n_arms, dtype=float)
            estimated_var = np.zeros(instance.n_arms, dtype=float)
            beta = np.asarray(instance.metadata["gumbel_beta"], dtype=float)
            for arm in range(instance.n_arms):
                if draws[arm] > 0:
                    samples = rng.gumbel(loc=instance.gumbel_location[arm], scale=beta[arm], size=int(draws[arm]))
                    rewards[arm] = float(np.sum(samples))
                    estimated_var[arm] = float(np.var(samples)) if draws[arm] > 1 else instance.true_obs_var[arm]
            gaussian_obs = instance.mean_scale * (rewards - draws * instance.base_reward)
            return rewards, draws.astype(float), gaussian_obs, estimated_var
    draws = np.maximum(0.0, instance.batch_sizes[t] * allocation)
    theta = instance.gaussian_theta.copy()
    obs_var = instance.true_obs_var.copy()
    if instance.reward_model == "asos" and instance.nonstationary and instance.asos_time_means is not None:
        idx = min(t, instance.asos_time_means.shape[0] - 1)
        theta = instance.asos_time_means[idx]
        obs_var = instance.asos_time_vars[idx]
    gaussian_obs = np.zeros(instance.n_arms, dtype=float)
    rewards = np.zeros(instance.n_arms, dtype=float)
    estimated_var = np.zeros(instance.n_arms, dtype=float)
    for arm in range(instance.n_arms):
        if allocation[arm] <= 1e-12:
            continue
        sd = math.sqrt(float(obs_var[arm]) / max(float(instance.batch_sizes[t] * allocation[arm]), 1e-12))
        gaussian_obs[arm] = rng.normal(theta[arm], sd)
        rewards[arm] = gaussian_obs[arm] * draws[arm]
        estimated_var[arm] = obs_var[arm]
    return rewards, draws, gaussian_obs, estimated_var


def run_paper_scenario(
    scenario: PaperScenario,
    policy_names: Iterable[str],
    rho_variants: Iterable[str],
    seed: int,
    profile: PaperProfile,
    data_path: Optional[Path] = None,
) -> list[PaperRunRecord]:
    records: list[PaperRunRecord] = []
    policies_to_run = list(policy_names) + list(rho_variants)
    for run_idx in range(scenario.n_runs):
        run_seed = int(seed + run_idx)
        instance = make_instance(scenario, run_seed, data_path=data_path)
        for policy_name in policies_to_run:
            policy = build_policy(policy_name, scenario, instance, profile)
            records.append(run_one_policy(scenario, instance, policy, run_seed))
    return records


def run_one_policy(scenario: PaperScenario, instance: PaperBanditInstance, policy: PaperPolicy, seed: int) -> PaperRunRecord:
    model = make_model(instance)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    rng = np.random.default_rng(int(seed))
    state = model.initial_state()
    allocation_path: list[list[float]] = []
    while state.t < scenario.horizon and state.t < model.horizon and not state.stopped:
        allocation_tensor = policy.allocate(state, model)
        allocation_tensor = model.validate_allocation(state, allocation_tensor)
        allocation = allocation_tensor.detach().cpu().numpy()
        allocation_path.append([float(x) for x in allocation.tolist()])
        rewards, draws, gaussian_obs, estimated_var = sample_observation(instance, allocation, state.t, rng)
        if hasattr(policy, "observe"):
            policy.observe(rewards=rewards, draws=draws, gaussian_observation=gaussian_obs)
        if scenario.estimate_variance_first_batch and state.t == 0 and estimated_var is not None:
            _replace_model_obs_var(model, estimated_var)
        observation = torch.as_tensor(gaussian_obs, dtype=torch.float64).reshape(instance.n_arms, 1)
        state = model.update(state, allocation_tensor, observation)

    if hasattr(policy, "selected_arm"):
        selected_arm = int(policy.selected_arm())
    else:
        selected_arm = state.selected_arm(model.target_idx)
    true_best_arm = int(np.argmax(instance.true_objective))
    simple_regret = float(np.max(instance.true_objective) - instance.true_objective[selected_arm])
    metadata = {
        **scenario.metadata,
        **instance.metadata,
        "reward_model": scenario.reward_model,
        "batch_size": scenario.batch_size,
        "horizon": scenario.horizon,
        "n_arms": scenario.n_arms,
        "obs_var": scenario.obs_var,
        "prior_pattern": scenario.prior_pattern,
    }
    return PaperRunRecord(
        figure_id=scenario.figure_id,
        scenario_id=scenario.scenario_id,
        policy_name=policy.name,
        policy_label=POLICY_LABELS.get(policy.name, policy.name),
        seed=int(seed),
        selected_arm=selected_arm,
        true_best_arm=true_best_arm,
        selected_true_best=selected_arm == true_best_arm,
        simple_regret=simple_regret,
        allocation_path=allocation_path,
        model_n_metrics=model.n_metrics,
        target_idx=model.target_idx,
        metadata=metadata,
    )


def _replace_model_obs_var(model: GaussianMetricModel, estimated_var: np.ndarray) -> None:
    clean = np.where(np.isfinite(estimated_var) & (estimated_var > 1e-12), estimated_var, np.nan)
    if np.all(np.isnan(clean)):
        return
    fill = np.nanmean(clean)
    clean = np.where(np.isnan(clean), fill, clean)
    obs_cov = torch.as_tensor(clean, dtype=model.prior_mean.dtype, device=model.prior_mean.device).reshape(model.n_arms, 1, 1)
    model.obs_cov = obs_cov
    model._obs_precision = torch.linalg.inv(obs_cov)


def build_policy(policy_name: str, scenario: PaperScenario, instance: PaperBanditInstance, profile: PaperProfile) -> PaperPolicy:
    if policy_name == "uniform":
        return UniformActivePolicy(name="uniform")
    if policy_name == "ts":
        return GaussianThompsonPolicy(n_samples=profile.ts_samples, name="ts")
    if policy_name == "ttts":
        return GaussianTopTwoThompsonPolicy(n_samples=profile.ts_samples, coin=0.5, name="ttts")
    if policy_name == "myopic":
        return MyopicLookaheadPolicy(epochs=profile.myopic_epochs, lr=0.08, num_zs=max(16, profile.ts_samples // 4), name="myopic")
    if policy_name == "oracle_beta_ts":
        alpha, beta = prior_alpha_beta(scenario.n_arms, scenario.batch_size, scenario.prior_pattern, "bernoulli")
        return OracleBetaThompsonPolicy(alpha, beta, top_two=False, n_samples=profile.ts_samples, name="oracle_beta_ts")
    if policy_name == "oracle_beta_ttts":
        alpha, beta = prior_alpha_beta(scenario.n_arms, scenario.batch_size, scenario.prior_pattern, "bernoulli")
        return OracleBetaThompsonPolicy(alpha, beta, top_two=True, n_samples=profile.ts_samples, name="oracle_beta_ttts")
    if policy_name == "successive_elimination":
        return SuccessiveEliminationPolicy(scenario.n_arms, const=math.sqrt(max(scenario.obs_var, 0.25)), delta=0.01)
    if policy_name == "ts_plus":
        return TsPlusPolicy(epochs=max(2, profile.myopic_epochs), n_rollouts=max(16, profile.ts_samples // 8), name="ts_plus")
    if policy_name in {
        "reduced_constant",
        "pathwise_constant",
        "pathwise_base_residual_regularized",
        "pathwise_free_sequence_regularized",
    }:
        policy = build_rho_policy(
            policy_name,
            target_idx=0,
            epochs=profile.rho_epochs,
            num_samples=profile.rho_num_samples,
            lr=profile.rho_lr,
            temporal_regularization=1e5,
            sample_method="sobol",
        )
        if scenario.planning_horizon is not None and scenario.planning_horizon != scenario.horizon:
            return PlanningHorizonPolicyAdapter(policy, planning_horizon=scenario.planning_horizon, batch_size=scenario.batch_size, name=policy_name)
        return policy
    raise ValueError(f"unknown paper policy: {policy_name}")


@dataclass(frozen=True)
class AsosSetting:
    key: tuple[str, str, str]
    time_since_start: tuple[float, ...]
    mean_c: tuple[float, ...]
    mean_t: tuple[float, ...]
    variance_c: tuple[float, ...]
    variance_t: tuple[float, ...]


def default_asos_data_path() -> Path:
    package_root = Path(__file__).resolve().parents[3]
    candidates = [
        package_root / "data" / "asos_digital_experiments_dataset.csv",
        package_root.parent / "data" / "asos_digital_experiments_dataset.csv",
        Path.cwd() / "data" / "asos_digital_experiments_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


@functools.lru_cache(maxsize=16)
def load_asos_settings(data_path: Optional[Path] = None, limit: Optional[int] = None) -> list[AsosSetting]:
    path = default_asos_data_path() if data_path is None else Path(data_path)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["experiment_id"], row["variant_id"], row["metric_id"])
            grouped.setdefault(key, []).append(row)
    settings = []
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: float(row["time_since_start"]))
        settings.append(
            AsosSetting(
                key=key,
                time_since_start=tuple(_safe_float(row["time_since_start"], 0.0) for row in rows),
                mean_c=tuple(_safe_float(row["mean_c"], 0.0) for row in rows),
                mean_t=tuple(_safe_float(row["mean_t"], 0.0) for row in rows),
                variance_c=tuple(max(_safe_float(row["variance_c"], 1.0), 1e-8) for row in rows),
                variance_t=tuple(max(_safe_float(row["variance_t"], 1.0), 1e-8) for row in rows),
            )
        )
    return settings if limit is None else settings[:limit]


def _safe_float(value: str, default: float) -> float:
    try:
        if value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def make_asos_instance(
    setting: AsosSetting,
    scenario: PaperScenario,
    rng: np.random.Generator,
    batch_sizes: np.ndarray,
) -> PaperBanditInstance:
    z_arms = np.array([0.0, 1.0] + [rng.normal(0.0, 1.0) for _ in range(scenario.n_arms - 2)])
    mean_c = np.asarray(setting.mean_c, dtype=float)
    mean_t = np.asarray(setting.mean_t, dtype=float)
    var_c = np.asarray(setting.variance_c, dtype=float)
    var_t = np.asarray(setting.variance_t, dtype=float)
    effect = mean_t - mean_c
    time_means = np.zeros((len(mean_c), scenario.n_arms), dtype=float)
    time_vars = np.zeros((len(mean_c), scenario.n_arms), dtype=float)
    for arm in range(scenario.n_arms):
        if arm == 0:
            time_means[:, arm] = mean_c
            time_vars[:, arm] = var_c
        elif arm == 1:
            time_means[:, arm] = mean_t
            time_vars[:, arm] = var_t
        else:
            time_means[:, arm] = mean_c + effect * z_arms[arm]
            time_vars[:, arm] = var_t
    if scenario.metadata.get("subtract", False):
        centered = time_means.copy()
        for idx in range(1, centered.shape[0]):
            centered[idx] = centered[idx] - np.mean(time_means[idx - 1])
        time_means = centered
    if scenario.metadata.get("demean", False):
        time_means = time_means - np.mean(time_means, axis=1, keepdims=True)
    horizon_means = time_means[: max(1, min(scenario.horizon, time_means.shape[0]))]
    true_objective = np.mean(horizon_means, axis=0)
    obs_var = np.mean(time_vars[: max(1, min(scenario.horizon, time_vars.shape[0]))], axis=0)
    return PaperBanditInstance(
        true_objective=true_objective,
        gaussian_theta=true_objective.copy(),
        true_obs_var=obs_var,
        believed_obs_var=obs_var.copy(),
        prior_mean=np.zeros(scenario.n_arms, dtype=float),
        prior_var=np.ones(scenario.n_arms, dtype=float),
        batch_sizes=batch_sizes,
        reward_model="asos",
        asos_time_means=time_means,
        asos_time_vars=time_vars,
        metadata={"asos_key": list(setting.key), "z_arms": z_arms.tolist()},
        nonstationary=scenario.nonstationary,
    )


def aggregate_records(records: list[PaperRunRecord]) -> dict[str, dict]:
    grouped: dict[tuple[str, str], list[PaperRunRecord]] = {}
    for record in records:
        grouped.setdefault((record.scenario_id, record.policy_name), []).append(record)
    aggregates: dict[str, dict] = {}
    scenario_uniform_regret: dict[str, float] = {}
    for (scenario_id, policy_name), group in grouped.items():
        regrets = np.asarray([record.simple_regret for record in group], dtype=float)
        selected_true_best = np.asarray([record.selected_true_best for record in group], dtype=float)
        entry = {
            "figure_id": group[0].figure_id,
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "policy_label": group[0].policy_label,
            "n_runs": len(group),
            "mean_simple_regret": float(np.mean(regrets)),
            "se_simple_regret": float(np.std(regrets, ddof=0) / math.sqrt(max(len(regrets), 1))),
            "selected_true_best_rate": float(np.mean(selected_true_best)),
            "model_n_metrics": group[0].model_n_metrics,
            "target_idx": group[0].target_idx,
            "metadata": group[0].metadata,
        }
        aggregates[f"{scenario_id}::{policy_name}"] = entry
        if policy_name == "uniform":
            scenario_uniform_regret[scenario_id] = entry["mean_simple_regret"]
    for entry in aggregates.values():
        uniform_regret = scenario_uniform_regret.get(entry["scenario_id"])
        if uniform_regret is not None and uniform_regret > 0:
            entry["simple_regret_percent_of_uniform"] = 100.0 * entry["mean_simple_regret"] / uniform_regret
        else:
            entry["simple_regret_percent_of_uniform"] = None
    return aggregates


def run_exact_gaussian_with_core_runner(scenario: PaperScenario, policy_names: Iterable[str], seed: int, profile: PaperProfile) -> list[PaperRunRecord]:
    """Small utility for tests that need to verify direct core-runner compatibility."""

    instance = make_instance(scenario, seed)
    model = make_model(instance)
    runner = ExperimentRunner(model, NoActiveSetRule(target_idx=0))
    results = []
    for policy_name in policy_names:
        policy = build_policy(policy_name, scenario, instance, profile)
        core_result = runner.run_one(ExperimentInstance(torch.as_tensor(instance.gaussian_theta, dtype=torch.float64).reshape(instance.n_arms, 1)), policy, seed)
        results.append(
            PaperRunRecord(
                figure_id=scenario.figure_id,
                scenario_id=scenario.scenario_id,
                policy_name=policy.name,
                policy_label=POLICY_LABELS.get(policy.name, policy.name),
                seed=seed,
                selected_arm=core_result.selected_arm,
                true_best_arm=core_result.true_best_arm,
                selected_true_best=core_result.selected_arm == core_result.true_best_arm,
                simple_regret=core_result.simple_regret,
                allocation_path=core_result.allocation_path,
                model_n_metrics=model.n_metrics,
                target_idx=model.target_idx,
                metadata=scenario.metadata,
            )
        )
    return results
