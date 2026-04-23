from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.stats
import torch
from torch import Tensor
from torch import nn, optim

from aexgym.metric import GaussianMetricModel, GaussianMetricState


@dataclass
class PaperScalarConfig:
    reward: str = "bernoulli"
    n_arms: int = 10
    horizon: int = 10
    mean_scale: float = 0.1
    s2: float = 0.25
    prior: str = "Flat"
    n_runs: int = 100
    seed: int = 1
    policies: list[str] = field(default_factory=lambda: ["uniform", "ts", "ttts", "myopic", "rho"])
    n_samples: int = 1000
    rho_num_zs: int = 1000
    rho_epochs: int = 20
    rho_lr: float = 1.0
    rho_eps: float = 1e-3
    legacy_shared_rng_order: bool = True
    output: str | None = None


def load_config(path: str | None) -> PaperScalarConfig:
    if path is None:
        return PaperScalarConfig()
    return PaperScalarConfig(**json.loads(Path(path).read_text()))


def one_hot_argmax(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    out[int(np.argmax(values))] = 1.0
    return out


def paper_priors(config: PaperScalarConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = (1.0 / config.mean_scale) ** 2
    perturb_scale = 1.0 / config.mean_scale
    if config.reward.lower() == "bernoulli":
        alpha = np.array([batch_size] * config.n_arms, dtype=float)
        beta = np.array([batch_size] * config.n_arms, dtype=float)
    elif config.reward.lower() == "gumbel":
        alpha = np.array([batch_size] * config.n_arms, dtype=float)
        beta = np.array([1.0 / batch_size] * config.n_arms, dtype=float)
    else:
        raise ValueError("reward must be 'bernoulli' or 'gumbel'")

    if config.prior == "Flat":
        pass
    elif config.prior == "Top One":
        alpha = alpha + np.array([perturb_scale] + [0.0] * (config.n_arms - 1))
    elif config.prior == "Top Half":
        alpha = alpha + np.array([perturb_scale] * (config.n_arms - config.n_arms // 2) + [0.0] * (config.n_arms // 2))
    elif config.prior == "Descending":
        alpha = alpha - np.array([perturb_scale / config.n_arms * i for i in range(config.n_arms)])
    else:
        raise ValueError("prior must be one of Flat, Top One, Top Half, Descending")

    if config.reward.lower() == "bernoulli":
        mu_0 = (alpha / (alpha + beta) - 0.5) * (1.0 / config.mean_scale)
        sigma2_0 = np.array([0.1] * config.n_arms, dtype=float)
    else:
        mu_0 = (alpha * beta - 1.0) * (1.0 / config.mean_scale)
        sigma2_0 = np.array([1.0] * config.n_arms, dtype=float)
    return alpha, beta, mu_0, sigma2_0


def make_model(mu_0: np.ndarray, sigma2_0: np.ndarray, s2: np.ndarray, horizon: int) -> GaussianMetricModel:
    prior_mean = torch.as_tensor(mu_0, dtype=torch.float64).reshape(-1, 1)
    prior_cov = torch.as_tensor(sigma2_0, dtype=torch.float64).reshape(-1, 1, 1)
    obs_cov = torch.as_tensor(s2, dtype=torch.float64).reshape(-1, 1, 1)
    return GaussianMetricModel(prior_mean, prior_cov, obs_cov, target_idx=0, batch_sizes=torch.ones(horizon, dtype=torch.float64))


class PaperBandit:
    def __init__(self, theta: np.ndarray, config: PaperScalarConfig) -> None:
        self.theta = theta
        self.k = len(theta)
        self.mean_scale = config.mean_scale
        self.batch_size = int((1.0 / config.mean_scale) ** 2)
        self.reward = config.reward.lower()
        if self.reward == "bernoulli":
            self.base_reward = 0.5
            self.gumbel_beta = None
        else:
            self.gumbel_beta = (np.sqrt(6.0) / np.pi) * np.sqrt(config.s2)
            self.base_reward = 1.0 + np.euler_gamma * self.gumbel_beta

    def sample_arms(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arm_draws = np.random.multinomial(self.batch_size, p)
        rewards = np.zeros(self.k)
        for arm in range(self.k):
            if self.reward == "bernoulli":
                rewards[arm] = np.random.binomial(1, self.theta[arm], size=arm_draws[arm]).sum()
            else:
                rewards[arm] = np.random.gumbel(self.theta[arm], self.gumbel_beta, size=arm_draws[arm]).sum()
        return rewards, arm_draws


class LegacyPolicy:
    def __init__(self, name: str) -> None:
        self.name = name

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        raise NotImplementedError


class LegacyUniform(LegacyPolicy):
    def __init__(self) -> None:
        super().__init__("uniform")

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        return torch.ones(state.n_arms, dtype=torch.float64) / state.n_arms


class LegacyTS(LegacyPolicy):
    def __init__(self, n_samples: int = 1000) -> None:
        super().__init__("ts")
        self.n_samples = n_samples

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        p = sample_posterior(_state_mu(state), _state_sigma2(state), self.n_samples)
        p = np.clip(p, 1e-10, None)
        p = p / p.sum()
        return torch.as_tensor(p, dtype=torch.float64)


class LegacyTTTS(LegacyPolicy):
    def __init__(self, n_samples: int = 1000, coin_p: float = 0.5, max_redraw: int = 10000) -> None:
        super().__init__("ttts")
        self.n_samples = n_samples
        self.coin_p = coin_p
        self.max_redraw = max_redraw

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        k = state.n_arms
        p = None
        found_other = False
        counter = 1
        while (not found_other) and counter <= self.max_redraw:
            p = sample_posterior(_state_mu(state), _state_sigma2(state), self.n_samples)
            if sum(1 for p_arm in p if p_arm > 0) > 1:
                found_other = True
            counter += 1
        if not found_other:
            p = p + np.array([0.01] * k)
            p = p / p.sum()
        p_cond = p / (1.0 - p)
        p_cond_list = list(p_cond)
        sum_other = [sum(p_cond_list[:i] + p_cond_list[i + 1 :]) for i in range(k)]
        top_two = p * (self.coin_p + (1.0 - self.coin_p) * np.array(sum_other))
        top_two = top_two / top_two.sum()
        return torch.as_tensor(top_two, dtype=torch.float64)


class LegacyMyopic(LegacyPolicy):
    def __init__(self, eps: float = 1e-3, n_max: int = 20, lr: float = 1.0, num_zs: int = 1000) -> None:
        super().__init__("myopic")
        self.eps = eps
        self.n_max = n_max
        self.lr = lr
        self.num_zs = num_zs

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        p = solve_legacy_state(
            mu=_state_mu(state),
            sigma2=_state_sigma2(state),
            s2=_model_s2(model),
            scale=1.0,
            eps=self.eps,
            n_max=self.n_max,
            lr=self.lr,
            num_zs=self.num_zs,
        )
        return torch.as_tensor(p / p.sum(), dtype=torch.float64)


class LegacyRho(LegacyPolicy):
    def __init__(self, eps: float = 1e-3, n_max: int = 20, lr: float = 1.0, num_zs: int = 1000, boost: float = 1.0) -> None:
        super().__init__("rho")
        self.eps = eps
        self.n_max = n_max
        self.lr = lr
        self.num_zs = num_zs
        self.boost = boost

    def allocate(self, state: GaussianMetricState, model: GaussianMetricModel) -> Tensor:
        residual = model.horizon - state.t
        s2_t = _model_s2(model) / (self.boost * residual)
        p = solve_legacy_state(
            mu=_state_mu(state),
            sigma2=_state_sigma2(state),
            s2=s2_t,
            scale=float(residual),
            eps=self.eps,
            n_max=self.n_max,
            lr=self.lr,
            num_zs=self.num_zs,
        )
        return torch.as_tensor(p / p.sum(), dtype=torch.float64)


def _state_mu(state: GaussianMetricState) -> np.ndarray:
    return state.mean[:, 0].detach().cpu().numpy()


def _state_sigma2(state: GaussianMetricState) -> np.ndarray:
    return state.cov[:, 0, 0].detach().cpu().numpy()


def _model_s2(model: GaussianMetricModel) -> np.ndarray:
    return model.obs_cov[:, 0, 0].detach().cpu().numpy()


def sample_posterior(mu: np.ndarray, sigma2: np.ndarray, n_samples: int) -> np.ndarray:
    samples = np.random.normal(mu, np.sqrt(sigma2), (n_samples, len(mu)))
    samples_arg_max = np.apply_along_axis(one_hot_argmax, 1, samples)
    return np.mean(samples_arg_max, axis=0)


def solve_legacy_state(mu: np.ndarray, sigma2: np.ndarray, s2: np.ndarray, scale: float, eps: float, n_max: int, lr: float, num_zs: int) -> np.ndarray:
    k = len(mu)
    sobol = torch.quasirandom.SobolEngine(k, scramble=True, seed=42)
    u = sobol.draw(num_zs)
    z = torch.tensor(scipy.stats.norm.ppf(u)).float()
    state = torch.tensor(np.vstack((mu, sigma2))).unsqueeze(0)
    s2_tensor = torch.tensor(s2).float()
    pnet = _LegacyPNet(k)
    optimizer = optim.Adam(pnet.parameters(), lr=lr)
    for _ in range(n_max):
        optimizer.zero_grad()
        loss = -legacy_q_value(pnet(state), state, s2_tensor, z)
        scaled_loss = scale * loss
        scaled_loss.backward()
        optimizer.step()
        grad_norm = torch.tensor(0.0)
        for param in pnet.parameters():
            grad_norm = grad_norm + param.grad.detach().data.norm(2)
        if grad_norm < eps:
            break
    return pnet(state)[0].detach().numpy()


class _LegacyPNet(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, k, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        del x
        fake_input = torch.tensor([[1.0]])
        return self.softmax(self.fc1(fake_input))


def legacy_q_value(p: Tensor, state: Tensor, s2: Tensor, z: Tensor) -> Tensor:
    mu = state[:, 0]
    sigma2 = state[:, 1]
    s2_p = torch.divide(s2, torch.clamp(p, min=1e-10))
    batch_size = mu.size()[0]
    s2_batch = s2_p.expand((batch_size, s2.size()[0]))
    new_sigma2 = (sigma2 * s2_batch) / (sigma2 + s2_batch)
    phi = torch.sqrt(torch.clamp(sigma2 - new_sigma2, min=1e-10))
    phi_nz = phi.unsqueeze(1).expand(batch_size, z.size()[0], s2.size()[0])
    z_batch = z.expand((batch_size, z.size()[0], s2.size()[0]))
    mu_nz = mu.unsqueeze(1).expand(batch_size, z.size()[0], s2.size()[0])
    new_mu_nz = torch.flatten(mu_nz + phi_nz * z_batch, 0, 1)
    return torch.max(new_mu_nz, dim=1).values.mean()


def make_policies(config: PaperScalarConfig) -> list[LegacyPolicy]:
    out: list[LegacyPolicy] = []
    for policy in config.policies:
        if policy == "uniform":
            out.append(LegacyUniform())
        elif policy == "ts":
            out.append(LegacyTS(n_samples=config.n_samples))
        elif policy in {"ttts", "top_two_ts"}:
            out.append(LegacyTTTS(n_samples=config.n_samples))
        elif policy in {"myopic", "kg"}:
            out.append(LegacyMyopic(eps=config.rho_eps, n_max=config.rho_epochs, lr=config.rho_lr, num_zs=config.rho_num_zs))
        elif policy == "rho":
            out.append(LegacyRho(eps=config.rho_eps, n_max=config.rho_epochs, lr=config.rho_lr, num_zs=config.rho_num_zs))
        else:
            raise ValueError(f"unknown policy: {policy}")
    return out


def run_policy_path(model: GaussianMetricModel, bandit: PaperBandit, policy: LegacyPolicy) -> tuple[int, list[list[float]]]:
    state = model.initial_state()
    allocations: list[list[float]] = []
    for _ in range(model.horizon):
        p = policy.allocate(state, model).detach().cpu().numpy().astype(float)
        p = p / p.sum()
        allocations.append([float(x) for x in p.tolist()])
        rewards, arm_draws = bandit.sample_arms(p)
        demeaned_rewards = rewards - arm_draws * bandit.base_reward
        aggregate_g = bandit.mean_scale * demeaned_rewards
        observation_y = aggregate_g
        observation = torch.as_tensor(observation_y, dtype=torch.float64).reshape(model.n_arms, 1)
        state = model.update(state, torch.as_tensor(p, dtype=torch.float64), observation)
    return state.selected_arm(model.target_idx), allocations


def sample_theta(config: PaperScalarConfig, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if config.reward.lower() == "bernoulli":
        return np.random.beta(alpha, beta)
    return np.random.gamma(alpha, beta)


def run_trial(config: PaperScalarConfig, seed: int, policies: list[LegacyPolicy]) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    alpha, beta, mu_0, sigma2_0 = paper_priors(config)
    s2 = np.array([0.25] * config.n_arms, dtype=float) if config.reward.lower() == "bernoulli" else np.array([config.s2] * config.n_arms, dtype=float)
    theta = sample_theta(config, alpha, beta)
    model = make_model(mu_0, sigma2_0, s2, config.horizon)
    bandit = PaperBandit(theta, config)
    trial = {
        "seed": seed,
        "theta": theta.tolist(),
        "true_best_arm": int(np.argmax(theta)),
        "policies": {},
    }
    for policy in policies:
        if not config.legacy_shared_rng_order:
            np.random.seed(seed)
            torch.manual_seed(seed)
            bandit = PaperBandit(theta, config)
        selected_arm, allocations = run_policy_path(model, bandit, policy)
        regret = float(np.max(theta) - theta[selected_arm])
        trial["policies"][policy.name] = {
            "selected_arm": selected_arm,
            "simple_regret": regret,
            "correct": float(selected_arm == int(np.argmax(theta))),
            "allocation_path": allocations,
        }
    return trial


def aggregate_trials(trials: list[dict], policies: list[LegacyPolicy]) -> dict:
    out = {}
    for policy in policies:
        regrets = np.array([trial["policies"][policy.name]["simple_regret"] for trial in trials], dtype=float)
        correct = np.array([trial["policies"][policy.name]["correct"] for trial in trials], dtype=float)
        out[policy.name] = {
            "avg_regret": float(np.mean(regrets)),
            "std_regret": float(np.std(regrets)),
            "90q_regret": float(np.quantile(regrets, 0.9)),
            "75q_regret": float(np.quantile(regrets, 0.75)),
            "avg_correct": float(np.mean(correct)),
        }
    return out


def run_config(config: PaperScalarConfig) -> dict:
    policies = make_policies(config)
    seeds = [config.seed + i for i in range(config.n_runs)]
    trials = [run_trial(config, seed, policies) for seed in seeds]
    return {
        "config": asdict(config),
        "aggregates": aggregate_trials(trials, policies),
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate old scalar paper experiments under the Gaussian metric framework.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output is not None:
        config.output = args.output
    payload = run_config(config)
    text = json.dumps(payload, indent=2)
    if config.output:
        Path(config.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
