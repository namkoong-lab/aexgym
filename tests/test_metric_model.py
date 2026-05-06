import torch

from aexgym.core import GaussianMetricModel, project_allocation


def test_j1_update_matches_scalar_recursion():
    prior_mean = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    prior_cov = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float64)
    obs_cov = torch.tensor([[[5.0]], [[7.0]]], dtype=torch.float64)
    model = GaussianMetricModel(prior_mean, prior_cov, obs_cov, target_metric_idx=0, batch_sizes=[4.0])
    state = model.initial_state()
    allocation = torch.tensor([0.25, 0.75], dtype=torch.float64)
    observation = torch.tensor([[1.5], [2.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)
    updated_mean = model.posterior_mean(updated)
    updated_cov = model.posterior_cov(updated)

    expected_var = []
    expected_mean = []
    for arm_idx in range(2):
        sigma2 = prior_cov[arm_idx, 0, 0]
        s2 = obs_cov[arm_idx, 0, 0]
        p = allocation[arm_idx]
        b = torch.tensor(4.0, dtype=torch.float64)
        post_var = s2 * sigma2 / (s2 + b * p * sigma2)
        post_mean = post_var * (prior_mean[arm_idx, 0] / sigma2 + b * p * observation[arm_idx, 0] / s2)
        expected_var.append(post_var)
        expected_mean.append(post_mean)

    assert torch.allclose(updated_mean[:, 0], torch.stack(expected_mean))
    assert torch.allclose(updated_cov[:, 0, 0], torch.stack(expected_var))
    assert torch.allclose(updated.n_eff, torch.tensor([1.0, 3.0], dtype=torch.float64))
    assert torch.allclose(updated.sum_g[:, 0], torch.tensor([1.5, 6.0], dtype=torch.float64))


def test_j2_update_matches_precision_formula():
    prior_mean = torch.tensor([[0.2, -0.1], [0.0, 0.0]], dtype=torch.float64)
    prior_cov = torch.tensor(
        [
            [[2.0, 0.3], [0.3, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    obs_cov = torch.tensor(
        [
            [[1.5, 0.2], [0.2, 0.8]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    model = GaussianMetricModel(prior_mean, prior_cov, obs_cov, target_metric_idx=0, batch_sizes=[3.0])
    state = model.initial_state()
    allocation = torch.tensor([0.4, 0.6], dtype=torch.float64)
    observation = torch.tensor([[1.0, -0.5], [0.0, 0.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)
    updated_mean = model.posterior_mean(updated)
    updated_cov = model.posterior_cov(updated)

    prior_precision = torch.linalg.inv(prior_cov[0])
    obs_precision = torch.linalg.inv(obs_cov[0])
    post_precision = prior_precision + 3.0 * 0.4 * obs_precision
    post_cov = torch.linalg.inv(post_precision)
    post_mean = post_cov @ (prior_precision @ prior_mean[0] + 3.0 * 0.4 * obs_precision @ observation[0])

    assert torch.allclose(updated_cov[0], post_cov)
    assert torch.allclose(updated_mean[0], post_mean)


def test_zero_allocation_and_inactive_arms_are_frozen():
    prior_mean = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    prior_cov = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float64)
    model = GaussianMetricModel(prior_mean, prior_cov, torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float64), target_metric_idx=0, batch_sizes=[1.0])
    state = model.initial_state(active=torch.tensor([True, True]))
    allocation = torch.tensor([0.0, 1.0], dtype=torch.float64)
    observation = torch.tensor([[5.0], [7.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)

    assert torch.allclose(updated.n_eff[0], state.n_eff[0])
    assert torch.allclose(updated.sum_g[0], state.sum_g[0])
    assert torch.allclose(model.posterior_mean(updated)[0], model.posterior_mean(state)[0])
    assert torch.allclose(model.posterior_cov(updated)[0], model.posterior_cov(state)[0])


def test_near_zero_allocation_is_stable():
    prior_mean = torch.zeros(2, 1, dtype=torch.float64)
    prior_cov = torch.eye(1, dtype=torch.float64).repeat(2, 1, 1)
    model = GaussianMetricModel(prior_mean, prior_cov, prior_cov.clone(), target_metric_idx=0, batch_sizes=[1.0])
    state = model.initial_state()
    allocation = torch.tensor([1e-20, 1.0], dtype=torch.float64)
    observation = torch.tensor([[100.0], [1.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)
    updated_mean = model.posterior_mean(updated)
    updated_cov = model.posterior_cov(updated)

    assert torch.isfinite(updated_mean).all()
    assert torch.isfinite(updated_cov).all()
    assert torch.allclose(updated.n_eff[0], state.n_eff[0])
    assert torch.allclose(updated.sum_g[0], state.sum_g[0])


def test_projection_preserves_fractional_active_weights():
    raw = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    active = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)

    projected = project_allocation(raw, active)

    assert torch.allclose(projected, torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=torch.float64))
