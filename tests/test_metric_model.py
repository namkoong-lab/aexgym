import torch

from aexgym.core import GaussianMetricModel


def test_j1_update_matches_scalar_recursion():
    prior_mean = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    prior_cov = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float64)
    obs_cov = torch.tensor([[[5.0]], [[7.0]]], dtype=torch.float64)
    model = GaussianMetricModel(prior_mean, prior_cov, obs_cov, target_idx=0, batch_sizes=[4.0])
    state = model.initial_state()
    allocation = torch.tensor([0.25, 0.75], dtype=torch.float64)
    observation = torch.tensor([[1.5], [2.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)

    expected_var = []
    expected_mean = []
    for arm in range(2):
        sigma2 = prior_cov[arm, 0, 0]
        s2 = obs_cov[arm, 0, 0]
        p = allocation[arm]
        b = torch.tensor(4.0, dtype=torch.float64)
        post_var = s2 * sigma2 / (s2 + b * p * sigma2)
        post_mean = post_var * (prior_mean[arm, 0] / sigma2 + b * p * observation[arm, 0] / s2)
        expected_var.append(post_var)
        expected_mean.append(post_mean)

    assert torch.allclose(updated.mean[:, 0], torch.stack(expected_mean))
    assert torch.allclose(updated.cov[:, 0, 0], torch.stack(expected_var))


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
    model = GaussianMetricModel(prior_mean, prior_cov, obs_cov, target_idx=0, batch_sizes=[3.0])
    state = model.initial_state()
    allocation = torch.tensor([0.4, 0.6], dtype=torch.float64)
    observation = torch.tensor([[1.0, -0.5], [0.0, 0.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)

    prior_precision = torch.linalg.inv(prior_cov[0])
    obs_precision = torch.linalg.inv(obs_cov[0])
    post_precision = prior_precision + 3.0 * 0.4 * obs_precision
    post_cov = torch.linalg.inv(post_precision)
    post_mean = post_cov @ (prior_precision @ prior_mean[0] + 3.0 * 0.4 * obs_precision @ observation[0])

    assert torch.allclose(updated.cov[0], post_cov)
    assert torch.allclose(updated.mean[0], post_mean)


def test_zero_allocation_and_inactive_arms_are_frozen():
    prior_mean = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    prior_cov = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float64)
    model = GaussianMetricModel(prior_mean, prior_cov, torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float64), target_idx=0, batch_sizes=[1.0])
    state = model.initial_state(active=torch.tensor([True, True]))
    allocation = torch.tensor([0.0, 1.0], dtype=torch.float64)
    observation = torch.tensor([[5.0], [7.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)

    assert torch.allclose(updated.mean[0], state.mean[0])
    assert torch.allclose(updated.cov[0], state.cov[0])


def test_near_zero_allocation_is_stable():
    prior_mean = torch.zeros(2, 1, dtype=torch.float64)
    prior_cov = torch.eye(1, dtype=torch.float64).repeat(2, 1, 1)
    model = GaussianMetricModel(prior_mean, prior_cov, prior_cov.clone(), target_idx=0, batch_sizes=[1.0])
    state = model.initial_state()
    allocation = torch.tensor([1e-20, 1.0], dtype=torch.float64)
    observation = torch.tensor([[100.0], [1.0]], dtype=torch.float64)

    updated = model.update(state, allocation, observation)

    assert torch.isfinite(updated.mean).all()
    assert torch.isfinite(updated.cov).all()
    assert torch.allclose(updated.mean[0], state.mean[0])
    assert torch.allclose(updated.cov[0], state.cov[0])
