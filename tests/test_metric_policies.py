import torch

from aexgym.core import GaussianMetricModel
from aexgym.policies import (
    GaussianThompsonPolicy,
    GaussianTopTwoThompsonPolicy,
    MyopicLookaheadPolicy,
    UniformActivePolicy,
    one_step_target_value,
)


def make_scalar_model():
    prior_mean = torch.tensor([[0.0], [0.1], [0.2]], dtype=torch.float64)
    prior_cov = torch.eye(1, dtype=torch.float64).repeat(3, 1, 1)
    obs_cov = torch.eye(1, dtype=torch.float64).repeat(3, 1, 1)
    return GaussianMetricModel(prior_mean, prior_cov, obs_cov, target_idx=0, batch_sizes=[1.0, 1.0, 1.0])


def assert_masked_simplex(allocation, active):
    assert torch.all(allocation[~active] == 0)
    assert torch.all(allocation[active] >= 0)
    assert torch.allclose(allocation.sum(), torch.tensor(1.0, dtype=allocation.dtype))


def test_policies_output_masked_simplex_and_no_inactive_mass():
    model = make_scalar_model()
    state = model.initial_state(active=torch.tensor([True, False, True]))
    policies = [
        UniformActivePolicy(),
        GaussianThompsonPolicy(n_samples=64),
        GaussianTopTwoThompsonPolicy(n_samples=64),
        MyopicLookaheadPolicy(epochs=3, num_zs=16, lr=0.05),
    ]

    for idx, policy in enumerate(policies):
        policy.reset(idx)
        allocation = policy.allocate(state, model)
        assert_masked_simplex(allocation, state.active)


def test_constant_rho_value_matches_old_scalar_reduced_formula():
    model = make_scalar_model()
    state = model.initial_state()
    allocation = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    z = torch.tensor([[0.0, 1.0, -1.0], [1.0, 0.0, -0.5]], dtype=torch.float64)
    budget = torch.tensor(3.0, dtype=torch.float64)

    value = one_step_target_value(state, model, allocation, z, residual_batch=budget)

    sigma2 = state.cov[:, 0, 0]
    s2 = model.obs_cov[:, 0, 0]
    phi = torch.sqrt((sigma2**2 * allocation * budget) / (s2 + sigma2 * allocation * budget))
    expected = torch.max(state.mean[:, 0].unsqueeze(0) + z * phi.unsqueeze(0), dim=1).values.mean()
    assert torch.allclose(value, expected)
