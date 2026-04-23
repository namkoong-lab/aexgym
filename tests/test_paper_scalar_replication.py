from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aexgym.experiments.paper_scalar_compare import compare_config
from aexgym.experiments.paper_scalar_replication import PaperScalarConfig, make_model
from aexgym.experiments.paper_scalar_variant_check import VariantCheckConfig, run_variant_trial


AES_ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_ROOT = AES_ROOT / "adaptive-experimentation"


def test_paper_scalar_update_matches_legacy_aggregate_recursion():
    mu = np.array([0.1, -0.2])
    sigma2 = np.array([0.4, 0.7])
    s2 = np.array([0.25, 0.5])
    allocation = np.array([0.3, 0.7])
    aggregate_g = np.array([0.05, -0.04])
    model = make_model(mu, sigma2, s2, horizon=1)

    state = model.update(
        model.initial_state(),
        torch.as_tensor(allocation, dtype=torch.float64),
        torch.as_tensor(aggregate_g, dtype=torch.float64).reshape(2, 1),
    )

    expected_sigma2 = (s2 * sigma2) / (s2 + allocation * sigma2)
    expected_mu = expected_sigma2 * ((mu / sigma2) + allocation * (aggregate_g / s2))
    assert np.allclose(state.mean[:, 0].numpy(), expected_mu)
    assert np.allclose(state.cov[:, 0, 0].numpy(), expected_sigma2)


@pytest.mark.skipif(not (ADAPTIVE_ROOT / "MAB" / "simulator.py").exists(), reason="legacy simulator is not available")
def test_paper_scalar_bernoulli_matches_legacy_simulator():
    config = PaperScalarConfig(
        reward="bernoulli",
        n_arms=4,
        horizon=2,
        mean_scale=0.1,
        n_runs=1,
        seed=31,
        policies=["uniform", "ts", "ttts", "myopic", "rho"],
        n_samples=32,
        rho_num_zs=32,
        rho_epochs=2,
    )

    payload = compare_config(config, ADAPTIVE_ROOT)

    assert payload["all_match"], payload["rows"]


@pytest.mark.skipif(not (ADAPTIVE_ROOT / "MAB" / "simulator.py").exists(), reason="legacy simulator is not available")
def test_paper_scalar_gumbel_matches_legacy_simulator():
    config = PaperScalarConfig(
        reward="gumbel",
        n_arms=4,
        horizon=2,
        mean_scale=0.1,
        s2=1.0,
        n_runs=1,
        seed=37,
        policies=["uniform", "ts", "ttts", "myopic", "rho"],
        n_samples=32,
        rho_num_zs=32,
        rho_epochs=2,
    )

    payload = compare_config(config, ADAPTIVE_ROOT)

    assert payload["all_match"], payload["rows"]


def test_multimetric_ignored_secondary_matches_scalar_reduced_rho():
    config = VariantCheckConfig(
        reward="bernoulli",
        n_arms=4,
        horizon=2,
        mean_scale=0.1,
        n_runs=2,
        seed=41,
        rho_num_zs=32,
        rho_epochs=2,
        rho_lr=1.0,
    )

    reduced_trials = [run_variant_trial(config, seed, "reduced_j1") for seed in range(config.seed, config.seed + config.n_runs)]
    ignored_trials = [run_variant_trial(config, seed, "reduced_j2_ignored") for seed in range(config.seed, config.seed + config.n_runs)]

    assert reduced_trials == [{**trial, "variant": "reduced_j1"} for trial in ignored_trials]
