import torch

from aexgym.experiments.guardrail_basic import GuardrailBasicConfig, run_config as run_guardrail
from aexgym.experiments.scalar_parity import SCALAR_SCENARIOS, ScalarParityConfig, run_config as run_scalar


def test_tiny_scalar_parity_run_logs_expected_fields():
    config = ScalarParityConfig(
        scenario="exact_gaussian",
        n_arms=3,
        horizon=2,
        n_runs=2,
        policies=["uniform", "ts"],
        seed=7,
    )

    payload = run_scalar(config)

    assert set(payload["runs"]) == {"uniform", "ts"}
    first = payload["runs"]["uniform"][0]
    for key in [
        "selected_arm",
        "true_best_arm",
        "stop_time",
        "simple_regret",
        "active_path",
        "allocation_path",
        "unsafe_exposure",
        "stop_reason",
    ]:
        assert key in first


def test_guardrail_run_removes_unsafe_treatment_and_records_exposure():
    config = GuardrailBasicConfig(
        horizon=3,
        batch_sizes=[1.0, 1.0, 1.0],
        n_runs=1,
        policies=["uniform"],
        seed=5,
        violation_prob_threshold=0.2,
    )

    payload = run_guardrail(config)
    run = payload["runs"]["uniform"][0]

    assert run["unsafe_exposure"] > 0
    assert any(2 not in active for active in run["active_path"][1:])


def test_reproducibility_with_fixed_seed():
    config = ScalarParityConfig(
        scenario="exact_gaussian",
        n_arms=3,
        horizon=2,
        n_runs=2,
        policies=["uniform"],
        seed=123,
    )

    payload_1 = run_scalar(config)
    payload_2 = run_scalar(config)

    assert payload_1["runs"] == payload_2["runs"]


def test_scalar_scenario_config_smoke():
    # ASOS requires a user-supplied data path, so smoke the registry but do not run it.
    assert "asos_scalar" in SCALAR_SCENARIOS
    for scenario in sorted(SCALAR_SCENARIOS - {"asos_scalar"}):
        config = ScalarParityConfig(
            scenario=scenario,
            n_arms=3,
            horizon=2,
            n_runs=1,
            policies=["uniform"],
            seed=3,
        )
        payload = run_scalar(config)
        assert "uniform" in payload["aggregates"]
