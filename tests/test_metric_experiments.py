import torch

from aexgym.experiments.parity.scalar import PARITY_SCENARIOS, ScalarParityConfig, run_config as run_scalar
from aexgym.experiments.parity.suite import ParitySweepConfig, run_sweep
from aexgym.experiments.revision.basic_guardrail import GuardrailBasicConfig, run_config as run_guardrail


def test_tiny_scalar_parity_run_logs_expected_fields():
    config = ScalarParityConfig(
        scenario="exact_gaussian",
        n_arms=3,
        horizon=2,
        n_runs=2,
        baseline_policies=["uniform", "ts"],
        rho_variants=[],
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
        baseline_policies=["uniform"],
        rho_variants=[],
        seed=123,
    )

    payload_1 = run_scalar(config)
    payload_2 = run_scalar(config)

    assert payload_1["runs"] == payload_2["runs"]


def test_scalar_scenario_config_smoke():
    for scenario in sorted(PARITY_SCENARIOS):
        config = ScalarParityConfig(
            scenario=scenario,
            n_arms=3,
            horizon=2,
            n_runs=1,
            baseline_policies=["uniform"],
            rho_variants=[],
            seed=3,
        )
        payload = run_scalar(config)
        assert "uniform" in payload["aggregates"]


def test_parity_sweep_collects_rho_variant_summaries():
    config = ParitySweepConfig(
        scenarios=["exact_gaussian"],
        n_arms=3,
        horizon=2,
        n_runs=1,
        baseline_policies=["uniform"],
        rho_variants=["reduced_constant", "pathwise_constant"],
        seed=19,
    )

    payload = run_sweep(config)

    assert set(payload["scenario_results"]) == {"exact_gaussian"}
    assert set(payload["rho_variant_summaries"]["exact_gaussian"]) == {"reduced_constant", "pathwise_constant"}
