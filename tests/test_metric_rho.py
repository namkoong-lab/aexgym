import torch

from aexgym.core import ActiveSetRule, GaussianMetricModel, SmoothingConfig
from aexgym.policies import (
    BasePlusResidualLogitParameterization,
    ConstantAllocationParameterization,
    FreeSequenceParameterization,
    NoSequenceRegularizer,
    PathwiseActiveSetRhoSimulation,
    ReducedTerminalRhoSimulation,
    RhoPolicy,
    TemporalUniformityRegularizer,
    one_step_target_value,
)


def make_scalar_model(batch_sizes=(1.0, 1.0)):
    prior_mean = torch.tensor([[0.1], [-0.2]], dtype=torch.float64)
    prior_cov = torch.tensor([[[0.4]], [[0.7]]], dtype=torch.float64)
    obs_cov = torch.tensor([[[0.25]], [[0.5]]], dtype=torch.float64)
    return GaussianMetricModel(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        obs_cov=obs_cov,
        target_metric_idx=0,
        batch_sizes=torch.tensor(batch_sizes, dtype=torch.float64),
    )


def make_multimetric_model(batch_sizes=(1.0, 1.0, 1.0)):
    prior_mean = torch.zeros((3, 2), dtype=torch.float64)
    prior_cov = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    prior_cov[:, 0, 0] = torch.tensor([0.3, 0.5, 0.4], dtype=torch.float64)
    prior_cov[:, 1, 1] = torch.tensor([0.6, 1.0, 0.7], dtype=torch.float64)
    obs_cov = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    obs_cov[:, 0, 0] = torch.tensor([0.8, 0.6, 0.7], dtype=torch.float64)
    obs_cov[:, 1, 1] = torch.tensor([0.8, 0.25, 0.9], dtype=torch.float64)
    return GaussianMetricModel(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        obs_cov=obs_cov,
        target_metric_idx=0,
        batch_sizes=torch.tensor(batch_sizes, dtype=torch.float64),
        control_arm_idx=0,
    )


def assert_masked_simplex_rows(sequence: torch.Tensor, active: torch.Tensor) -> None:
    active = active > 0
    for row in sequence:
        assert torch.all(row[~active] == 0)
        assert torch.all(row[active] >= 0)
        assert torch.allclose(row.sum(), torch.tensor(1.0, dtype=row.dtype))


def test_reduced_simulator_matches_scalar_reduced_formula_and_cumulative_information():
    model = make_scalar_model(batch_sizes=(1.0, 1.0))
    state = model.initial_state()
    simulation = ReducedTerminalRhoSimulation()
    z = torch.tensor([[0.25, -0.5], [1.2, -0.7]], dtype=torch.float64)
    sequence_a = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.float64)
    sequence_b = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64)

    value_a = simulation.evaluate(state, model, sequence_a, z, smoothing=SmoothingConfig(), terminal_beta=float("inf"))
    value_b = simulation.evaluate(state, model, sequence_b, z, smoothing=SmoothingConfig(), terminal_beta=float("inf"))
    expected = one_step_target_value(
        state,
        model,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        z,
        residual_batch=torch.tensor(2.0, dtype=torch.float64),
    )

    assert torch.allclose(value_a, expected)
    assert torch.allclose(value_b, expected)


def test_pathwise_simulator_distinguishes_same_total_budget_with_different_order_under_pruning():
    model = make_multimetric_model(batch_sizes=(1.0, 1.0))
    state = model.initial_state(active=torch.tensor([True, True, False]))
    rule = ActiveSetRule(
        target_metric_idx=0,
        control_arm_idx=0,
        guardrail_metric_indices=[1],
        guardrail_floors=[-0.25],
        violation_prob_threshold=0.8,
    )
    simulation = PathwiseActiveSetRhoSimulation(rule)
    sequence_early_treatment = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    sequence_late_treatment = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    prepared = torch.zeros((1, 2, 3, 2), dtype=torch.float64)
    prepared[0, 0, 1, 0] = 2.0
    prepared[0, 0, 1, 1] = -2.5
    prepared[0, 1, 1, 0] = 2.0
    prepared[0, 1, 1, 1] = 0.0

    early_value = simulation.evaluate(
        state,
        model,
        sequence_early_treatment,
        prepared,
        smoothing=SmoothingConfig(beta=float("inf")),
        terminal_beta=float("inf"),
    )
    late_value = simulation.evaluate(
        state,
        model,
        sequence_late_treatment,
        prepared,
        smoothing=SmoothingConfig(beta=float("inf")),
        terminal_beta=float("inf"),
    )

    assert late_value > early_value + 1e-6


def test_pathwise_simulator_continues_after_hard_control_collapse():
    prior_mean = torch.tensor([[0.0, 0.0], [-1.0, 0.0], [-1.2, 0.0]], dtype=torch.float64)
    prior_cov = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1) * 0.01
    obs_cov = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    model = GaussianMetricModel(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        obs_cov=obs_cov,
        target_metric_idx=0,
        batch_sizes=torch.tensor([1.0, 1.0], dtype=torch.float64),
        control_arm_idx=0,
    )
    state = model.initial_state()
    rule = ActiveSetRule(
        target_metric_idx=0,
        control_arm_idx=0,
        control_prob_threshold=0.95,
        shutdown_prob_threshold=0.05,
        best_arm_num_samples=0,
    )
    simulation = PathwiseActiveSetRhoSimulation(rule)
    sequence = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    prepared = torch.zeros((1, 2, 3, 2), dtype=torch.float64)
    prepared[0, 1, 0, 0] = 3.0

    value = simulation.evaluate(
        state,
        model,
        sequence,
        prepared,
        smoothing=SmoothingConfig(beta=float("inf")),
        terminal_beta=float("inf"),
    )

    one_step = model.posterior_random_walk(state, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), prepared[0, 0])
    collapsed = one_step.replace(active=rule.evaluate(one_step, model, smoothing=SmoothingConfig()).active)
    one_step_value = model.terminal_value(collapsed)
    two_step = model.posterior_random_walk(collapsed, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), prepared[0, 1])
    two_step = two_step.replace(active=rule.evaluate(two_step, model, smoothing=SmoothingConfig()).active)

    assert torch.allclose(value, model.terminal_value(two_step))
    assert value > one_step_value + 1e-6


def test_parameterizations_initialize_to_uniform_constant_plan():
    model = make_multimetric_model(batch_sizes=(1.0, 1.0, 1.0))
    state = model.initial_state(active=torch.tensor([True, False, True]))
    expected_row = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64)

    constant = ConstantAllocationParameterization().realize(
        state,
        model,
        ConstantAllocationParameterization().initialize(state, model, horizon=3),
    )
    base_plus_residual = BasePlusResidualLogitParameterization().realize(
        state,
        model,
        BasePlusResidualLogitParameterization().initialize(state, model, horizon=3),
    )
    free_sequence = FreeSequenceParameterization().realize(
        state,
        model,
        FreeSequenceParameterization().initialize(state, model, horizon=3),
    )

    for realized in (constant, base_plus_residual, free_sequence):
        assert realized.sequence.shape == (3, 3)
        assert_masked_simplex_rows(realized.sequence, state.active)
        assert torch.allclose(realized.sequence, expected_row.unsqueeze(0).repeat(3, 1))

    assert torch.allclose(constant.base_allocation, expected_row)
    assert torch.allclose(base_plus_residual.base_allocation, expected_row)
    assert torch.allclose(constant.residual_sequence, torch.zeros_like(constant.sequence))
    assert torch.allclose(base_plus_residual.residual_sequence, torch.zeros_like(base_plus_residual.sequence))
    assert free_sequence.base_allocation is None
    assert free_sequence.residual_sequence is None


def test_temporal_uniformity_regularizer_is_zero_on_constant_sequence_and_depends_only_on_realized_sequence():
    model = make_multimetric_model(batch_sizes=(1.0, 1.0, 1.0))
    state = model.initial_state(active=torch.tensor([True, False, True]))
    regularizer = TemporalUniformityRegularizer(weight=10.0)
    constant_sequence = ConstantAllocationParameterization().realize(
        state,
        model,
        ConstantAllocationParameterization().initialize(state, model, horizon=3),
    ).sequence
    free_sequence = FreeSequenceParameterization().realize(
        state,
        model,
        {"logits": torch.tensor([[2.0, -1.0], [0.0, 0.0], [-1.0, 2.0]], dtype=torch.float64, requires_grad=True)},
    ).sequence

    assert torch.allclose(regularizer.penalty(constant_sequence), torch.tensor(0.0, dtype=torch.float64))
    assert regularizer.penalty(free_sequence) > 0
    assert torch.allclose(
        regularizer.penalty(constant_sequence),
        regularizer.penalty(constant_sequence.clone()),
    )


def test_rho_policy_reduced_constant_no_regularizer_records_constant_plan_and_objective_scale():
    model = make_multimetric_model(batch_sizes=(0.5, 2.0, 1.5))
    state = model.initial_state(active=torch.tensor([True, False, True])).replace(t=1)
    policy = RhoPolicy(
        simulator=ReducedTerminalRhoSimulation(),
        parameterization=ConstantAllocationParameterization(),
        regularizer=NoSequenceRegularizer(),
        epochs=2,
        lr=0.1,
        num_samples=8,
        name="constant_rho",
    )
    policy.reset(7)
    allocation = policy.allocate(state, model)

    assert_masked_simplex_rows(policy.last_plan.sequence, state.active)
    assert torch.allclose(policy.last_plan.sequence[0], policy.last_plan.sequence[1])
    assert torch.allclose(policy.last_plan.first_allocation, allocation)
    assert torch.allclose(policy.last_plan.base_allocation, allocation)
    assert torch.allclose(policy.last_plan.residual_sequence, torch.zeros_like(policy.last_plan.sequence))
    assert policy.last_plan.regularization_penalty == 0.0
    assert policy.last_plan.objective_scale == 3.5
    assert len(policy.last_plan.optimization_trace) == policy.epochs
    assert len(policy.last_optimization_trace) == policy.epochs
    assert {"epoch", "loss", "objective_value", "simulation_value", "regularization_penalty", "objective_scale"} <= set(policy.last_optimization_trace[0])


def test_rho_policy_optimization_seed_is_configurable_independent_of_reset_seed():
    model = make_multimetric_model(batch_sizes=(1.0, 1.0))
    state = model.initial_state(active=torch.tensor([True, True, True]))

    def run_policy(optimization_seed: int, reset_seed: int):
        policy = RhoPolicy(
            simulator=ReducedTerminalRhoSimulation(sample_method="normal"),
            parameterization=ConstantAllocationParameterization(),
            regularizer=NoSequenceRegularizer(),
            epochs=2,
            lr=0.05,
            num_samples=16,
            optimization_seed=optimization_seed,
        )
        policy.reset(reset_seed)
        policy.allocate(state, model)
        return policy.last_plan

    first = run_policy(optimization_seed=31, reset_seed=1)
    same_optimization_seed = run_policy(optimization_seed=31, reset_seed=999)
    different_optimization_seed = run_policy(optimization_seed=32, reset_seed=1)

    assert first.optimization_seed == 31
    assert same_optimization_seed.optimization_seed == 31
    assert different_optimization_seed.optimization_seed == 32
    assert first.optimization_trace == same_optimization_seed.optimization_trace
    assert first.optimization_trace[0]["loss"] != different_optimization_seed.optimization_trace[0]["loss"]


def test_rho_policy_pathwise_base_plus_residual_records_sequence_base_and_residuals():
    model = make_multimetric_model(batch_sizes=(0.5, 2.0, 1.5))
    state = model.initial_state(active=torch.tensor([True, True, False])).replace(t=1)
    rule = ActiveSetRule(target_metric_idx=0, control_arm_idx=0)
    policy = RhoPolicy(
        simulator=PathwiseActiveSetRhoSimulation(rule),
        parameterization=BasePlusResidualLogitParameterization(),
        regularizer=TemporalUniformityRegularizer(weight=100.0),
        epochs=2,
        lr=0.05,
        num_samples=4,
        name="pathwise_rho",
    )
    policy.reset(11)
    allocation = policy.allocate(state, model)

    assert_masked_simplex_rows(policy.last_plan.sequence, state.active)
    assert torch.allclose(policy.last_plan.first_allocation, allocation)
    assert policy.last_plan.sequence.shape == (2, 3)
    assert policy.last_plan.base_allocation.shape == (3,)
    assert policy.last_plan.residual_sequence.shape == (2, 3)
    assert policy.last_plan.objective_scale == 3.5
    assert len(policy.last_plan.optimization_trace) == policy.epochs


def test_smooth_pathwise_rollout_has_gradients_through_active():
    model = make_multimetric_model(batch_sizes=(1.0, 1.0))
    state = model.initial_state(active=torch.tensor([True, True, False]))
    rule = ActiveSetRule(
        target_metric_idx=0,
        control_arm_idx=0,
        guardrail_metric_indices=[1],
        guardrail_floors=[-0.25],
        violation_prob_threshold=0.5,
    )
    simulation = PathwiseActiveSetRhoSimulation(rule)
    logits = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    selector = torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=3).to(torch.float64)
    sequence = torch.softmax(logits, dim=1) @ selector
    prepared = torch.zeros((2, 2, 3, 2), dtype=torch.float64)
    prepared[:, :, 1, 1] = -0.5

    value = simulation.evaluate(
        state,
        model,
        sequence,
        prepared,
        smoothing=SmoothingConfig(beta=5.0),
        terminal_beta=10.0,
    )
    value.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.any(torch.abs(logits.grad) > 0)
