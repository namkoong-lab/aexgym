import math

import torch

from aexgym.core import ActiveSetRule, GaussianMetricModel, SmoothingConfig


def make_model_and_state(mean, active=None, prior_var=0.01, t=1):
    mean = torch.tensor(mean, dtype=torch.float64)
    cov = torch.eye(mean.shape[1], dtype=torch.float64).repeat(mean.shape[0], 1, 1) * prior_var
    model = GaussianMetricModel(
        prior_mean=mean,
        prior_cov=cov,
        obs_cov=torch.eye(mean.shape[1], dtype=torch.float64).repeat(mean.shape[0], 1, 1),
        target_metric_idx=0,
        batch_sizes=[1.0, 1.0, 1.0],
        control_arm_idx=0,
    )
    state = model.initial_state(active=torch.ones(mean.shape[0], dtype=torch.float64) if active is None else active).replace(t=t)
    return model, state


def test_guardrail_pruning_removes_only_active_noncontrol_arm_idxs_hard():
    model, state = make_model_and_state(
        [
            [0.0, 0.0],
            [0.1, -1.0],
            [0.2, 0.1],
            [0.3, -1.0],
        ],
        active=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64),
    )
    rule = ActiveSetRule(target_metric_idx=0, control_arm_idx=0, guardrail_metric_indices=[1], guardrail_floors=[-0.25], violation_prob_threshold=0.8)

    decision = rule.evaluate(state, model, smoothing=SmoothingConfig(beta=math.inf))

    assert decision.active.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_smooth_guardrail_rule_emits_continuous_active():
    model, state = make_model_and_state(
        [
            [0.0, 0.0],
            [0.1, -0.25],
            [0.2, 0.1],
        ],
        prior_var=0.2,
    )
    rule = ActiveSetRule(target_metric_idx=0, control_arm_idx=0, guardrail_metric_indices=[1], guardrail_floors=[-0.25], violation_prob_threshold=0.8)

    decision = rule.evaluate(state, model, smoothing=SmoothingConfig(beta=5.0))

    assert torch.all(decision.active >= 0)
    assert torch.all(decision.active <= 1)
    assert decision.active[0] == 1.0
    assert 0.0 < decision.active[1] < 1.0


def test_control_is_preserved_and_all_treatments_failing_collapses_to_control():
    model, state = make_model_and_state(
        [
            [0.0, -2.0],
            [0.3, -2.0],
            [0.4, -2.0],
        ]
    )
    rule = ActiveSetRule(target_metric_idx=0, control_arm_idx=0, guardrail_metric_indices=[1], guardrail_floors=[-0.25], violation_prob_threshold=0.8)

    decision = rule.evaluate(state, model, smoothing=SmoothingConfig(beta=math.inf))

    assert decision.active.tolist() == [1.0, 0.0, 0.0]


def test_single_active_arm_remains_an_active_set_decision_only():
    model, state = make_model_and_state([[0.0, 0.0], [0.2, 0.0]], active=torch.tensor([0.0, 1.0], dtype=torch.float64))
    rule = ActiveSetRule(target_metric_idx=0, control_arm_idx=0, guardrail_metric_indices=[1], guardrail_floors=[-0.25], violation_prob_threshold=0.8)

    decision = rule.evaluate(state, model, smoothing=SmoothingConfig(beta=math.inf))

    assert decision.active.tolist() == [0.0, 1.0]


def test_control_and_winner_collapse_are_active_set_updates_only():
    model, state = make_model_and_state(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.1, 0.0],
        ],
        prior_var=0.01,
        t=1,
    )
    shutdown_rule = ActiveSetRule(
        target_metric_idx=0,
        control_arm_idx=0,
        control_prob_threshold=0.95,
        shutdown_prob_threshold=0.05,
    )
    shutdown = shutdown_rule.evaluate(state, model, smoothing=SmoothingConfig(beta=math.inf))
    assert shutdown.active.tolist() == [1.0, 0.0, 0.0]
    assert shutdown.diagnostics["control_collapse_weight"] == 1.0

    success_model, success_state = make_model_and_state(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
        ],
        prior_var=0.01,
        t=1,
    )
    success_rule = ActiveSetRule(
        target_metric_idx=0,
        control_arm_idx=0,
        early_success_prob_threshold=0.8,
        min_success_epoch=2,
        best_arm_num_samples=0,
    )
    blocked = success_rule.evaluate(success_state, success_model, smoothing=SmoothingConfig(beta=math.inf))
    assert "winner_collapse_weight" not in blocked.diagnostics

    later = success_state.replace(t=2)
    success = success_rule.evaluate(later, success_model, smoothing=SmoothingConfig(beta=math.inf))
    assert success.active.sum() == 1.0
    assert success.diagnostics["winner_collapse_weight"] == 1.0
