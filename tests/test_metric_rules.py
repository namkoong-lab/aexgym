import torch

from aexgym.metric import ActiveSetRule, GaussianMetricState


def make_state(mean, active=None):
    mean = torch.tensor(mean, dtype=torch.float64)
    cov = torch.eye(mean.shape[1], dtype=torch.float64).repeat(mean.shape[0], 1, 1) * 0.01
    if active is None:
        active = torch.ones(mean.shape[0], dtype=torch.bool)
    return GaussianMetricState(mean=mean, cov=cov, active=active, t=1)


def test_guardrail_pruning_removes_only_active_noncontrol_arms():
    state = make_state(
        [
            [0.0, 0.0],
            [0.1, -1.0],
            [0.2, 0.1],
            [0.3, -1.0],
        ],
        active=torch.tensor([True, True, True, False]),
    )
    rule = ActiveSetRule(target_idx=0, control_arm=0, guardrail_indices=[1], guardrail_floors=[-0.25], violation_prob_threshold=0.8)

    updated = rule.apply(state)

    assert updated.active.tolist() == [True, False, True, False]
    assert not updated.stopped


def test_control_is_preserved_and_all_treatments_failing_collapses_to_control():
    state = make_state(
        [
            [0.0, -2.0],
            [0.3, -2.0],
            [0.4, -2.0],
        ]
    )
    rule = ActiveSetRule(target_idx=0, control_arm=0, guardrail_indices=[1], guardrail_floors=[-0.25], violation_prob_threshold=0.8)

    updated = rule.apply(state)

    assert updated.active.tolist() == [True, False, False]
    assert updated.stopped
    assert updated.stop_reason == "singleton_active"


def test_singleton_active_set_is_absorbing():
    state = make_state([[0.0, 0.0], [0.2, -2.0]], active=torch.tensor([False, True]))
    rule = ActiveSetRule(target_idx=0, control_arm=0, guardrail_indices=[1], guardrail_floors=[-0.25], violation_prob_threshold=0.8)

    updated = rule.apply(state)

    assert updated.active.tolist() == [False, True]
    assert updated.stopped
