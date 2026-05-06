import matplotlib
import pytest
import torch

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from aexgym.core import GaussianMetricModel
from aexgym.experiments.revision.plotting import plot_two_metric_trajectory


def test_plot_two_metric_trajectory_builds_stage_transition_grid():
    model = GaussianMetricModel(
        prior_mean=torch.zeros(2, 2, dtype=torch.float64),
        prior_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        obs_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        target_metric_idx=0,
        batch_sizes=torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    state0 = model.initial_state()
    allocation0 = torch.tensor([0.6, 0.4], dtype=torch.float64)
    state1 = model.update(
        state0,
        allocation0,
        torch.tensor([[0.5, -0.2], [0.1, 0.3]], dtype=torch.float64),
    )
    state1 = state1.replace(active=torch.tensor([1.0, 0.0], dtype=torch.float64))
    allocation1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    state2 = model.update(
        state1,
        allocation1,
        torch.tensor([[0.3, -0.1], [0.0, 0.0]], dtype=torch.float64),
    )

    fig, axes = plot_two_metric_trajectory(
        model,
        states=[state0, state1, state2],
        allocations=[allocation0, allocation1],
        metric_names=("Primary", "Secondary"),
    )

    assert axes.shape == (5, 2)
    assert len(fig.axes) == 10
    assert axes[0, 0].get_title() == "Arm 0"
    assert any(text.get_text() == "Stage 0" for text in axes[0, 0].texts)
    assert any(text.get_text() == "0 to 1" for text in axes[1, 0].texts)
    assert any(text.get_text() == "Stage 1" for text in axes[2, 0].texts)
    assert not any(label.get_visible() for label in axes[0, 0].get_yticklabels())
    assert not any(label.get_visible() for label in axes[1, 0].get_yticklabels())
    assert any(label.get_visible() for label in axes[0, 1].get_yticklabels())
    assert any(label.get_visible() for label in axes[1, 1].get_yticklabels())
    assert axes[0, 1].get_ylabel() == "Secondary"
    assert axes[1, 1].get_ylabel() == "Sample Size"
    assert axes[4, 0].get_xlabel() == "Primary"
    assert axes[4, 1].get_xlabel() == "Primary"
    plt.close(fig)


def test_plot_two_metric_trajectory_adds_optimization_column():
    model = GaussianMetricModel(
        prior_mean=torch.zeros(2, 2, dtype=torch.float64),
        prior_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        obs_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        target_metric_idx=0,
        batch_sizes=torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    state0 = model.initial_state()
    allocation0 = torch.tensor([0.6, 0.4], dtype=torch.float64)
    state1 = model.update(state0, allocation0, torch.tensor([[0.5, -0.2], [0.1, 0.3]], dtype=torch.float64))
    allocation1 = torch.tensor([0.7, 0.3], dtype=torch.float64)
    state2 = model.update(state1, allocation1, torch.tensor([[0.3, -0.1], [0.0, 0.0]], dtype=torch.float64))
    histories = [
        [{"epoch": 1.0, "loss": -1.0}, {"epoch": 2.0, "loss": -1.2}],
        [{"epoch": 1.0, "loss": -0.8}],
    ]

    fig, axes = plot_two_metric_trajectory(
        model,
        states=[state0, state1, state2],
        allocations=[allocation0, allocation1],
        optimization_histories=histories,
        optimization_max_epochs=3,
    )

    assert axes.shape == (5, 3)
    assert axes[0, 2].get_title() == "RHO Loss"
    assert axes[1, 2].get_xlabel() == ""
    assert axes[3, 2].get_xlabel() == "Epoch"
    plt.close(fig)


def test_plot_two_metric_trajectory_adds_planned_sequence_column():
    model = GaussianMetricModel(
        prior_mean=torch.zeros(2, 2, dtype=torch.float64),
        prior_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        obs_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        target_metric_idx=0,
        batch_sizes=torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    state0 = model.initial_state()
    allocation0 = torch.tensor([0.6, 0.4], dtype=torch.float64)
    state1 = model.update(state0, allocation0, torch.tensor([[0.5, -0.2], [0.1, 0.3]], dtype=torch.float64))
    state1 = state1.replace(active=torch.tensor([1.0, 0.0], dtype=torch.float64))
    allocation1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    state2 = model.update(state1, allocation1, torch.tensor([[0.3, -0.1], [0.0, 0.0]], dtype=torch.float64))
    histories = [
        [{"epoch": 1.0, "loss": -1.0}, {"epoch": 2.0, "loss": -1.2}],
        [{"epoch": 1.0, "loss": -0.8}],
    ]
    planned_sequences = [
        torch.tensor([[0.6, 0.4], [0.25, 0.75]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
    ]

    fig, axes = plot_two_metric_trajectory(
        model,
        states=[state0, state1, state2],
        allocations=[allocation0, allocation1],
        optimization_histories=histories,
        optimization_max_epochs=3,
        planned_sequences=planned_sequences,
    )

    assert axes.shape == (5, 4)
    assert axes[0, 2].get_title() == "RHO Loss"
    assert axes[0, 3].get_title() == "RHO Plan"
    assert axes[3, 3].get_xlabel() == "Arm Index"
    assert [label.get_text() for label in axes[3, 3].get_xticklabels()] == ["0"]
    assert len(axes[1, 3].patches) == 4
    plt.close(fig)


def test_plot_two_metric_trajectory_rejects_non_two_metric_model():
    model = GaussianMetricModel(
        prior_mean=torch.zeros(2, 1, dtype=torch.float64),
        prior_cov=torch.eye(1, dtype=torch.float64).repeat(2, 1, 1),
        obs_cov=torch.eye(1, dtype=torch.float64).repeat(2, 1, 1),
        target_metric_idx=0,
        batch_sizes=torch.tensor([1.0], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="exactly two metrics"):
        plot_two_metric_trajectory(model, states=[model.initial_state()], allocations=[])


def test_plot_two_metric_trajectory_rejects_mismatched_optimization_histories():
    model = GaussianMetricModel(
        prior_mean=torch.zeros(2, 2, dtype=torch.float64),
        prior_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        obs_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        target_metric_idx=0,
        batch_sizes=torch.tensor([1.0], dtype=torch.float64),
    )
    state = model.initial_state()
    allocation = torch.tensor([0.5, 0.5], dtype=torch.float64)
    next_state = model.update(state, allocation, torch.zeros(2, 2, dtype=torch.float64))

    with pytest.raises(ValueError, match="optimization_histories"):
        plot_two_metric_trajectory(
            model,
            states=[state, next_state],
            allocations=[allocation],
            optimization_histories=[],
        )


def test_plot_two_metric_trajectory_rejects_mismatched_planned_sequences():
    model = GaussianMetricModel(
        prior_mean=torch.zeros(2, 2, dtype=torch.float64),
        prior_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        obs_cov=torch.eye(2, dtype=torch.float64).repeat(2, 1, 1),
        target_metric_idx=0,
        batch_sizes=torch.tensor([1.0], dtype=torch.float64),
    )
    state = model.initial_state()
    allocation = torch.tensor([0.5, 0.5], dtype=torch.float64)
    next_state = model.update(state, allocation, torch.zeros(2, 2, dtype=torch.float64))

    with pytest.raises(ValueError, match="planned_sequences"):
        plot_two_metric_trajectory(
            model,
            states=[state, next_state],
            allocations=[allocation],
            planned_sequences=[],
        )
