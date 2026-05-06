from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping, Optional

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from torch import Tensor

from aexgym.core import GaussianMetricModel, GaussianMetricState


def plot_two_metric_trajectory(
    model: GaussianMetricModel,
    states: Sequence[GaussianMetricState],
    allocations: Sequence[Tensor | Sequence[float]],
    *,
    metric_names: tuple[str, str] = ("Metric 0", "Metric 1"),
    arm_names: Optional[Sequence[str]] = None,
    confidence: float = 0.9,
    state_cell_size: tuple[float, float] = (2.3, 1.7),
    transition_width: float | None = None,
    optimization_histories: Optional[Sequence[Sequence[Mapping[str, float]]]] = None,
    optimization_max_epochs: Optional[int] = None,
    optimization_width: float = 1.9,
    planned_sequences: Optional[Sequence[Tensor | Sequence[Sequence[float]]]] = None,
    planned_width: float = 2.2,
) -> tuple[Figure, np.ndarray]:
    """Plot a two-metric trajectory as arm columns and stage/transition rows.

    Main columns are arms. Optional right-side columns show RHO optimization
    diagnostics. Stage rows show posterior confidence ellipses, posterior
    means, empirical/raw means, effective sample sizes, and active weights.
    Transition rows show executed and planned effective sample mass.
    """

    import matplotlib.pyplot as plt

    if model.n_metrics != 2:
        raise ValueError("plot_two_metric_trajectory requires a model with exactly two metrics")
    if len(states) == 0:
        raise ValueError("states must be nonempty")
    if len(allocations) != len(states) - 1:
        raise ValueError("allocations must have length len(states) - 1")
    if optimization_histories is not None and len(optimization_histories) != len(allocations):
        raise ValueError("optimization_histories must have length len(allocations)")
    if planned_sequences is not None and len(planned_sequences) != len(allocations):
        raise ValueError("planned_sequences must have length len(allocations)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    transition_height = state_cell_size[1] if transition_width is None else transition_width
    if state_cell_size[0] <= 0 or state_cell_size[1] <= 0 or transition_height <= 0 or optimization_width <= 0 or planned_width <= 0:
        raise ValueError("state_cell_size, transition_width, optimization_width, and planned_width must be positive")

    n_arms = states[0].n_arms
    for state in states:
        if state.n_arms != n_arms or state.n_metrics != 2:
            raise ValueError("all states must have the same arms and exactly two metrics")
    if arm_names is None:
        arm_names = tuple(f"Arm {arm}" for arm in range(n_arms))
    if len(arm_names) != n_arms:
        raise ValueError("arm_names must match the number of arms")

    allocation_arrays = [_as_1d_numpy(allocation, n_arms) for allocation in allocations]
    planned_arrays = None if planned_sequences is None else [_as_2d_numpy(sequence, n_arms) for sequence in planned_sequences]
    stage_data = [_stage_arrays(model, state) for state in states]
    radius2 = float(chi2.ppf(confidence, df=2))
    xlim, ylim = _trajectory_limits(stage_data, radius2)

    include_optimization_column = optimization_histories is not None
    include_planned_column = planned_arrays is not None
    n_rows = 2 * len(states) - 1
    n_cols = n_arms + int(include_optimization_column) + int(include_planned_column)
    height_ratios = [state_cell_size[1] if row % 2 == 0 else transition_height for row in range(n_rows)]
    width_ratios = [state_cell_size[0]] * n_arms
    if include_optimization_column:
        width_ratios.append(optimization_width)
    if include_planned_column:
        width_ratios.append(planned_width)
    figsize = (sum(width_ratios), len(states) * state_cell_size[1] + len(allocations) * transition_height)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "width_ratios": width_ratios},
        constrained_layout=True,
    )

    for arm_idx in range(n_arms):
        for stage_idx, state in enumerate(states):
            row = 2 * stage_idx
            _plot_stage_cell(
                axes[row, arm_idx],
                state=state,
                stage_data=stage_data[stage_idx],
                arm_idx=arm_idx,
                confidence_radius2=radius2,
                xlim=xlim,
                ylim=ylim,
                metric_names=metric_names,
                stage_label=f"Stage {state.t}",
                show_ylabel=arm_idx == 0,
                show_y_ticks=arm_idx == n_arms - 1,
                show_xlabel=stage_idx == len(states) - 1,
                show_x_label=stage_idx == len(states) - 1,
            )
            if row == 0:
                axes[row, arm_idx].set_title(arm_names[arm_idx], fontsize=10)

        for transition_idx, allocation in enumerate(allocation_arrays):
            row = 2 * transition_idx + 1
            _plot_transition_cell(
                axes[row, arm_idx],
                state_before=states[transition_idx],
                state_after=states[transition_idx + 1],
                allocation=allocation,
                batch_size=float(model.batch_size(states[transition_idx].t).detach().cpu().item()),
                arm_idx=arm_idx,
                transition_label=_transition_label(states[transition_idx], states[transition_idx + 1]),
                show_ylabel=arm_idx == 0,
                show_y_ticks=arm_idx == n_arms - 1,
                show_xlabel=transition_idx == len(allocation_arrays) - 1,
            )

    if include_optimization_column:
        optimization_col = n_arms
        max_epochs = _optimization_max_epochs(optimization_histories, optimization_max_epochs)
        for row in range(n_rows):
            ax = axes[row, optimization_col]
            if row % 2 == 0:
                ax.axis("off")
                if row == 0:
                    ax.set_title("RHO Loss", fontsize=10)
                continue
            transition_idx = (row - 1) // 2
            _plot_optimization_cell(
                ax,
                history=optimization_histories[transition_idx],
                max_epochs=max_epochs,
                transition_label=_transition_label(states[transition_idx], states[transition_idx + 1]),
                show_xlabel=transition_idx == len(allocation_arrays) - 1,
            )

    if include_planned_column:
        planned_col = n_cols - 1
        for row in range(n_rows):
            ax = axes[row, planned_col]
            if row % 2 == 0:
                ax.axis("off")
                if row == 0:
                    ax.set_title("RHO Plan", fontsize=10)
                continue
            transition_idx = (row - 1) // 2
            _plot_planned_sequence_cell(
                ax,
                model=model,
                state=states[transition_idx],
                sequence=planned_arrays[transition_idx],
                show_xlabel=transition_idx == len(allocation_arrays) - 1,
            )

    return fig, axes


def _as_1d_numpy(value: Tensor | Sequence[float], n_arms: int) -> np.ndarray:
    if isinstance(value, Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value, dtype=float)
    array = np.asarray(array, dtype=float).reshape(-1)
    if array.shape != (n_arms,):
        raise ValueError("each allocation must have shape (n_arms,)")
    return array


def _as_2d_numpy(value: Tensor | Sequence[Sequence[float]], n_arms: int) -> np.ndarray:
    if isinstance(value, Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value, dtype=float)
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[1] != n_arms:
        raise ValueError("each planned sequence must have shape (remaining_stages, n_arms)")
    return array


def _stage_arrays(model: GaussianMetricModel, state: GaussianMetricState) -> dict[str, np.ndarray]:
    mean, cov = model.posterior_moments(state)
    empirical = model.empirical_mean(state)
    return {
        "mean": mean.detach().cpu().numpy(),
        "cov": cov.detach().cpu().numpy(),
        "empirical": empirical.detach().cpu().numpy(),
        "n_eff": state.n_eff.detach().cpu().numpy(),
        "active": state.active.detach().cpu().numpy(),
    }


def _trajectory_limits(stage_data: Sequence[dict[str, np.ndarray]], radius2: float) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for data in stage_data:
        means = data["mean"]
        covs = data["cov"]
        empirical = data["empirical"]
        for mean, cov, raw in zip(means, covs, empirical):
            vals = np.linalg.eigvalsh(_symmetrize(cov))
            radius = float(np.sqrt(max(float(vals.max()), 0.0) * radius2))
            xs.extend([float(mean[0] - radius), float(mean[0] + radius), float(raw[0])])
            ys.extend([float(mean[1] - radius), float(mean[1] + radius), float(raw[1])])
    return _padded_limits(xs), _padded_limits(ys)


def _padded_limits(values: Sequence[float]) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high):
        return -1.0, 1.0
    if abs(high - low) < 1e-12:
        low -= 1.0
        high += 1.0
    pad = 0.08 * (high - low)
    return low - pad, high + pad


def _plot_stage_cell(
    ax: Axes,
    *,
    state: GaussianMetricState,
    stage_data: dict[str, np.ndarray],
    arm_idx: int,
    confidence_radius2: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    metric_names: tuple[str, str],
    stage_label: str,
    show_ylabel: bool,
    show_y_ticks: bool,
    show_xlabel: bool,
    show_x_label: bool,
) -> None:
    mean = stage_data["mean"][arm_idx]
    cov = _symmetrize(stage_data["cov"][arm_idx])
    empirical = stage_data["empirical"][arm_idx]
    n_eff = float(stage_data["n_eff"][arm_idx])
    active = float(stage_data["active"][arm_idx])
    is_active = active > 0.0

    active_color = "#2563eb"
    inactive_color = "#9ca3af"
    raw_color = "#111827" if n_eff > 0 else "#9ca3af"

    ax.set_facecolor("#ffffff" if is_active else "#f3f4f6")
    ellipse = _confidence_ellipse(mean, cov, confidence_radius2)
    ellipse.set_edgecolor(active_color if is_active else inactive_color)
    ellipse.set_linewidth(2.0 if is_active else 1.25)
    ellipse.set_alpha(0.95 if is_active else 0.55)
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], s=26, color=active_color if is_active else inactive_color, zorder=3)
    ax.scatter(empirical[0], empirical[1], s=34, marker="x", color=raw_color, linewidths=1.5, zorder=4)

    for spine in ax.spines.values():
        spine.set_color(active_color if is_active else inactive_color)
        spine.set_linewidth(1.6 if is_active else 0.9)

    ax.text(
        0.03,
        0.97,
        f"Active={active:.2g}\nSamples={n_eff:.2g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    ax.text(
        0.03,
        0.03,
        f"Posterior=({mean[0]:.2g}, {mean[1]:.2g})\nMean=({empirical[0]:.2g}, {empirical[1]:.2g})",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    _configure_y_axis(ax, show_ticks=show_y_ticks, label=metric_names[1] if show_y_ticks else None)
    if show_ylabel:
        _set_row_label(ax, stage_label)
    if show_xlabel:
        ax.set_xlabel(metric_names[0] if show_x_label else "", fontsize=8)
    else:
        ax.set_xticklabels([])


def _plot_transition_cell(
    ax: Axes,
    *,
    state_before: GaussianMetricState,
    state_after: GaussianMetricState,
    allocation: np.ndarray,
    batch_size: float,
    arm_idx: int,
    transition_label: str,
    show_ylabel: bool,
    show_y_ticks: bool,
    show_xlabel: bool,
) -> None:
    pi = float(allocation[arm_idx])
    effective_samples = batch_size * pi
    before_active = float(state_before.active[arm_idx].detach().cpu().item())
    after_active = float(state_after.active[arm_idx].detach().cpu().item())
    active_color = "#0f766e"
    inactive_color = "#d1d5db"
    drop_color = "#dc2626"
    color = active_color if before_active > 0 else inactive_color

    ax.set_facecolor("#ffffff" if before_active > 0 else "#f3f4f6")
    ax.bar([0.0], [effective_samples], width=0.5, color=color, edgecolor="#111827", linewidth=0.4)
    ax.axhline(0.0, color="#111827", linewidth=0.7)
    ax.axhline(batch_size, color="#e5e7eb", linewidth=0.7)
    if before_active > 0 and after_active <= 0:
        ax.text(0.97, 0.90, "dropped", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=drop_color)
    ax.text(
        0.05,
        0.88,
        f"New Samples={effective_samples:.2g}\nBatch Fraction={pi:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )

    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(0.0, batch_size)
    ax.set_xticks([])
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    _configure_y_axis(ax, show_ticks=show_y_ticks, label="Sample Size" if show_y_ticks else None)
    if show_ylabel:
        _set_row_label(ax, transition_label)
    if show_xlabel:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("")


def _optimization_max_epochs(histories: Sequence[Sequence[Mapping[str, float]]], configured_max_epochs: Optional[int]) -> int:
    observed = 0
    for history in histories:
        for step in history:
            observed = max(observed, int(step.get("epoch", 0)))
    if configured_max_epochs is not None:
        observed = max(observed, int(configured_max_epochs))
    return max(observed, 1)


def _transition_label(before: GaussianMetricState, after: GaussianMetricState) -> str:
    return f"{before.t} to {after.t}"


def _configure_y_axis(ax: Axes, *, show_ticks: bool = True, label: str | None = None) -> None:
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", left=False, labelleft=False, right=show_ticks, labelright=show_ticks, labelsize=6)
    if label is not None:
        ax.set_ylabel(label, fontsize=8)


def _move_y_axis_right(ax: Axes, label: str | None = None) -> None:
    _configure_y_axis(ax, show_ticks=True, label=label)


def _set_row_label(ax: Axes, label: str) -> None:
    text = ax.text(
        -0.18,
        0.5,
        label,
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=0,
        fontsize=9,
        clip_on=False,
    )
    text.set_in_layout(True)


def _plot_optimization_cell(
    ax: Axes,
    *,
    history: Sequence[Mapping[str, float]],
    max_epochs: int,
    transition_label: str,
    show_xlabel: bool,
) -> None:
    ax.set_facecolor("#ffffff")
    if len(history) == 0:
        _move_y_axis_right(ax)
        ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#6b7280")
        ax.set_xlim(0.5, max_epochs + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    epochs = np.asarray([float(step.get("epoch", idx + 1)) for idx, step in enumerate(history)], dtype=float)
    losses = np.asarray([float(step["loss"]) for step in history], dtype=float)
    ax.plot(epochs, losses, color="#7c3aed", linewidth=1.2)
    ax.scatter(epochs[-1], losses[-1], s=12, color="#7c3aed", zorder=3)
    if int(epochs[-1]) < max_epochs:
        ax.axvspan(epochs[-1], max_epochs, color="#e5e7eb", alpha=0.55, linewidth=0)
        ax.axvline(epochs[-1], color="#9ca3af", linewidth=0.8, linestyle="--")

    low = float(np.nanmin(losses))
    high = float(np.nanmax(losses))
    if np.isfinite(low) and np.isfinite(high):
        if abs(high - low) < 1e-12:
            pad = max(abs(high), 1.0) * 0.05
        else:
            pad = 0.10 * (high - low)
        ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(0.5, max_epochs + 0.5)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=6)
    _move_y_axis_right(ax, label="Loss")
    ax.text(
        0.04,
        0.88,
        f"{transition_label}\n{len(history)}/{max_epochs}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    if show_xlabel:
        ax.set_xlabel("Epoch", fontsize=8)
    else:
        ax.set_xticklabels([])


def _plot_planned_sequence_cell(
    ax: Axes,
    *,
    model: GaussianMetricModel,
    state: GaussianMetricState,
    sequence: np.ndarray,
    show_xlabel: bool,
) -> None:
    ax.set_facecolor("#ffffff")
    if sequence.shape[0] == 0:
        _move_y_axis_right(ax)
        ax.text(0.5, 0.5, "no plan", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#6b7280")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    active = state.active.detach().cpu().numpy() > 0
    active_arm_idx = np.flatnonzero(active)
    if active_arm_idx.size == 0:
        _move_y_axis_right(ax)
        ax.text(0.5, 0.5, "empty active set", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#991b1b")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    start = int(state.t)
    horizon = int(sequence.shape[0])
    budgets = model.batch_sizes[start : start + horizon].detach().cpu().numpy().astype(float)
    if budgets.shape[0] != horizon:
        raise ValueError("planned sequence is longer than the model's remaining batch schedule")
    investments = np.clip(sequence, a_min=0.0, a_max=None) * budgets[:, None]
    planned = investments[:, active_arm_idx]

    x = np.arange(active_arm_idx.size)
    bottoms = np.zeros(active_arm_idx.size)
    colors = _planned_stage_colors(horizon)
    for h in range(horizon):
        ax.bar(
            x,
            planned[h],
            bottom=bottoms,
            width=0.72,
            color=colors[h],
            edgecolor="#111827",
            linewidth=0.25,
        )
        bottoms = bottoms + planned[h]

    max_total = float(np.max(bottoms)) if bottoms.size else 0.0
    ymax = max(max_total, 1e-12) * 1.12
    ax.set_ylim(0.0, ymax)
    ax.set_xlim(-0.6, active_arm_idx.size - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(arm_idx)) for arm_idx in active_arm_idx], fontsize=7)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    _move_y_axis_right(ax, label="Planned Samples")
    if horizon > 1:
        ax.text(
            0.96,
            0.90,
            "bottom=next",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
    if show_xlabel:
        ax.set_xlabel("Arm Index", fontsize=8)


def _planned_stage_colors(horizon: int) -> list[tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    if horizon <= 1:
        return [cmap(0.25)]
    return [cmap(0.18 + 0.70 * h / max(horizon - 1, 1)) for h in range(horizon)]


def _confidence_ellipse(mean: np.ndarray, cov: np.ndarray, radius2: float) -> Ellipse:
    vals, vecs = np.linalg.eigh(_symmetrize(cov))
    vals = np.clip(vals, a_min=0.0, a_max=None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * np.sqrt(vals * radius2)
    return Ellipse(xy=(float(mean[0]), float(mean[1])), width=float(width), height=float(height), angle=angle, fill=False)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
