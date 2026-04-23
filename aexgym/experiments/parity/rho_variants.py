from __future__ import annotations

from dataclasses import dataclass

from aexgym.core import ActiveSetRule
from aexgym.policies import (
    BasePlusResidualLogitParameterization,
    ConstantAllocationParameterization,
    FreeSequenceParameterization,
    NoSequenceRegularizer,
    PathwiseStoppedRhoSimulation,
    ReducedTerminalRhoSimulation,
    RhoPolicy,
    TemporalUniformityRegularizer,
)


@dataclass(frozen=True)
class ParityRhoVariantSpec:
    name: str
    description: str
    simulator: str
    parameterization: str
    regularizer: str


PARITY_RHO_VARIANTS: dict[str, ParityRhoVariantSpec] = {
    "reduced_constant": ParityRhoVariantSpec(
        name="reduced_constant",
        description="Paper reduction with a single constant allocation over the residual horizon.",
        simulator="reduced",
        parameterization="constant",
        regularizer="none",
    ),
    "pathwise_constant": ParityRhoVariantSpec(
        name="pathwise_constant",
        description="Pathwise rollout simulation with a constant allocation parameterization.",
        simulator="pathwise",
        parameterization="constant",
        regularizer="none",
    ),
    "pathwise_base_residual_regularized": ParityRhoVariantSpec(
        name="pathwise_base_residual_regularized",
        description="Pathwise rollout simulation with base-plus-residual logits and temporal uniformity regularization.",
        simulator="pathwise",
        parameterization="base_residual",
        regularizer="temporal_uniformity",
    ),
    "pathwise_free_sequence_regularized": ParityRhoVariantSpec(
        name="pathwise_free_sequence_regularized",
        description="Pathwise rollout simulation with a free allocation sequence and temporal uniformity regularization.",
        simulator="pathwise",
        parameterization="free_sequence",
        regularizer="temporal_uniformity",
    ),
}

DEFAULT_PARITY_RHO_VARIANTS: tuple[str, ...] = tuple(PARITY_RHO_VARIANTS)


def build_rho_policy(
    variant: str,
    *,
    target_idx: int = 0,
    epochs: int,
    num_samples: int,
    lr: float,
    temporal_regularization: float,
    name: str | None = None,
) -> RhoPolicy:
    spec = PARITY_RHO_VARIANTS[variant]
    if spec.simulator == "reduced":
        simulator = ReducedTerminalRhoSimulation()
    elif spec.simulator == "pathwise":
        simulator = PathwiseStoppedRhoSimulation(ActiveSetRule(target_idx=target_idx, stop_on_singleton=False))
    else:
        raise ValueError(f"unknown simulator: {spec.simulator}")

    if spec.parameterization == "constant":
        parameterization = ConstantAllocationParameterization()
    elif spec.parameterization == "base_residual":
        parameterization = BasePlusResidualLogitParameterization()
    elif spec.parameterization == "free_sequence":
        parameterization = FreeSequenceParameterization()
    else:
        raise ValueError(f"unknown parameterization: {spec.parameterization}")

    if spec.regularizer == "none":
        regularizer = NoSequenceRegularizer()
    elif spec.regularizer == "temporal_uniformity":
        regularizer = TemporalUniformityRegularizer(weight=temporal_regularization)
    else:
        raise ValueError(f"unknown regularizer: {spec.regularizer}")

    return RhoPolicy(
        simulator=simulator,
        parameterization=parameterization,
        regularizer=regularizer,
        epochs=epochs,
        num_samples=num_samples,
        lr=lr,
        name=variant if name is None else name,
    )
