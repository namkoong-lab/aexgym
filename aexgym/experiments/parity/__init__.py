"""Parity experiments for reproducing the previous paper with the new code."""

from .rho_variants import DEFAULT_PARITY_RHO_VARIANTS, PARITY_RHO_VARIANTS, ParityRhoVariantSpec, build_rho_policy
from .scalar import BASELINE_POLICIES, PARITY_SCENARIOS, ScalarParityConfig, make_policies, run_config
from .suite import PARITY_EXPERIMENT_GROUPS, ParitySweepConfig, run_sweep
from .paper_manifest import PAPER_FIGURE_MANIFEST, TABLE1_REFERENCE_PERCENT_OF_UNIFORM

__all__ = [
    "BASELINE_POLICIES",
    "DEFAULT_PARITY_RHO_VARIANTS",
    "PAPER_FIGURE_MANIFEST",
    "PARITY_EXPERIMENT_GROUPS",
    "PARITY_RHO_VARIANTS",
    "PARITY_SCENARIOS",
    "ParityRhoVariantSpec",
    "ParitySweepConfig",
    "ScalarParityConfig",
    "TABLE1_REFERENCE_PERCENT_OF_UNIFORM",
    "build_rho_policy",
    "make_policies",
    "run_config",
    "run_sweep",
]
