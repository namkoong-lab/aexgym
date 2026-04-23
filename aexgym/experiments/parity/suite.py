from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aexgym.experiments.parity.rho_variants import DEFAULT_PARITY_RHO_VARIANTS
from aexgym.experiments.parity.scalar import BASELINE_POLICIES, PARITY_SCENARIOS, ScalarParityConfig, run_config


PARITY_EXPERIMENT_GROUPS: dict[str, list[str]] = {
    "synthetic_scalar": sorted(PARITY_SCENARIOS),
}


@dataclass
class ParitySweepConfig:
    scenarios: list[str] = field(default_factory=lambda: list(PARITY_EXPERIMENT_GROUPS["synthetic_scalar"]))
    n_arms: int = 10
    horizon: int = 5
    batch_size: float = 1.0
    n_runs: int = 20
    seed: int = 1
    prior_mean: float = 0.0
    prior_var: float = 0.1
    obs_var: float = 1.0
    mean_scale: float = 0.1
    prior_pattern: str = "flat"
    baseline_policies: list[str] = field(default_factory=lambda: list(BASELINE_POLICIES))
    rho_variants: list[str] = field(default_factory=lambda: list(DEFAULT_PARITY_RHO_VARIANTS))
    rho_epochs: int = 60
    rho_num_samples: int = 256
    rho_lr: float = 0.08
    rho_temporal_regularization: float = 300.0
    output: str | None = None


def load_config(path: str | None) -> ParitySweepConfig:
    if path is None:
        return ParitySweepConfig()
    return ParitySweepConfig(**json.loads(Path(path).read_text()))


def build_scalar_config(config: ParitySweepConfig, scenario: str) -> ScalarParityConfig:
    return ScalarParityConfig(
        scenario=scenario,
        n_arms=config.n_arms,
        horizon=config.horizon,
        batch_size=config.batch_size,
        n_runs=config.n_runs,
        seed=config.seed,
        prior_mean=config.prior_mean,
        prior_var=config.prior_var,
        obs_var=config.obs_var,
        mean_scale=config.mean_scale,
        prior_pattern=config.prior_pattern,
        baseline_policies=list(config.baseline_policies),
        rho_variants=list(config.rho_variants),
        rho_epochs=config.rho_epochs,
        rho_num_samples=config.rho_num_samples,
        rho_lr=config.rho_lr,
        rho_temporal_regularization=config.rho_temporal_regularization,
    )


def run_sweep(config: ParitySweepConfig) -> dict:
    invalid = sorted(set(config.scenarios) - set(PARITY_SCENARIOS))
    if invalid:
        raise ValueError(f"unknown parity scenarios: {invalid}")

    scenario_results = {}
    rho_variant_summaries = {}
    for scenario in config.scenarios:
        payload = run_config(build_scalar_config(config, scenario))
        scenario_results[scenario] = payload
        rho_variant_summaries[scenario] = {
            variant: payload["aggregates"][variant]
            for variant in config.rho_variants
            if variant in payload["aggregates"]
        }

    return {
        "config": asdict(config),
        "groups": PARITY_EXPERIMENT_GROUPS,
        "scenario_results": scenario_results,
        "rho_variant_summaries": rho_variant_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the parity sweep across the maintained old-paper scenarios.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print available parity scenario names.")
    parser.add_argument("--list-rho-variants", action="store_true", help="Print available parity RHO variants.")
    args = parser.parse_args()

    if args.list_scenarios:
        print(json.dumps(sorted(PARITY_SCENARIOS), indent=2))
        return
    if args.list_rho_variants:
        print(json.dumps(list(DEFAULT_PARITY_RHO_VARIANTS), indent=2))
        return

    config = load_config(args.config)
    if args.output is not None:
        config.output = args.output
    payload = run_sweep(config)
    output_text = json.dumps(payload, indent=2)
    if config.output:
        Path(config.output).write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
