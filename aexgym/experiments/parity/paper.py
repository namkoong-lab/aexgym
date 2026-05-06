from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from aexgym.experiments.parity.paper_manifest import PAPER_FIGURE_MANIFEST, manifest_as_dict
from aexgym.experiments.parity.paper_simulation import (
    PAPER_PROFILES,
    POLICY_LABELS,
    PaperProfile,
    PaperRunRecord,
    PaperScenario,
    aggregate_records,
    default_asos_data_path,
    load_asos_settings,
    run_paper_scenario,
)
from aexgym.experiments.parity.rho_variants import DEFAULT_PARITY_RHO_VARIANTS


DEFAULT_PAPER_RHO_VARIANTS: tuple[str, ...] = DEFAULT_PARITY_RHO_VARIANTS


@dataclass
class PaperParityRunConfig:
    figures: list[str] = field(default_factory=lambda: ["all"])
    profile: str = "smoke"
    rho_variants: list[str] = field(default_factory=lambda: list(DEFAULT_PAPER_RHO_VARIANTS))
    output_dir: str = "results/parity"
    num_workers: int = 1
    seed: int = 0
    skip_plots: bool = False
    policies: Optional[list[str]] = None
    data_path: Optional[str] = None


def resolve_figures(figures: Iterable[str]) -> list[str]:
    requested = list(figures)
    if not requested or requested == ["all"] or "all" in requested:
        return list(PAPER_FIGURE_MANIFEST)
    invalid = sorted(set(requested) - set(PAPER_FIGURE_MANIFEST))
    if invalid:
        raise ValueError(f"unknown paper figure ids: {invalid}")
    return requested


def build_scenarios(figure_id: str, profile: PaperProfile, seed: int, data_path: Optional[Path] = None) -> list[PaperScenario]:
    n_runs = profile.n_runs
    horizons = profile.horizons
    if figure_id == "bern_batch_bsr":
        return [
            PaperScenario(
                scenario_id=f"bern_k100_batch{batch}_h{horizon}_flat",
                figure_id=figure_id,
                reward_model="bernoulli",
                n_arms=100,
                horizon=horizon,
                batch_size=batch,
                n_runs=n_runs,
                obs_var=0.25,
                prior_var=0.1,
                prior_pattern="flat",
                policies=("uniform", "ts", "ttts", "myopic", "oracle_beta_ts", "oracle_beta_ttts", "successive_elimination"),
            )
            for batch in (100.0, 10000.0)
            for horizon in horizons
        ]
    if figure_id == "scaling_bsr":
        scenarios: list[PaperScenario] = []
        for reward_model, obs_var in (("bernoulli", 0.25), ("gumbel", 1.0), ("bernoulli_gaussian", 0.25), ("gumbel_gaussian", 1.0)):
            for batch in (100.0, 1000.0, 10000.0):
                scenarios.append(
                    PaperScenario(
                        scenario_id=f"{reward_model}_scaling_k100_batch{batch:g}_h{max(horizons)}",
                        figure_id=figure_id,
                        reward_model=reward_model,
                        n_arms=100,
                        horizon=max(horizons),
                        batch_size=batch,
                        n_runs=n_runs,
                        obs_var=obs_var,
                        prior_var=0.1 if "bernoulli" in reward_model else 1.0,
                        prior_pattern="flat",
                        policies=("uniform", "ts"),
                    )
                )
        return scenarios
    if figure_id == "rho_explore":
        return [
            PaperScenario(
                scenario_id=f"rho_explore_k5_h{horizon}",
                figure_id=figure_id,
                reward_model="gaussian",
                n_arms=5,
                horizon=horizon,
                batch_size=1.0,
                n_runs=1,
                obs_var=1.0,
                prior_var=0.25,
                prior_pattern="descending",
                policies=("ts",),
                metadata={"diagnostic": True, "prior_mean_perturb": 0.35},
            )
            for horizon in (1, 10)
        ]
    if figure_id == "gse_compare_table":
        return [
            _table_scenario("baseline_gumbel_k10_batch100_s2_1_flat", "gumbel_gaussian", 10, 100.0, 1.0, "flat", n_runs, max(horizons)),
            _table_scenario("bernoulli_k10_batch100_s2_0.25_flat", "bernoulli_gaussian", 10, 100.0, 0.25, "flat", n_runs, max(horizons)),
            _table_scenario("gumbel_k100_batch100_s2_1_flat", "gumbel_gaussian", 100, 100.0, 1.0, "flat", n_runs, max(horizons)),
            _table_scenario("gumbel_k10_batch10000_s2_1_flat", "gumbel_gaussian", 10, 10000.0, 1.0, "flat", n_runs, max(horizons)),
            _table_scenario("gumbel_k10_batch100_s2_0.2_flat", "gumbel_gaussian", 10, 100.0, 0.2, "flat", n_runs, max(horizons)),
            _table_scenario("gumbel_k10_batch100_s2_5_flat", "gumbel_gaussian", 10, 100.0, 5.0, "flat", n_runs, max(horizons)),
            _table_scenario("gumbel_k10_batch100_s2_1_top_one", "gumbel_gaussian", 10, 100.0, 1.0, "top_one", n_runs, max(horizons)),
            _table_scenario("gumbel_k10_batch100_s2_1_descending", "gumbel_gaussian", 10, 100.0, 1.0, "descending", n_runs, max(horizons)),
        ]
    if figure_id == "gse_compare":
        scenarios = [
            PaperScenario(
                scenario_id=f"gse_gumbel_k100_batch10000_s2_1_h{horizon}",
                figure_id=figure_id,
                reward_model="gumbel_gaussian",
                n_arms=100,
                horizon=horizon,
                batch_size=10000.0,
                n_runs=n_runs,
                obs_var=1.0,
                prior_var=1.0,
                prior_pattern="flat",
                policies=("uniform", "ts", "ttts", "myopic"),
            )
            for horizon in horizons
        ]
        scenarios.extend(
            PaperScenario(
                scenario_id=f"gse_gumbel_k100_batch10000_s2_{obs_var:g}_h10",
                figure_id=figure_id,
                reward_model="gumbel_gaussian",
                n_arms=100,
                horizon=max(horizons),
                batch_size=10000.0,
                n_runs=n_runs,
                obs_var=obs_var,
                prior_var=1.0,
                prior_pattern="flat",
                policies=("uniform", "ts", "ttts", "myopic"),
                metadata={"panel": "variance"},
            )
            for obs_var in (0.2, 1.0, 5.0)
        )
        return scenarios
    if figure_id == "num_arm_bsr":
        return [
            PaperScenario(
                scenario_id=f"bern_k{k}_batch100_h{horizon}_flat",
                figure_id=figure_id,
                reward_model="bernoulli",
                n_arms=k,
                horizon=horizon,
                batch_size=100.0,
                n_runs=n_runs,
                obs_var=0.25,
                prior_var=0.1,
                prior_pattern="flat",
                policies=("uniform", "ts", "ttts", "myopic", "oracle_beta_ts", "oracle_beta_ttts", "successive_elimination"),
            )
            for k in (10, 100)
            for horizon in horizons
        ]
    if figure_id == "bar_var_and_prior":
        scenarios = [
            PaperScenario(
                scenario_id=f"gumbel_k100_batch100_s2_{obs_var:g}_h10_flat",
                figure_id=figure_id,
                reward_model="gumbel",
                n_arms=100,
                horizon=max(horizons),
                batch_size=100.0,
                n_runs=n_runs,
                obs_var=obs_var,
                prior_var=1.0,
                prior_pattern="flat",
                policies=("uniform", "ts", "ttts", "myopic", "successive_elimination"),
                metadata={"panel": "variance"},
            )
            for obs_var in (0.2, 1.0, 5.0)
        ]
        scenarios.extend(
            PaperScenario(
                scenario_id=f"bern_k100_batch100_h10_{prior}",
                figure_id=figure_id,
                reward_model="bernoulli",
                n_arms=100,
                horizon=max(horizons),
                batch_size=100.0,
                n_runs=n_runs,
                obs_var=0.25,
                prior_var=0.1,
                prior_pattern=prior,
                policies=("uniform", "ts", "ttts", "myopic", "oracle_beta_ts", "oracle_beta_ttts", "successive_elimination"),
                metadata={"panel": "prior"},
            )
            for prior in ("flat", "top_one", "top_half", "descending")
        )
        return scenarios
    if figure_id == "gumbel_batch_bsr":
        return [
            PaperScenario(
                scenario_id=f"gumbel_k100_batch{batch:g}_s2_1_h{horizon}_flat",
                figure_id=figure_id,
                reward_model="gumbel",
                n_arms=100,
                horizon=horizon,
                batch_size=batch,
                n_runs=n_runs,
                obs_var=1.0,
                prior_var=1.0,
                prior_pattern="flat",
                policies=("uniform", "ts", "ttts", "myopic", "successive_elimination"),
            )
            for batch in (100.0, 10000.0)
            for horizon in horizons
        ]
    if figure_id == "var_perturb":
        return [
            PaperScenario(
                scenario_id=f"gumbel_k100_batch10000_s2_1_varsigma_{sigma:g}",
                figure_id=figure_id,
                reward_model="gumbel",
                n_arms=100,
                horizon=max(horizons),
                batch_size=10000.0,
                n_runs=n_runs,
                obs_var=1.0,
                prior_var=1.0,
                prior_pattern="flat",
                variance_perturbation=sigma,
                policies=("uniform", "ts", "ttts", "myopic", "successive_elimination"),
            )
            for sigma in (0.25, 0.5, 1.0)
        ]
    if figure_id == "horizon_misspecification":
        return [
            PaperScenario(
                scenario_id=f"gaussian_hactual{actual}_plan{planned}",
                figure_id=figure_id,
                reward_model="gaussian",
                n_arms=100,
                horizon=actual,
                batch_size=100.0,
                n_runs=n_runs,
                obs_var=0.2,
                prior_var=1e-3,
                prior_pattern="flat",
                planning_horizon=planned,
                policies=("uniform", "ts"),
            )
            for actual in horizons
            for planned in (5, 10)
        ]
    if figure_id in {"asos", "asos_nonstationary"}:
        settings = load_asos_settings(data_path=data_path or default_asos_data_path(), limit=profile.max_asos_settings)
        schedules = (
            ("standard", "poisson", 10000.0, False),
            ("decreasing", "decreasing", 2000.0, False),
            ("decreasing_var", "decreasing", 2000.0, True),
        )
        if figure_id == "asos_nonstationary":
            schedules = (("nonstationary", "poisson", 10000.0, False),)
        scenarios = []
        for setting_idx, _setting in enumerate(settings):
            for horizon in horizons:
                for schedule_name, schedule, batch_size, estimate_var in schedules:
                    scenarios.append(
                        PaperScenario(
                            scenario_id=f"{figure_id}_setting{setting_idx}_{schedule_name}_h{horizon}",
                            figure_id=figure_id,
                            reward_model="asos",
                            n_arms=10,
                            horizon=horizon,
                            batch_size=batch_size,
                            n_runs=n_runs,
                            batch_schedule=schedule,
                            estimate_variance_first_batch=estimate_var,
                            asos_setting_index=setting_idx,
                            nonstationary=figure_id == "asos_nonstationary",
                            policies=("uniform", "ts", "ttts", "myopic"),
                            metadata={"schedule": schedule_name, "setting_index": setting_idx},
                        )
                    )
        return scenarios
    if figure_id == "alt_gse_compare":
        return [
            PaperScenario(
                scenario_id=f"alt_gse_gumbel_k100_batch10000_s2_{obs_var:g}_h{horizon}",
                figure_id=figure_id,
                reward_model="gumbel_gaussian",
                n_arms=100,
                horizon=horizon,
                batch_size=10000.0,
                n_runs=n_runs,
                obs_var=obs_var,
                prior_var=1.0,
                prior_pattern="flat",
                policies=("uniform", "ts", "ttts", "myopic", "ts_plus"),
            )
            for obs_var in (1.0, 0.2, 5.0)
            for horizon in horizons
        ]
    raise ValueError(f"unknown paper figure id: {figure_id}")


def _table_scenario(
    scenario_id: str,
    reward_model: str,
    n_arms: int,
    batch_size: float,
    obs_var: float,
    prior_pattern: str,
    n_runs: int,
    horizon: int,
) -> PaperScenario:
    return PaperScenario(
        scenario_id=scenario_id,
        figure_id="gse_compare_table",
        reward_model=reward_model,
        n_arms=n_arms,
        horizon=horizon,
        batch_size=batch_size,
        n_runs=n_runs,
        obs_var=obs_var,
        prior_var=0.1 if "bernoulli" in reward_model else 1.0,
        prior_pattern=prior_pattern,
        policies=("uniform", "ts", "ttts", "myopic"),
    )


def run_figures(config: PaperParityRunConfig) -> dict:
    profile = PAPER_PROFILES[config.profile]
    figures = resolve_figures(config.figures)
    data_path = None if config.data_path is None else Path(config.data_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "aggregates").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest_as_dict(), indent=2))

    all_payload: dict[str, dict] = {}
    for figure_id in figures:
        scenarios = build_scenarios(figure_id, profile, seed=config.seed, data_path=data_path)
        records: list[PaperRunRecord] = []
        for scenario in scenarios:
            policy_names = list(config.policies) if config.policies is not None else list(scenario.policies)
            records.extend(
                run_paper_scenario(
                    scenario,
                    policy_names=policy_names,
                    rho_variants=config.rho_variants,
                    seed=config.seed,
                    profile=profile,
                    data_path=data_path,
                )
            )
        aggregates = aggregate_records(records)
        figure_payload = {
            "figure_id": figure_id,
            "manifest": PAPER_FIGURE_MANIFEST[figure_id].to_dict(),
            "profile": asdict(profile),
            "scenarios": [asdict(scenario) for scenario in scenarios],
            "aggregates": aggregates,
            "rho_variant_summary": {
                key: row
                for key, row in aggregates.items()
                if row["policy_name"] in DEFAULT_PAPER_RHO_VARIANTS
            },
            "raw_path": str(output_dir / "raw" / f"{figure_id}.jsonl"),
            "aggregate_path": str(output_dir / "aggregates" / f"{figure_id}.json"),
            "aggregate_csv_path": str(output_dir / "aggregates" / f"{figure_id}.csv"),
        }
        _write_records(output_dir / "raw" / f"{figure_id}.jsonl", records)
        _write_aggregate_csv(output_dir / "aggregates" / f"{figure_id}.csv", list(aggregates.values()))
        (output_dir / "aggregates" / f"{figure_id}.json").write_text(json.dumps(figure_payload, indent=2))
        if not config.skip_plots:
            figure_payload["plot_path"] = _write_plot(output_dir / "figures" / f"{figure_id}.png", figure_id, list(aggregates.values()))
        all_payload[figure_id] = figure_payload

    summary = {
        "config": asdict(config),
        "figures": figures,
        "profile": asdict(profile),
        "output_dir": str(output_dir),
        "results": {
            figure_id: {
                "n_scenarios": len(payload["scenarios"]),
                "n_aggregate_rows": len(payload["aggregates"]),
                "aggregate_path": payload["aggregate_path"],
                "aggregate_csv_path": payload["aggregate_csv_path"],
                "raw_path": payload["raw_path"],
                "plot_path": payload.get("plot_path"),
            }
            for figure_id, payload in all_payload.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_records(path: Path, records: list[PaperRunRecord]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")


def _write_aggregate_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "figure_id",
        "scenario_id",
        "policy_name",
        "policy_label",
        "n_runs",
        "mean_simple_regret",
        "se_simple_regret",
        "selected_true_best_rate",
        "simple_regret_percent_of_uniform",
        "model_n_metrics",
        "target_metric_idx",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_plot(path: Path, figure_id: str, rows: list[dict]) -> dict[str, str] | str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised only when optional plotting import is unavailable.
        return f"plot skipped: {exc}"
    if not rows:
        return "plot skipped: no rows"
    policies = sorted({row["policy_name"] for row in rows})
    has_horizon = all("horizon" in row.get("metadata", {}) for row in rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    y_key = "simple_regret_percent_of_uniform"
    if has_horizon:
        for policy in policies:
            policy_rows = sorted([row for row in rows if row["policy_name"] == policy], key=lambda row: (row["metadata"].get("horizon", 0), row["scenario_id"]))
            xs = [row["metadata"].get("horizon", idx) for idx, row in enumerate(policy_rows)]
            ys = [row.get(y_key) if row.get(y_key) is not None else row["mean_simple_regret"] for row in policy_rows]
            ax.plot(xs, ys, marker="o", label=POLICY_LABELS.get(policy, policy))
        ax.set_xlabel("Horizon")
    else:
        labels = [row["scenario_id"] for row in rows]
        xs = range(len(rows))
        ys = [row.get(y_key) if row.get(y_key) is not None else row["mean_simple_regret"] for row in rows]
        ax.bar(xs, ys)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Simple regret (% of uniform)" if any(row.get(y_key) is not None for row in rows) else "Simple regret")
    ax.set_title(figure_id)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(path), "pdf": str(pdf_path)}


def parse_csv_list(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run maintained paper-figure parity experiments.")
    parser.add_argument("--figure", action="append", default=None, help="Figure id to run. Repeatable. Use 'all' for every figure.")
    parser.add_argument("--profile", choices=sorted(PAPER_PROFILES), default="smoke", help="Run profile.")
    parser.add_argument("--rho-variants", default=",".join(DEFAULT_PAPER_RHO_VARIANTS), help="Comma-separated RHO variants to run.")
    parser.add_argument("--policies", default=None, help="Optional comma-separated non-RHO policy override.")
    parser.add_argument("--output-dir", default="results/parity", help="Output directory.")
    parser.add_argument("--num-workers", type=int, default=1, help="Reserved for future parallel execution; current implementation runs serially.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--data-path", default=None, help="Optional ASOS CSV path.")
    parser.add_argument("--skip-plots", action="store_true", help="Write JSON outputs without plotting.")
    parser.add_argument("--list-figures", action="store_true", help="Print figure manifest and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print scenario plan without running simulations.")
    args = parser.parse_args()

    if args.list_figures:
        print(json.dumps(manifest_as_dict(), indent=2))
        return

    figures = resolve_figures(args.figure or ["all"])
    rho_variants = parse_csv_list(args.rho_variants)
    policies = parse_csv_list(args.policies) or None
    config = PaperParityRunConfig(
        figures=figures,
        profile=args.profile,
        rho_variants=rho_variants,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        seed=args.seed,
        skip_plots=args.skip_plots,
        policies=policies,
        data_path=args.data_path,
    )

    if args.dry_run:
        profile = PAPER_PROFILES[config.profile]
        data_path = None if config.data_path is None else Path(config.data_path)
        payload = {
            "config": asdict(config),
            "scenarios": {
                figure_id: [asdict(scenario) for scenario in build_scenarios(figure_id, profile, seed=config.seed, data_path=data_path)]
                for figure_id in figures
            },
        }
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(run_figures(config), indent=2))


if __name__ == "__main__":
    main()
