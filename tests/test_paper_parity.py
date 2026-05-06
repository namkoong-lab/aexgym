import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from aexgym.core import GaussianMetricModel
from aexgym.experiments.parity.paper import PAPER_PROFILES, PaperParityRunConfig, build_scenarios, run_figures
from aexgym.experiments.parity.paper_manifest import PAPER_FIGURE_MANIFEST
from aexgym.experiments.parity.paper_report import build_report
from aexgym.experiments.parity.paper_simulation import (
    aggregate_records,
    load_asos_settings,
    make_instance,
    make_model,
    run_paper_scenario,
)
from aexgym.policies import ReducedTerminalRhoSimulation


def test_paper_manifest_contains_all_empirical_figures():
    expected = {
        "bern_batch_bsr",
        "scaling_bsr",
        "rho_explore",
        "gse_compare_table",
        "gse_compare",
        "num_arm_bsr",
        "bar_var_and_prior",
        "gumbel_batch_bsr",
        "var_perturb",
        "horizon_misspecification",
        "asos",
        "asos_nonstationary",
        "alt_gse_compare",
    }

    assert set(PAPER_FIGURE_MANIFEST) == expected


def test_paper_scenarios_use_j1_target_metric_for_gaussian_policy_path():
    profile = PAPER_PROFILES["smoke"]
    scenario = build_scenarios("gse_compare_table", profile, seed=0)[0]
    instance = make_instance(scenario, seed=0)
    model = make_model(instance)

    assert isinstance(model, GaussianMetricModel)
    assert model.n_metrics == 1
    assert model.target_metric_idx == 0


def test_sobol_reduced_terminal_prepare_is_deterministic():
    model = GaussianMetricModel(
        prior_mean=torch.zeros((3, 1), dtype=torch.float64),
        prior_cov=torch.eye(1, dtype=torch.float64).repeat(3, 1, 1),
        obs_cov=torch.eye(1, dtype=torch.float64).repeat(3, 1, 1),
        target_metric_idx=0,
        batch_sizes=torch.ones(2, dtype=torch.float64),
    )
    state = model.initial_state()
    sim_a = ReducedTerminalRhoSimulation(sample_method="sobol", sobol_seed=42)
    sim_b = ReducedTerminalRhoSimulation(sample_method="sobol", sobol_seed=42)

    assert torch.allclose(sim_a.prepare(state, model, 2, 8, None), sim_b.prepare(state, model, 2, 8, None))


def test_tiny_paper_scenario_run_records_target_metric_and_aggregates():
    profile = PAPER_PROFILES["smoke"]
    scenario = build_scenarios("gse_compare_table", profile, seed=0)[0]
    scenario = replace(scenario, n_arms=3, n_runs=1, policies=("uniform", "ts"))

    records = run_paper_scenario(
        scenario,
        policy_names=scenario.policies,
        rho_variants=["reduced_constant"],
        seed=3,
        profile=profile,
    )
    aggregates = aggregate_records(records)

    assert {record.policy_name for record in records} == {"uniform", "ts", "reduced_constant"}
    assert all(record.model_n_metrics == 1 and record.target_metric_idx == 0 for record in records)
    assert all("selected_true_best_rate" in row for row in aggregates.values())


def test_asos_loader_returns_settings_from_surviving_csv():
    settings = load_asos_settings(limit=2)

    assert len(settings) == 2
    assert len(settings[0].mean_c) > 0
    assert len(settings[0].mean_c) == len(settings[0].variance_c)


def test_paper_cli_smoke_writes_structured_outputs(tmp_path):
    output_dir = tmp_path / "parity"
    summary = run_figures(
        PaperParityRunConfig(
            figures=["gse_compare_table"],
            profile="smoke",
            rho_variants=["reduced_constant"],
            output_dir=str(output_dir),
            seed=0,
            skip_plots=True,
            policies=["uniform", "ts"],
        )
    )

    aggregate_path = Path(summary["results"]["gse_compare_table"]["aggregate_path"])
    raw_path = Path(summary["results"]["gse_compare_table"]["raw_path"])
    assert aggregate_path.exists()
    assert raw_path.exists()
    payload = json.loads(aggregate_path.read_text())
    assert payload["manifest"]["manuscript_label"] == "table:gse_compare"


def test_paper_report_handles_smoke_outputs(tmp_path):
    output_dir = tmp_path / "parity"
    run_figures(
        PaperParityRunConfig(
            figures=["gse_compare_table"],
            profile="smoke",
            rho_variants=["reduced_constant"],
            output_dir=str(output_dir),
            seed=0,
            skip_plots=True,
            policies=["uniform", "ts"],
        )
    )

    report = build_report(output_dir)

    assert "table1" in report
    assert report["table1"]


def test_paper_cli_help_smoke():
    completed = subprocess.run(
        [sys.executable, "-m", "aexgym.experiments.parity.paper", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--figure" in completed.stdout
