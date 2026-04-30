from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PaperFigureManifestEntry:
    figure_id: str
    manuscript_label: str
    description: str
    old_output_files: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PAPER_FIGURE_MANIFEST: dict[str, PaperFigureManifestEntry] = {
    "bern_batch_bsr": PaperFigureManifestEntry(
        figure_id="bern_batch_bsr",
        manuscript_label="fig:bern-bsr-batch",
        description="Intro Bernoulli Beta experiment comparing batch sizes 100 and 10000.",
        old_output_files=("reg_Bernoulli_100_100_1_Flat.png", "reg_Bernoulli_10000_100_1_Flat.png"),
    ),
    "scaling_bsr": PaperFigureManifestEntry(
        figure_id="scaling_bsr",
        manuscript_label="fig:scaling-bsr",
        description="Simple-regret scaling histograms for Bernoulli and Gamma-Gumbel experiments.",
        old_output_files=("scaling_bern_ts.jpg", "scaling_gumbel_ts.jpg"),
    ),
    "rho_explore": PaperFigureManifestEntry(
        figure_id="rho_explore",
        manuscript_label="fig:rho_explore",
        description="Fixed posterior-state diagnostic comparing TS, one-step RHO, and long-horizon RHO allocations.",
        old_output_files=("prob_comparison_alt.png",),
        notes="If the exact old state fixture is unavailable, this is regenerated as a deterministic diagnostic with the same qualitative purpose.",
    ),
    "gse_compare_table": PaperFigureManifestEntry(
        figure_id="gse_compare_table",
        manuscript_label="table:gse_compare",
        description="Gaussian-policy table reporting simple regret as a percent of uniform.",
        old_output_files=(),
    ),
    "gse_compare": PaperFigureManifestEntry(
        figure_id="gse_compare",
        manuscript_label="fig:gse_compare",
        description="Gaussian sequential experiment comparisons over horizon and measurement variance.",
        old_output_files=("gse_reg_Gumbel_10000_100_1_Flat.png", "gse_bar_var_Gumbel_10000_100_Flat.png"),
    ),
    "num_arm_bsr": PaperFigureManifestEntry(
        figure_id="num_arm_bsr",
        manuscript_label="fig:num_arm_bsr",
        description="Bernoulli Beta experiment comparing K=10 and K=100 arms.",
        old_output_files=("reg_Bernoulli_100_10_1_Flat.png", "reg_Bernoulli_100_100_1_Flat.png"),
    ),
    "bar_var_and_prior": PaperFigureManifestEntry(
        figure_id="bar_var_and_prior",
        manuscript_label="fig:bar_var",
        description="Gamma-Gumbel measurement-noise bars and Bernoulli prior-shape bars.",
        old_output_files=("bar_var_Gumbel_100_100_Flat_no_legend.png", "bar_prior_Bernoulli_100_100_0.25.png"),
    ),
    "gumbel_batch_bsr": PaperFigureManifestEntry(
        figure_id="gumbel_batch_bsr",
        manuscript_label="fig:gumbel_batch_bsr",
        description="Gamma-Gumbel experiment comparing batch sizes 100 and 10000.",
        old_output_files=("reg_Gumbel_100_100_1_Flat.png", "reg_Gumbel_10000_100_1_Flat.png"),
    ),
    "var_perturb": PaperFigureManifestEntry(
        figure_id="var_perturb",
        manuscript_label="fig:var_perturb",
        description="Gamma-Gumbel unknown-variance robustness experiment.",
        old_output_files=("reg_Gumbel_10000_100_1_Flat_var.png",),
    ),
    "horizon_misspecification": PaperFigureManifestEntry(
        figure_id="horizon_misspecification",
        manuscript_label="fig:horizon-misspecification",
        description="Gaussian experiment where RHO plans with T'=5 or T'=10 while actual horizon varies.",
        old_output_files=("focused_simple_regret_analysis.png", "simple_regret_comparison.png"),
    ),
    "asos": PaperFigureManifestEntry(
        figure_id="asos",
        manuscript_label="fig:asos",
        description="ASOS normal-reward experiments with standard, decreasing, and variance-estimated batches.",
        old_output_files=(
            "avg_simple_regret_standard_False.png",
            "avg_simple_regret_decreasing_False.png",
            "avg_simple_regret_decreasing_True.png",
            "instances_standard_8_False.png",
            "instances_decreasing_8_False.png",
            "instances_decreasing_8_True.png",
        ),
    ),
    "asos_nonstationary": PaperFigureManifestEntry(
        figure_id="asos_nonstationary",
        manuscript_label="asos_nonstationary_text",
        description="Text/table replication of the nonstationary ASOS comparison.",
        old_output_files=(),
    ),
    "alt_gse_compare": PaperFigureManifestEntry(
        figure_id="alt_gse_compare",
        manuscript_label="fig:alt_gse_compare",
        description="Appendix Gaussian sequential experiment comparing alternative policies.",
        old_output_files=("alt_reg_Gumbel_10000_100_1.0_Flat.png", "alt_gse_bar_var_Gumbel_10000_100_Flat.png"),
    ),
}


TABLE1_REFERENCE_PERCENT_OF_UNIFORM: dict[str, dict[str, float]] = {
    "baseline_gumbel_k10_batch100_s2_1_flat": {"ts": 70.1, "ttts": 68.7, "myopic": 66.5, "rho": 58.8},
    "bernoulli_k10_batch100_s2_0.25_flat": {"ts": 80.9, "ttts": 82.1, "myopic": 76.0, "rho": 73.1},
    "gumbel_k100_batch100_s2_1_flat": {"ts": 73.4, "ttts": 74.1, "myopic": 64.3, "rho": 55.8},
    "gumbel_k10_batch10000_s2_1_flat": {"ts": 69.1, "ttts": 71.8, "myopic": 69.1, "rho": 64.2},
    "gumbel_k10_batch100_s2_0.2_flat": {"ts": 51.6, "ttts": 46.8, "myopic": 49.7, "rho": 41.5},
    "gumbel_k10_batch100_s2_5_flat": {"ts": 89.1, "ttts": 88.9, "myopic": 80.2, "rho": 78.6},
    "gumbel_k10_batch100_s2_1_top_one": {"ts": 60.0, "ttts": 59.3, "myopic": 60.0, "rho": 52.6},
    "gumbel_k10_batch100_s2_1_descending": {"ts": 62.8, "ttts": 65.0, "myopic": 63.6, "rho": 56.6},
}


def manifest_as_dict() -> dict[str, dict]:
    return {figure_id: entry.to_dict() for figure_id, entry in PAPER_FIGURE_MANIFEST.items()}
