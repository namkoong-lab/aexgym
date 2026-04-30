from __future__ import annotations

import argparse
import json
from pathlib import Path

from aexgym.experiments.parity.paper_manifest import TABLE1_REFERENCE_PERCENT_OF_UNIFORM


POLICY_ALIASES = {
    "reduced_constant": "rho",
    "ts": "ts",
    "ttts": "ttts",
    "myopic": "myopic",
}


def build_report(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    aggregate_path = results_dir / "aggregates" / "gse_compare_table.json"
    report = {
        "results_dir": str(results_dir),
        "table1": [],
        "rank_order": [],
        "warnings": [],
    }
    if not aggregate_path.exists():
        report["warnings"].append(f"missing aggregate file: {aggregate_path}")
        return report

    payload = json.loads(aggregate_path.read_text())
    aggregates = payload.get("aggregates", {})
    by_scenario_policy = {}
    for row in aggregates.values():
        by_scenario_policy[(row["scenario_id"], row["policy_name"])] = row

    for scenario_id, reference_values in TABLE1_REFERENCE_PERCENT_OF_UNIFORM.items():
        old_order = [name for name, _ in sorted(reference_values.items(), key=lambda item: item[1])]
        new_values = {}
        for policy_name, old_value in reference_values.items():
            new_policy = "reduced_constant" if policy_name == "rho" else policy_name
            row = by_scenario_policy.get((scenario_id, new_policy))
            if row is None:
                report["warnings"].append(f"missing row for {scenario_id}::{new_policy}")
                continue
            new_value = row.get("simple_regret_percent_of_uniform")
            if new_value is not None:
                new_values[policy_name] = new_value
            diff = None if new_value is None else new_value - old_value
            rel_diff = None if new_value is None or old_value == 0 else diff / old_value
            report["table1"].append(
                {
                    "scenario_id": scenario_id,
                    "policy": policy_name,
                    "old_percent_of_uniform": old_value,
                    "new_percent_of_uniform": new_value,
                    "absolute_difference": diff,
                    "relative_difference": rel_diff,
                }
            )
            if rel_diff is not None and abs(rel_diff) > 0.25:
                report["warnings"].append(
                    f"large table1 deviation for {scenario_id}::{policy_name}: old={old_value:.3g}, new={new_value:.3g}"
                )
        new_order = [name for name, _ in sorted(new_values.items(), key=lambda item: item[1])]
        report["rank_order"].append(
            {
                "scenario_id": scenario_id,
                "old_order": old_order,
                "new_order": new_order,
                "complete": set(new_values) == set(reference_values),
                "exact_order_match": new_order == old_order,
                "best_policy_match": bool(new_order and old_order and new_order[0] == old_order[0]),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regenerated parity aggregates against manuscript reference values.")
    parser.add_argument("--results-dir", default="results/parity", help="Directory written by aexgym.experiments.parity.paper.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    report = build_report(args.results_dir)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
