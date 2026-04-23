from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "aexgym"


def test_import_smoke_for_supported_namespaces():
    for module_name in [
        "aexgym",
        "aexgym.core",
        "aexgym.policies",
        "aexgym.experiments",
        "aexgym.experiments.parity",
        "aexgym.experiments.revision",
        "aexgym.legacy_parity",
    ]:
        importlib.import_module(module_name)


def test_maintained_code_does_not_import_removed_or_legacy_namespaces():
    forbidden = [
        "aexgym.agent",
        "aexgym.env",
        "aexgym.model",
        "aexgym.objectives",
        "aexgym.metric",
        "aexgym.legacy_parity",
    ]
    for subdir in ["core", "policies", "experiments"]:
        for path in (PACKAGE_ROOT / subdir).rglob("*.py"):
            text = path.read_text()
            for pattern in forbidden:
                assert pattern not in text, f"{path} still references {pattern}"


def test_cli_help_smoke():
    modules = [
        "aexgym.experiments.parity.scalar",
        "aexgym.experiments.parity.suite",
        "aexgym.experiments.revision.basic_guardrail",
        "aexgym.legacy_parity.scalar_replication",
        "aexgym.legacy_parity.compare_adaptive",
        "aexgym.legacy_parity.variant_check",
    ]
    for module_name in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
