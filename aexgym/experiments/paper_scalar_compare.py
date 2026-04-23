from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import io
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from aexgym.experiments.paper_scalar_replication import (
    PaperScalarConfig,
    load_config,
    make_policies,
    paper_priors,
    run_trial,
)


def install_legacy_import_stubs() -> None:
    """Stub legacy dependencies unused by direct simulator.eval_instance calls."""

    sys.modules.setdefault("cvxpy", types.ModuleType("cvxpy"))
    pathos = sys.modules.setdefault("pathos", types.ModuleType("pathos"))
    multiprocessing = types.ModuleType("pathos.multiprocessing")

    class ProcessingPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "ProcessingPool":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def amap(self, fn, jobs):
            class Result:
                def get(self_nonlocal):
                    return [fn(job) for job in jobs]

            return Result()

    multiprocessing.ProcessingPool = ProcessingPool
    pathos.multiprocessing = multiprocessing
    sys.modules.setdefault("pathos.multiprocessing", multiprocessing)


def import_legacy_simulator(adaptive_root: Path):
    install_legacy_import_stubs()
    mab_path = adaptive_root / "MAB"
    if not (mab_path / "simulator.py").exists():
        raise FileNotFoundError(f"could not find legacy simulator at {mab_path / 'simulator.py'}")
    sys.path.insert(0, str(mab_path))
    return importlib.import_module("simulator")


def legacy_policy_dict(simulator, config: PaperScalarConfig) -> dict:
    policies = {}
    for name in config.policies:
        if name == "uniform":
            policies[name] = simulator.uniform
        elif name == "ts":
            policies[name] = functools.partial(simulator.ts, n_samples=config.n_samples)
        elif name in {"ttts", "top_two_ts"}:
            policies[name] = functools.partial(simulator.top_two_ts, n_samples=config.n_samples)
        elif name in {"myopic", "kg"}:
            policies[name] = functools.partial(
                simulator.kg,
                eps=config.rho_eps,
                n_max=config.rho_epochs,
                lr=config.rho_lr,
                num_zs=config.rho_num_zs,
            )
        elif name == "rho":
            policies[name] = functools.partial(
                simulator.rho,
                eps=config.rho_eps,
                n_max=config.rho_epochs,
                lr=config.rho_lr,
                num_zs=config.rho_num_zs,
            )
        else:
            raise ValueError(f"unknown policy: {name}")
    return policies


def run_legacy_trial(simulator, config: PaperScalarConfig, seed: int) -> dict:
    alpha, beta, mu_0, sigma2_0 = paper_priors(config)
    s2 = (
        np.array([0.25] * config.n_arms, dtype=float)
        if config.reward.lower() == "bernoulli"
        else np.array([config.s2] * config.n_arms, dtype=float)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return simulator.eval_instance(
            seed=seed,
            s2=s2,
            alpha=alpha,
            beta=beta,
            mu_0=mu_0,
            sigma2_0=sigma2_0,
            T=config.horizon,
            name="paper_scalar_compare",
            prior=config.prior,
            mean_scale=config.mean_scale,
            delta=0,
            c=0,
            policies=legacy_policy_dict(simulator, config),
            f_bern=config.reward.lower() == "bernoulli",
        )


def compare_config(config: PaperScalarConfig, adaptive_root: str | Path) -> dict:
    simulator = import_legacy_simulator(Path(adaptive_root).resolve())
    policies = make_policies(config)
    rows = []
    for seed in range(config.seed, config.seed + config.n_runs):
        legacy = run_legacy_trial(simulator, config, seed)
        current = run_trial(config, seed, policies)
        for name in config.policies:
            old_regret = float(legacy[name]["simple_regret"])
            new_regret = float(current["policies"][name]["simple_regret"])
            old_correct = float(legacy[name]["1_correct"])
            new_correct = float(current["policies"][name]["correct"])
            rows.append(
                {
                    "seed": seed,
                    "policy": name,
                    "old_regret": old_regret,
                    "new_regret": new_regret,
                    "regret_diff": new_regret - old_regret,
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "correct_diff": new_correct - old_correct,
                }
            )

    max_abs_regret_diff = max((abs(row["regret_diff"]) for row in rows), default=0.0)
    max_abs_correct_diff = max((abs(row["correct_diff"]) for row in rows), default=0.0)
    return {
        "config": asdict(config),
        "adaptive_root": str(Path(adaptive_root).resolve()),
        "max_abs_regret_diff": max_abs_regret_diff,
        "max_abs_correct_diff": max_abs_correct_diff,
        "all_match": max_abs_regret_diff <= 1e-12 and max_abs_correct_diff <= 1e-12,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare new scalar replication against adaptive-experimentation/MAB/simulator.py.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--adaptive-root", type=str, default="../adaptive-experimentation", help="Path to the adaptive-experimentation repo.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = compare_config(config, args.adaptive_root)
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
