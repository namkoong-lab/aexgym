# AExGym

This branch is a paper-focused codebase for the revised Gaussian metric
experimentation model. The maintained package surface is organized around:

- `aexgym.core`: posterior state, Gaussian conjugate model, active-set rules,
  and the shared experiment runner
- `aexgym.policies`: standard policies and the compositional RHO implementation
- `aexgym.experiments`: maintained experiment entry points for scalar parity and
  the basic guardrail experiment
- `aexgym.legacy_parity`: temporary validation code used to compare against the
  older scalar implementation in the sibling `adaptive-experimentation` repo

The old contextual benchmarking stack, notebook workflows, and Hydra scripts are
not part of the supported execution path on this branch.

## Installation

Create the environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate aexgym
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

## Maintained Entry Points

Run the maintained experiments with:

```bash
python -m aexgym.experiments.scalar_parity --help
python -m aexgym.experiments.guardrail_basic --help
```

Run the temporary parity validation tools with:

```bash
python -m aexgym.legacy_parity.scalar_replication --help
python -m aexgym.legacy_parity.compare_adaptive --help
python -m aexgym.legacy_parity.variant_check --help
```

## Data

The repo keeps the raw ASOS aggregate experiment file under `data/` for possible
future use. It is not currently wired into the maintained noncontextual
experiment pipeline on this branch.

## Verification

```bash
python -m pytest -q
python -m compileall -q aexgym tests
```
