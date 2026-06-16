# DE-VJEPA Independent Verification Package

This repository contains the code, configurations, tests, compact reported
results, and paper needed for an independent implementation and audit of
Decision-Equivalent VJEPA (DE-VJEPA).

## Claim Under Test

DE-VJEPA augments variational latent prediction with a policy-induced KL term
between the action distributions produced from predicted and true future
latents. The reported evidence supports improved offline decision preservation
across most tested MetaWorld and PushT conditions, including cases where latent
MSE worsens.

The closed-loop evidence is negative: the tested planners do not establish a
statistically significant aggregate success improvement over VJEPA or direct
behavior cloning. This package must not be used to claim otherwise.

## Included

- `src/devjepa/`: model, losses, data, training, evaluation, and analysis code.
- `configs/`: exact smoke, primary comparison, ablation, and planning configs.
- `scripts/`: experiment launchers, analysis, and paper-figure generation.
- `tests/`: unit tests for losses, data, analysis, progress, and closed-loop code.
- `results/`: compact summary, paired, statistics, validation, and manifest CSVs.
- `tables/` and `plots/`: derived paper artifacts.
- `main.tex` and the compiled paper PDF.
- `EXTERNAL_ARTIFACT_INVENTORY.csv`: inventory of excluded datasets, checkpoints,
  raw outputs, and per-sample result files.
- `REPRODUCE.md`: staged instructions for audit and independent reruns.

## Not Included

Raw datasets, model checkpoints, experiment output directories, and large
per-sample CSVs are excluded from Git. They must be regenerated independently
or restored from a separately published artifact archive. The compact CSVs are
enough to audit reported aggregates and regenerate paper figures, but they are
not evidence of an independent rerun.

## Quick Verification

Use Linux or WSL2 with Python 3.10-3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[dev]"

python verify_release.py
pytest -q
python scripts/make_paper_figures.py
```

For an NVIDIA GPU under WSL2, use `bash scripts/setup_wsl.sh` instead of the
generic installation commands.

## Independent Rerun

Start with the smoke test:

```bash
bash scripts/run_smoke.sh
```

Then follow `REPRODUCE.md`. Record the Git commit, package versions, GPU,
operating system, generated data checksums, and every config used. Compare new
results to the tracked CSVs without overwriting them.

## Hardware

The reported runs used an NVIDIA RTX 4090. Data collection and MuJoCo rendering
are substantially CPU-bound; encoding and training are GPU-bound. The code is
configured for Linux/WSL2 rather than native Windows.

## License

No software or dataset license is asserted in this package. Select a license
only after checking the redistribution terms of MetaWorld, PushT, pretrained
backbones, and any externally published artifacts.
