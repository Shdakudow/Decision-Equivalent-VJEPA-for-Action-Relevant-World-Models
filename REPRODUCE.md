# Independent Reproduction Protocol

This protocol separates source validation, numerical audit, smoke testing, and
independent experimental reproduction. Passing an earlier stage does not imply
that a later stage has passed.

## 1. Record the Environment

Run from the repository root:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > verifier_environment.txt
nvidia-smi
```

Recommended platform: Ubuntu under WSL2, Python 3.10-3.12, and an NVIDIA GPU.
The original runs used an RTX 4090.

## 2. Validate the Release

```bash
python verify_release.py
pytest -q
```

`verify_release.py` checks required files, SHA-256 checksums, referenced
graphics, result schemas, and the absence of accidentally bundled large files.

## 3. Audit Reported Aggregates

Regenerate the paper figures from the tracked compact CSVs:

```bash
python scripts/make_paper_figures.py
```

Inspect the paired and inferential files directly:

```text
results/cross_method_metaworld_paired.csv
results/cross_method_pusht_paired.csv
results/vjepa_bjepa_de_comparison_paired.csv
results/closed_loop_rollout_statistics.csv
results/planner_ablation_statistics.csv
results/cem_mpc_progress512_statistics.csv
```

This stage audits the submitted aggregates. It does not independently reproduce
training or evaluation.

## 4. Run the End-to-End Smoke Test

```bash
bash scripts/setup_wsl.sh
source .venv/bin/activate
bash scripts/run_smoke.sh
```

Confirm that data collection, encoding, policy training, predictor training,
and evaluation all complete. Smoke-test values are not expected to match the
paper because the dataset and training budget are intentionally small.

## 5. Reproduce the Primary Offline Comparison

The primary five-task MetaWorld comparison uses three training seeds and
independent evaluator policies:

```bash
source .venv/bin/activate
export MUJOCO_GL=egl

devjepa-run-experiment --config configs/pilot_reproduction.yaml
devjepa-run-experiment --config configs/core_vjepa_bjepa_de.yaml
```

The first invocation collects and encodes data if the configured archives are
absent. Do not reuse the tracked compact CSVs as inputs to model training.

For PushT, prepare the dataset and then run:

```bash
python scripts/prepare_pusht.py --config configs/pusht.yaml
devjepa-run-experiment --config configs/cross_method_pusht.yaml
python scripts/combine_cross_dataset.py
```

Use the exact seeds, visual shifts, task order, and source/evaluator separation
specified in the YAML files. Preserve sample alignment between clean and shifted
archives.

## 6. Reproduce Diagnostics

Run the decision-loss and policy-geometry ablations:

```bash
devjepa-run-experiment --config configs/lambda_sweep.yaml
devjepa-run-experiment --config configs/policy_geometry_ablation.yaml
devjepa-run-experiment --config configs/kl_vs_action_mse.yaml
```

Run autoregressive prediction:

```bash
python scripts/run_multistep_training.py \
  --config configs/multistep_de_training.yaml
python scripts/run_multistep_diagnostic.py \
  --config configs/multistep_diagnostic.yaml
```

## 7. Reproduce Closed-Loop Evaluation

Closed-loop runs require the checkpoints produced by the corresponding offline
experiments:

```bash
python scripts/run_closed_loop.py --config configs/closed_loop_main.yaml
python scripts/run_closed_loop.py --config configs/planner_ablation.yaml
python scripts/train_progress_head.py --config configs/progress_head.yaml
python scripts/run_closed_loop.py \
  --config configs/cem_mpc_progress512_main.yaml
```

The expected scientific outcome is cautious: DE-VJEPA may be directionally
better in some cells, especially horizon-three brightness, but the aggregate
success improvement is not statistically significant. A verification report
must retain negative results and all tested conditions.

## 8. Reporting a Verification

Publish:

1. Git commit and environment file.
2. Checksums for generated datasets and checkpoints.
3. Exact commands and resolved YAML configs.
4. All seeds, failed runs, resumptions, and exclusions.
5. Raw per-sample or per-episode outputs.
6. Aggregate tables with uncertainty and paired tests.
7. Deviations from the supplied implementation.

Label the outcome precisely as source audit, aggregate audit, partial
reproduction, or full independent reproduction.
