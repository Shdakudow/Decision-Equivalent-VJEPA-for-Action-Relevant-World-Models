#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pilot.yaml}"
OUTPUT="${2:-outputs/hybrid_sweep}"

for seed in 42 1000 10000; do
  devjepa-train-bc --config "$CONFIG" \
    --set "output.dir=$OUTPUT" \
    --set "experiment.seed=$seed" \
    --set policy.quality=strong
done

for spec in \
  "l001 0.01" \
  "l003 0.03" \
  "l01 0.1" \
  "l03 0.3" \
  "l1 1.0"; do
  read -r tag weight <<<"$spec"
  devjepa-train --config "$CONFIG" \
    --set "output.dir=$OUTPUT" \
    --set experiment.seed=42 \
    --set experiment.method=hybrid \
    --set "experiment.tag=$tag" \
    --set policy.quality=strong \
    --set "loss.decision_weight=$weight"
  devjepa-evaluate-independent --config "$CONFIG" \
    --set "output.dir=$OUTPUT" \
    --set experiment.method=hybrid \
    --set "experiment.tag=$tag" \
    --set "evaluation.methods=[hybrid]" \
    --set "evaluation.source_seeds=[42]" \
    --set "evaluation.evaluator_seeds=[1000,10000]"
done
