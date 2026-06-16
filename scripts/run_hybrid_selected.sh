#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pilot.yaml}"
OUTPUT="${2:-outputs/hybrid_sweep}"
TAG="${3:-l001}"
WEIGHT="${4:-0.01}"

for seed in 1000 10000; do
  devjepa-train --config "$CONFIG" \
    --set "output.dir=$OUTPUT" \
    --set "experiment.seed=$seed" \
    --set experiment.method=hybrid \
    --set "experiment.tag=$TAG" \
    --set policy.quality=strong \
    --set "loss.decision_weight=$WEIGHT"
done

devjepa-evaluate-independent --config "$CONFIG" \
  --set "output.dir=$OUTPUT" \
  --set experiment.method=hybrid \
  --set "experiment.tag=$TAG" \
  --set "evaluation.methods=[hybrid]" \
  --set "evaluation.source_seeds=[42,1000,10000]" \
  --set "evaluation.evaluator_seeds=[42,1000,10000]"
