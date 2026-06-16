#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pilot.yaml}"
OUTPUT="${2:-outputs/prior_pilot}"

for seed in 42 1000 10000; do
  devjepa-train-bc --config "$CONFIG" \
    --set "output.dir=$OUTPUT" \
    --set "experiment.seed=$seed" \
    --set policy.quality=strong

  for method in vjepa de_vjepa de_vjepa_band; do
    devjepa-train --config "$CONFIG" \
      --set "output.dir=$OUTPUT" \
      --set "experiment.seed=$seed" \
      --set policy.quality=strong \
      --set "experiment.method=$method"
    devjepa-evaluate --config "$CONFIG" \
      --set "output.dir=$OUTPUT" \
      --set "experiment.seed=$seed" \
      --set policy.quality=strong \
      --set "experiment.method=$method"
  done
done
