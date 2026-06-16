#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-egl}"
CONFIG="${1:-configs/pilot.yaml}"
SEED="${2:-42}"

if [[ ! -f data/metaworld_pilot_raw.npz ]]; then
  devjepa-collect --config "$CONFIG" --set "experiment.seed=$SEED"
fi
if [[ ! -f data/metaworld_pilot_resnet18.npz ]]; then
  devjepa-encode --config "$CONFIG" --set "experiment.seed=$SEED"
fi

for quality in strong weak random; do
  devjepa-train-bc --config "$CONFIG" \
    --set "experiment.seed=$SEED" \
    --set "policy.quality=$quality"
  for method in vjepa de_vjepa de_vjepa_band; do
    devjepa-train --config "$CONFIG" \
      --set "experiment.seed=$SEED" \
      --set "policy.quality=$quality" \
      --set "experiment.method=$method"
    devjepa-evaluate --config "$CONFIG" \
      --set "experiment.seed=$SEED" \
      --set "policy.quality=$quality" \
      --set "experiment.method=$method"
  done
done

# Planning-oriented multi-step ablation with the strongest reference policy.
for horizon in 3 5; do
  devjepa-train --config "$CONFIG" \
    --set "experiment.seed=$SEED" \
    --set policy.quality=strong \
    --set experiment.method=de_vjepa \
    --set "predictor.horizon=$horizon"
  devjepa-evaluate --config "$CONFIG" \
    --set "experiment.seed=$SEED" \
    --set policy.quality=strong \
    --set experiment.method=de_vjepa \
    --set "predictor.horizon=$horizon"
done
