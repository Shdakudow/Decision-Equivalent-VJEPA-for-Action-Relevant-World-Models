#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-egl}"
CONFIG="${1:-configs/smoke.yaml}"

devjepa-collect --config "$CONFIG"
devjepa-encode --config "$CONFIG"
devjepa-train-bc --config "$CONFIG"
devjepa-train --config "$CONFIG"
devjepa-evaluate --config "$CONFIG"
