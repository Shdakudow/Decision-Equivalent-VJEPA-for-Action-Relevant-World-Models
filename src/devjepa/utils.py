from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(config: dict[str, Any]) -> torch.device:
    requested = config.get("runtime", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check WSL NVIDIA support and PyTorch.")
    return torch.device(requested)


def output_dir(config: dict[str, Any], suffix: str | None = None) -> Path:
    path = Path(config["output"]["dir"])
    if suffix:
        path /= suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(data: dict[str, Any], path: Path) -> None:
    clean = {key: value for key, value in data.items() if not key.startswith("_")}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def autocast_context(device: torch.device, enabled: bool = True):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")
