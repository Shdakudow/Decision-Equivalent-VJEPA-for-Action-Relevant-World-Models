from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from devjepa.config import load_config
from devjepa.encode import _build_encoder
from devjepa.utils import device_from_config


def run(config: dict, output: Path, samples: int) -> Path:
    raw = np.load(config["data"]["raw_path"], allow_pickle=False)
    images = raw["images"][:samples]
    model, transform, dimension = _build_encoder("dinov2_vits14")
    device = device_from_config(config)
    model.eval().requires_grad_(False).to(device)
    batch = torch.stack([transform(Image.fromarray(image)) for image in images]).to(device)
    with torch.inference_mode():
        features = model(batch).float().cpu().numpy()
    if features.shape != (len(images), dimension):
        raise RuntimeError(f"Unexpected DINOv2 feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise FloatingPointError("Non-finite DINOv2 features")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, latents=features)
    reloaded = np.load(output, allow_pickle=False)["latents"]
    if reloaded.shape != features.shape or not np.isfinite(reloaded).all():
        raise RuntimeError("Saved DINOv2 features failed reload validation")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/dinov2_smoke.npz"))
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    print(run(load_config(args.config), args.output, args.samples))


if __name__ == "__main__":
    main()
