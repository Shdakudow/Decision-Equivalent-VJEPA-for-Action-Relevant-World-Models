from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from devjepa.config import load_config, parser
from devjepa.encode import _build_encoder
from devjepa.utils import autocast_context, device_from_config, seed_everything


SHIFTS = ("brightness", "desaturate", "blur")
SHIFT_ALIASES = {"desaturation": "desaturate"}


class ShiftedImageDataset(Dataset):
    def __init__(self, images: np.ndarray, transform, shift: str):
        self.images = images
        self.transform = transform
        self.shift = shift

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.fromarray(self.images[index])
        if self.shift == "brightness":
            image = ImageEnhance.Brightness(image).enhance(0.55)
        elif self.shift == "desaturate":
            image = ImageEnhance.Color(image).enhance(0.15)
        elif self.shift == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        else:
            raise ValueError(f"Unknown visual shift: {self.shift}")
        return self.transform(image)


def encode_shift(config: dict[str, Any], shift: str) -> Path:
    shift = SHIFT_ALIASES.get(shift, shift)
    if shift not in SHIFTS:
        supported = (*SHIFTS, *SHIFT_ALIASES)
        raise ValueError(f"Supported shifts: {', '.join(supported)}")
    seed_everything(int(config["experiment"]["seed"]))
    raw_path = Path(config["data"]["raw_path"])
    base_path = Path(config["data"]["encoded_path"])
    output = base_path.with_name(f"{base_path.stem}_{shift}{base_path.suffix}")
    raw = np.load(raw_path, allow_pickle=False)
    model, transform, _ = _build_encoder(config["encoder"]["name"])
    device = device_from_config(config)
    model.eval().requires_grad_(False).to(device)
    loader = DataLoader(
        ShiftedImageDataset(raw["images"], transform, shift),
        batch_size=int(config["encoder"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    outputs = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=f"encoding {shift}"):
            with autocast_context(device, bool(config["runtime"].get("amp", True))):
                outputs.append(model(images.to(device)).float().cpu())
    payload = {key: raw[key] for key in raw.files if key != "images"}
    payload["latents"] = torch.cat(outputs).numpy().astype(np.float32)
    np.savez_compressed(output, **payload)
    return output


def main() -> None:
    args = parser("Encode deterministic visual shifts").parse_args()
    config = load_config(args.config, args.set)
    for shift in SHIFTS:
        print(encode_shift(config, shift))


if __name__ == "__main__":
    main()
