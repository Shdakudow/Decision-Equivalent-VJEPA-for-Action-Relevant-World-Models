from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2
from tqdm import tqdm

from devjepa.config import load_config, parser
from devjepa.utils import autocast_context, device_from_config, seed_everything


class ImageDataset(Dataset):
    def __init__(self, images: np.ndarray, transform):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(Image.fromarray(self.images[index]))


def _build_encoder(name: str) -> tuple[nn.Module, Any, int]:
    if name == "resnet18_imagenet":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        return model, weights.transforms(), 512
    if name == "dinov2_vits14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        return model, transform, 384
    raise ValueError(f"Unsupported encoder: {name}")


def encode(config: dict[str, Any]) -> Path:
    seed_everything(int(config["experiment"]["seed"]))
    raw_path = Path(config["data"]["raw_path"])
    encoded_path = Path(config["data"]["encoded_path"])
    encoded_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.load(raw_path, allow_pickle=False)
    model, transform, _ = _build_encoder(config["encoder"]["name"])
    device = device_from_config(config)
    model.eval().requires_grad_(False).to(device)
    loader = DataLoader(
        ImageDataset(raw["images"], transform),
        batch_size=int(config["encoder"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["workers"]),
        pin_memory=device.type == "cuda",
    )
    outputs = []
    with torch.inference_mode():
        for images in tqdm(loader, desc="encoding"):
            with autocast_context(device, bool(config["runtime"].get("amp", True))):
                outputs.append(model(images.to(device, non_blocking=True)).float().cpu())
    latents = torch.cat(outputs).numpy().astype(np.float32)
    payload = {key: raw[key] for key in raw.files if key != "images"}
    payload["latents"] = latents
    np.savez_compressed(encoded_path, **payload)
    return encoded_path


def main() -> None:
    args = parser("Encode MetaWorld frames with a frozen visual encoder").parse_args()
    print(encode(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
