from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from devjepa.config import load_config, parser
from devjepa.data import Normalization, load_configured_archive, split_archive
from devjepa.models import GaussianPolicy
from devjepa.utils import device_from_config, output_dir, save_yaml, seed_everything


def _normalized_policy_input(latents: torch.Tensor, normalization: Normalization) -> torch.Tensor:
    return (latents - normalization.latent_mean) / normalization.latent_std


def train_bc(config: dict[str, Any]) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    archive = load_configured_archive(config)
    train, validation = split_archive(archive, float(config["data"]["validation_fraction"]), seed)
    normalization = Normalization.fit(train["latents"].float(), train["actions"].float())
    latent_dim = train["latents"].shape[-1]
    action_dim = train["actions"].shape[-1]
    num_tasks = len(config["data"]["tasks"])
    policy = GaussianPolicy(
        latent_dim,
        action_dim,
        num_tasks,
        int(config["policy"]["hidden_dim"]),
        int(config["policy"]["task_dim"]),
    )
    quality = config["policy"].get("quality", "strong")
    if quality == "random":
        epochs = 0
    elif quality == "weak":
        epochs = int(config["policy"].get("weak_epochs", 2))
    else:
        epochs = int(config["policy"]["epochs"])
    device = device_from_config(config)
    policy.to(device)
    dataset = TensorDataset(
        _normalized_policy_input(train["latents"].float(), normalization),
        train["actions"].float(),
        train["task_ids"].long(),
    )
    fraction = float(config["policy"].get("data_fraction", 1.0))
    if quality == "weak":
        fraction = float(config["policy"].get("weak_data_fraction", fraction))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("policy data fraction must be in (0, 1]")
    if fraction < 1.0:
        generator = torch.Generator().manual_seed(seed)
        count = max(1, int(len(dataset) * fraction))
        dataset, _ = torch.utils.data.random_split(
            dataset, [count, len(dataset) - count], generator=generator
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["policy"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["runtime"]["workers"]),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["policy"]["learning_rate"]),
        weight_decay=float(config["policy"]["weight_decay"]),
    )
    policy.train()
    for _ in trange(epochs, desc=f"BC ({quality})"):
        for z, actions, task_ids in loader:
            distribution = policy.distribution(z.to(device), task_ids.to(device))
            loss = -distribution.log_prob(actions.to(device)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
    destination = output_dir(config, f"policy_{quality}_seed{seed}") / "policy.pt"
    torch.save(
        {
            "model": policy.cpu().state_dict(),
            "normalization": normalization.state_dict(),
            "latent_dim": latent_dim,
            "action_dim": action_dim,
            "num_tasks": num_tasks,
            "quality": quality,
        },
        destination,
    )
    save_yaml(config, destination.parent / "resolved_config.yaml")
    return destination


def main() -> None:
    args = parser("Train the frozen Gaussian reference policy").parse_args()
    print(train_bc(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
