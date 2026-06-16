from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from devjepa.data import Normalization
from devjepa.models import GaussianPolicy, VariationalPredictor


def load_policy(config: dict[str, Any], device: torch.device) -> tuple[GaussianPolicy, Normalization]:
    quality = config["policy"].get("quality", "strong")
    seed = int(config["experiment"]["seed"])
    policy_root = Path(config["policy"].get("checkpoint_dir", config["output"]["dir"]))
    path = policy_root / f"policy_{quality}_seed{seed}" / "policy.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = GaussianPolicy(
        checkpoint["latent_dim"],
        checkpoint["action_dim"],
        checkpoint["num_tasks"],
        int(config["policy"]["hidden_dim"]),
        int(config["policy"]["task_dim"]),
    )
    policy.load_state_dict(checkpoint["model"])
    policy.eval().requires_grad_(False).to(device)
    normalization = Normalization.from_state_dict(checkpoint["normalization"])
    normalization = Normalization(
        latent_mean=normalization.latent_mean.to(device),
        latent_std=normalization.latent_std.to(device),
        action_mean=normalization.action_mean.to(device),
        action_std=normalization.action_std.to(device),
    )
    return policy, normalization


def build_predictor(
    config: dict[str, Any],
    latent_dim: int,
    action_dim: int,
    prior_mean: torch.Tensor | None = None,
    prior_variance: torch.Tensor | None = None,
) -> VariationalPredictor:
    method = config["experiment"]["method"]
    bjepa = method in {"bjepa", "de_bjepa"}
    return VariationalPredictor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        num_tasks=len(config["data"]["tasks"]),
        hidden_dim=int(config["predictor"]["hidden_dim"]),
        stochastic_dim=int(config["predictor"]["stochastic_dim"]),
        task_dim=int(config["predictor"]["task_dim"]),
        horizon=int(config["predictor"]["horizon"]),
        bjepa=bjepa,
        bjepa_prior_type=config.get("bjepa", {}).get("prior_type", "none"),
        use_actions=bool(config.get("predictor", {}).get("use_actions", True)),
        architecture=str(config.get("predictor", {}).get("architecture", "mlp")),
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )
