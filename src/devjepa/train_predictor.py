from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from devjepa.config import load_config, parser
from devjepa.data import TransitionDataset, load_configured_archive, split_archive
from devjepa.losses import predictor_loss
from devjepa.models import posterior_kl
from devjepa.runtime import build_predictor, load_policy
from devjepa.utils import device_from_config, output_dir, save_json, save_yaml, seed_everything


def _normalize_batch(batch, normalization, device):
    return {
        "z": ((batch["z"].to(device) - normalization.latent_mean) / normalization.latent_std),
        "target": (
            (batch["target"].to(device) - normalization.latent_mean) / normalization.latent_std
        ),
        "actions": (
            (batch["actions"].to(device) - normalization.action_mean) / normalization.action_std
        ),
        "task_id": batch["task_id"].to(device),
    }


def _task_prior_statistics(train, normalization, num_tasks):
    normalized = (train["latents"].float() - normalization.latent_mean.cpu()) / (
        normalization.latent_std.cpu()
    )
    means = []
    variances = []
    for task_id in range(num_tasks):
        values = normalized[train["task_ids"] == task_id]
        if len(values) < 2:
            raise ValueError(f"Not enough samples to fit task prior for task {task_id}")
        means.append(values.mean(0))
        variances.append(values.var(0, unbiased=False).clamp_min(1e-4))
    return torch.stack(means), torch.stack(variances)


def train_predictor(config: dict[str, Any]) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = device_from_config(config)
    policy, normalization = load_policy(config, device)
    archive = load_configured_archive(config)
    train, validation = split_archive(archive, float(config["data"]["validation_fraction"]), seed)
    horizon = int(config["predictor"]["horizon"])
    train_dataset = TransitionDataset(train, horizon)
    validation_dataset = TransitionDataset(validation, horizon)
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["predictor"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["runtime"]["workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=512, shuffle=False)
    prior_mean = prior_variance = None
    if (
        method := config["experiment"]["method"]
    ) in {"bjepa", "de_bjepa"} and config.get("bjepa", {}).get(
        "prior_type", "none"
    ) == "empirical_task_gaussian":
        prior_mean, prior_variance = _task_prior_statistics(
            train, normalization, len(config["data"]["tasks"])
        )
    predictor = build_predictor(
        config,
        archive["latents"].shape[-1],
        archive["actions"].shape[-1],
        prior_mean,
        prior_variance,
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["predictor"]["learning_rate"]),
        weight_decay=float(config["predictor"]["weight_decay"]),
    )
    epochs = int(config["predictor"]["epochs"])
    method = config["experiment"]["method"]
    tag = config["experiment"].get("tag")
    run_name = f"{method}_{config['policy']['quality']}_h{horizon}_seed{seed}"
    if tag:
        run_name = f"{run_name}_{tag}"
    destination = output_dir(config, run_name)
    history = []

    for epoch in trange(epochs, desc=run_name):
        predictor.train()
        totals: dict[str, float] = defaultdict(float)
        count = 0
        max_batches = config["predictor"].get("max_batches")
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _normalize_batch(raw_batch, normalization, device)
            posterior_result = predictor(
                batch["z"], batch["actions"], batch["task_id"], batch["target"], sample=True
            )
            prior_result = predictor(
                batch["z"], batch["actions"], batch["task_id"], target=None, sample=False
            )
            loss, metrics = predictor_loss(
                method,
                prior_result.predicted,
                batch["target"],
                batch["task_id"],
                prior_result.posterior_mean,
                prior_result.posterior_logvar,
                policy,
                0.0,
                float(config["loss"]["epsilon0"]),
                float(config["loss"]["uncertainty_scale"]),
                float(config["loss"]["gamma"]),
                float(config["loss"].get("decision_weight", 1.0)),
                prior_result.predictive_logvar,
                config.get("bjepa", {}).get("loss_type", "nll"),
                config["loss"].get("band_type", "entropy_band"),
            )
            variational_kl = posterior_kl(
                posterior_result.posterior_mean, posterior_result.posterior_logvar
            ).mean()
            loss = loss + float(config["loss"]["beta"]) * variational_kl
            metrics["variational_kl"] = variational_kl.detach()
            metrics["loss"] = loss.detach()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), float(config["predictor"]["gradient_clip"])
            )
            optimizer.step()
            for key, value in metrics.items():
                totals[key] += float(value)
            count += 1
        validation_metrics = _validate(
            predictor, policy, normalization, validation_loader, config, device
        )
        row = {"epoch": epoch + 1}
        row.update({f"train_{key}": value / max(count, 1) for key, value in totals.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        save_json(history, destination / "history.json")
        torch.save(
            {
                "model": predictor.cpu().state_dict(),
                "latent_dim": archive["latents"].shape[-1],
                "action_dim": archive["actions"].shape[-1],
                "method": method,
                "bjepa_prior_type": config.get("bjepa", {}).get("prior_type", "none"),
                "bjepa_loss_type": config.get("bjepa", {}).get("loss_type", "nll"),
                "horizon": horizon,
                "decision_weight": float(config["loss"].get("decision_weight", 0.0)),
                "epoch": epoch + 1,
            },
            destination / "predictor.pt",
        )
        predictor.to(device)
    save_yaml(config, destination / "resolved_config.yaml")
    return destination / "predictor.pt"


@torch.inference_mode()
def _validate(predictor, policy, normalization, loader, config, device):
    predictor.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for raw_batch in loader:
        batch = _normalize_batch(raw_batch, normalization, device)
        result = predictor(
            batch["z"], batch["actions"], batch["task_id"], target=None, sample=False
        )
        _, metrics = predictor_loss(
            config["experiment"]["method"],
            result.predicted,
            batch["target"],
            batch["task_id"],
            result.posterior_mean,
            result.posterior_logvar,
            policy,
            float(config["loss"]["beta"]),
            float(config["loss"]["epsilon0"]),
            float(config["loss"]["uncertainty_scale"]),
            float(config["loss"]["gamma"]),
            float(config["loss"].get("decision_weight", 1.0)),
            result.predictive_logvar,
            config.get("bjepa", {}).get("loss_type", "nll"),
            config["loss"].get("band_type", "entropy_band"),
        )
        for key, value in metrics.items():
            totals[key] += float(value)
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parser("Train VJEPA or DE-VJEPA predictor").parse_args()
    print(train_predictor(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
