from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from devjepa.data import load_configured_archive, split_archive
from devjepa.models import ProgressHead
from devjepa.utils import device_from_config, seed_everything


def trajectory_progress_labels(episode_ids: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(len(episode_ids), dtype=torch.float32)
    for episode_id in torch.unique(episode_ids):
        indices = torch.where(episode_ids == episode_id)[0]
        if len(indices) > 1:
            labels[indices] = torch.linspace(0.0, 1.0, len(indices))
    if not torch.isfinite(labels).all():
        raise FloatingPointError("Non-finite trajectory progress labels")
    return labels


def _correlation(target: np.ndarray, predicted: np.ndarray, kind: str) -> float:
    if len(target) < 2 or np.std(target) == 0 or np.std(predicted) == 0:
        return np.nan
    result = stats.spearmanr(target, predicted) if kind == "spearman" else stats.pearsonr(
        target, predicted
    )
    return float(result.statistic)


def _evaluate(
    model: ProgressHead,
    latents: torch.Tensor,
    task_ids: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    loader = DataLoader(
        TensorDataset(latents, task_ids, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.inference_mode():
        for batch_latents, batch_tasks, _ in loader:
            normalized = (batch_latents.to(device) - mean) / std
            predictions.append(model(normalized, batch_tasks.to(device)).cpu())
    prediction = torch.cat(predictions).numpy()
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite progress-head predictions")
    return labels.numpy(), prediction


def _validation_rows(
    seed: int,
    tasks: list[str],
    task_ids: torch.Tensor,
    target: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    task_values = task_ids.numpy()
    for task_id, task in [(-1, "ALL"), *enumerate(tasks)]:
        mask = np.ones(len(target), dtype=bool) if task_id == -1 else task_values == task_id
        rows.append(
            {
                "seed": seed,
                "task": task,
                "label_type": "trajectory_progress",
                "samples": int(mask.sum()),
                "mse": float(np.mean((target[mask] - predicted[mask]) ** 2)),
                "pearson": _correlation(target[mask], predicted[mask], "pearson"),
                "spearman": _correlation(target[mask], predicted[mask], "spearman"),
            }
        )
    return rows


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Trajectory-progress head validation. The target is normalized position within a successful demonstration, not true task value.}",
        r"\label{tab:progress-head-validation}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Task & MSE $\downarrow$ & Pearson $\uparrow$ & Spearman $\uparrow$ \\",
        r"\midrule",
    ]
    for task, group in frame.groupby("task", sort=False):
        escaped_task = task.replace("_", r"\_")
        lines.append(
            f"{escaped_task} & {group['mse'].mean():.4f} & "
            f"{group['pearson'].mean():.3f} & {group['spearman'].mean():.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _plot_calibration(target: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    bins = pd.qcut(predicted, q=min(10, len(np.unique(predicted))), duplicates="drop")
    frame = pd.DataFrame({"target": target, "predicted": predicted, "bin": bins})
    calibration = frame.groupby("bin", observed=True)[["target", "predicted"]].mean()
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.plot(calibration["predicted"], calibration["target"], "o-", label="Validation")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axis.set_xlabel("Predicted trajectory progress")
    axis.set_ylabel("Observed trajectory progress")
    axis.grid(alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_by_task(
    target: np.ndarray,
    predicted: np.ndarray,
    task_ids: np.ndarray,
    tasks: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.2), squeeze=False)
    for task_id, task in enumerate(tasks):
        axis = axes[0, task_id]
        mask = task_ids == task_id
        axis.scatter(target[mask], predicted[mask], s=5, alpha=0.2)
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.set_title(task.replace("-v3", ""))
        axis.set_xlabel("Target progress")
        axis.set_ylabel("Predicted progress")
        axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train_progress_heads(config: dict[str, Any]) -> tuple[Path, Path]:
    section = config.get("progress_head", config.get("value_head"))
    if section is None:
        raise KeyError("Config requires a progress_head or value_head section")
    device = device_from_config(config)
    archive = load_configured_archive(config)
    tasks = list(config["data"]["tasks"])
    output_dir = Path(section.get("checkpoint_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = Path(section.get("validation_output", "results/value_head_validation.csv"))
    summary_path = Path(
        section.get("summary_output", "results/value_head_validation_summary.csv")
    )
    table_path = Path(section.get("table_output", "tables/value_head_validation_table.tex"))
    plot_path = Path(section.get("plot_output", "plots/value_head_calibration.png"))
    by_task_plot_path = Path(
        section.get("by_task_plot_output", "plots/value_head_by_task.png")
    )
    checkpoint_prefix = str(section.get("checkpoint_prefix", "value_head"))
    all_rows = []
    calibration_target = []
    calibration_prediction = []
    calibration_task_ids = []
    for seed in [int(value) for value in section.get("seeds", [42])]:
        seed_everything(seed)
        train, validation = split_archive(
            archive, float(section.get("validation_fraction", 0.2)), seed
        )
        train_labels = trajectory_progress_labels(train["episode_ids"])
        validation_labels = trajectory_progress_labels(validation["episode_ids"])
        latent_mean = train["latents"].float().mean(0).to(device)
        latent_std = train["latents"].float().std(0).clamp_min(1e-5).to(device)
        model = ProgressHead(
            latent_dim=train["latents"].shape[-1],
            num_tasks=len(tasks),
            hidden_dim=int(section.get("hidden_dim", 256)),
            task_dim=int(section.get("task_dim", 32)),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(section.get("learning_rate", 3e-4)),
            weight_decay=float(section.get("weight_decay", 1e-4)),
        )
        batch_size = int(section.get("batch_size", 256))
        loader = DataLoader(
            TensorDataset(train["latents"].float(), train["task_ids"].long(), train_labels),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        model.train()
        for _ in range(int(section.get("epochs", 30))):
            for latents, task_ids, labels in loader:
                normalized = (latents.to(device) - latent_mean) / latent_std
                prediction = model(normalized, task_ids.to(device))
                loss = nn.functional.mse_loss(prediction, labels.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite progress-head loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        target, predicted = _evaluate(
            model,
            validation["latents"].float(),
            validation["task_ids"].long(),
            validation_labels,
            latent_mean,
            latent_std,
            device,
            batch_size,
        )
        all_rows.extend(
            _validation_rows(seed, tasks, validation["task_ids"], target, predicted)
        )
        calibration_target.append(target)
        calibration_prediction.append(predicted)
        calibration_task_ids.append(validation["task_ids"].numpy())
        torch.save(
            {
                "model": model.state_dict(),
                "latent_mean": latent_mean.cpu(),
                "latent_std": latent_std.cpu(),
                "latent_dim": int(train["latents"].shape[-1]),
                "num_tasks": len(tasks),
                "hidden_dim": int(section.get("hidden_dim", 256)),
                "task_dim": int(section.get("task_dim", 32)),
                "tasks": tasks,
                "label_type": "trajectory_progress",
                "seed": seed,
            },
            output_dir / f"{checkpoint_prefix}_seed_{seed}.pt",
        )
    results = pd.DataFrame(all_rows)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(validation_path, index=False)
    summary = (
        results.groupby(["task", "label_type"], as_index=False)[
            ["mse", "pearson", "spearman"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    _write_table(results, table_path)
    combined_target = np.concatenate(calibration_target)
    combined_prediction = np.concatenate(calibration_prediction)
    combined_task_ids = np.concatenate(calibration_task_ids)
    _plot_calibration(combined_target, combined_prediction, plot_path)
    _plot_by_task(
        combined_target,
        combined_prediction,
        combined_task_ids,
        tasks,
        by_task_plot_path,
    )
    return validation_path, table_path


def load_progress_head(
    checkpoint_dir: str | Path,
    seed: int,
    device: torch.device,
    checkpoint_prefix: str = "value_head",
) -> tuple[ProgressHead, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(
        Path(checkpoint_dir) / f"{checkpoint_prefix}_seed_{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = ProgressHead(
        checkpoint["latent_dim"],
        checkpoint["num_tasks"],
        checkpoint["hidden_dim"],
        checkpoint["task_dim"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False).to(device)
    return (
        model,
        checkpoint["latent_mean"].to(device),
        checkpoint["latent_std"].to(device),
    )
