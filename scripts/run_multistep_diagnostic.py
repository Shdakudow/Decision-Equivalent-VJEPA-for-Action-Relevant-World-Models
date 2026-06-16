from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader

from devjepa.closed_loop import _load_predictor, _policy_config
from devjepa.config import load_config
from devjepa.data import TransitionDataset, load_archive, split_archive
from devjepa.runtime import load_policy
from devjepa.utils import device_from_config, seed_everything


def _write_table(summary: pd.DataFrame, path: Path) -> None:
    overall = (
        summary.groupby(["method", "horizon"])[
            ["latent_mse", "decision_kl", "action_mse"]
        ]
        .mean()
        .reset_index()
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Autoregressive multi-step prediction diagnostic. Lower is better.}",
        r"\label{tab:multistep-diagnostic}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{$K$} & \textbf{Latent MSE} & \textbf{Decision KL} & \textbf{Action MSE} \\",
        r"\midrule",
    ]
    for row in overall.itertuples():
        escaped_method = row.method.replace("_", r"\_")
        lines.append(
            f"{escaped_method} & {row.horizon} & "
            f"{row.latent_mse:.6f} & {row.decision_kl:.4f} & "
            f"{row.action_mse:.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _plot(summary: pd.DataFrame, metric: str, path: Path) -> None:
    overall = summary.groupby(["method", "horizon"])[metric].mean().reset_index()
    figure, axis = plt.subplots(figsize=(6, 4.5))
    for method, group in overall.groupby("method"):
        axis.plot(group["horizon"], group[metric], marker="o", label=method)
    axis.set_xlabel("Prediction horizon K")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_xticks(sorted(overall["horizon"].unique()))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


@torch.inference_mode()
def run(config: dict) -> tuple[Path, Path]:
    device = device_from_config(config)
    section = config["diagnostic"]
    horizons = sorted(int(value) for value in section["horizons"])
    max_horizon = max(horizons)
    selected_tasks = set(config["data"]["selected_tasks"])
    task_names = list(config["data"]["tasks"])
    archive = load_archive(config["data"]["encoded_path"])
    evaluator, evaluator_norm = load_policy(
        _policy_config(config, int(config["closed_loop"]["policy_seed"])), device
    )
    rows = []
    for seed in [int(value) for value in section["seeds"]]:
        seed_everything(seed)
        _, validation = split_archive(
            archive, float(config["data"]["validation_fraction"]), seed
        )
        dataset = TransitionDataset(validation, max_horizon)
        loader = DataLoader(
            dataset,
            batch_size=int(section["batch_size"]),
            shuffle=False,
            num_workers=int(config["runtime"].get("workers", 0)),
        )
        predictors = {}
        for method in section["methods"]:
            model = "de_vjepa" if "de_vjepa" in method else "vjepa"
            predictors[method] = _load_predictor(
                config,
                model,
                seed,
                archive["latents"].shape[-1],
                archive["actions"].shape[-1],
                device,
            )
        sample_offset = 0
        for batch in loader:
            batch_task_ids = batch["task_id"]
            keep = torch.tensor(
                [task_names[int(task)] in selected_tasks for task in batch_task_ids],
                dtype=torch.bool,
            )
            if not keep.any():
                continue
            task_ids = batch_task_ids[keep].to(device)
            current_raw = batch["z"][keep].to(device)
            target_raw = batch["target"][keep].to(device)
            actions_raw = batch["actions"][keep].to(device)
            for method, (predictor, normalization) in predictors.items():
                z = (current_raw - normalization.latent_mean) / normalization.latent_std
                actions = (
                    actions_raw - normalization.action_mean
                ) / normalization.action_std
                result = predictor(z, actions, task_ids, target=None, sample=False)
                predicted_raw = (
                    result.predicted * normalization.latent_std
                    + normalization.latent_mean
                )
                for horizon in horizons:
                    index = horizon - 1
                    predicted = predicted_raw[:, index]
                    target = target_raw[:, index]
                    latent_mse = (predicted - target).square().mean(-1)
                    predicted_eval = (
                        predicted - evaluator_norm.latent_mean
                    ) / evaluator_norm.latent_std
                    target_eval = (
                        target - evaluator_norm.latent_mean
                    ) / evaluator_norm.latent_std
                    predicted_dist = evaluator.distribution(predicted_eval, task_ids)
                    target_dist = evaluator.distribution(target_eval, task_ids)
                    decision_kl = kl_divergence(target_dist, predicted_dist).sum(-1)
                    action_mse = (
                        target_dist.mean - predicted_dist.mean
                    ).square().mean(-1)
                    for item in range(len(task_ids)):
                        rows.append(
                            {
                                "method": method,
                                "train_seed": seed,
                                "evaluator_seed": int(
                                    config["closed_loop"]["policy_seed"]
                                ),
                                "task": task_names[int(task_ids[item])],
                                "sample_id": sample_offset + item,
                                "horizon": horizon,
                                "latent_mse": float(latent_mse[item].cpu()),
                                "decision_kl": float(decision_kl[item].cpu()),
                                "action_mse": float(action_mse[item].cpu()),
                            }
                        )
            sample_offset += int(keep.sum())
            max_samples = section.get("max_samples")
            if max_samples is not None and sample_offset >= int(max_samples):
                break
    raw = pd.DataFrame(rows)
    slopes = (
        raw.groupby(["method", "train_seed", "task", "sample_id"])
        .apply(
            lambda group: pd.Series(
                {
                    "latent_compounding_error_slope": np.polyfit(
                        group["horizon"], group["latent_mse"], 1
                    )[0],
                    "decision_compounding_error_slope": np.polyfit(
                        group["horizon"], group["decision_kl"], 1
                    )[0],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    raw = raw.merge(slopes, on=["method", "train_seed", "task", "sample_id"])
    raw_path = Path(section["raw_output"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    summary = (
        raw.groupby(["method", "task", "horizon"])
        .agg(
            latent_mse=("latent_mse", "mean"),
            decision_kl=("decision_kl", "mean"),
            action_mse=("action_mse", "mean"),
            latent_compounding_error_slope=(
                "latent_compounding_error_slope",
                "mean",
            ),
            decision_compounding_error_slope=(
                "decision_compounding_error_slope",
                "mean",
            ),
            count=("sample_id", "count"),
        )
        .reset_index()
    )
    summary_path = Path(section["summary_output"])
    summary.to_csv(summary_path, index=False)
    plots = Path(section["plots_dir"])
    _plot(
        summary,
        "decision_kl",
        plots / "multistep_decision_kl_vs_horizon.png",
    )
    _plot(
        summary,
        "latent_mse",
        plots / "multistep_latent_mse_vs_horizon.png",
    )
    _write_table(summary, Path(section["table_output"]))
    return raw_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    print(run(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
