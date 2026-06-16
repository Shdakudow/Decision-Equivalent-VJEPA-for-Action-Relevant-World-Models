from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from devjepa.config import load_config, parser
from devjepa.data import TransitionDataset, load_archive, split_archive
from devjepa.models import gaussian_policy_kl
from devjepa.runtime import build_predictor, load_policy
from devjepa.train_predictor import _normalize_batch
from devjepa.utils import device_from_config, output_dir, save_json, seed_everything


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float((pearsonr if kind == "pearson" else spearmanr)(x, y).statistic)


@torch.inference_mode()
def evaluate(config: dict[str, Any]) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = device_from_config(config)
    policy, normalization = load_policy(config, device)
    archive = load_archive(config["data"]["encoded_path"])
    _, validation = split_archive(archive, float(config["data"]["validation_fraction"]), seed)
    dataset = TransitionDataset(validation, int(config["predictor"]["horizon"]))
    loader = DataLoader(dataset, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False)
    predictor = build_predictor(
        config, archive["latents"].shape[-1], archive["actions"].shape[-1]
    )
    method = config["experiment"]["method"]
    run_name = (
        f"{method}_{config['policy']['quality']}_h{config['predictor']['horizon']}_seed{seed}"
    )
    tag = config["experiment"].get("tag")
    if tag:
        run_name = f"{run_name}_{tag}"
    run_dir = output_dir(config, run_name)
    checkpoint = torch.load(run_dir / "predictor.pt", map_location="cpu", weights_only=True)
    predictor.load_state_dict(checkpoint["model"])
    predictor.eval().to(device)
    rows = []
    offset = 0
    for raw_batch in loader:
        batch = _normalize_batch(raw_batch, normalization, device)
        result = predictor(
            batch["z"], batch["actions"], batch["task_id"], target=None, sample=False
        )
        final_prediction = result.predicted[:, -1]
        final_target = batch["target"][:, -1]
        latent_error = (final_prediction - final_target).square().mean(-1)
        decision_kl = gaussian_policy_kl(
            policy, final_target, final_prediction, batch["task_id"]
        )
        batch_size = len(latent_error)
        for index in range(batch_size):
            rows.append(
                {
                    "task_id": int(raw_batch["task_id"][index]),
                    "task": config["data"]["tasks"][int(raw_batch["task_id"][index])],
                    "latent_error": float(latent_error[index]),
                    "decision_kl": float(decision_kl[index]),
                    "success": float(raw_batch["success"][index, -1]),
                    "example": offset + index,
                }
            )
        offset += batch_size
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "predictions.csv", index=False)
    metrics: dict[str, Any] = {
        "method": method,
        "policy_quality": config["policy"]["quality"],
        "horizon": int(config["predictor"]["horizon"]),
        "examples": len(frame),
        "latent_error": float(frame.latent_error.mean()),
        "decision_kl": float(frame.decision_kl.mean()),
        "success_label_rate": float(frame.success.mean()),
        "pearson_latent_vs_decision": _correlation(
            frame.latent_error.to_numpy(), frame.decision_kl.to_numpy(), "pearson"
        ),
        "spearman_latent_vs_decision": _correlation(
            frame.latent_error.to_numpy(), frame.decision_kl.to_numpy(), "spearman"
        ),
        "pearson_latent_vs_success": _correlation(
            frame.latent_error.to_numpy(), frame.success.to_numpy(), "pearson"
        ),
        "pearson_decision_vs_success": _correlation(
            frame.decision_kl.to_numpy(), frame.success.to_numpy(), "pearson"
        ),
        "per_task": {},
    }
    for task, group in frame.groupby("task"):
        metrics["per_task"][task] = {
            "latent_error": float(group.latent_error.mean()),
            "decision_kl": float(group.decision_kl.mean()),
            "success_label_rate": float(group.success.mean()),
        }
    save_json(metrics, run_dir / "evaluation.json")
    _plot(frame, run_dir / "prediction_action_gap.png", method)
    return run_dir / "evaluation.json"


def _plot(frame: pd.DataFrame, path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for task, group in frame.groupby("task"):
        axis.scatter(group.latent_error, group.decision_kl, s=10, alpha=0.5, label=task)
    axis.set_xlabel("Normalized latent MSE")
    axis.set_ylabel("Reference-policy forward KL")
    axis.set_title(title)
    axis.set_xscale("symlog", linthresh=1e-5)
    axis.set_yscale("symlog", linthresh=1e-5)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def aggregate(config: dict[str, Any]) -> Path:
    output = Path(config["output"]["dir"])
    rows = []
    for path in output.glob("*/evaluation.json"):
        import json

        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append({key: value for key, value in metrics.items() if key != "per_task"})
    frame = pd.DataFrame(rows)
    destination = output / "summary.csv"
    frame.to_csv(destination, index=False)
    return destination


def main() -> None:
    args = parser("Evaluate prediction-action gap diagnostics").parse_args()
    config = load_config(args.config, args.set)
    print(evaluate(config))
    print(aggregate(config))


if __name__ == "__main__":
    main()
