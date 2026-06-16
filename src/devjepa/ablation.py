from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader, Dataset

from devjepa.config import load_config, parser
from devjepa.data import TransitionDataset, load_configured_archive, split_archive
from devjepa.runtime import build_predictor, load_policy
from devjepa.train_bc import train_bc
from devjepa.train_predictor import train_predictor
from devjepa.utils import device_from_config, seed_everything


class PairedShiftDataset(Dataset):
    def __init__(self, shifted: TransitionDataset, clean: TransitionDataset):
        if len(shifted) != len(clean):
            raise ValueError("Shifted and clean validation datasets must be aligned")
        self.shifted = shifted
        self.clean = clean

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int):
        return self.shifted[index], self.clean[index]


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "_")


def _policy_quality(reference_type: str) -> str:
    return {
        "none": "strong",
        "random": "random",
        "random_policy": "random",
        "weak": "weak",
        "weak_bc": "weak",
        "strong": "strong",
        "strong_bc": "strong",
    }[reference_type]


def _apply_overrides(
    config: dict[str, Any],
    values: dict[str, Any],
    prefix: str = "",
) -> None:
    for key, value in values.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "." not in key:
            _apply_overrides(config, value, dotted_key)
            continue
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value


def _policy_config(
    base: dict[str, Any], seed: int, quality: str, output_dir: Path
) -> dict[str, Any]:
    config = deepcopy(base)
    config["experiment"]["seed"] = seed
    config["policy"]["quality"] = quality
    config["output"]["dir"] = str(output_dir)
    return config


def _ensure_policy(config: dict[str, Any]) -> Path:
    seed = int(config["experiment"]["seed"])
    quality = config["policy"]["quality"]
    checkpoint = Path(config["output"]["dir"]) / f"policy_{quality}_seed{seed}" / "policy.pt"
    return checkpoint if checkpoint.exists() else train_bc(config)


def _predictor_run_name(config: dict[str, Any]) -> str:
    name = (
        f"{config['experiment']['method']}_{config['policy']['quality']}"
        f"_h{config['predictor']['horizon']}_seed{config['experiment']['seed']}"
    )
    tag = config["experiment"].get("tag")
    return f"{name}_{tag}" if tag else name


def _condition_tag(condition: dict[str, Any], timestamp: str) -> str:
    return "_".join(
        (
            _slug(condition["name"]),
            f"lambda{_slug(condition.get('lambda_de', condition.get('lambda', 0.0)))}",
            _slug(condition.get("bjepa_prior_type", "none")),
            _slug(condition.get("reference_policy_type", "none")),
            "actions" if condition.get("use_actions", True) else "noactions",
            _slug(condition.get("architecture", "mlp")),
            timestamp,
        )
    )


def _archive_path(base_path: Path, shift: str) -> Path:
    if shift == "clean":
        return base_path
    if shift == "desaturation":
        shift = "desaturate"
    return base_path.with_name(f"{base_path.stem}_{shift}{base_path.suffix}")


@torch.inference_mode()
def evaluate_condition(
    config: dict[str, Any],
    checkpoint_path: Path | None,
    evaluator_seeds: list[int],
    shifts: list[str],
    condition: dict[str, Any],
    timestamp: str,
) -> list[dict[str, Any]]:
    train_seed = int(config["experiment"]["seed"])
    if train_seed in evaluator_seeds:
        evaluator_seeds = [seed for seed in evaluator_seeds if seed != train_seed]
    if not evaluator_seeds:
        raise ValueError("At least one evaluator seed must differ from the training seed")

    device = device_from_config(config)
    source_policy, source_norm = load_policy(config, device)
    del source_policy
    evaluators = {}
    for evaluator_seed in evaluator_seeds:
        evaluator_config = _policy_config(
            config, evaluator_seed, "strong", Path(config["output"]["dir"])
        )
        evaluators[evaluator_seed] = load_policy(evaluator_config, device)

    clean_archive = load_configured_archive(config)
    _, clean_validation = split_archive(
        clean_archive, float(config["data"]["validation_fraction"]), train_seed
    )
    clean_dataset = TransitionDataset(clean_validation, horizon=1)
    predictor = None
    if condition["objective"] != "persistence":
        if checkpoint_path is None:
            raise ValueError("A trained checkpoint is required for non-analytic methods")
        predictor = build_predictor(
            config, clean_archive["latents"].shape[-1], clean_archive["actions"].shape[-1]
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        predictor.load_state_dict(checkpoint["model"])
        predictor.eval().to(device)

    rows: list[dict[str, Any]] = []
    base_path = Path(config["data"]["encoded_path"])
    for shift in shifts:
        shift_path = _archive_path(base_path, shift)
        if not shift_path.exists():
            raise FileNotFoundError(f"Missing encoded visual shift: {shift_path}")
        shifted_archive = load_configured_archive(config, shift_path)
        _, shifted_validation = split_archive(
            shifted_archive, float(config["data"]["validation_fraction"]), train_seed
        )
        shifted_dataset = TransitionDataset(shifted_validation, horizon=1)
        loader = DataLoader(
            PairedShiftDataset(shifted_dataset, clean_dataset),
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["runtime"].get("workers", 0)),
        )
        sample_offset = 0
        for shifted, clean in loader:
            z = (shifted["z"].to(device) - source_norm.latent_mean) / source_norm.latent_std
            actions = (
                shifted["actions"].to(device) - source_norm.action_mean
            ) / source_norm.action_std
            task_ids = shifted["task_id"].to(device)
            if predictor is None:
                prediction_result = None
                prediction = z
            else:
                prediction_result = predictor(
                    z, actions, task_ids, target=None, sample=False
                )
                prediction = prediction_result.predicted[:, -1]
            predicted_raw = prediction * source_norm.latent_std + source_norm.latent_mean
            target_raw = clean["target"][:, -1].to(device)
            latent_mse = (predicted_raw - target_raw).square().mean(-1)
            if (
                prediction_result is not None
                and prediction_result.predictive_logvar is not None
            ):
                predictive_logvar = prediction_result.predictive_logvar[:, -1].clamp(
                    -10.0, 5.0
                )
                predictive_variance = predictive_logvar.exp().clamp_min(1e-6)
                normalized_target = (
                    target_raw - source_norm.latent_mean
                ) / source_norm.latent_std
                nll = 0.5 * (
                    torch.log(
                        torch.as_tensor(
                            2.0 * torch.pi,
                            device=device,
                            dtype=prediction.dtype,
                        )
                    )
                    + predictive_logvar
                    + (normalized_target - prediction).square() / predictive_variance
                ).mean(-1)
                uncertainty = predictive_variance.mean(-1)
            else:
                nll = torch.full_like(latent_mse, torch.nan)
                uncertainty = torch.full_like(latent_mse, torch.nan)

            for evaluator_seed, (policy, normalization) in evaluators.items():
                if evaluator_seed == train_seed:
                    raise AssertionError("Evaluator policy leaked from the training seed")
                predicted = (
                    predicted_raw - normalization.latent_mean
                ) / normalization.latent_std
                target = (target_raw - normalization.latent_mean) / normalization.latent_std
                target_distribution = policy.distribution(target, task_ids)
                predicted_distribution = policy.distribution(predicted, task_ids)
                decision_kl = kl_divergence(
                    target_distribution, predicted_distribution
                ).sum(-1)
                action_mse = (
                    target_distribution.mean - predicted_distribution.mean
                ).square().mean(-1)
                for index in range(len(latent_mse)):
                    task = config["data"]["tasks"][int(task_ids[index])]
                    run_id = "_".join(
                        (
                            _slug(condition["name"]),
                            f"lambda{_slug(condition.get('lambda_de', condition.get('lambda', 0.0)))}",
                            _slug(condition.get("bjepa_prior_type", "none")),
                            _slug(condition.get("reference_policy_type", "none")),
                            _slug(task),
                            f"seed{train_seed}",
                            f"eval{evaluator_seed}",
                            _slug(shift),
                            timestamp,
                        )
                    )
                    rows.append(
                        {
                            "run_id": run_id,
                            "sample_id": sample_offset + index,
                            "dataset": config.get("dataset_name", "unknown"),
                            "task": task,
                            "train_seed": train_seed,
                            "evaluator_seed": evaluator_seed,
                            "method": condition["name"],
                            "lambda": float(
                                condition.get("lambda_de", condition.get("lambda", 0.0))
                            ),
                            "lambda_de": float(
                                condition.get("lambda_de", condition.get("lambda", 0.0))
                            ),
                            "bjepa_prior_type": condition.get(
                                "bjepa_prior_type", "none"
                            ),
                            "bjepa_loss_type": condition.get(
                                "bjepa_loss_type", "nll"
                            ),
                            "reference_policy_type": condition.get(
                                "reference_policy_type", "none"
                            ),
                            "shift": shift,
                            "latent_mse": float(latent_mse[index]),
                            "decision_kl": float(decision_kl[index]),
                            "action_mse": float(action_mse[index]),
                            "nll": float(nll[index]),
                            "uncertainty": float(uncertainty[index]),
                        }
                    )
            sample_offset += len(latent_mse)
    return rows


def run_experiment(experiment_config: dict[str, Any]) -> Path:
    config_path = Path(experiment_config["_config_path"])
    base_path = Path(experiment_config["base_config"])
    if not base_path.is_absolute():
        base_path = (config_path.parent.parent / base_path).resolve()
    base = load_config(base_path)
    _apply_overrides(base, experiment_config.get("base_overrides", {}))
    base["dataset_name"] = experiment_config.get(
        "dataset_name", experiment_config["name"]
    )
    base["data"]["archive_tasks"] = list(
        experiment_config.get("archive_tasks", base["data"]["tasks"])
    )
    base["data"]["tasks"] = list(experiment_config["tasks"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_root = Path(experiment_config["artifacts_dir"]) / (
        f"{experiment_config['name']}_{timestamp}"
    )
    artifact_root.mkdir(parents=True, exist_ok=False)
    seeds = [int(seed) for seed in experiment_config["seeds"]]
    evaluator_seeds = [int(seed) for seed in experiment_config["evaluator_seeds"]]
    conditions = list(experiment_config["conditions"])

    required_policies = {(seed, "strong") for seed in set(seeds + evaluator_seeds)}
    for condition in conditions:
        reference_type = condition.get("reference_policy_type", "none")
        if reference_type != "none":
            required_policies.update(
                (seed, _policy_quality(reference_type)) for seed in seeds
            )
    for seed, quality in sorted(required_policies):
        _ensure_policy(_policy_config(base, seed, quality, artifact_root))

    rows = []
    for train_seed in seeds:
        for condition in conditions:
            reference_type = condition.get("reference_policy_type", "none")
            training_quality = _policy_quality(reference_type)
            run_config = _policy_config(base, train_seed, training_quality, artifact_root)
            run_config["experiment"]["method"] = condition["objective"]
            run_config["experiment"]["tag"] = _condition_tag(condition, timestamp)
            run_config["loss"]["decision_weight"] = float(
                condition.get("lambda_de", condition.get("lambda", 0.0))
            )
            for key in ("epsilon0", "uncertainty_scale", "band_type"):
                if key in condition:
                    run_config["loss"][key] = condition[key]
            run_config.setdefault("bjepa", {})
            run_config["bjepa"]["prior_type"] = condition.get(
                "bjepa_prior_type", "none"
            )
            run_config["bjepa"]["loss_type"] = condition.get(
                "bjepa_loss_type", "nll"
            )
            run_config["predictor"]["use_actions"] = bool(
                condition.get("use_actions", True)
            )
            run_config["predictor"]["architecture"] = condition.get(
                "architecture", "mlp"
            )
            checkpoint = None
            if condition["objective"] != "persistence":
                checkpoint = (
                    artifact_root / _predictor_run_name(run_config) / "predictor.pt"
                )
                if not checkpoint.exists():
                    checkpoint = train_predictor(run_config)
            rows.extend(
                evaluate_condition(
                    run_config,
                    checkpoint,
                    evaluator_seeds,
                    list(experiment_config["shifts"]),
                    condition,
                    timestamp,
                )
            )

    output_path = Path(experiment_config["output"]["raw"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        backup = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        output_path.replace(backup)
    frame = pd.DataFrame(rows)
    numeric = frame[["latent_mse", "decision_kl", "action_mse"]]
    if frame.empty or not torch.isfinite(torch.from_numpy(numeric.to_numpy())).all():
        raise RuntimeError("Experiment produced empty or non-finite metrics")
    if (frame["train_seed"] == frame["evaluator_seed"]).any():
        raise RuntimeError("Independent evaluation invariant violated")
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parser("Run a config-driven DE-VJEPA ablation").parse_args()
    print(run_experiment(load_config(args.config, args.set)))
