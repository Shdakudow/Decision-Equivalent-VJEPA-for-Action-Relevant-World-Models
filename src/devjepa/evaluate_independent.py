from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader

from devjepa.config import load_config, parser
from devjepa.data import TransitionDataset, load_archive, split_archive
from devjepa.runtime import build_predictor, load_policy
from devjepa.utils import device_from_config, seed_everything


def _policy_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    result = deepcopy(config)
    result["experiment"]["seed"] = seed
    result["policy"]["quality"] = "strong"
    return result


@torch.inference_mode()
def evaluate_independent(config: dict[str, Any]) -> Path:
    device = device_from_config(config)
    source_seeds = [
        int(seed)
        for seed in config["evaluation"].get("source_seeds", [42, 1000, 10000])
    ]
    evaluator_seeds = [
        int(seed)
        for seed in config["evaluation"].get("evaluator_seeds", [42, 1000, 10000])
    ]
    methods = list(
        config["evaluation"].get("methods", ["vjepa", "de_vjepa", "de_vjepa_band"])
    )
    base_path = Path(config["data"]["encoded_path"])
    archives = {"clean": load_archive(base_path)}
    for shift in ("brightness", "desaturate", "blur"):
        path = base_path.with_name(f"{base_path.stem}_{shift}{base_path.suffix}")
        archives[shift] = load_archive(path)

    rows = []
    tag = config["experiment"].get("tag")
    for source_seed in source_seeds:
        seed_everything(source_seed)
        source_config = _policy_config(config, source_seed)
        source_policy, source_norm = load_policy(source_config, device)
        del source_policy
        _, clean_validation = split_archive(
            archives["clean"], float(config["data"]["validation_fraction"]), source_seed
        )
        evaluators = {}
        for evaluator_seed in evaluator_seeds:
            if evaluator_seed == source_seed:
                continue
            evaluators[evaluator_seed] = load_policy(
                _policy_config(config, evaluator_seed), device
            )

        for method in methods:
            run_config = deepcopy(source_config)
            run_config["experiment"]["method"] = method
            run_config["predictor"]["horizon"] = 1
            predictor = build_predictor(
                run_config,
                archives["clean"]["latents"].shape[-1],
                archives["clean"]["actions"].shape[-1],
            )
            run_name = f"{method}_strong_h1_seed{source_seed}"
            if tag:
                run_name = f"{run_name}_{tag}"
            checkpoint = torch.load(
                Path(config["output"]["dir"]) / run_name / "predictor.pt",
                map_location="cpu",
                weights_only=True,
            )
            predictor.load_state_dict(checkpoint["model"])
            predictor.eval().to(device)

            for shift, archive in archives.items():
                _, shifted_validation = split_archive(
                    archive, float(config["data"]["validation_fraction"]), source_seed
                )
                shifted_dataset = TransitionDataset(shifted_validation, horizon=1)
                clean_dataset = TransitionDataset(clean_validation, horizon=1)
                loader = DataLoader(
                    list(zip(shifted_dataset, clean_dataset)),
                    batch_size=int(config["evaluation"]["batch_size"]),
                    shuffle=False,
                )
                totals = {
                    evaluator_seed: {"kl": 0.0, "action_mse": 0.0, "count": 0}
                    for evaluator_seed in evaluators
                }
                latent_sum = 0.0
                latent_count = 0
                for shifted, clean in loader:
                    z = (
                        shifted["z"].to(device) - source_norm.latent_mean
                    ) / source_norm.latent_std
                    actions = (
                        shifted["actions"].to(device) - source_norm.action_mean
                    ) / source_norm.action_std
                    task_ids = shifted["task_id"].to(device)
                    prediction = predictor(
                        z, actions, task_ids, target=None, sample=False
                    ).predicted[:, -1]
                    predicted_raw = (
                        prediction * source_norm.latent_std + source_norm.latent_mean
                    )
                    target_raw = clean["target"][:, -1].to(device)
                    latent_sum += float((predicted_raw - target_raw).square().mean().cpu())
                    latent_count += 1

                    for evaluator_seed, (policy, normalization) in evaluators.items():
                        predicted = (
                            predicted_raw - normalization.latent_mean
                        ) / normalization.latent_std
                        target = (
                            target_raw - normalization.latent_mean
                        ) / normalization.latent_std
                        target_distribution = policy.distribution(target, task_ids)
                        predicted_distribution = policy.distribution(predicted, task_ids)
                        kl = kl_divergence(
                            target_distribution, predicted_distribution
                        ).sum(-1)
                        action_mse = (
                            target_distribution.mean - predicted_distribution.mean
                        ).square().mean(-1)
                        totals[evaluator_seed]["kl"] += float(kl.sum().cpu())
                        totals[evaluator_seed]["action_mse"] += float(
                            action_mse.sum().cpu()
                        )
                        totals[evaluator_seed]["count"] += len(kl)

                for evaluator_seed, total in totals.items():
                    rows.append(
                        {
                            "source_seed": source_seed,
                            "evaluator_seed": evaluator_seed,
                            "method": method,
                            "shift": shift,
                            "latent_mse_raw": latent_sum / max(latent_count, 1),
                            "independent_decision_kl": total["kl"] / total["count"],
                            "independent_action_mse": total["action_mse"]
                            / total["count"],
                        }
                    )

    output = Path(config["output"]["dir"])
    details = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    details.to_csv(output / f"independent_results{suffix}.csv", index=False)
    grouped = (
        details.groupby(["method", "shift"])[
            ["latent_mse_raw", "independent_decision_kl", "independent_action_mse"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    destination = output / f"independent_mean_std{suffix}.csv"
    grouped.to_csv(destination, index=False)
    print(grouped.to_string(index=False))
    return destination


def main() -> None:
    args = parser("Evaluate prior predictions with independent policies").parse_args()
    print(evaluate_independent(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
