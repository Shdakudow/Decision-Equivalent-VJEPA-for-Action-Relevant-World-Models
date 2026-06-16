from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.distributions import kl_divergence

from devjepa.collect import _make_env
from devjepa.config import load_config, parser
from devjepa.data import Normalization
from devjepa.encode import _build_encoder
from devjepa.progress import load_progress_head
from devjepa.runtime import build_predictor, load_policy
from devjepa.utils import device_from_config, seed_everything


SHIFT_ALIASES = {"desaturation": "desaturate"}
METHOD_TO_MODEL = {
    "bc_plus_vjepa": "vjepa",
    "bc_plus_de_vjepa": "de_vjepa",
    "bc_plus_bjepa": "bjepa",
    "bc_plus_de_bjepa": "de_bjepa",
    # Backward-compatible names used by the first diagnostic pilot.
    "vjepa": "vjepa",
    "de_vjepa": "de_vjepa",
    "bjepa": "bjepa",
    "de_bjepa": "de_bjepa",
    "bc_plus_vjepa_value_planner": "vjepa",
    "bc_plus_de_vjepa_value_planner": "de_vjepa",
    "bc_plus_vjepa_cem": "vjepa",
    "bc_plus_de_vjepa_cem": "de_vjepa",
}
PLANNER_SCORES = {
    "entropy_score",
    "decision_consistency_score",
    "action_mean_consistency_score",
    "low_uncertainty_score",
    "random_candidate_control",
}


def _method_requires_predictor(method: str) -> bool:
    return method != "bc_only"


def _validate_independent_policy(
    policy_seed: int, predictor_seeds: list[int], required: bool
) -> None:
    if required and policy_seed in predictor_seeds:
        raise ValueError("Evaluation policy seed must differ from predictor training seeds")


def _shift_image(image: Image.Image, condition: str) -> Image.Image:
    condition = SHIFT_ALIASES.get(condition, condition)
    if condition == "clean":
        return image
    if condition == "brightness":
        return ImageEnhance.Brightness(image).enhance(0.55)
    if condition == "desaturate":
        return ImageEnhance.Color(image).enhance(0.15)
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if condition == "background_shift":
        pixels = np.asarray(image).copy()
        height, width, _ = pixels.shape
        channel_range = pixels.max(axis=-1) - pixels.min(axis=-1)
        border = np.zeros((height, width), dtype=bool)
        border[: height // 5] = True
        border[:, : width // 12] = True
        border[:, -width // 12 :] = True
        background = border & (channel_range < 55)
        tint = np.asarray([45, 95, 155], dtype=np.float32)
        pixels[background] = (
            0.35 * pixels[background].astype(np.float32) + 0.65 * tint
        ).clip(0, 255).astype(np.uint8)
        return Image.fromarray(pixels)
    raise ValueError(f"Unknown visual condition: {condition}")


def _experiment_root(
    config: dict[str, Any],
    artifacts_dir: str | Path | None = None,
    experiment_name: str | None = None,
) -> Path:
    artifacts_dir = Path(
        artifacts_dir or config["closed_loop"]["artifacts_dir"]
    )
    experiment_name = str(
        experiment_name or config["closed_loop"]["source_experiment"]
    )
    roots = sorted(
        artifacts_dir.glob(f"{experiment_name}_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not roots:
        raise FileNotFoundError(f"No experiment roots matching {experiment_name}_*")
    return roots[0]


def _find_run(
    config: dict[str, Any],
    model_name: str,
    seed: int,
    lambda_de: float | None = None,
    artifacts_dir: str | Path | None = None,
    source_experiment: str | None = None,
) -> Path:
    prefixes = {
        "vjepa": f"vjepa_strong_h1_seed{seed}_vjepa_",
        "de_vjepa": f"hybrid_strong_h1_seed{seed}_de_vjepa_",
        "bjepa": f"bjepa_strong_h1_seed{seed}_bjepa_",
        "de_bjepa": f"de_bjepa_strong_h1_seed{seed}_de_bjepa_",
    }
    if model_name not in prefixes:
        raise ValueError(f"No checkpoint mapping for model: {model_name}")
    root = _experiment_root(config, artifacts_dir, source_experiment)
    candidates = sorted(root.glob(f"{prefixes[model_name]}*/predictor.pt"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {model_name} seed {seed}")
    lambda_de = float(
        config["closed_loop"].get("lambda_de", 0.01)
        if lambda_de is None
        else lambda_de
    )
    if model_name in {"de_vjepa", "de_bjepa"}:
        token = str(lambda_de).replace(".", "p")
        matched = [path for path in candidates if f"lambda{token}_" in str(path.parent)]
        if matched:
            candidates = matched
        elif lambda_de != 0.01:
            raise FileNotFoundError(
                f"No {model_name} checkpoint with lambda_de={lambda_de}; "
                "train it or use lambda_de=0.01"
            )
    return candidates[-1]


def _policy_config(
    config: dict[str, Any],
    seed: int,
    artifacts_dir: str | Path | None = None,
    source_experiment: str | None = None,
) -> dict[str, Any]:
    result = deepcopy(config)
    result["experiment"]["seed"] = seed
    result["policy"]["quality"] = "strong"
    result["output"]["dir"] = str(
        _experiment_root(config, artifacts_dir, source_experiment)
    )
    return result


def _load_predictor(
    config: dict[str, Any],
    model_name: str,
    seed: int,
    latent_dim: int,
    action_dim: int,
    device: torch.device,
    lambda_de: float | None = None,
    artifacts_dir: str | Path | None = None,
    source_experiment: str | None = None,
) -> tuple[torch.nn.Module, Normalization]:
    checkpoint = torch.load(
        _find_run(
            config,
            model_name,
            seed,
            lambda_de,
            artifacts_dir,
            source_experiment,
        ),
        map_location="cpu",
        weights_only=True,
    )
    run_config = deepcopy(config)
    run_config["experiment"]["method"] = model_name
    run_config["experiment"]["seed"] = seed
    run_config["predictor"]["horizon"] = 1
    if model_name in {"bjepa", "de_bjepa"}:
        run_config.setdefault("bjepa", {})["prior_type"] = "empirical_task_gaussian"
    predictor = build_predictor(run_config, latent_dim, action_dim)
    predictor.load_state_dict(checkpoint["model"])
    predictor.eval().requires_grad_(False).to(device)
    _, normalization = load_policy(
        _policy_config(config, seed, artifacts_dir, source_experiment), device
    )
    return predictor, normalization


def _method_specs(section: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for value in section["methods"]:
        if isinstance(value, str):
            specs.append(
                {
                    "name": value,
                    "model": METHOD_TO_MODEL.get(value),
                    "lambda_de": float(section.get("lambda_de", 0.01)),
                    "artifacts_dir": section["artifacts_dir"],
                    "source_experiment": section["source_experiment"],
                }
            )
        else:
            spec = dict(value)
            spec.setdefault("model", METHOD_TO_MODEL.get(spec["name"]))
            spec.setdefault("lambda_de", float(section.get("lambda_de", 0.01)))
            spec.setdefault("artifacts_dir", section["artifacts_dir"])
            spec.setdefault("source_experiment", section["source_experiment"])
            specs.append(spec)
    return specs


def _predict_next(
    predictor: torch.nn.Module,
    normalization: Normalization,
    latent_raw: torch.Tensor,
    actions_raw: torch.Tensor,
    task_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch = actions_raw.shape[0]
    latent_batch = latent_raw.expand(batch, -1)
    normalized_latent = (
        latent_batch - normalization.latent_mean
    ) / normalization.latent_std
    normalized_actions = (
        actions_raw - normalization.action_mean
    ) / normalization.action_std
    result = predictor(
        normalized_latent,
        normalized_actions.unsqueeze(1),
        task_ids.expand(batch),
        target=None,
        sample=False,
    )
    predicted = result.predicted[:, -1]
    predictive_variance = None
    if result.predictive_logvar is not None:
        predictive_variance = result.predictive_logvar[:, -1].clamp(-10.0, 5.0).exp().mean(-1)
    return (
        predicted * normalization.latent_std + normalization.latent_mean,
        predictive_variance,
    )


def _generate_candidate_actions(
    bc_action: torch.Tensor,
    count: int,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1:
        raise ValueError("num_candidate_actions must be positive")
    if count == 1:
        return bc_action.clone()
    noise = torch.randn(
        count - 1,
        bc_action.shape[-1],
        generator=generator,
        device=bc_action.device,
    )
    return torch.cat(
        (bc_action, bc_action.expand(count - 1, -1) + noise_std * noise),
        dim=0,
    ).clamp(-1.0, 1.0)


def _planner_scores(
    planner_score: str,
    candidates: torch.Tensor,
    bc_action: torch.Tensor,
    current_distribution,
    predicted_raw: torch.Tensor,
    predictive_variance: torch.Tensor | None,
    task_ids: torch.Tensor,
    policy: torch.nn.Module,
    policy_normalization: Normalization,
    prior_weight: float,
) -> torch.Tensor:
    if planner_score not in PLANNER_SCORES:
        raise ValueError(f"Unknown planner_score: {planner_score}")
    predicted_policy = (
        predicted_raw - policy_normalization.latent_mean
    ) / policy_normalization.latent_std
    distributions = policy.distribution(
        predicted_policy, task_ids.expand(len(candidates))
    )
    deviation = (candidates - bc_action).square().mean(-1)
    if planner_score == "entropy_score":
        model_score = distributions.entropy().sum(-1)
    elif planner_score == "decision_consistency_score":
        current = torch.distributions.Normal(
            current_distribution.mean.expand_as(distributions.mean),
            current_distribution.stddev.expand_as(distributions.stddev),
        )
        model_score = kl_divergence(current, distributions).sum(-1)
    elif planner_score == "action_mean_consistency_score":
        model_score = (distributions.mean - bc_action).square().mean(-1)
    elif planner_score == "low_uncertainty_score":
        if predictive_variance is None:
            raise ValueError(
                "low_uncertainty_score requires a predictor with predictive variance"
            )
        model_score = predictive_variance
    else:
        # Selection happens in _candidate_action; keep a finite score for logging/tests.
        model_score = torch.zeros_like(deviation)
    scores = model_score + prior_weight * deviation
    if not torch.isfinite(scores).all():
        raise FloatingPointError(f"Non-finite planner scores for {planner_score}")
    return scores


def _candidate_action(
    predictor: torch.nn.Module,
    normalization: Normalization,
    latent_raw: torch.Tensor,
    bc_action: torch.Tensor,
    task_ids: torch.Tensor,
    current_distribution,
    policy: torch.nn.Module,
    policy_normalization: Normalization,
    config: dict[str, Any],
    planner_score: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    count = int(config["closed_loop"].get("num_candidate_actions", 32))
    noise_std = float(config["closed_loop"].get("candidate_noise_std", 0.12))
    prior_weight = float(config["closed_loop"].get("candidate_prior_weight", 0.1))
    candidates = _generate_candidate_actions(bc_action, count, noise_std, generator)
    predicted_raw, predictive_variance = _predict_next(
        predictor, normalization, latent_raw, candidates, task_ids
    )
    scores = _planner_scores(
        planner_score,
        candidates,
        bc_action,
        current_distribution,
        predicted_raw,
        predictive_variance,
        task_ids,
        policy,
        policy_normalization,
        prior_weight,
    )
    if planner_score == "random_candidate_control":
        best = int(
            torch.randint(
                0,
                len(candidates),
                (1,),
                generator=generator,
                device=candidates.device,
            )
        )
    else:
        best = int(scores.argmin())
    return (
        candidates[best : best + 1],
        predicted_raw[best : best + 1],
        float(scores[best].cpu()),
    )


def _progress_values(
    progress_head: torch.nn.Module,
    progress_mean: torch.Tensor,
    progress_std: torch.Tensor,
    predicted_raw: torch.Tensor,
    task_ids: torch.Tensor,
) -> torch.Tensor:
    values = progress_head(
        (predicted_raw - progress_mean) / progress_std,
        task_ids.expand(len(predicted_raw)),
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("Non-finite progress-head values")
    return values


def _value_candidate_action(
    predictor: torch.nn.Module,
    normalization: Normalization,
    latent_raw: torch.Tensor,
    bc_action: torch.Tensor,
    task_ids: torch.Tensor,
    progress_head: torch.nn.Module,
    progress_mean: torch.Tensor,
    progress_std: torch.Tensor,
    config: dict[str, Any],
    alpha: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    section = config["closed_loop"]
    candidates = _generate_candidate_actions(
        bc_action,
        int(section.get("num_candidate_actions", 32)),
        float(section.get("candidate_noise_std", 0.12)),
        generator,
    )
    predicted_raw, _ = _predict_next(
        predictor, normalization, latent_raw, candidates, task_ids
    )
    values = _progress_values(
        progress_head, progress_mean, progress_std, predicted_raw, task_ids
    )
    deviation = (candidates - bc_action).square().mean(-1)
    scores = -values + alpha * deviation
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite value-planner scores")
    best = int(scores.argmin())
    return (
        candidates[best : best + 1],
        predicted_raw[best : best + 1],
        float(scores[best].cpu()),
        float(values[best].cpu()),
        float(values[0].cpu()),
    )


def _initialize_cem_population(
    bc_action: torch.Tensor,
    population_size: int,
    horizon: int,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if population_size < 2 or horizon < 1:
        raise ValueError("CEM requires population_size >= 2 and horizon >= 1")
    mean = bc_action.expand(population_size, horizon, -1)
    noise = torch.randn(
        population_size,
        horizon,
        bc_action.shape[-1],
        generator=generator,
        device=bc_action.device,
    )
    population = (mean + noise_std * noise).clamp(-1.0, 1.0)
    population[0] = bc_action.expand(horizon, -1)
    return population


def _select_cem_elites(
    population: torch.Tensor,
    scores: torch.Tensor,
    elite_fraction: float,
) -> torch.Tensor:
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite CEM scores")
    count = max(1, int(math.ceil(len(population) * elite_fraction)))
    return population[torch.topk(scores, count, largest=False).indices]


def _score_action_sequences(
    predictor: torch.nn.Module,
    normalization: Normalization,
    latent_raw: torch.Tensor,
    sequences: torch.Tensor,
    bc_action: torch.Tensor,
    task_ids: torch.Tensor,
    progress_head: torch.nn.Module,
    progress_mean: torch.Tensor,
    progress_std: torch.Tensor,
    gamma: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current = latent_raw
    discounted_value = torch.zeros(len(sequences), device=sequences.device)
    first_prediction = None
    for step in range(sequences.shape[1]):
        current, _ = _predict_next(
            predictor, normalization, current, sequences[:, step], task_ids
        )
        if first_prediction is None:
            first_prediction = current
        discounted_value += (gamma**step) * _progress_values(
            progress_head, progress_mean, progress_std, current, task_ids
        )
    deviation = (
        sequences - bc_action.view(1, 1, -1)
    ).square().mean(dim=(1, 2))
    scores = -discounted_value + alpha * deviation
    return scores, discounted_value, first_prediction


def _cem_action(
    predictor: torch.nn.Module,
    normalization: Normalization,
    latent_raw: torch.Tensor,
    bc_action: torch.Tensor,
    task_ids: torch.Tensor,
    progress_head: torch.nn.Module,
    progress_mean: torch.Tensor,
    progress_std: torch.Tensor,
    config: dict[str, Any],
    horizon: int,
    alpha: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    section = config["closed_loop"]
    population_size = int(section.get("population_size", 128))
    iterations = int(section.get("cem_iterations", 3))
    elite_fraction = float(section.get("elite_fraction", 0.1))
    initial_std = float(section.get("action_noise_init_std", 0.25))
    minimum_std = float(section.get("cem_min_std", 0.02))
    gamma = float(section.get("planning_gamma", 0.95))
    mean = bc_action.expand(horizon, -1).clone()
    std = torch.full_like(mean, initial_std)
    best_sequence = mean
    best_score = torch.tensor(float("inf"), device=bc_action.device)
    best_value = torch.tensor(float("nan"), device=bc_action.device)
    best_prediction = latent_raw
    for iteration in range(iterations):
        if iteration == 0:
            population = _initialize_cem_population(
                bc_action, population_size, horizon, initial_std, generator
            )
        else:
            noise = torch.randn(
                population_size,
                horizon,
                bc_action.shape[-1],
                generator=generator,
                device=bc_action.device,
            )
            population = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1.0, 1.0)
            population[0] = mean.clamp(-1.0, 1.0)
        scores, values, first_predictions = _score_action_sequences(
            predictor,
            normalization,
            latent_raw,
            population,
            bc_action,
            task_ids,
            progress_head,
            progress_mean,
            progress_std,
            gamma,
            alpha,
        )
        elites = _select_cem_elites(population, scores, elite_fraction)
        mean = elites.mean(0)
        std = elites.std(0, unbiased=False).clamp_min(minimum_std)
        index = int(scores.argmin())
        if scores[index] < best_score:
            best_score = scores[index]
            best_value = values[index]
            best_sequence = population[index]
            best_prediction = first_predictions[index : index + 1]
    bc_prediction, _ = _predict_next(
        predictor, normalization, latent_raw, bc_action, task_ids
    )
    bc_value = _progress_values(
        progress_head, progress_mean, progress_std, bc_prediction, task_ids
    )
    return (
        best_sequence[:1],
        best_prediction,
        float(best_score.cpu()),
        float(best_value.cpu()),
        float(bc_value[0].cpu()),
    )


def _encode_frame(
    env,
    condition: str,
    image_size: int,
    transform,
    encoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    frame = Image.fromarray(np.asarray(env.render())).resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    image = transform(_shift_image(frame, condition)).unsqueeze(0).to(device)
    return encoder(image).float()


def _finite_action(action: torch.Tensor) -> np.ndarray:
    if not torch.isfinite(action).all():
        raise FloatingPointError("Planner produced a non-finite action")
    return action.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()


@torch.inference_mode()
def run_closed_loop(config: dict[str, Any]) -> tuple[Path, Path]:
    section = config["closed_loop"]
    device = device_from_config(config)
    method_specs = _method_specs(section)
    methods = [spec["name"] for spec in method_specs]
    specs_by_name = {spec["name"]: spec for spec in method_specs}
    tasks = list(section.get("tasks", config["data"]["tasks"]))
    archive_tasks = list(config["data"]["tasks"])
    seeds = [int(seed) for seed in section["seeds"]]
    policy_seed = int(section["policy_seed"])
    _validate_independent_policy(
        policy_seed,
        seeds,
        bool(section.get("require_independent_policy", True)),
    )
    conditions = list(section.get("visual_conditions", section.get("shifts", ["clean"])))
    episodes = int(
        section.get("episodes_per_task_seed", section.get("episodes_per_condition", 5))
    )
    max_steps = int(section.get("max_episode_steps", config["data"]["max_steps"]))
    planner_type = str(section.get("planner_type", "bc_action"))
    if planner_type not in {
        "bc_action",
        "one_step_candidate_planner",
        "one_step_value_planner",
        "cem_mpc",
    }:
        raise ValueError(f"Unknown planner_type: {planner_type}")
    planner_scores = list(section.get("planner_scores", ["entropy_score"]))
    unknown_scores = sorted(set(planner_scores) - PLANNER_SCORES)
    if unknown_scores and planner_type == "one_step_candidate_planner":
        raise ValueError(f"Unknown planner scores: {unknown_scores}")
    planner_alphas = [float(value) for value in section.get("planner_alphas", [0.1])]
    planning_horizons = [int(value) for value in section.get("planning_horizons", [3])]

    encoder, transform, latent_dim = _build_encoder(config["encoder"]["name"])
    encoder.eval().requires_grad_(False).to(device)
    policy, policy_norm = load_policy(_policy_config(config, policy_seed), device)
    action_dim = int(policy.mean.out_features)
    predictors: dict[tuple[str, int], tuple[torch.nn.Module, Normalization]] = {}
    for spec in method_specs:
        method = spec["name"]
        if not _method_requires_predictor(method):
            continue
        if not spec.get("model"):
            raise ValueError(f"Unknown closed-loop method: {method}")
        for seed in seeds:
            predictors[(method, seed)] = _load_predictor(
                config,
                str(spec["model"]),
                seed,
                latent_dim,
                action_dim,
                device,
                float(spec["lambda_de"]),
                spec["artifacts_dir"],
                spec["source_experiment"],
            )
    progress_heads = {}
    if planner_type in {"one_step_value_planner", "cem_mpc"}:
        for seed in seeds:
            progress_heads[seed] = load_progress_head(
                section.get("value_head_checkpoint_dir", "checkpoints"),
                int(section.get("value_head_seed", seed)),
                device,
                str(section.get("value_head_checkpoint_prefix", "value_head")),
            )

    raw_path = Path(
        section.get(
            "raw_output",
            section.get("output", "results/closed_loop_rollout_raw.csv"),
        )
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    resume = bool(section.get("resume", True))
    existing = pd.read_csv(raw_path) if resume and raw_path.exists() else pd.DataFrame()
    if not existing.empty and "planner_score" not in existing:
        existing["planner_score"] = np.where(
            existing["method"] == "bc_only", "bc_only", "entropy_score"
        )
    if not existing.empty and "run_timestamp" not in existing:
        existing["run_timestamp"] = "legacy"
    if not existing.empty and "planner_alpha" not in existing:
        existing["planner_alpha"] = 0.0
    if not existing.empty and "planning_horizon" not in existing:
        existing["planning_horizon"] = 0
    run_timestamp = (
        str(existing["run_timestamp"].iloc[0])
        if not existing.empty and "run_timestamp" in existing
        else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    completed_keys: set[tuple[Any, ...]] = set()
    if not existing.empty:
        completed_keys = set(
            existing[
                [
                    "method",
                    "planner_score",
                    "planner_alpha",
                    "planning_horizon",
                    "task",
                    "seed",
                    "visual_condition",
                    "episode_id",
                    "environment_seed",
                ]
            ].itertuples(index=False, name=None)
        )
    rows: list[dict[str, Any]] = existing.to_dict("records")
    base_seed = int(section.get("environment_seed", 2026))
    image_size = int(config["data"]["image_size"])
    camera = str(config["data"].get("camera", "corner2"))
    for task in tasks:
        task_id = archive_tasks.index(task)
        task_tensor = torch.tensor([task_id], device=device)
        for condition_id, condition in enumerate(conditions):
            for seed_index, seed in enumerate(seeds):
                for episode in range(episodes):
                    reset_seed = (
                        base_seed
                        + task_id * 100000
                        + condition_id * 10000
                        + seed_index * 1000
                        + episode
                    )
                    jobs = (
                        [("bc_only", "bc_only", 0.0, 0)]
                        if "bc_only" in methods
                        else []
                    )
                    if planner_type == "one_step_value_planner":
                        jobs.extend(
                            (method, "value_score", alpha, 1)
                            for alpha in planner_alphas
                            for method in methods
                            if method != "bc_only"
                        )
                    elif planner_type == "cem_mpc":
                        jobs.extend(
                            (method, "cem_value", alpha, horizon)
                            for horizon in planning_horizons
                            for alpha in planner_alphas
                            for method in methods
                            if method != "bc_only"
                        )
                    else:
                        jobs.extend(
                            (method, planner_score, 0.0, 1)
                            for planner_score in planner_scores
                            for method in methods
                            if method != "bc_only"
                        )
                    for method, planner_score, planner_alpha, planning_horizon in jobs:
                        method_lambda = float(specs_by_name[method]["lambda_de"])
                        episode_key = (
                            method,
                            planner_score,
                            planner_alpha,
                            planning_horizon,
                            task,
                            seed,
                            condition,
                            episode,
                            reset_seed,
                        )
                        if episode_key in completed_keys:
                            continue
                        seed_everything(reset_seed)
                        env = _make_env(task, reset_seed, camera)
                        success = False
                        total_return = 0.0
                        action_norms: list[float] = []
                        entropies: list[float] = []
                        latent_errors: list[float] = []
                        decision_kls: list[float] = []
                        action_errors: list[float] = []
                        action_deviations: list[float] = []
                        planner_score_values: list[float] = []
                        predicted_values: list[float] = []
                        bc_predicted_values: list[float] = []
                        final_info: dict[str, Any] = {}
                        terminated = truncated = False
                        steps = 0
                        try:
                            env.reset(seed=reset_seed)
                            latent_raw = _encode_frame(
                                env, condition, image_size, transform, encoder, device
                            )
                            for step in range(max_steps):
                                latent_policy = (
                                    latent_raw - policy_norm.latent_mean
                                ) / policy_norm.latent_std
                                distribution = policy.distribution(latent_policy, task_tensor)
                                bc_action = distribution.mean
                                entropies.append(float(distribution.entropy().sum().cpu()))
                                action = bc_action
                                predicted_raw = None
                                if method != "bc_only":
                                    predictor, predictor_norm = predictors[(method, seed)]
                                    if planner_type == "one_step_candidate_planner":
                                        candidate_generator = torch.Generator(
                                            device=device
                                        ).manual_seed(reset_seed * 1000 + step)
                                        action, predicted_raw, selected_score = _candidate_action(
                                            predictor,
                                            predictor_norm,
                                            latent_raw,
                                            bc_action,
                                            task_tensor,
                                            distribution,
                                            policy,
                                            policy_norm,
                                            config,
                                            planner_score,
                                            candidate_generator,
                                        )
                                        planner_score_values.append(selected_score)
                                    elif planner_type == "one_step_value_planner":
                                        candidate_generator = torch.Generator(
                                            device=device
                                        ).manual_seed(reset_seed * 1000 + step)
                                        head, head_mean, head_std = progress_heads[seed]
                                        (
                                            action,
                                            predicted_raw,
                                            selected_score,
                                            selected_value,
                                            bc_predicted_value,
                                        ) = _value_candidate_action(
                                            predictor,
                                            predictor_norm,
                                            latent_raw,
                                            bc_action,
                                            task_tensor,
                                            head,
                                            head_mean,
                                            head_std,
                                            config,
                                            planner_alpha,
                                            candidate_generator,
                                        )
                                        planner_score_values.append(selected_score)
                                        predicted_values.append(selected_value)
                                        bc_predicted_values.append(bc_predicted_value)
                                    elif planner_type == "cem_mpc":
                                        candidate_generator = torch.Generator(
                                            device=device
                                        ).manual_seed(reset_seed * 1000 + step)
                                        head, head_mean, head_std = progress_heads[seed]
                                        (
                                            action,
                                            predicted_raw,
                                            selected_score,
                                            selected_value,
                                            bc_predicted_value,
                                        ) = _cem_action(
                                            predictor,
                                            predictor_norm,
                                            latent_raw,
                                            bc_action,
                                            task_tensor,
                                            head,
                                            head_mean,
                                            head_std,
                                            config,
                                            planning_horizon,
                                            planner_alpha,
                                            candidate_generator,
                                        )
                                        planner_score_values.append(selected_score)
                                        predicted_values.append(selected_value)
                                        bc_predicted_values.append(bc_predicted_value)
                                    else:
                                        predicted_raw, _ = _predict_next(
                                            predictor,
                                            predictor_norm,
                                            latent_raw,
                                            bc_action,
                                            task_tensor,
                                        )
                                action_deviations.append(
                                    float((action - bc_action).square().mean().sqrt().cpu())
                                )
                                action_array = _finite_action(action)
                                action_norms.append(float(np.linalg.norm(action_array)))
                                _, reward, terminated, truncated, final_info = env.step(
                                    action_array
                                )
                                total_return += float(reward)
                                steps = step + 1
                                success = success or bool(final_info.get("success", False))
                                next_latent_raw = _encode_frame(
                                    env, condition, image_size, transform, encoder, device
                                )
                                if predicted_raw is not None:
                                    latent_errors.append(
                                        float((predicted_raw - next_latent_raw).square().mean().cpu())
                                    )
                                    target_policy = (
                                        next_latent_raw - policy_norm.latent_mean
                                    ) / policy_norm.latent_std
                                    predicted_policy = (
                                        predicted_raw - policy_norm.latent_mean
                                    ) / policy_norm.latent_std
                                    target_distribution = policy.distribution(
                                        target_policy, task_tensor
                                    )
                                    predicted_distribution = policy.distribution(
                                        predicted_policy, task_tensor
                                    )
                                    decision_kls.append(
                                        float(
                                            kl_divergence(
                                                target_distribution,
                                                predicted_distribution,
                                            )
                                            .sum(-1)
                                            .cpu()
                                        )
                                    )
                                    action_errors.append(
                                        float(
                                            (
                                                target_distribution.mean
                                                - predicted_distribution.mean
                                            )
                                            .square()
                                            .mean()
                                            .cpu()
                                        )
                                    )
                                latent_raw = next_latent_raw
                                if terminated or truncated or success:
                                    break
                        finally:
                            env.close()
                        if success:
                            failure_reason = ""
                        elif truncated or steps >= max_steps:
                            failure_reason = "timeout"
                        elif terminated:
                            failure_reason = "terminated"
                        else:
                            failure_reason = "no_success"
                        final_distance = final_info.get("obj_to_target", np.nan)
                        run_id = "_".join(
                            (
                                str(config.get("experiment_name", "closed_loop")),
                                method,
                                planner_score,
                                f"h{planning_horizon}",
                                f"alpha{str(planner_alpha).replace('.', 'p')}",
                                f"lambda{str(method_lambda).replace('.', 'p')}",
                                task,
                                f"seed{seed}",
                                condition,
                                run_timestamp,
                            )
                        )
                        row = {
                                "run_id": run_id,
                                "run_timestamp": run_timestamp,
                                "method": method,
                                "planner_score": planner_score,
                                "planner_alpha": planner_alpha,
                                "planning_horizon": planning_horizon,
                                "task": task,
                                "seed": seed,
                                "predictor_seed": seed,
                                "policy_seed": policy_seed,
                                "visual_condition": condition,
                                "episode_id": episode,
                                "environment_seed": reset_seed,
                                "planner_type": planner_type,
                                "num_candidate_actions": int(
                                    section.get("num_candidate_actions", 0)
                                ),
                                "lambda_de": method_lambda,
                                "success": float(success),
                                "return": total_return,
                                "episode_length": steps,
                                "final_distance_to_goal": float(final_distance),
                                "failure_reason": failure_reason,
                                "mean_action_norm": float(np.mean(action_norms)),
                                "mean_action_deviation_from_bc": float(
                                    np.mean(action_deviations)
                                ),
                                "mean_selected_planner_score": (
                                    float(np.mean(planner_score_values))
                                    if planner_score_values
                                    else np.nan
                                ),
                                "mean_predicted_value": (
                                    float(np.mean(predicted_values))
                                    if predicted_values
                                    else np.nan
                                ),
                                "mean_predicted_progress_selected": (
                                    float(np.mean(predicted_values))
                                    if predicted_values
                                    else np.nan
                                ),
                                "mean_predicted_progress_bc_action": (
                                    float(np.mean(bc_predicted_values))
                                    if bc_predicted_values
                                    else np.nan
                                ),
                                "mean_policy_entropy": float(np.mean(entropies)),
                                "mean_prediction_latent_mse": (
                                    float(np.mean(latent_errors)) if latent_errors else np.nan
                                ),
                                "mean_decision_kl": (
                                    float(np.mean(decision_kls)) if decision_kls else np.nan
                                ),
                                "mean_action_mse": (
                                    float(np.mean(action_errors)) if action_errors else np.nan
                                ),
                            }
                        rows.append(row)
                        pd.DataFrame([row]).to_csv(
                            raw_path,
                            mode="a",
                            header=not raw_path.exists() or raw_path.stat().st_size == 0,
                            index=False,
                        )
                        completed_keys.add(episode_key)

    details = pd.DataFrame(rows)
    details = details.drop_duplicates(
        [
            "method",
            "planner_score",
            "planner_alpha",
            "planning_horizon",
            "task",
            "seed",
            "visual_condition",
            "episode_id",
            "environment_seed",
        ],
        keep="last",
    )
    details.to_csv(raw_path, index=False)
    metrics = [
        "success",
        "return",
        "episode_length",
        "final_distance_to_goal",
        "mean_action_norm",
        "mean_action_deviation_from_bc",
        "mean_selected_planner_score",
        "mean_predicted_value",
        "mean_predicted_progress_selected",
        "mean_predicted_progress_bc_action",
        "mean_policy_entropy",
        "mean_prediction_latent_mse",
        "mean_decision_kl",
        "mean_action_mse",
    ]
    summary = (
        details.groupby(
            [
                "method",
                "planner_score",
                "planner_alpha",
                "planning_horizon",
                "task",
                "visual_condition",
            ]
        )[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    for metric in metrics:
        summary[f"{metric}_stderr"] = (
            summary[f"{metric}_std"]
            / summary[f"{metric}_count"].clip(lower=1).map(math.sqrt)
        )
    summary_path = Path(
        section.get(
            "summary_output",
            "results/closed_loop_rollout_summary.csv",
        )
    )
    summary.to_csv(summary_path, index=False)
    return raw_path, summary_path


def main() -> None:
    args = parser("Run closed-loop MetaWorld evaluation").parse_args()
    raw, summary = run_closed_loop(load_config(args.config, args.set))
    print(raw)
    print(summary)


if __name__ == "__main__":
    main()
