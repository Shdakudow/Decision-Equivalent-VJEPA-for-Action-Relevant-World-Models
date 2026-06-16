from __future__ import annotations

import math

import torch

from devjepa.models import GaussianPolicy, gaussian_policy_kl, policy_entropy, posterior_kl


def predictor_loss(
    method: str,
    predicted: torch.Tensor,
    target: torch.Tensor,
    task_ids: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    policy: GaussianPolicy,
    beta: float,
    epsilon0: float,
    uncertainty_scale: float,
    gamma: float,
    decision_weight: float = 1.0,
    predictive_logvar: torch.Tensor | None = None,
    bjepa_loss_type: str = "nll",
    band_type: str = "entropy_band",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, horizon, latent_dim = predicted.shape
    flat_predicted = predicted.reshape(batch_size * horizon, latent_dim)
    flat_target = target.reshape(batch_size * horizon, latent_dim)
    flat_tasks = task_ids[:, None].expand(-1, horizon).reshape(-1)
    weights = gamma ** torch.arange(horizon, device=predicted.device, dtype=predicted.dtype)
    weights = weights[None, :].expand(batch_size, -1).reshape(-1)
    weights = weights / weights.mean()
    latent_error = (flat_predicted - flat_target).square().mean(-1)
    decision_kl = gaussian_policy_kl(policy, flat_target, flat_predicted, flat_tasks)
    target_mean = policy.distribution(flat_target, flat_tasks).mean
    predicted_mean = policy.distribution(flat_predicted, flat_tasks).mean
    action_mean_mse = (target_mean - predicted_mean).square().mean(-1)
    if predictive_logvar is None:
        predictive_nll = torch.zeros_like(latent_error)
        predictive_variance = torch.zeros_like(latent_error)
    else:
        flat_logvar = predictive_logvar.reshape(batch_size * horizon, latent_dim).clamp(
            -10.0, 5.0
        )
        variance = flat_logvar.exp().clamp_min(1e-6)
        predictive_nll = 0.5 * (
            math.log(2.0 * math.pi) + flat_logvar + (flat_target - flat_predicted).square() / variance
        ).mean(-1)
        predictive_variance = variance.mean(-1)
        if not torch.isfinite(predictive_nll).all():
            raise FloatingPointError("Non-finite BJEPA negative log likelihood")
    if method == "vjepa":
        prediction_term = latent_error
    elif method == "cosine_jepa":
        prediction_term = 1.0 - torch.nn.functional.cosine_similarity(
            flat_predicted, flat_target, dim=-1
        )
    elif method == "hybrid":
        prediction_term = latent_error + decision_weight * decision_kl
    elif method == "hybrid_action_mse":
        prediction_term = latent_error + decision_weight * action_mean_mse
    elif method == "bjepa":
        if predictive_logvar is None:
            raise ValueError("BJEPA requires predictive log variance")
        prediction_term = predictive_nll if bjepa_loss_type == "nll" else latent_error
    elif method == "de_bjepa":
        if predictive_logvar is None:
            raise ValueError("DE-BJEPA requires predictive log variance")
        base = predictive_nll if bjepa_loss_type == "nll" else latent_error
        prediction_term = base + decision_weight * decision_kl
    elif method == "de_vjepa":
        prediction_term = decision_kl
    elif method == "de_vjepa_band":
        if band_type == "constant_band":
            epsilon = torch.full_like(decision_kl, epsilon0)
        elif band_type == "entropy_band":
            uncertainty = policy_entropy(policy, flat_predicted, flat_tasks)
            epsilon = epsilon0 + uncertainty_scale * uncertainty.detach()
        else:
            raise ValueError(f"Unknown uncertainty band type: {band_type}")
        prediction_term = latent_error + decision_weight * torch.relu(
            decision_kl - epsilon
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    variational_kl = posterior_kl(posterior_mean, posterior_logvar).reshape(-1)
    loss = (prediction_term * weights).mean() + beta * (variational_kl * weights).mean()
    return loss, {
        "loss": loss.detach(),
        "prediction": prediction_term.mean().detach(),
        "latent_mse": latent_error.mean().detach(),
        "decision_kl": decision_kl.mean().detach(),
        "action_mean_mse": action_mean_mse.mean().detach(),
        "nll": predictive_nll.mean().detach(),
        "predictive_variance": predictive_variance.mean().detach(),
        "decision_weight": torch.as_tensor(decision_weight),
        "variational_kl": variational_kl.mean().detach(),
    }
