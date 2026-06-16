from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


class TaskEmbedding(nn.Module):
    def __init__(self, num_tasks: int, size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, size)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_ids)


class GaussianPolicy(nn.Module):
    def __init__(
        self, latent_dim: int, action_dim: int, num_tasks: int, hidden_dim: int, task_dim: int
    ):
        super().__init__()
        self.tasks = TaskEmbedding(num_tasks, task_dim)
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim + task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, z: torch.Tensor, task_ids: torch.Tensor) -> Normal:
        hidden = self.backbone(torch.cat((z, self.tasks(task_ids)), dim=-1))
        mean = torch.tanh(self.mean(hidden))
        log_std = self.log_std(hidden).clamp(-5.0, 1.0)
        return Normal(mean, log_std.exp())

    def forward(self, z: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        return self.distribution(z, task_ids).mean


class ProgressHead(nn.Module):
    """Weak task-progress proxy for successful demonstration trajectories."""

    def __init__(
        self,
        latent_dim: int,
        num_tasks: int,
        hidden_dim: int = 256,
        task_dim: int = 32,
    ):
        super().__init__()
        self.tasks = TaskEmbedding(num_tasks, task_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        logits = self.network(torch.cat((z, self.tasks(task_ids)), dim=-1))
        return torch.sigmoid(logits).squeeze(-1)


@dataclass
class PredictorOutput:
    predicted: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_logvar: torch.Tensor
    predictive_mean: torch.Tensor | None = None
    predictive_logvar: torch.Tensor | None = None
    dynamics_mean: torch.Tensor | None = None
    dynamics_logvar: torch.Tensor | None = None
    prior_mean: torch.Tensor | None = None
    prior_logvar: torch.Tensor | None = None


def gaussian_product_of_experts(
    dynamics_mean: torch.Tensor,
    dynamics_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    dynamics_logvar = dynamics_logvar.clamp(-10.0, 5.0)
    prior_logvar = prior_logvar.clamp(-10.0, 5.0)
    dynamics_variance = dynamics_logvar.exp().clamp_min(epsilon)
    prior_variance = prior_logvar.exp().clamp_min(epsilon)
    dynamics_precision = dynamics_variance.reciprocal()
    prior_precision = prior_variance.reciprocal()
    combined_precision = (dynamics_precision + prior_precision).clamp_min(epsilon)
    combined_variance = combined_precision.reciprocal().clamp_min(epsilon)
    combined_mean = combined_variance * (
        dynamics_precision * dynamics_mean + prior_precision * prior_mean
    )
    combined_logvar = combined_variance.log().clamp(-10.0, 5.0)
    for name, value in (
        ("PoE precision", combined_precision),
        ("PoE variance", combined_variance),
        ("PoE mean", combined_mean),
    ):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {name}")
    return combined_mean, combined_logvar


class VariationalPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_tasks: int,
        hidden_dim: int,
        stochastic_dim: int,
        task_dim: int,
        horizon: int,
        bjepa: bool = False,
        bjepa_prior_type: str = "none",
        use_actions: bool = True,
        architecture: str = "mlp",
        prior_mean: torch.Tensor | None = None,
        prior_variance: torch.Tensor | None = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.bjepa = bjepa
        self.bjepa_prior_type = bjepa_prior_type
        self.use_actions = use_actions
        self.architecture = architecture
        self.tasks = TaskEmbedding(num_tasks, task_dim)
        context_dim = latent_dim + action_dim + task_dim
        self.posterior = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_dim * 2),
        )
        if bjepa:
            self.dynamics_backbone = nn.Sequential(
                nn.Linear(context_dim + stochastic_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.dynamics_mean = nn.Linear(hidden_dim, latent_dim)
            self.dynamics_logvar = nn.Linear(hidden_dim, latent_dim)
            if bjepa_prior_type == "learned_task_prior":
                self.task_prior_mean = nn.Embedding(num_tasks, latent_dim)
                self.task_prior_logvar = nn.Embedding(num_tasks, latent_dim)
            elif bjepa_prior_type == "empirical_task_gaussian":
                if prior_mean is None or prior_variance is None:
                    prior_mean = torch.zeros(num_tasks, latent_dim)
                    prior_variance = torch.ones(num_tasks, latent_dim)
                self.register_buffer("empirical_prior_mean", prior_mean.float())
                self.register_buffer(
                    "empirical_prior_logvar",
                    prior_variance.float().clamp_min(1e-6).log().clamp(-10.0, 5.0),
                )
            elif bjepa_prior_type != "none":
                raise ValueError(f"Unknown BJEPA prior type: {bjepa_prior_type}")
        else:
            if architecture == "linear":
                self.predictor = nn.Linear(context_dim + stochastic_dim, latent_dim)
            elif architecture == "mlp":
                self.predictor = nn.Sequential(
                    nn.Linear(context_dim + stochastic_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
            else:
                raise ValueError(f"Unknown predictor architecture: {architecture}")

    def _prior_expert(
        self, current: torch.Tensor, task_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bjepa_prior_type == "empirical_task_gaussian":
            return (
                self.empirical_prior_mean[task_ids],
                self.empirical_prior_logvar[task_ids],
            )
        if self.bjepa_prior_type == "learned_task_prior":
            return (
                self.task_prior_mean(task_ids),
                self.task_prior_logvar(task_ids).clamp(-10.0, 5.0),
            )
        return torch.zeros_like(current), torch.full_like(current, 5.0)

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        task_ids: torch.Tensor,
        target: torch.Tensor | None = None,
        sample: bool = True,
    ) -> PredictorOutput:
        predictions = []
        means = []
        logvars = []
        predictive_means = []
        predictive_logvars = []
        dynamics_means = []
        dynamics_logvars = []
        prior_means = []
        prior_logvars = []
        current = z
        task_embedding = self.tasks(task_ids)
        for step in range(actions.shape[1]):
            action = actions[:, step] if self.use_actions else torch.zeros_like(actions[:, step])
            context = torch.cat((current, action, task_embedding), dim=-1)
            if target is None:
                mean = torch.zeros(
                    z.shape[0], self.posterior[-1].out_features // 2, device=z.device
                )
                logvar = torch.zeros_like(mean)
            else:
                mean, logvar = self.posterior(
                    torch.cat((context, target[:, step]), dim=-1)
                ).chunk(2, dim=-1)
                logvar = logvar.clamp(-10.0, 5.0)
            noise = torch.randn_like(mean) if sample else torch.zeros_like(mean)
            stochastic = mean + noise * torch.exp(0.5 * logvar)
            predictor_input = torch.cat((context, stochastic), dim=-1)
            if self.bjepa:
                hidden = self.dynamics_backbone(predictor_input)
                dynamics_mean = current + self.dynamics_mean(hidden)
                dynamics_logvar = self.dynamics_logvar(hidden).clamp(-10.0, 5.0)
                prior_mean, prior_logvar = self._prior_expert(current, task_ids)
                predictive_mean, predictive_logvar = gaussian_product_of_experts(
                    dynamics_mean,
                    dynamics_logvar,
                    prior_mean,
                    prior_logvar,
                )
                predictive_noise = torch.randn_like(predictive_mean) if sample else 0.0
                current = predictive_mean + predictive_noise * torch.exp(
                    0.5 * predictive_logvar
                )
                predictive_means.append(predictive_mean)
                predictive_logvars.append(predictive_logvar)
                dynamics_means.append(dynamics_mean)
                dynamics_logvars.append(dynamics_logvar)
                prior_means.append(prior_mean)
                prior_logvars.append(prior_logvar)
            else:
                current = current + self.predictor(predictor_input)
            predictions.append(current)
            means.append(mean)
            logvars.append(logvar)
        output = PredictorOutput(
            torch.stack(predictions, dim=1),
            torch.stack(means, dim=1),
            torch.stack(logvars, dim=1),
        )
        if self.bjepa:
            output.predictive_mean = torch.stack(predictive_means, dim=1)
            output.predictive_logvar = torch.stack(predictive_logvars, dim=1)
            output.dynamics_mean = torch.stack(dynamics_means, dim=1)
            output.dynamics_logvar = torch.stack(dynamics_logvars, dim=1)
            output.prior_mean = torch.stack(prior_means, dim=1)
            output.prior_logvar = torch.stack(prior_logvars, dim=1)
        return output


def posterior_kl(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(-1)


def gaussian_policy_kl(
    policy: GaussianPolicy,
    target_z: torch.Tensor,
    predicted_z: torch.Tensor,
    task_ids: torch.Tensor,
) -> torch.Tensor:
    target_distribution = policy.distribution(target_z, task_ids)
    predicted_distribution = policy.distribution(predicted_z, task_ids)
    return kl_divergence(target_distribution, predicted_distribution).sum(-1)


def policy_entropy(policy: GaussianPolicy, z: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
    entropy = policy.distribution(z, task_ids).entropy().sum(-1)
    return (entropy / math.log(2.0 * math.pi * math.e)).clamp_min(0.0)
