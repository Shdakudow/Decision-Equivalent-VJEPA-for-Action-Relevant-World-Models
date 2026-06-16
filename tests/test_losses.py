import torch

from devjepa.losses import predictor_loss
from devjepa.models import (
    GaussianPolicy,
    VariationalPredictor,
    gaussian_policy_kl,
    gaussian_product_of_experts,
)


def test_policy_kl_is_zero_for_identical_latents():
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    z = torch.randn(5, 8)
    tasks = torch.tensor([0, 1, 0, 1, 0])
    value = gaussian_policy_kl(policy, z, z, tasks)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)


def test_uncertainty_band_is_nonnegative():
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    predicted = torch.randn(5, 1, 8, requires_grad=True)
    target = torch.randn(5, 1, 8)
    tasks = torch.tensor([0, 1, 0, 1, 0])
    mean = torch.zeros(5, 3)
    logvar = torch.zeros(5, 3)
    loss, metrics = predictor_loss(
        "de_vjepa_band",
        predicted,
        target,
        tasks,
        mean[:, None],
        logvar[:, None],
        policy,
        0.01,
        0.1,
        1.0,
        0.95,
    )
    assert loss >= 0
    assert metrics["prediction"] >= 0
    loss.backward()


def test_hybrid_combines_latent_and_decision_losses():
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    predicted = torch.randn(5, 1, 8, requires_grad=True)
    target = torch.randn(5, 1, 8)
    tasks = torch.tensor([0, 1, 0, 1, 0])
    mean = torch.zeros(5, 1, 3)
    logvar = torch.zeros(5, 1, 3)
    loss, metrics = predictor_loss(
        "hybrid",
        predicted,
        target,
        tasks,
        mean,
        logvar,
        policy,
        0.0,
        0.1,
        1.0,
        0.95,
        decision_weight=0.1,
    )
    expected = metrics["latent_mse"] + 0.1 * metrics["decision_kl"]
    assert torch.allclose(loss, expected)


def test_hybrid_action_mse_combines_latent_and_policy_mean_losses():
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    predicted = torch.randn(5, 1, 8, requires_grad=True)
    target = torch.randn(5, 1, 8)
    tasks = torch.tensor([0, 1, 0, 1, 0])
    mean = torch.zeros(5, 1, 3)
    logvar = torch.zeros(5, 1, 3)
    loss, metrics = predictor_loss(
        "hybrid_action_mse",
        predicted,
        target,
        tasks,
        mean,
        logvar,
        policy,
        0.0,
        0.1,
        1.0,
        0.95,
        decision_weight=0.1,
    )
    expected = metrics["latent_mse"] + 0.1 * metrics["action_mean_mse"]
    assert torch.allclose(loss, expected)
    loss.backward()


def test_cosine_jepa_loss_is_finite_and_differentiable():
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    predicted = torch.randn(5, 1, 8, requires_grad=True)
    target = torch.randn(5, 1, 8)
    tasks = torch.tensor([0, 1, 0, 1, 0])
    mean = torch.zeros(5, 1, 3)
    logvar = torch.zeros(5, 1, 3)
    loss, metrics = predictor_loss(
        "cosine_jepa",
        predicted,
        target,
        tasks,
        mean,
        logvar,
        policy,
        0.0,
        0.1,
        1.0,
        0.95,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["latent_mse"])
    loss.backward()


def test_no_action_predictor_ignores_action_values():
    predictor = VariationalPredictor(
        latent_dim=8,
        action_dim=4,
        num_tasks=2,
        hidden_dim=16,
        stochastic_dim=3,
        task_dim=4,
        horizon=1,
        use_actions=False,
    )
    z = torch.randn(5, 8)
    tasks = torch.tensor([0, 1, 0, 1, 0])
    first = predictor(z, torch.randn(5, 1, 4), tasks, sample=False).predicted
    second = predictor(z, torch.randn(5, 1, 4), tasks, sample=False).predicted
    assert torch.allclose(first, second)


def test_gaussian_product_of_experts_is_finite_and_more_confident():
    dynamics_mean = torch.tensor([[1.0, 2.0]])
    dynamics_logvar = torch.zeros_like(dynamics_mean)
    prior_mean = torch.tensor([[3.0, 4.0]])
    prior_logvar = torch.zeros_like(prior_mean)
    mean, logvar = gaussian_product_of_experts(
        dynamics_mean, dynamics_logvar, prior_mean, prior_logvar
    )
    assert torch.allclose(mean, torch.tensor([[2.0, 3.0]]), atol=1e-6)
    assert torch.allclose(logvar.exp(), torch.full_like(logvar, 0.5), atol=1e-6)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(logvar).all()


def test_bjepa_predictor_and_nll_are_finite():
    predictor = VariationalPredictor(
        latent_dim=8,
        action_dim=4,
        num_tasks=2,
        hidden_dim=16,
        stochastic_dim=3,
        task_dim=4,
        horizon=1,
        bjepa=True,
        bjepa_prior_type="empirical_task_gaussian",
        prior_mean=torch.zeros(2, 8),
        prior_variance=torch.ones(2, 8),
    )
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    z = torch.randn(5, 8)
    actions = torch.randn(5, 1, 4)
    target = torch.randn(5, 1, 8)
    tasks = torch.tensor([0, 1, 0, 1, 0])
    result = predictor(z, actions, tasks, target=None, sample=False)
    loss, metrics = predictor_loss(
        "de_bjepa",
        result.predictive_mean,
        target,
        tasks,
        result.posterior_mean,
        result.posterior_logvar,
        policy,
        0.0,
        0.1,
        1.0,
        0.95,
        decision_weight=0.01,
        predictive_logvar=result.predictive_logvar,
        bjepa_loss_type="nll",
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["nll"])
    assert metrics["predictive_variance"] > 0
    loss.backward()


def test_multistep_predictor_rollout_is_autoregressive():
    predictor = VariationalPredictor(
        latent_dim=8,
        action_dim=4,
        num_tasks=2,
        hidden_dim=16,
        stochastic_dim=3,
        task_dim=4,
        horizon=3,
    )
    z = torch.randn(2, 8)
    actions = torch.randn(2, 3, 4)
    tasks = torch.tensor([0, 1])
    full = predictor(z, actions, tasks, target=None, sample=False).predicted
    first = predictor(z, actions[:, :1], tasks, target=None, sample=False).predicted[:, -1]
    second = predictor(
        first, actions[:, 1:2], tasks, target=None, sample=False
    ).predicted[:, -1]
    third = predictor(
        second, actions[:, 2:3], tasks, target=None, sample=False
    ).predicted[:, -1]
    manual = torch.stack((first, second, third), dim=1)
    assert torch.allclose(full, manual, atol=1e-6)
