import inspect

import numpy as np
import pytest
import torch
from PIL import Image

from devjepa.closed_loop import (
    _generate_candidate_actions,
    _initialize_cem_population,
    _cem_action,
    _method_requires_predictor,
    _planner_scores,
    _select_cem_elites,
    _shift_image,
    _validate_independent_policy,
    _value_candidate_action,
)
from devjepa.data import Normalization
from devjepa.models import GaussianPolicy


def test_candidate_actions_have_valid_shape_range_and_pairing():
    action = torch.tensor([[0.95, -0.95, 0.0, 0.2]])
    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)
    first = _generate_candidate_actions(action, 32, 0.12, first_generator)
    second = _generate_candidate_actions(action, 32, 0.12, second_generator)
    assert first.shape == (32, 4)
    assert torch.all(first <= 1.0)
    assert torch.all(first >= -1.0)
    assert torch.equal(first, second)
    assert torch.equal(first[0], action[0])


def test_cem_population_shape_range_and_pairing():
    action = torch.tensor([[0.9, -0.9, 0.0, 0.2]])
    first = _initialize_cem_population(
        action, 32, 3, 0.25, torch.Generator().manual_seed(91)
    )
    second = _initialize_cem_population(
        action, 32, 3, 0.25, torch.Generator().manual_seed(91)
    )
    assert first.shape == (32, 3, 4)
    assert torch.equal(first, second)
    assert torch.all(first <= 1.0)
    assert torch.all(first >= -1.0)


def test_cem_elite_selection_is_finite():
    population = torch.randn(20, 3, 4)
    scores = torch.linspace(-1.0, 1.0, 20)
    elites = _select_cem_elites(population, scores, 0.1)
    assert elites.shape == (2, 3, 4)
    assert torch.isfinite(elites).all()


@pytest.mark.parametrize(
    "score_name",
    [
        "entropy_score",
        "decision_consistency_score",
        "action_mean_consistency_score",
        "random_candidate_control",
    ],
)
def test_planner_scores_are_finite(score_name):
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    candidates = torch.randn(6, 4).clamp(-1, 1)
    bc_action = candidates[:1]
    tasks = torch.tensor([0])
    current = policy.distribution(torch.randn(1, 8), tasks)
    predicted = torch.randn(6, 8)
    normalization = Normalization(
        torch.zeros(8), torch.ones(8), torch.zeros(4), torch.ones(4)
    )
    scores = _planner_scores(
        score_name,
        candidates,
        bc_action,
        current,
        predicted,
        None,
        tasks,
        policy,
        normalization,
        0.1,
    )
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()


def test_low_uncertainty_score_requires_variance():
    policy = GaussianPolicy(8, 4, 2, 16, 4)
    candidates = torch.zeros(2, 4)
    tasks = torch.tensor([0])
    normalization = Normalization(
        torch.zeros(8), torch.ones(8), torch.zeros(4), torch.ones(4)
    )
    with pytest.raises(ValueError, match="predictive variance"):
        _planner_scores(
            "low_uncertainty_score",
            candidates,
            candidates[:1],
            policy.distribution(torch.zeros(1, 8), tasks),
            torch.zeros(2, 8),
            None,
            tasks,
            policy,
            normalization,
            0.1,
        )


def test_bc_only_bypasses_predictor():
    assert not _method_requires_predictor("bc_only")
    assert _method_requires_predictor("bc_plus_vjepa")
    assert _method_requires_predictor("bc_plus_de_vjepa")
    assert _method_requires_predictor("bc_plus_vjepa_value_planner")
    assert _method_requires_predictor("bc_plus_de_vjepa_cem")


def test_visual_shifts_are_deterministic():
    array = np.full((32, 32, 3), 120, dtype=np.uint8)
    image = Image.fromarray(array)
    for condition in ("clean", "brightness", "background_shift", "blur"):
        first = np.asarray(_shift_image(image, condition))
        second = np.asarray(_shift_image(image, condition))
        assert np.array_equal(first, second)


def test_independent_policy_seed_rejects_leakage():
    with pytest.raises(ValueError, match="must differ"):
        _validate_independent_policy(42, [42, 1000], True)
    _validate_independent_policy(7, [42, 1000], True)


def test_value_planners_cannot_receive_future_true_latent():
    for planner in (_value_candidate_action, _cem_action):
        parameters = set(inspect.signature(planner).parameters)
        assert "target" not in parameters
        assert "next_latent" not in parameters
        assert "future_latent" not in parameters
