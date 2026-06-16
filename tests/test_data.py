import numpy as np
import torch

from devjepa.data import TransitionDataset, select_tasks


def test_transition_dataset_does_not_cross_episode_boundary():
    archive = {
        "latents": torch.randn(6, 4),
        "actions": torch.randn(6, 2),
        "task_ids": torch.zeros(6, dtype=torch.long),
        "episode_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "success": torch.zeros(6),
    }
    dataset = TransitionDataset(archive, horizon=2)
    assert dataset.indices == [0, 3]
    assert dataset[0]["target"].shape == (2, 4)


def test_select_tasks_filters_and_remaps_task_ids():
    archive = {
        "latents": torch.randn(6, 4),
        "actions": torch.randn(6, 2),
        "task_ids": torch.tensor([0, 0, 1, 1, 2, 2]),
        "episode_ids": torch.arange(6),
    }
    selected = select_tasks(
        archive,
        ["reach-v3", "button-press-v3", "drawer-open-v3"],
        ["drawer-open-v3", "reach-v3"],
    )
    assert selected["task_ids"].tolist() == [1, 1, 0, 0]
    assert len(selected["latents"]) == 4


def test_dinov2_feature_archive_can_be_saved_and_reloaded(tmp_path):
    path = tmp_path / "dinov2_features.npz"
    features = np.random.default_rng(42).normal(size=(4, 384)).astype(np.float32)
    np.savez_compressed(path, latents=features)
    loaded = np.load(path, allow_pickle=False)["latents"]
    assert loaded.shape == (4, 384)
    assert np.isfinite(loaded).all()
