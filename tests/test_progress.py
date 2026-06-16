import torch

from devjepa.models import ProgressHead
from devjepa.progress import trajectory_progress_labels


def test_progress_head_forward_shape_and_range():
    model = ProgressHead(latent_dim=8, num_tasks=3, hidden_dim=16, task_dim=4)
    output = model(torch.randn(5, 8), torch.tensor([0, 1, 2, 0, 1]))
    assert output.shape == (5,)
    assert torch.isfinite(output).all()
    assert ((0.0 <= output) & (output <= 1.0)).all()


def test_trajectory_progress_labels_are_finite_and_episode_local():
    labels = trajectory_progress_labels(torch.tensor([3, 3, 3, 8, 8]))
    assert torch.isfinite(labels).all()
    assert torch.allclose(labels, torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0]))
