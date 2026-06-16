from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Normalization:
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    @classmethod
    def fit(cls, latents: torch.Tensor, actions: torch.Tensor) -> "Normalization":
        return cls(
            latents.mean(0),
            latents.std(0).clamp_min(1e-5),
            actions.mean(0),
            actions.std(0).clamp_min(1e-5),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.__dict__

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "Normalization":
        return cls(**state)


def load_archive(path: str | Path) -> dict[str, torch.Tensor]:
    raw = np.load(path, allow_pickle=False)
    return {key: torch.from_numpy(raw[key]) for key in raw.files}


def select_tasks(
    archive: dict[str, torch.Tensor],
    archive_tasks: list[str],
    selected_tasks: list[str],
) -> dict[str, torch.Tensor]:
    if selected_tasks == archive_tasks:
        return archive
    unknown = sorted(set(selected_tasks) - set(archive_tasks))
    if unknown:
        raise ValueError(f"Tasks not present in archive task order: {unknown}")
    old_to_new = {
        archive_tasks.index(task): new_id for new_id, task in enumerate(selected_tasks)
    }
    mask = torch.zeros_like(archive["task_ids"], dtype=torch.bool)
    for old_id in old_to_new:
        mask |= archive["task_ids"] == old_id
    selected = {key: value[mask] for key, value in archive.items()}
    remapped = selected["task_ids"].clone()
    for old_id, new_id in old_to_new.items():
        remapped[selected["task_ids"] == old_id] = new_id
    selected["task_ids"] = remapped
    return selected


def load_configured_archive(config: dict, path: str | Path | None = None) -> dict[str, torch.Tensor]:
    archive = load_archive(path or config["data"]["encoded_path"])
    selected_tasks = list(config["data"]["tasks"])
    archive_tasks = list(config["data"].get("archive_tasks", selected_tasks))
    return select_tasks(archive, archive_tasks, selected_tasks)


class TransitionDataset(Dataset):
    def __init__(self, archive: dict[str, torch.Tensor], horizon: int = 1):
        self.latents = archive["latents"].float()
        self.actions = archive["actions"].float()
        self.task_ids = archive["task_ids"].long()
        self.episode_ids = archive["episode_ids"].long()
        self.success = archive.get("success", torch.zeros(len(self.actions))).float()
        self.horizon = horizon
        self.indices = [
            i
            for i in range(len(self.actions) - horizon)
            if self.episode_ids[i] == self.episode_ids[i + horizon]
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        return {
            "z": self.latents[index],
            "actions": self.actions[index : index + self.horizon],
            "target": self.latents[index + 1 : index + self.horizon + 1],
            "task_id": self.task_ids[index],
            "success": self.success[index + 1 : index + self.horizon + 1],
        }


def split_archive(
    archive: dict[str, torch.Tensor], validation_fraction: float, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    episode_ids = archive["episode_ids"].numpy()
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    validation_ids = set(unique[: max(1, int(len(unique) * validation_fraction))].tolist())
    validation_mask = np.array([episode in validation_ids for episode in episode_ids])
    train_mask = ~validation_mask
    train = {key: value[torch.from_numpy(train_mask)] for key, value in archive.items()}
    validation = {key: value[torch.from_numpy(validation_mask)] for key, value in archive.items()}
    return train, validation
