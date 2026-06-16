from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from devjepa.config import load_config, parser
from devjepa.utils import seed_everything


def _make_env(task: str, seed: int, camera: str):
    import gymnasium as gym
    import metaworld  # noqa: F401

    return gym.make(
        "Meta-World/MT1",
        env_name=task,
        seed=seed,
        render_mode="rgb_array",
        camera_name=camera,
    )


def _expert_policy(task: str):
    from metaworld.policies import ENV_POLICY_MAP

    if task not in ENV_POLICY_MAP:
        raise KeyError(f"No scripted MetaWorld policy for {task}")
    return ENV_POLICY_MAP[task]()


def collect(config: dict[str, Any]) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    tasks = list(config["data"]["tasks"])
    episodes_per_task = int(config["data"]["episodes_per_task"])
    max_steps = int(config["data"]["max_steps"])
    camera = config["data"].get("camera", "corner2")
    image_size = int(config["data"]["image_size"])
    output = Path(config["data"]["raw_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    task_ids: list[int] = []
    episode_ids: list[int] = []
    successes: list[float] = []
    episode = 0

    from PIL import Image

    for task_id, task in enumerate(tasks):
        policy = _expert_policy(task)
        env = _make_env(task, seed + task_id, camera)
        try:
            for local_episode in tqdm(range(episodes_per_task), desc=task):
                observation, _ = env.reset(seed=seed + task_id * 10000 + local_episode)
                episode_success = False
                episode_start = len(successes)
                for _ in range(max_steps):
                    frame = env.render()
                    frame = np.asarray(
                        Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                    action = np.asarray(policy.get_action(observation), dtype=np.float32)
                    next_observation, _, terminated, truncated, info = env.step(action)
                    episode_success = episode_success or bool(info.get("success", False))
                    frames.append(frame)
                    observations.append(np.asarray(observation, dtype=np.float32))
                    actions.append(action)
                    task_ids.append(task_id)
                    episode_ids.append(episode)
                    successes.append(float(episode_success))
                    observation = next_observation
                    if terminated or truncated or episode_success:
                        break
                for index in range(episode_start, len(successes)):
                    successes[index] = float(episode_success)
                episode += 1
        finally:
            env.close()

    np.savez_compressed(
        output,
        images=np.stack(frames),
        observations=np.stack(observations),
        actions=np.stack(actions),
        task_ids=np.asarray(task_ids, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        success=np.asarray(successes, dtype=np.float32),
    )
    return output


def main() -> None:
    args = parser("Collect scripted MetaWorld demonstrations").parse_args()
    path = collect(load_config(args.config, args.set))
    print(path)


if __name__ == "__main__":
    main()
