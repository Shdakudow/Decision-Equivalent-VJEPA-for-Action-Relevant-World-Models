from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _vectors(series: pd.Series) -> np.ndarray:
    return np.stack(series.map(np.asarray).to_numpy()).astype(np.float32)


def _decode_video(path: Path, frames: int, height: int, width: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg did not expose stdout")
    images = np.empty((frames, height, width, 3), dtype=np.uint8)
    frame_bytes = height * width * 3
    for index in range(frames):
        payload = process.stdout.read(frame_bytes)
        if len(payload) != frame_bytes:
            process.kill()
            raise RuntimeError(f"Video ended at frame {index} of {frames}")
        images[index] = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    trailing = process.stdout.read(1)
    return_code = process.wait()
    if return_code != 0 or trailing:
        raise RuntimeError(f"Unexpected ffmpeg output (return code {return_code})")
    return images


def prepare(source: Path, output: Path) -> Path:
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(source / "data" / "chunk-000" / "file-000.parquet")
    frame_count = int(info["total_frames"])
    image_shape = info["features"]["observation.image"]["shape"]
    if len(table) != frame_count:
        raise ValueError(f"Metadata has {frame_count} frames but parquet has {len(table)}")
    images = _decode_video(
        source / "videos" / "observation.image" / "chunk-000" / "file-000.mp4",
        frame_count,
        int(image_shape[0]),
        int(image_shape[1]),
    )
    # PushT actions are absolute pixel targets in a 512x512 workspace. The
    # Gaussian BC head uses a bounded tanh mean, so map coordinates to [-1, 1].
    actions = _vectors(table["action"]) / 256.0 - 1.0
    success = table["next.success"].to_numpy(dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images=images,
        observations=_vectors(table["observation.state"]),
        actions=actions,
        task_ids=np.zeros(frame_count, dtype=np.int64),
        episode_ids=table["episode_index"].to_numpy(dtype=np.int64),
        success=success,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LeRobot PushT to DE-VJEPA format")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.output))


if __name__ == "__main__":
    main()
