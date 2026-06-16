from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "RELEASE_MANIFEST_SHA256.csv"
MAX_TRACKED_BYTES = 50 * 1024 * 1024

REQUIRED_PATHS = (
    "pyproject.toml",
    "main.tex",
    "src/devjepa/losses.py",
    "src/devjepa/models.py",
    "src/devjepa/train_predictor.py",
    "src/devjepa/evaluate_independent.py",
    "configs/pilot_reproduction.yaml",
    "configs/core_vjepa_bjepa_de.yaml",
    "configs/closed_loop_main.yaml",
    "scripts/run_smoke.sh",
    "scripts/make_paper_figures.py",
    "tests/test_losses.py",
    "results/cross_dataset_comparison_summary.csv",
    "results/closed_loop_rollout_statistics.csv",
)

RESULT_COLUMNS = {
    "results/cross_dataset_comparison_summary.csv": {
        "dataset",
        "method",
        "shift",
        "metric",
        "mean",
    },
    "results/closed_loop_rollout_statistics.csv": {
        "candidate",
        "baseline",
        "metric",
        "visual_condition",
        "mean_difference",
        "paired_t_pvalue",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, str]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {
            row["path"]: row["sha256"]
            for row in csv.DictReader(handle)
        }


def check_required_paths() -> list[str]:
    return [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]


def check_manifest() -> list[str]:
    failures: list[str] = []
    for relative, expected in load_manifest().items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing manifest file: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    return failures


def check_result_schemas() -> list[str]:
    failures: list[str] = []
    for relative, required in RESULT_COLUMNS.items():
        with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
            columns = set(next(csv.reader(handle)))
        missing = required - columns
        if missing:
            failures.append(
                f"{relative} missing columns: {', '.join(sorted(missing))}"
            )
    return failures


def check_graphics() -> list[str]:
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    references = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    return [
        f"missing graphic: {relative}"
        for relative in references
        if not (ROOT / relative).is_file()
    ]


def check_large_files() -> list[str]:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            size = path.stat().st_size
            if size > MAX_TRACKED_BYTES:
                failures.append(
                    f"file exceeds {MAX_TRACKED_BYTES // (1024 * 1024)} MiB: "
                    f"{path.relative_to(ROOT)}"
                )
    return failures


def main() -> None:
    failures = []
    failures.extend(f"missing required file: {path}" for path in check_required_paths())
    failures.extend(check_manifest())
    failures.extend(check_result_schemas())
    failures.extend(check_graphics())
    failures.extend(check_large_files())

    if failures:
        print("Release verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print(f"Release verification passed ({len(load_manifest())} checksummed files).")
    print("This validates package integrity, not independent experimental replication.")


if __name__ == "__main__":
    main()
