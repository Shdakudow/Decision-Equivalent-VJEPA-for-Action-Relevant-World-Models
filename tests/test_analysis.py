from pathlib import Path

import pandas as pd
import pytest

from devjepa.analysis import paired_comparisons, summarize


def test_summary_pairs_by_task_and_evaluator(tmp_path: Path):
    rows = []
    for method, offset in (("latent_only", 0.0), ("hybrid", -0.1)):
        for train_seed, evaluator_seed in ((1, 2), (2, 1)):
            for sample_id in range(3):
                rows.append(
                    {
                        "task": "reach-v3",
                        "train_seed": train_seed,
                        "evaluator_seed": evaluator_seed,
                        "method": method,
                        "lambda": 0.0 if method == "latent_only" else 0.01,
                        "reference_policy_type": "none"
                        if method == "latent_only"
                        else "strong",
                        "shift": "clean",
                        "latent_mse": 1.0 - offset,
                        "decision_kl": 1.0 + offset,
                        "action_mse": 1.0 + offset,
                        "sample_id": sample_id,
                    }
                )
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    summary = summarize(path, "latent_only")
    hybrid_kl = summary[
        (summary["method"] == "hybrid") & (summary["metric"] == "decision_kl")
    ].iloc[0]
    assert hybrid_kl["pairs"] == 2
    assert hybrid_kl["relative_change_percent"] < 0


def test_explicit_method_comparisons_use_task_pairs(tmp_path: Path):
    rows = []
    for task in ("reach-v3", "drawer-open-v3"):
        for method, value in (("vjepa", 1.0), ("bjepa", 0.8)):
            rows.append(
                {
                    "task": task,
                    "train_seed": 1,
                    "evaluator_seed": 2,
                    "method": method,
                    "lambda": 0.0,
                    "reference_policy_type": "none",
                    "shift": "clean",
                    "latent_mse": value,
                    "decision_kl": value,
                    "action_mse": value,
                }
            )
    path = tmp_path / "comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    comparisons = paired_comparisons(path, [["bjepa", "vjepa"]], pair_by_task=True)
    row = comparisons[comparisons["metric"] == "latent_mse"].iloc[0]
    assert row["pairs"] == 2
    assert row["relative_change_percent"] == pytest.approx(-20.0)
