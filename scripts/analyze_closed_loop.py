from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = (
    "success",
    "return",
    "episode_length",
    "final_distance_to_goal",
    "mean_action_norm",
    "mean_action_deviation_from_bc",
    "mean_selected_planner_score",
    "mean_predicted_value",
    "mean_predicted_progress_selected",
    "mean_predicted_progress_bc_action",
    "mean_policy_entropy",
    "mean_prediction_latent_mse",
    "mean_decision_kl",
    "mean_action_mse",
)
COMPARISONS = (
    ("bc_plus_de_vjepa", "bc_plus_vjepa"),
    ("bc_plus_de_bjepa", "bc_plus_bjepa"),
    ("bc_plus_de_vjepa", "bc_only"),
    ("bc_plus_de_vjepa_value_planner", "bc_plus_vjepa_value_planner"),
    ("bc_plus_de_vjepa_value_planner", "bc_only"),
    ("bc_plus_de_vjepa_cem", "bc_plus_vjepa_cem"),
    ("bc_plus_de_vjepa_cem", "bc_only"),
)


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def _paired_test(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "mean_difference": np.nan,
            "stderr": np.nan,
            "paired_t_pvalue": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "paired_units": 0,
        }
    ci_low, ci_high = _bootstrap_mean(values, rng)
    return {
        "mean_difference": float(values.mean()),
        "stderr": (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        ),
        "paired_t_pvalue": (
            float(stats.ttest_1samp(values, 0.0).pvalue) if len(values) > 1 else np.nan
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "paired_units": len(values),
    }


def _write_latex(summary: pd.DataFrame, destination: Path) -> None:
    overall = summary[summary["task"] == "ALL"].copy()
    variant_keys = [
        key
        for key in ("planner_score", "planner_alpha", "planning_horizon")
        if key in overall
    ]
    row_keys = list(
        dict.fromkeys(
            tuple(row)
            for row in overall[["method", *variant_keys]].itertuples(
                index=False, name=None
            )
        )
    )
    conditions = list(dict.fromkeys(overall["visual_condition"]))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Closed-loop MetaWorld success rate (\%) with 95\% episode-bootstrap confidence intervals.}",
        r"\label{tab:closed-loop-rollout}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(conditions) + "}",
        r"\toprule",
        r"\textbf{Method} & "
        + " & ".join(rf"\textbf{{{condition.replace('_', ' ').title()}}}" for condition in conditions)
        + r" \\",
        r"\midrule",
    ]
    for row_key in row_keys:
        method, *variant_values = row_key
        cells = []
        for condition in conditions:
            row = overall[
                (overall["method"] == method)
                & (overall["visual_condition"] == condition)
            ]
            for key, value in zip(variant_keys, variant_values):
                row = row[row[key] == value]
            if row.empty:
                cells.append("--")
            else:
                item = row.iloc[0]
                cells.append(
                    f"{100 * item['success_mean']:.1f} "
                    f"[{100 * item['success_ci_low']:.1f}, "
                    f"{100 * item['success_ci_high']:.1f}]"
                )
        details = []
        for key, value in zip(variant_keys, variant_values):
            if key == "planner_score" and value in {"default", "bc_only"}:
                continue
            if key == "planner_alpha" and float(value) == 0.0:
                continue
            if key == "planning_horizon" and int(value) == 0:
                continue
            details.append(f"{key.replace('planner_', '')}={value}")
        label = method if not details else f"{method}: {', '.join(details)}"
        lines.append(label.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")


def analyze(
    input_path: Path,
    summary_path: Path,
    statistics_path: Path,
    table_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(input_path)
    rng = np.random.default_rng(2026)
    metrics = [metric for metric in METRICS if metric in data]
    rows = []
    variant_keys = [
        key
        for key in ("planner_score", "planner_alpha", "planning_horizon")
        if key in data
    ]
    has_planner_scores = "planner_score" in data
    group_keys = ["method", *variant_keys]
    group_keys.extend(["task", "visual_condition"])
    for keys, group in data.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, keys))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
            row[f"{metric}_stderr"] = (
                float(finite.std(ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else np.nan
            )
            row[f"{metric}_count"] = len(finite)
        row["success_ci_low"], row["success_ci_high"] = _bootstrap_mean(
            group["success"].to_numpy(dtype=float), rng
        )
        rows.append(row)
    overall_keys = ["method", *variant_keys]
    overall_keys.append("visual_condition")
    for keys, group in data.groupby(overall_keys, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(overall_keys, keys))
        row["task"] = "ALL"
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
            row[f"{metric}_stderr"] = (
                float(finite.std(ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else np.nan
            )
            row[f"{metric}_count"] = len(finite)
        row["success_ci_low"], row["success_ci_high"] = _bootstrap_mean(
            group["success"].to_numpy(dtype=float), rng
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    unit_keys = ["method", *variant_keys]
    unit_keys.extend(["task", "seed", "visual_condition"])
    units = data.groupby(unit_keys, as_index=False)[["success", "return"]].mean()
    stats_rows = []
    for candidate, baseline in COMPARISONS:
        candidate_rows = units[units["method"] == candidate]
        baseline_rows = units[units["method"] == baseline]
        if candidate_rows.empty or baseline_rows.empty:
            continue
        variants = candidate_rows[variant_keys].drop_duplicates()
        for _, variant in variants.iterrows():
            left = candidate_rows.copy()
            for key in variant_keys:
                left = left[left[key] == variant[key]]
            right = baseline_rows.copy()
            if baseline != "bc_only":
                for key in variant_keys:
                    right = right[right[key] == variant[key]]
            if left.empty or right.empty:
                continue
            candidate_score = str(variant.get("planner_score", "default"))
            baseline_score = (
                "bc_only"
                if baseline == "bc_only"
                else str(variant.get("planner_score", "default"))
            )
            for metric in ("success", "return"):
                index = ["task", "seed", "visual_condition"]
                left_values = left.set_index(index)[metric]
                right_values = right.set_index(index)[metric]
                difference = (left_values - right_values).dropna()
                base = {
                    "candidate": candidate,
                    "candidate_planner_score": candidate_score,
                    "baseline": baseline,
                    "baseline_planner_score": baseline_score,
                    "planner_alpha": variant.get("planner_alpha", np.nan),
                    "planning_horizon": variant.get("planning_horizon", np.nan),
                    "metric": metric,
                }
                stats_rows.append(
                    {
                        **base,
                        "visual_condition": "ALL",
                        **_paired_test(difference.to_numpy(dtype=float), rng),
                    }
                )
                for condition, values in difference.groupby(level="visual_condition"):
                    stats_rows.append(
                        {
                            **base,
                            "visual_condition": condition,
                            **_paired_test(values.to_numpy(dtype=float), rng),
                        }
                    )
    statistics = pd.DataFrame(stats_rows)
    statistics.to_csv(statistics_path, index=False)
    _write_latex(summary, table_path)
    return summary, statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/closed_loop_rollout_summary.csv"),
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("results/closed_loop_rollout_statistics.csv"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("tables/closed_loop_rollout_table.tex"),
    )
    args = parser.parse_args()
    summary, statistics = analyze(
        args.input, args.summary, args.statistics, args.table
    )
    print(summary[summary["task"] == "ALL"].to_string(index=False))
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
