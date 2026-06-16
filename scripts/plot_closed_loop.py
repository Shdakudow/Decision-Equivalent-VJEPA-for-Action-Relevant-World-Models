from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _grouped_bars(
    frame: pd.DataFrame,
    value: str,
    ylabel: str,
    destination: Path,
    x_key: str = "visual_condition",
) -> None:
    if frame.empty:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.text(
            0.5,
            0.5,
            "No shifted condition available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        return
    methods = list(dict.fromkeys(frame["method"]))
    groups = list(dict.fromkeys(frame[x_key]))
    x = np.arange(len(groups))
    width = 0.8 / max(len(methods), 1)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for index, method in enumerate(methods):
        values = []
        errors = []
        for group in groups:
            row = frame[(frame["method"] == method) & (frame[x_key] == group)]
            values.append(float(row[value].iloc[0]) if not row.empty else np.nan)
            stderr = value.replace("_mean", "_stderr")
            errors.append(float(row[stderr].iloc[0]) if not row.empty else np.nan)
        axis.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=method,
        )
    axis.set_xticks(x, [group.replace("_", " ").title() for group in groups])
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def _sweep_plot(
    frame: pd.DataFrame,
    x_key: str,
    value: str,
    ylabel: str,
    destination: Path,
) -> None:
    frame = frame[frame["planner_score"] != "bc_only"].copy()
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for method, group in frame.groupby("method", sort=False):
        points = group.groupby(x_key, as_index=False)[value].mean().sort_values(x_key)
        axis.plot(points[x_key], points[value], "o-", label=method)
    axis.set_xlabel(x_key.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot(input_path: Path, output_dir: Path, prefix: str = "closed_loop") -> None:
    summary = pd.read_csv(input_path)
    numeric_summary = summary.copy()
    if "planner_score" in summary:
        detail = summary["planner_score"].astype(str)
        if "planner_alpha" in summary:
            detail += "\na=" + summary["planner_alpha"].astype(str)
        if "planning_horizon" in summary:
            detail += ", H=" + summary["planning_horizon"].astype(str)
        summary["plot_method"] = np.where(
            summary["planner_score"] == "bc_only", summary["method"], summary["method"] + "\n" + detail
        )
        summary["method"] = summary["plot_method"]
    overall = summary[summary["task"] == "ALL"]
    per_task = summary[summary["task"] != "ALL"].copy()
    _grouped_bars(
        overall,
        "success_mean",
        "Success rate",
        output_dir / f"{prefix}_success.png",
    )
    per_task["task_condition"] = (
        per_task["task"].str.replace("-v3", "", regex=False)
        + "\n"
        + per_task["visual_condition"].str.replace("_", " ", regex=False)
    )
    _grouped_bars(
        per_task,
        "success_mean",
        "Success rate",
        output_dir / f"{prefix}_by_task.png",
        x_key="task_condition",
    )
    _grouped_bars(
        overall,
        "return_mean",
        "Average return",
        output_dir / f"{prefix}_return.png",
    )

    clean = overall[overall["visual_condition"] == "clean"].set_index("method")
    gap_rows = []
    for _, row in overall[overall["visual_condition"] != "clean"].iterrows():
        if row["method"] not in clean.index:
            continue
        gap_rows.append(
            {
                "method": row["method"],
                "visual_condition": row["visual_condition"],
                "success_gap_mean": row["success_mean"]
                - clean.loc[row["method"], "success_mean"],
                "success_gap_stderr": np.sqrt(
                    row["success_stderr"] ** 2
                    + clean.loc[row["method"], "success_stderr"] ** 2
                ),
            }
        )
    _grouped_bars(
        pd.DataFrame(gap_rows),
        "success_gap_mean",
        "Success-rate change from clean",
        output_dir / f"{prefix}_condition_gap.png",
    )
    if "mean_decision_kl_mean" in overall:
        diagnostics = overall[overall["mean_decision_kl_mean"].notna()]
        _grouped_bars(
            diagnostics,
            "mean_decision_kl_mean",
            "Online decision KL",
            output_dir / f"{prefix}_online_decision_kl.png",
        )
    numeric_overall = numeric_summary[numeric_summary["task"] == "ALL"]
    if prefix.startswith("value_planner") and "planner_alpha" in numeric_overall:
        _sweep_plot(
            numeric_overall,
            "planner_alpha",
            "success_mean",
            "Success rate",
            output_dir / f"{prefix}_alpha_sweep.png",
        )
        points = numeric_overall[
            numeric_overall.get("mean_predicted_value_mean", pd.Series(dtype=float)).notna()
        ]
        fig, axis = plt.subplots(figsize=(6, 4.5))
        for method, group in points.groupby("method", sort=False):
            axis.scatter(
                group["mean_predicted_value_mean"],
                group["success_mean"],
                label=method,
            )
        axis.set_xlabel("Mean selected predicted progress")
        axis.set_ylabel("Success rate")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_value_vs_success.png", dpi=200)
        fig.savefig(output_dir / f"{prefix}_progress_vs_success.png", dpi=200)
        plt.close(fig)
    if prefix.startswith("cem_mpc") and "planning_horizon" in numeric_overall:
        _sweep_plot(
            numeric_overall,
            "planning_horizon",
            "success_mean",
            "Success rate",
            output_dir / f"{prefix}_success_by_horizon.png",
        )
        _sweep_plot(
            numeric_overall,
            "planning_horizon",
            "return_mean",
            "Average return",
            output_dir / f"{prefix}_return_by_horizon.png",
        )
        comparison = numeric_overall[
            numeric_overall["method"].isin(
                ["bc_plus_vjepa_cem", "bc_plus_de_vjepa_cem"]
            )
        ]
        _sweep_plot(
            comparison,
            "planning_horizon",
            "success_mean",
            "Success rate",
            output_dir / f"{prefix}_de_vs_vjepa.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--prefix", default="closed_loop")
    args = parser.parse_args()
    plot(args.input, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
