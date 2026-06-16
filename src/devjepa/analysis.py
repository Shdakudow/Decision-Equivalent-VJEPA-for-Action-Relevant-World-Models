from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from devjepa.config import load_config, parser


DEFAULT_METRICS = ("latent_mse", "decision_kl", "action_mse")
DEFAULT_PAIR_KEYS = ["train_seed", "evaluator_seed", "shift"]
DEFAULT_CONDITION_KEYS = ["method", "lambda", "reference_policy_type"]


def _safe_wilcoxon(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    differences = left.to_numpy() - right.to_numpy()
    if len(differences) == 0 or (differences == 0).all():
        return 0.0, 1.0
    result = wilcoxon(left, right)
    return float(result.statistic), float(result.pvalue)


def summarize(
    raw_path: Path,
    baseline_method: str,
    pair_by_task: bool = False,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    metrics = [
        metric
        for metric in (*DEFAULT_METRICS, "nll", "uncertainty")
        if metric in raw.columns and raw[metric].notna().any()
    ]
    pair_keys = list(DEFAULT_PAIR_KEYS)
    if pair_by_task:
        pair_keys.insert(0, "task")
    condition_keys = list(DEFAULT_CONDITION_KEYS)
    for key in ("bjepa_prior_type", "bjepa_loss_type"):
        if key in raw.columns:
            condition_keys.append(key)
    paired = (
        raw.groupby(condition_keys + pair_keys, dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    baseline = paired[paired["method"] == baseline_method].set_index(pair_keys)
    rows: list[dict[str, Any]] = []
    for condition, group in paired.groupby(condition_keys, dropna=False):
        condition_values = dict(zip(condition_keys, condition))
        method = condition_values["method"]
        candidate = group.set_index(pair_keys)
        for shift in sorted(group["shift"].unique()):
            shifted = candidate[candidate.index.get_level_values("shift") == shift]
            shifted_baseline = baseline[
                baseline.index.get_level_values("shift") == shift
            ]
            common = shifted.index.intersection(shifted_baseline.index)
            for metric in metrics:
                values = shifted.loc[common, metric]
                base_values = shifted_baseline.loc[common, metric]
                t_stat = t_p = w_stat = w_p = math.nan
                if method != baseline_method and len(common) >= 2:
                    t_result = ttest_rel(values, base_values)
                    t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)
                    w_stat, w_p = _safe_wilcoxon(values, base_values)
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row = {
                        **condition_values,
                        "shift": shift,
                        "metric": metric,
                        "pairs": len(values),
                        "mean": mean,
                        "std": std,
                        "stderr": std / math.sqrt(len(values)) if len(values) else math.nan,
                        "baseline_mean": float(base_values.mean()),
                        "relative_change_percent": (
                            100.0 * (mean - float(base_values.mean())) / float(base_values.mean())
                            if float(base_values.mean()) != 0.0
                            else math.nan
                        ),
                        "paired_t": t_stat,
                        "paired_t_p": t_p,
                        "wilcoxon_w": w_stat,
                        "wilcoxon_p": w_p,
                    }
                rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(
    raw_path: Path,
    comparisons: list[list[str]],
    pair_by_task: bool = True,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    metrics = [
        metric
        for metric in (*DEFAULT_METRICS, "nll", "uncertainty")
        if metric in raw.columns and raw[metric].notna().any()
    ]
    pair_keys = list(DEFAULT_PAIR_KEYS)
    if pair_by_task:
        pair_keys.insert(0, "task")
    paired = raw.groupby(["method"] + pair_keys, dropna=False)[metrics].mean().reset_index()
    rows = []
    for candidate_method, baseline_method in comparisons:
        candidate = paired[paired["method"] == candidate_method].set_index(pair_keys)
        baseline = paired[paired["method"] == baseline_method].set_index(pair_keys)
        for shift in sorted(set(candidate.reset_index()["shift"])):
            left = candidate[candidate.index.get_level_values("shift") == shift]
            right = baseline[baseline.index.get_level_values("shift") == shift]
            common = left.index.intersection(right.index)
            for metric in metrics:
                values = left.loc[common, metric].dropna()
                base_values = right.loc[common, metric].reindex(values.index)
                t_stat = t_p = w_stat = w_p = math.nan
                if len(values) >= 2:
                    t_result = ttest_rel(values, base_values)
                    t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)
                    w_stat, w_p = _safe_wilcoxon(values, base_values)
                mean_change = float((values - base_values).mean()) if len(values) else math.nan
                change_std = (
                    float((values - base_values).std(ddof=1)) if len(values) > 1 else 0.0
                )
                rows.append(
                    {
                        "candidate": candidate_method,
                        "baseline": baseline_method,
                        "shift": shift,
                        "metric": metric,
                        "pairs": len(values),
                        "candidate_mean": float(values.mean()) if len(values) else math.nan,
                        "baseline_mean": float(base_values.mean()) if len(values) else math.nan,
                        "mean_change": mean_change,
                        "change_stderr": change_std / math.sqrt(len(values))
                        if len(values)
                        else math.nan,
                        "relative_change_percent": 100.0
                        * mean_change
                        / float(base_values.mean())
                        if len(values) and float(base_values.mean()) != 0.0
                        else math.nan,
                        "paired_t": t_stat,
                        "paired_t_p": t_p,
                        "wilcoxon_w": w_stat,
                        "wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(rows)


def _plot_lambda(
    summary: pd.DataFrame,
    output_dir: Path,
    prefix: str = "lambda",
) -> None:
    for metric in DEFAULT_METRICS:
        figure, axis = plt.subplots(figsize=(6, 4))
        data = summary[summary["metric"] == metric]
        for shift, group in data.groupby("shift"):
            group = group.sort_values("lambda")
            axis.errorbar(
                group["lambda"],
                group["mean"],
                yerr=group["stderr"],
                marker="o",
                capsize=3,
                label=shift,
            )
        axis.set_xscale("symlog", linthresh=1e-4)
        axis.set_xlabel("Decision regularizer weight")
        axis.set_ylabel(metric.replace("_", " "))
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"{prefix}_vs_{metric}.png"
        _preserve_existing(path)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def _plot_bars(summary: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    for metric in DEFAULT_METRICS:
        data = summary[summary["metric"] == metric].copy()
        data["label"] = data["method"] + "\n" + data["shift"]
        figure, axis = plt.subplots(figsize=(max(7, len(data) * 0.7), 4))
        axis.bar(data["label"], data["mean"], yerr=data["stderr"], capsize=3)
        axis.set_ylabel(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path = output_dir / f"{prefix}_{metric}.png"
        _preserve_existing(path)
        figure.savefig(path, dpi=180)
        plt.close(figure)


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "method",
        "lambda",
        "reference_policy_type",
        "shift",
        "metric",
        "mean",
        "stderr",
        "relative_change_percent",
        "paired_t_p",
        "wilcoxon_p",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [column for column in columns if column in summary.columns]
    path.write_text(
        summary[columns].to_latex(index=False, float_format="%.4g", escape=True),
        encoding="utf-8",
    )


def _plot_uncertainty_calibration(raw_path: Path, output_dir: Path) -> None:
    raw = pd.read_csv(raw_path)
    data = raw.dropna(subset=["uncertainty"]).copy()
    if data.empty:
        return
    data["variance_bin"] = data.groupby(["method", "shift"])["uncertainty"].transform(
        lambda values: pd.qcut(values, q=min(10, values.nunique()), duplicates="drop")
    )
    grouped = (
        data.groupby(["method", "shift", "variance_bin"], observed=True)
        .agg(
            uncertainty=("uncertainty", "mean"),
            latent_mse=("latent_mse", "mean"),
            decision_kl=("decision_kl", "mean"),
        )
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (method, shift), group in grouped.groupby(["method", "shift"]):
        label = f"{method} / {shift}"
        axes[0].plot(group["uncertainty"], group["latent_mse"], marker="o", label=label)
        axes[1].plot(group["uncertainty"], group["decision_kl"], marker="o", label=label)
    axes[0].set_xlabel("Mean predictive variance")
    axes[0].set_ylabel("Latent MSE")
    axes[1].set_xlabel("Mean predictive variance")
    axes[1].set_ylabel("Decision KL")
    axes[1].legend(fontsize=6)
    figure.tight_layout()
    path = output_dir / "uncertainty_vs_error.png"
    _preserve_existing(path)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _preserve_existing(path: Path) -> None:
    if not path.exists():
        return
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path.replace(path.with_name(f"{path.stem}_{timestamp}{path.suffix}"))


def analyze(config: dict[str, Any]) -> Path:
    raw_path = Path(config["input"])
    summary = summarize(
        raw_path,
        config.get("baseline_method", "latent_only"),
        bool(config.get("pair_by_task", False)),
    )
    summary_path = Path(config["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _preserve_existing(summary_path)
    summary.to_csv(summary_path, index=False)
    table_path = Path(config["table"])
    _preserve_existing(table_path)
    _write_latex(summary, table_path)
    plot_dir = Path(config["plots_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    kind = config.get("plot_kind", "bars")
    if kind == "lambda":
        _plot_lambda(
            summary,
            plot_dir,
            prefix=str(config.get("plot_prefix", "lambda")),
        )
    else:
        _plot_bars(summary, plot_dir, config.get("plot_prefix", kind))
    _plot_uncertainty_calibration(raw_path, plot_dir)
    comparisons = config.get("comparisons")
    if comparisons:
        comparison_path = Path(
            config.get(
                "comparisons_output",
                summary_path.with_name(f"{summary_path.stem}_comparisons.csv"),
            )
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        _preserve_existing(comparison_path)
        paired_comparisons(
            raw_path,
            comparisons,
            bool(config.get("comparison_pair_by_task", True)),
        ).to_csv(comparison_path, index=False)
    return summary_path


def main() -> None:
    args = parser("Analyze DE-VJEPA ablation results").parse_args()
    config = load_config(args.config, args.set)
    print(analyze(config.get("analysis", config)))
