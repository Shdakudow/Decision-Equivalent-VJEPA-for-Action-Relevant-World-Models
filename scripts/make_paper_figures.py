from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

COLORS = {
    "vjepa": "#6B7280",
    "de": "#0072B2",
    "bjepa": "#D55E00",
    "de_bjepa": "#009E73",
    "bc": "#4B5563",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(PLOTS / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def offline_results() -> None:
    core = pd.read_csv(RESULTS / "vjepa_bjepa_de_comparison_summary.csv")
    clean = core[
        (core["shift"] == "clean")
        & core["metric"].isin(["latent_mse", "decision_kl"])
    ]
    clean = clean.pivot(index="method", columns="metric", values="relative_change_percent")

    mw = pd.read_csv(RESULTS / "cross_method_metaworld_paired.csv")
    pt = pd.read_csv(RESULTS / "cross_method_pusht_paired.csv")
    paired = pd.concat([mw.assign(dataset="MetaWorld-5"), pt.assign(dataset="PushT")])
    paired = paired[
        (paired["candidate"] == "de_vjepa")
        & (paired["baseline"] == "vjepa")
        & (paired["metric"] == "decision_kl")
    ].copy()
    order = ["clean", "brightness", "blur", "desaturation"]
    paired["shift"] = pd.Categorical(paired["shift"], categories=order, ordered=True)
    paired = paired.sort_values(["dataset", "shift"])

    multi = pd.read_csv(RESULTS / "multistep_diagnostic_summary.csv")
    multi = (
        multi.groupby(["method", "horizon"], as_index=False)
        .apply(lambda x: pd.Series({"decision_kl": np.average(x["decision_kl"], weights=x["count"])}))
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.75))

    ax = axes[0]
    labels = {
        "vjepa": "VJEPA",
        "bjepa": "BJEPA",
        "de_vjepa": "DE-VJEPA",
        "de_bjepa": "DE-BJEPA",
    }
    for method, row in clean.iterrows():
        color = COLORS.get(method, COLORS["vjepa"])
        ax.scatter(
            row["latent_mse"],
            row["decision_kl"],
            s=55 if method == "de_vjepa" else 38,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            labels[method],
            (row["latent_mse"], row["decision_kl"]),
            xytext=(4, 4 if method != "de_bjepa" else -11),
            textcoords="offset points",
            fontsize=7.3,
        )
    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.axvline(0, color="#999999", linewidth=0.7)
    ax.set_xlabel("Latent MSE change vs. VJEPA (%)")
    ax.set_ylabel("Decision KL change vs. VJEPA (%)")
    ax.set_title("(a) Clean MetaWorld trade-off", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.04,
        "better decisions",
        transform=ax.transAxes,
        color=COLORS["de"],
        fontsize=7,
    )

    ax = axes[1]
    x = np.arange(len(order))
    width = 0.36
    for offset, dataset, color in [
        (-width / 2, "MetaWorld-5", COLORS["de"]),
        (width / 2, "PushT", COLORS["de_bjepa"]),
    ]:
        subset = paired[paired["dataset"] == dataset].set_index("shift").reindex(order)
        ax.bar(
            x + offset,
            subset["relative_change_percent"],
            width,
            color=color,
            label=dataset,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, ["Clean", "Bright.", "Blur", "Desat."], rotation=20)
    ax.set_ylabel("DE-VJEPA decision KL change (%) $\\downarrow$")
    ax.set_title("(b) Independent-policy evaluation", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower left")

    ax = axes[2]
    for method, label, color, marker in [
        ("bc_plus_vjepa", "VJEPA", COLORS["vjepa"], "o"),
        ("bc_plus_de_vjepa", "DE-VJEPA", COLORS["de"], "o"),
    ]:
        subset = multi[multi["method"] == method].sort_values("horizon")
        ax.plot(
            subset["horizon"],
            subset["decision_kl"],
            marker=marker,
            linewidth=2,
            markersize=5,
            label=label,
            color=color,
        )
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Autoregressive horizon")
    ax.set_ylabel("Decision KL")
    ax.set_title("(c) Advantage persists to $K=5$", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    fig.tight_layout(w_pad=2.0)
    _save(fig, "paper_offline_results")


def policy_geometry() -> None:
    data = pd.read_csv(RESULTS / "policy_geometry_ablation_summary.csv")
    data = data[data["metric"] == "decision_kl"].copy()
    methods = ["random_policy_KL", "weak_BC_policy_KL", "strong_BC_policy_KL"]
    labels = ["Random policy", "Weak BC", "Strong BC"]
    shifts = [("clean", "Clean"), ("brightness", "Brightness")]
    x = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.7, 2.75))
    for index, (shift, label) in enumerate(shifts):
        values = [
            data[(data["method"] == method) & (data["shift"] == shift)][
                "relative_change_percent"
            ].iloc[0]
            for method in methods
        ]
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=[COLORS["vjepa"], COLORS["de_bjepa"]][index],
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Decision KL change vs. latent-only (%)")
    ax.set_title("Useful geometry requires a competent reference policy\n(lower is better)",
                 fontweight="bold", fontsize=9)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "paper_policy_geometry")


def closed_loop() -> None:
    data = pd.read_csv(RESULTS / "closed_loop_rollout_summary.csv")
    aggregate = (
        data.groupby(["method", "visual_condition"])
        .apply(
            lambda x: pd.Series(
                {
                    "success": np.average(x["success_mean"], weights=x["success_count"]),
                    "count": x["success_count"].sum(),
                }
            )
        )
        .reset_index()
    )
    conditions = ["clean", "brightness", "background_shift"]
    condition_labels = ["Clean", "Brightness", "Background"]
    methods = ["bc_only", "bc_plus_vjepa", "bc_plus_de_vjepa"]
    labels = ["BC only", "BC + VJEPA", "BC + DE-VJEPA"]
    colors = [COLORS["bc"], COLORS["vjepa"], COLORS["de"]]
    x = np.arange(len(conditions))
    width = 0.24

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        subset = aggregate[aggregate["method"] == method].set_index("visual_condition")
        means = subset.reindex(conditions)["success"].to_numpy()
        counts = subset.reindex(conditions)["count"].to_numpy()
        errors = np.sqrt(means * (1 - means) / counts)
        ax.bar(
            x + (index - 1) * width,
            means * 100,
            width,
            yerr=errors * 100,
            capsize=2.5,
            label=label,
            color=color,
        )
    ax.set_xticks(x, condition_labels)
    ax.set_ylabel("Episode success (%)")
    ax.set_ylim(0, 90)
    ax.set_title("Offline decision gains do not yield significant control gains", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.text(
        0.99,
        0.60,
        "DE vs. VJEPA overall:\n+1.30 points, $p=0.473$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.25"},
    )
    fig.tight_layout()
    _save(fig, "paper_closed_loop")


def ablation_statistics() -> None:
    sweep = pd.read_csv(RESULTS / "lambda_sweep_summary.csv")
    matching = pd.read_csv(RESULTS / "kl_vs_action_mse_summary.csv")
    rollout = pd.read_csv(RESULTS / "closed_loop_rollout_statistics.csv")
    cem = pd.read_csv(RESULTS / "cem_mpc_progress512_statistics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8))

    ax = axes[0]
    clean = sweep[sweep["shift"] == "clean"]
    for metric, label, color, marker in [
        ("latent_mse", "Latent MSE", COLORS["bjepa"], "s"),
        ("decision_kl", "Decision KL", COLORS["de"], "o"),
        ("action_mse", "Action MSE", COLORS["de_bjepa"], "^"),
    ]:
        subset = clean[clean["metric"] == metric].sort_values("lambda")
        ax.plot(
            subset["lambda"],
            subset["relative_change_percent"],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )
        significant = subset["paired_t_p"] < 0.05
        ax.scatter(
            subset.loc[significant, "lambda"],
            subset.loc[significant, "relative_change_percent"],
            s=58,
            facecolors="none",
            edgecolors=color,
            linewidth=1.1,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xticks([0, 1e-4, 1e-3, 1e-2, 1e-1], ["0", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax.set_yticks([-20, -10, 0, 10, 20, 50, 100])
    ax.set_yticklabels(["-20", "-10", "0", "10", "20", "50", "100"])
    ax.set_xlabel("Decision weight $\\lambda$")
    ax.set_ylabel("Change vs. latent-only (%)")
    ax.set_title("(a) Fidelity is tunable (rings: $p<.05$)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left",
              bbox_to_anchor=(0.0, 0.88))

    ax = axes[1]
    rows = []
    for shift in ["clean", "brightness"]:
        for method, label in [("policy_kl", "Policy KL"), ("action_mse_1e-1", "Action mean")]:
            if method == "policy_kl":
                geometry = pd.read_csv(RESULTS / "policy_geometry_ablation_summary.csv")
                row = geometry[
                    (geometry["method"] == "strong_BC_policy_KL")
                    & (geometry["shift"] == shift)
                    & (geometry["metric"] == "decision_kl")
                ].iloc[0]
            else:
                row = matching[
                    (matching["method"] == method)
                    & (matching["shift"] == shift)
                    & (matching["metric"] == "decision_kl")
                ].iloc[0]
            rows.append((shift, label, row["relative_change_percent"], row["paired_t_p"]))
    x = np.arange(2)
    width = 0.34
    for index, (label, color) in enumerate([("Policy KL", COLORS["de"]), ("Action mean", COLORS["vjepa"])]):
        values = [next(r[2] for r in rows if r[0] == shift and r[1] == label) for shift in ["clean", "brightness"]]
        bars = ax.bar(x + (index - 0.5) * width, values, width, label=label, color=color)
        for bar, shift in zip(bars, ["clean", "brightness"]):
            pvalue = next(r[3] for r in rows if r[0] == shift and r[1] == label)
            if pvalue < 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1.2, "*", ha="center", va="top", color="white", fontweight="bold")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, ["Clean", "Brightness"])
    ax.set_ylabel("Decision KL change (%)")
    ax.set_title("(b) Distribution $>$ mean matching (* $p<.05$)",
                 loc="left", fontweight="bold")
    ax.set_ylim(top=4.0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.98),
              ncol=2, fontsize=7.0, columnspacing=1.2, handletextpad=0.4)

    ax = axes[2]
    forest = []
    overall = rollout[
        (rollout["candidate"] == "bc_plus_de_vjepa")
        & (rollout["baseline"] == "bc_plus_vjepa")
        & (rollout["metric"] == "success")
        & (rollout["visual_condition"] == "ALL")
    ].iloc[0]
    forest.append(("One-step planner", overall))
    brightness = rollout[
        (rollout["candidate"] == "bc_plus_de_vjepa")
        & (rollout["baseline"] == "bc_plus_vjepa")
        & (rollout["metric"] == "success")
        & (rollout["visual_condition"] == "brightness")
    ].iloc[0]
    forest.append(("One-step, bright.", brightness))
    cem_row = cem[
        (cem["candidate"] == "bc_plus_de_vjepa_cem")
        & (cem["baseline"] == "bc_plus_vjepa_cem")
        & (cem["planner_alpha"] == 0.1)
        & (cem["planning_horizon"] == 3)
        & (cem["metric"] == "success")
        & (cem["visual_condition"] == "ALL")
    ].iloc[0]
    forest.append(("CEM $H=3$", cem_row))
    cem_bright = cem[
        (cem["candidate"] == "bc_plus_de_vjepa_cem")
        & (cem["baseline"] == "bc_plus_vjepa_cem")
        & (cem["planner_alpha"] == 0.1)
        & (cem["planning_horizon"] == 3)
        & (cem["metric"] == "success")
        & (cem["visual_condition"] == "brightness")
    ].iloc[0]
    forest.append(("CEM $H=3$, bright.", cem_bright))
    y = np.arange(len(forest))[::-1]
    means = np.array([row["mean_difference"] * 100 for _, row in forest])
    lows = np.array([row["bootstrap_ci_low"] * 100 for _, row in forest])
    highs = np.array([row["bootstrap_ci_high"] * 100 for _, row in forest])
    ax.errorbar(
        means,
        y,
        xerr=np.vstack((means - lows, highs - means)),
        fmt="o",
        color=COLORS["de"],
        ecolor="#555555",
        capsize=2.5,
        markersize=5,
    )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, [name for name, _ in forest])
    ax.set_xlabel("DE-VJEPA success difference (points)")
    ax.set_title("(c) Closed-loop effects remain uncertain", loc="left", fontweight="bold")
    for yi, (_, row) in zip(y, forest):
        ax.text(highs[len(forest) - 1 - yi] + 0.7, yi, f"$p={row['paired_t_pvalue']:.3f}$", va="center", fontsize=6.7)

    fig.tight_layout(w_pad=2.2)
    _save(fig, "paper_ablation_statistics")


def baseline_heatmap() -> None:
    data = pd.read_csv(RESULTS / "cross_dataset_comparison_summary.csv")
    data = data[data["metric"] == "decision_kl"].copy()
    methods = [
        "persistence",
        "linear_dynamics",
        "no_action_vjepa",
        "cosine_jepa",
        "action_mse",
        "bjepa",
        "de_vjepa",
        "de_bjepa",
        "random_policy_kl",
        "weak_policy_kl",
    ]
    labels = [
        "Persistence",
        "Linear",
        "No action",
        "Cosine",
        "Action mean",
        "BJEPA",
        "DE-VJEPA",
        "DE-BJEPA",
        "Random KL",
        "Weak-BC KL",
    ]
    shifts = ["clean", "brightness", "blur", "desaturation"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)
    for ax, dataset, title in zip(axes, ["metaworld-5", "lerobot-pusht"], ["MetaWorld-5", "PushT"]):
        subset = data[data["dataset"] == dataset]
        matrix = np.full((len(methods), len(shifts)), np.nan)
        for j, shift in enumerate(shifts):
            vjepa = subset[(subset["method"] == "vjepa") & (subset["shift"] == shift)]["mean"].iloc[0]
            for i, method in enumerate(methods):
                row = subset[(subset["method"] == method) & (subset["shift"] == shift)]
                if not row.empty:
                    matrix[i, j] = 100 * (row["mean"].iloc[0] - vjepa) / vjepa
        clipped = np.clip(matrix, -50, 50)
        image = ax.imshow(clipped, cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
        for i in range(len(methods)):
            for j in range(len(shifts)):
                if np.isfinite(matrix[i, j]):
                    value = matrix[i, j]
                    text = "0" if abs(value) < 0.5 else f"{value:+.0f}"
                    color = "white" if abs(clipped[i, j]) > 28 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=6.6, color=color)
        ax.set_xticks(range(len(shifts)), ["Clean", "Bright.", "Blur", "Desat."], rotation=25)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Decision KL change vs. VJEPA (%)")
    axes[0].set_yticks(range(len(methods)), labels)
    # Keep the color scale in a dedicated margin so it cannot cover PushT cells.
    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.20, top=0.88, wspace=0.08)
    colorbar_ax = fig.add_axes([0.905, 0.20, 0.018, 0.68])
    figure_colorbar = fig.colorbar(image, cax=colorbar_ax)
    figure_colorbar.set_label("lower is better", fontsize=8)
    fig.suptitle("Extended baseline matrix: independent decision KL", fontweight="bold", y=1.01)
    _save(fig, "paper_baseline_heatmap")


def planner_ablation_appendix() -> None:
    data = pd.read_csv(RESULTS / "planner_ablation_summary.csv")
    data = data[data["task"] == "ALL"].copy()
    conditions = ["clean", "brightness", "background_shift"]
    condition_labels = ["Clean", "Brightness", "Background shift"]

    methods = [
        ("bc_only", "bc_only", "BC only", COLORS["bc"]),
        ("bc_plus_vjepa", "entropy_score", "VJEPA + entropy", "#7a8a98"),
        ("bc_plus_de_vjepa", "entropy_score", "DE-VJEPA + entropy", "#3b6ea5"),
        ("bc_plus_vjepa", "decision_consistency_score", "VJEPA + decision", "#b5651d"),
        ("bc_plus_de_vjepa", "decision_consistency_score", "DE-VJEPA + decision", "#d59a4d"),
        ("bc_plus_vjepa", "action_mean_consistency_score", "VJEPA + action mean", "#5c8a5c"),
        ("bc_plus_de_vjepa", "action_mean_consistency_score", "DE-VJEPA + action mean", "#88b888"),
        ("bc_plus_vjepa", "random_candidate_control", "VJEPA + random", "#666666"),
        ("bc_plus_de_vjepa", "random_candidate_control", "DE-VJEPA + random", "#aaaaaa"),
    ]

    def _draw(metric_mean: str, metric_err: str, ylabel: str, title: str, name: str) -> None:
        fig, ax = plt.subplots(figsize=(7.4, 3.2))
        n = len(methods)
        x = np.arange(len(conditions))
        width = 0.8 / n
        for index, (method, score, label, color) in enumerate(methods):
            means, errs = [], []
            for cond in conditions:
                row = data[
                    (data["method"] == method)
                    & (data["planner_score"] == score)
                    & (data["visual_condition"] == cond)
                ]
                if row.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                else:
                    means.append(row[metric_mean].iloc[0])
                    errs.append(row[metric_err].iloc[0] if metric_err else 0.0)
            ax.bar(
                x + (index - (n - 1) / 2) * width,
                means,
                width,
                yerr=errs if metric_err else None,
                capsize=1.5,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x, condition_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(
            frameon=False,
            fontsize=7,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        fig.tight_layout()
        _save(fig, name)

    _draw(
        "success_mean",
        "success_stderr",
        "Episode success rate",
        "Planner-ablation success rates",
        "planner_ablation_success",
    )
    _draw(
        "mean_decision_kl_mean",
        "mean_decision_kl_stderr",
        "Online decision KL $\\downarrow$",
        "Planner-ablation online decision KL",
        "planner_ablation_online_decision_kl",
    )


def multistep_appendix() -> None:
    data = pd.read_csv(RESULTS / "multistep_diagnostic_summary.csv")
    agg = (
        data.groupby(["method", "horizon"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "decision_kl": np.average(x["decision_kl"], weights=x["count"]),
                    "latent_mse": np.average(x["latent_mse"], weights=x["count"]),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    methods = [
        ("bc_plus_vjepa", "VJEPA", COLORS["vjepa"]),
        ("bc_plus_de_vjepa", "DE-VJEPA", COLORS["de"]),
    ]

    def _draw(field: str, ylabel: str, title: str, name: str) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        for method, label, color in methods:
            sub = agg[agg["method"] == method].sort_values("horizon")
            ax.plot(
                sub["horizon"],
                sub[field],
                marker="o",
                linewidth=2,
                markersize=6,
                color=color,
                label=label,
            )
        ax.set_xticks([1, 3, 5])
        ax.set_xlabel("Prediction horizon $K$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        _save(fig, name)

    _draw(
        "decision_kl",
        "Decision KL $\\downarrow$",
        "Autoregressive decision KL",
        "multistep_decision_kl_vs_horizon",
    )
    _draw(
        "latent_mse",
        "Latent MSE $\\downarrow$",
        "Autoregressive latent MSE",
        "multistep_latent_mse_vs_horizon",
    )


def cem_progress_appendix() -> None:
    data = pd.read_csv(RESULTS / "cem_mpc_progress512_summary.csv")
    data = data[data["task"] == "ALL"].copy()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    horizons = [1, 3, 5]
    bc_row = data[
        (data["method"] == "bc_only") & (data["visual_condition"] == "clean")
    ].iloc[0]
    ax.axhline(
        bc_row["success_mean"],
        color=COLORS["bc"],
        linestyle="--",
        linewidth=1.2,
        label="BC only (clean)",
        alpha=0.7,
    )
    for method, label, color in [
        ("bc_plus_vjepa_cem", "VJEPA + CEM", COLORS["vjepa"]),
        ("bc_plus_de_vjepa_cem", "DE-VJEPA + CEM", COLORS["de"]),
    ]:
        means, errs = [], []
        for h in horizons:
            row = data[
                (data["method"] == method)
                & (data["planner_alpha"] == 0.1)
                & (data["planning_horizon"] == h)
                & (data["visual_condition"] == "clean")
            ]
            means.append(row["success_mean"].iloc[0] if not row.empty else np.nan)
            errs.append(row["success_stderr"].iloc[0] if not row.empty else 0.0)
        ax.errorbar(
            horizons,
            means,
            yerr=errs,
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_xticks(horizons)
    ax.set_xlabel("Planning horizon $H$")
    ax.set_ylabel("Clean success rate $\\uparrow$")
    ax.set_title("Progress-aware CEM: success vs. horizon ($\\alpha{=}0.1$)",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "cem_mpc_progress512_success_by_horizon")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    conditions = ["brightness", "background_shift"]
    condition_labels = ["Brightness", "Background shift"]
    methods = [
        ("bc_only", "bc_only", 0.0, 0, "BC only", COLORS["bc"]),
        ("bc_plus_vjepa_cem", "cem_value", 0.1, 1, "VJEPA, $H{=}1$", "#7a8a98"),
        ("bc_plus_de_vjepa_cem", "cem_value", 0.1, 1, "DE-VJEPA, $H{=}1$", "#3b6ea5"),
        ("bc_plus_vjepa_cem", "cem_value", 0.1, 3, "VJEPA, $H{=}3$", "#b5651d"),
        ("bc_plus_de_vjepa_cem", "cem_value", 0.1, 3, "DE-VJEPA, $H{=}3$", "#d59a4d"),
        ("bc_plus_vjepa_cem", "cem_value", 0.1, 5, "VJEPA, $H{=}5$", "#5c8a5c"),
        ("bc_plus_de_vjepa_cem", "cem_value", 0.1, 5, "DE-VJEPA, $H{=}5$", "#88b888"),
    ]
    clean = {}
    for method, score, alpha, h, _, _ in methods:
        row = data[
            (data["method"] == method)
            & (data["planner_score"] == score)
            & (data["planner_alpha"] == alpha)
            & (data["planning_horizon"] == h)
            & (data["visual_condition"] == "clean")
        ]
        clean[(method, h)] = row["success_mean"].iloc[0] if not row.empty else np.nan

    n = len(methods)
    x = np.arange(len(conditions))
    width = 0.8 / n
    for index, (method, score, alpha, h, label, color) in enumerate(methods):
        diffs, errs = [], []
        for cond in conditions:
            row = data[
                (data["method"] == method)
                & (data["planner_score"] == score)
                & (data["planner_alpha"] == alpha)
                & (data["planning_horizon"] == h)
                & (data["visual_condition"] == cond)
            ]
            if row.empty:
                diffs.append(np.nan)
                errs.append(0.0)
            else:
                diffs.append(row["success_mean"].iloc[0] - clean[(method, h)])
                errs.append(row["success_stderr"].iloc[0])
        ax.bar(
            x + (index - (n - 1) / 2) * width,
            diffs,
            width,
            yerr=errs,
            capsize=1.5,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, condition_labels)
    ax.set_ylabel("Success-rate change from clean")
    ax.set_title("Progress-aware CEM: visual-shift gap by horizon",
                 fontweight="bold")
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout()
    _save(fig, "cem_mpc_progress512_condition_gap")


def planner_ablation_combined() -> None:
    data = pd.read_csv(RESULTS / "planner_ablation_summary.csv")
    data = data[data["task"] == "ALL"].copy()
    conditions = ["clean", "brightness", "background_shift"]
    condition_labels = ["Clean", "Brightness", "Background shift"]

    methods = [
        ("bc_only", "bc_only", "BC only", COLORS["bc"]),
        ("bc_plus_vjepa", "entropy_score", "VJEPA + entropy", "#7a8a98"),
        ("bc_plus_de_vjepa", "entropy_score", "DE-VJEPA + entropy", "#3b6ea5"),
        ("bc_plus_vjepa", "decision_consistency_score", "VJEPA + decision", "#b5651d"),
        ("bc_plus_de_vjepa", "decision_consistency_score", "DE-VJEPA + decision", "#d59a4d"),
        ("bc_plus_vjepa", "action_mean_consistency_score", "VJEPA + action mean", "#5c8a5c"),
        ("bc_plus_de_vjepa", "action_mean_consistency_score", "DE-VJEPA + action mean", "#88b888"),
        ("bc_plus_vjepa", "random_candidate_control", "VJEPA + random", "#666666"),
        ("bc_plus_de_vjepa", "random_candidate_control", "DE-VJEPA + random", "#aaaaaa"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4))
    panels = [
        (axes[0], "success_mean", "success_stderr",
         "Episode success rate $\\uparrow$",
         "(a) Planner-ablation success rates"),
        (axes[1], "mean_decision_kl_mean", "mean_decision_kl_stderr",
         "Online decision KL $\\downarrow$",
         "(b) Planner-ablation online decision KL"),
    ]
    n = len(methods)
    x = np.arange(len(conditions))
    width = 0.8 / n
    for ax, mean_col, err_col, ylabel, title in panels:
        for index, (method, score, label, color) in enumerate(methods):
            means, errs = [], []
            for cond in conditions:
                row = data[
                    (data["method"] == method)
                    & (data["planner_score"] == score)
                    & (data["visual_condition"] == cond)
                ]
                if row.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                else:
                    means.append(row[mean_col].iloc[0])
                    errs.append(row[err_col].iloc[0])
            ax.bar(
                x + (index - (n - 1) / 2) * width,
                means,
                width,
                yerr=errs,
                capsize=1.5,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x, condition_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        frameon=False,
        fontsize=8,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(0.905, 0.5),
    )
    fig.tight_layout(rect=(0, 0, 0.90, 1))
    _save(fig, "planner_ablation_appendix")


def multistep_combined() -> None:
    data = pd.read_csv(RESULTS / "multistep_diagnostic_summary.csv")
    agg = (
        data.groupby(["method", "horizon"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "decision_kl": np.average(x["decision_kl"], weights=x["count"]),
                    "latent_mse": np.average(x["latent_mse"], weights=x["count"]),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    methods = [
        ("bc_plus_vjepa", "VJEPA", COLORS["vjepa"]),
        ("bc_plus_de_vjepa", "DE-VJEPA", COLORS["de"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    panels = [
        (axes[0], "decision_kl", "Decision KL $\\downarrow$",
         "(a) Autoregressive decision KL"),
        (axes[1], "latent_mse", "Latent MSE $\\downarrow$",
         "(b) Autoregressive latent MSE"),
    ]
    for ax, field, ylabel, title in panels:
        for method, label, color in methods:
            sub = agg[agg["method"] == method].sort_values("horizon")
            ax.plot(
                sub["horizon"],
                sub[field],
                marker="o",
                linewidth=2,
                markersize=7,
                color=color,
                label=label,
            )
        ax.set_xticks([1, 3, 5])
        ax.set_xlabel("Prediction horizon $K$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "multistep_appendix")


def cem_progress_combined() -> None:
    data = pd.read_csv(RESULTS / "cem_mpc_progress512_summary.csv")
    data = data[data["task"] == "ALL"].copy()
    horizons = [1, 3, 5]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4),
                             gridspec_kw={"width_ratios": [1.0, 1.3]})

    ax = axes[0]
    bc_row = data[
        (data["method"] == "bc_only") & (data["visual_condition"] == "clean")
    ].iloc[0]
    ax.axhline(
        bc_row["success_mean"],
        color=COLORS["bc"],
        linestyle="--",
        linewidth=1.2,
        label="BC only (clean)",
        alpha=0.7,
    )
    for method, label, color in [
        ("bc_plus_vjepa_cem", "VJEPA + CEM", COLORS["vjepa"]),
        ("bc_plus_de_vjepa_cem", "DE-VJEPA + CEM", COLORS["de"]),
    ]:
        means, errs = [], []
        for h in horizons:
            row = data[
                (data["method"] == method)
                & (data["planner_alpha"] == 0.1)
                & (data["planning_horizon"] == h)
                & (data["visual_condition"] == "clean")
            ]
            means.append(row["success_mean"].iloc[0] if not row.empty else np.nan)
            errs.append(row["success_stderr"].iloc[0] if not row.empty else 0.0)
        ax.errorbar(
            horizons, means, yerr=errs,
            marker="o", linewidth=2, markersize=7, capsize=3,
            color=color, label=label,
        )
    ax.set_xticks(horizons)
    ax.set_xlabel("Planning horizon $H$")
    ax.set_ylabel("Clean success rate $\\uparrow$")
    ax.set_title("(a) Progress-aware CEM: success vs. horizon ($\\alpha{=}0.1$)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    conditions = ["brightness", "background_shift"]
    condition_labels = ["Brightness", "Background shift"]
    methods = [
        ("bc_only", "bc_only", 0.0, 0, "BC only", COLORS["bc"]),
        ("bc_plus_vjepa_cem", "cem_value", 0.1, 1, "VJEPA, $H{=}1$", "#7a8a98"),
        ("bc_plus_de_vjepa_cem", "cem_value", 0.1, 1, "DE-VJEPA, $H{=}1$", "#3b6ea5"),
        ("bc_plus_vjepa_cem", "cem_value", 0.1, 3, "VJEPA, $H{=}3$", "#b5651d"),
        ("bc_plus_de_vjepa_cem", "cem_value", 0.1, 3, "DE-VJEPA, $H{=}3$", "#d59a4d"),
        ("bc_plus_vjepa_cem", "cem_value", 0.1, 5, "VJEPA, $H{=}5$", "#5c8a5c"),
        ("bc_plus_de_vjepa_cem", "cem_value", 0.1, 5, "DE-VJEPA, $H{=}5$", "#88b888"),
    ]
    clean = {}
    for method, score, alpha, h, _, _ in methods:
        row = data[
            (data["method"] == method)
            & (data["planner_score"] == score)
            & (data["planner_alpha"] == alpha)
            & (data["planning_horizon"] == h)
            & (data["visual_condition"] == "clean")
        ]
        clean[(method, h)] = row["success_mean"].iloc[0] if not row.empty else np.nan

    n = len(methods)
    x = np.arange(len(conditions))
    width = 0.8 / n
    for index, (method, score, alpha, h, label, color) in enumerate(methods):
        diffs, errs = [], []
        for cond in conditions:
            row = data[
                (data["method"] == method)
                & (data["planner_score"] == score)
                & (data["planner_alpha"] == alpha)
                & (data["planning_horizon"] == h)
                & (data["visual_condition"] == cond)
            ]
            if row.empty:
                diffs.append(np.nan)
                errs.append(0.0)
            else:
                diffs.append(row["success_mean"].iloc[0] - clean[(method, h)])
                errs.append(row["success_stderr"].iloc[0])
        ax.bar(
            x + (index - (n - 1) / 2) * width,
            diffs, width, yerr=errs, capsize=1.5,
            label=label, color=color, edgecolor="white", linewidth=0.4,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, condition_labels)
    ax.set_ylabel("Success-rate change from clean")
    ax.set_title("(b) Progress-aware CEM: visual-shift gap by horizon",
                 loc="left", fontweight="bold")
    ax.legend(
        frameon=False, fontsize=7.5, ncol=1,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    _save(fig, "cem_mpc_progress512_appendix")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    _style()
    offline_results()
    policy_geometry()
    closed_loop()
    ablation_statistics()
    baseline_heatmap()
    planner_ablation_appendix()
    multistep_appendix()
    cem_progress_appendix()
    planner_ablation_combined()
    multistep_combined()
    cem_progress_combined()


if __name__ == "__main__":
    main()
