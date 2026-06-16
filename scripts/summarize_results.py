from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    output = Path("outputs/pilot")
    rows = []
    for path in sorted(output.glob("*/evaluation.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        match = re.search(r"_seed(\d+)$", path.parent.name)
        rows.append(
            {
                "run": path.parent.name,
                "seed": int(match.group(1)) if match else -1,
                "method": result["method"],
                "policy_quality": result["policy_quality"],
                "horizon": result["horizon"],
                "latent_error": result["latent_error"],
                "decision_kl": result["decision_kl"],
                "pearson_latent_vs_decision": result["pearson_latent_vs_decision"],
                "spearman_latent_vs_decision": result["spearman_latent_vs_decision"],
                "success_label_rate": result["success_label_rate"],
            }
        )

    details = pd.DataFrame(rows)
    details.to_csv(output / "results_by_seed.csv", index=False)
    metrics = [
        "latent_error",
        "decision_kl",
        "pearson_latent_vs_decision",
        "spearman_latent_vs_decision",
        "success_label_rate",
    ]
    grouped = (
        details.groupby(["method", "policy_quality", "horizon"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    grouped.to_csv(output / "results_mean_std.csv", index=False)
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
