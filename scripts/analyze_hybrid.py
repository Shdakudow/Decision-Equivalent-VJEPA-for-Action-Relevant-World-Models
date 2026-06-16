from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


def main() -> None:
    baseline = pd.read_csv("outputs/prior_pilot/independent_results.csv")
    baseline = baseline[baseline.method == "vjepa"]
    hybrid = pd.read_csv("outputs/hybrid_sweep/independent_results_l001.csv")
    index = ["source_seed", "evaluator_seed", "shift"]
    baseline = baseline.set_index(index)
    hybrid = hybrid.set_index(index)
    rows = []
    for shift in sorted(hybrid.reset_index()["shift"].unique()):
        left = baseline.xs(shift, level="shift")
        right = hybrid.xs(shift, level="shift")
        for metric in (
            "latent_mse_raw",
            "independent_decision_kl",
            "independent_action_mse",
        ):
            base = left[metric]
            value = right[metric]
            test = ttest_rel(value, base)
            rows.append(
                {
                    "shift": shift,
                    "metric": metric,
                    "pairs": len(base),
                    "vjepa_mean": base.mean(),
                    "hybrid_mean": value.mean(),
                    "relative_change_percent": 100.0
                    * (value.mean() - base.mean())
                    / base.mean(),
                    "paired_t": test.statistic,
                    "paired_p": test.pvalue,
                }
            )
    result = pd.DataFrame(rows)
    output = Path("outputs/hybrid_sweep/hybrid_vs_vjepa_paired.csv")
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
