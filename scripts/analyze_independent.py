from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


def main() -> None:
    output = Path("outputs/prior_pilot")
    frame = pd.read_csv(output / "independent_results.csv")
    index = ["source_seed", "evaluator_seed", "shift"]
    baseline = frame[frame.method == "vjepa"].set_index(index)
    rows = []
    for method in ("de_vjepa", "de_vjepa_band"):
        candidate = frame[frame.method == method].set_index(index)
        for shift in sorted(frame["shift"].unique()):
            left = baseline.xs(shift, level="shift")
            right = candidate.xs(shift, level="shift")
            for metric in ("independent_decision_kl", "independent_action_mse"):
                base = left[metric]
                value = right[metric]
                test = ttest_rel(value, base)
                rows.append(
                    {
                        "method": method,
                        "shift": shift,
                        "metric": metric,
                        "pairs": len(base),
                        "vjepa_mean": base.mean(),
                        "method_mean": value.mean(),
                        "relative_change_percent": 100.0
                        * (value.mean() - base.mean())
                        / base.mean(),
                        "paired_t": test.statistic,
                        "paired_p": test.pvalue,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output / "independent_paired_tests.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
