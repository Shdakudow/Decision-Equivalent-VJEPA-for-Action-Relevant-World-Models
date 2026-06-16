from pathlib import Path

import pandas as pd


def main() -> None:
    results = Path("results")
    metaworld = pd.read_csv(results / "cross_method_metaworld_summary.csv")
    metaworld.insert(0, "dataset", "metaworld-5")
    pusht = pd.read_csv(results / "cross_method_pusht_summary.csv")
    pusht.insert(0, "dataset", "lerobot-pusht")
    output = results / "cross_dataset_comparison_summary.csv"
    pd.concat((metaworld, pusht), ignore_index=True).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
