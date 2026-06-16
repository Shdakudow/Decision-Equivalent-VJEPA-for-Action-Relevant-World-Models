from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from devjepa.config import load_config, parser
from devjepa.train_predictor import train_predictor


def run(config: dict) -> Path:
    section = config["multistep_training"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for seed in [int(value) for value in section["seeds"]]:
        for variant in section["variants"]:
            run_config = deepcopy(config)
            run_config["experiment"]["seed"] = seed
            run_config["experiment"]["method"] = variant["method"]
            run_config["experiment"]["tag"] = (
                f"{variant['name']}_lambda{str(variant['lambda_de']).replace('.', 'p')}_"
                f"{timestamp}"
            )
            run_config["predictor"]["horizon"] = int(variant["horizon"])
            run_config["loss"]["decision_weight"] = float(variant["lambda_de"])
            checkpoint = train_predictor(run_config)
            rows.append(
                {
                    "run_id": run_config["experiment"]["tag"],
                    "name": variant["name"],
                    "method": variant["method"],
                    "seed": seed,
                    "horizon": int(variant["horizon"]),
                    "lambda_de": float(variant["lambda_de"]),
                    "checkpoint": str(checkpoint),
                }
            )
    output = Path(section["manifest_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    args = parser("Train the multi-step DE-VJEPA grid").parse_args()
    print(run(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
