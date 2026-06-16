from __future__ import annotations

from devjepa.ablation import run_experiment
from devjepa.config import load_config, parser


def main() -> None:
    args = parser("Run a config-driven DE-VJEPA ablation").parse_args()
    print(run_experiment(load_config(args.config, args.set)))


if __name__ == "__main__":
    main()
