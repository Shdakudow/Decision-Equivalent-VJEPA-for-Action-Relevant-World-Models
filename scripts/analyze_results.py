from __future__ import annotations

from devjepa.analysis import analyze
from devjepa.config import load_config, parser


def main() -> None:
    args = parser("Analyze DE-VJEPA ablation results").parse_args()
    config = load_config(args.config, args.set)
    print(analyze(config.get("analysis", config)))


if __name__ == "__main__":
    main()
