from devjepa.config import load_config, parser
from devjepa.progress import train_progress_heads


def main() -> None:
    args = parser("Train a trajectory-progress head").parse_args()
    validation, table = train_progress_heads(load_config(args.config, args.set))
    print(validation)
    print(table)


if __name__ == "__main__":
    main()
