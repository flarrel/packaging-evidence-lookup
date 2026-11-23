"""Commands for local packaging research."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(".work"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("medical", help="prepare the PneumoniaMNIST image benchmark")
    args = parser.parse_args()
    if args.command == "medical":
        from .medical import run

        print(json.dumps(run(args.work), indent=2))


if __name__ == "__main__":
    main()
