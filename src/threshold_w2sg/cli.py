"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .runner import (
    recompute_state_evolution,
    regenerate_artifacts,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="threshold-w2sg",
        description="Reproduce threshold-renormalization experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a configured experiment")
    run.add_argument("--config", required=True, help="path to YAML configuration")
    render = subparsers.add_parser(
        "render", help="regenerate summaries and artifacts from saved raw results"
    )
    render.add_argument("--config", required=True, help="path to YAML configuration")
    recompute = subparsers.add_parser(
        "recompute-se",
        help="recompute state evolution and derived artifacts from saved fits",
    )
    recompute.add_argument(
        "--config", required=True, help="path to YAML configuration"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        metadata = run_experiment(load_config(args.config))
        print(json.dumps(metadata, indent=2, sort_keys=True))
    elif args.command == "render":
        metadata = regenerate_artifacts(load_config(args.config))
        print(json.dumps(metadata, indent=2, sort_keys=True))
    elif args.command == "recompute-se":
        metadata = recompute_state_evolution(load_config(args.config))
        print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
