from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multi_view_world_dataset.pipeline import inspect_simulator_runtime, run_simulator_probe
from multi_view_world_dataset.utils.config import load_yaml_config
from multi_view_world_dataset.utils.runtime import resolve_runtime_paths
from multi_view_world_dataset.utils.serialization import to_jsonable


def _machine_arguments(parser: argparse.ArgumentParser, *, output: bool = False) -> None:
    parser.add_argument("--behavior-root", help="External BEHAVIOR-1K repository root")
    if output:
        parser.add_argument("--output-root", help="Generated data root (never inferred from source paths)")
    parser.add_argument("--cache-root", help="Optional generated cache root")


def _load(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    config = load_yaml_config(args.config)
    runtime = resolve_runtime_paths(
        config,
        behavior_root=getattr(args, "behavior_root", None),
        output_root=getattr(args, "output_root", None),
        cache_root=getattr(args, "cache_root", None),
    )
    return config, runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvwd", description="MultiViewWorldDataset generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate frozen dataset semantics")
    validate.add_argument("--config", required=True)

    inspect = subparsers.add_parser("inspect-runtime", help="Inspect installed simulator APIs, scenes, and assets")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--skip-asset-stat", action="store_true", help="Resolve but do not stat the Nova Carter URI")
    _machine_arguments(inspect)

    smoke = subparsers.add_parser("simulator-smoke", help="Run headless scene/snapshot/BEV/3-robot/overlap probe")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--scene", help="Installed scene ID; defaults to first dynamically discovered scene")
    _machine_arguments(smoke, output=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_yaml_config(args.config)
        print(json.dumps({"status": "ok", "profile": config["profile"]}, indent=2))
        return 0
    config, runtime = _load(args)
    if args.command == "inspect-runtime":
        report = inspect_simulator_runtime(runtime, config, verify_assets=not args.skip_asset_stat)
        print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
        return 0
    if args.command == "simulator-smoke":
        output, findings = run_simulator_probe(runtime, config, scene_id=args.scene)
        print(json.dumps({"status": findings["status"], "output": str(output)}, indent=2))
        return 0 if findings["status"] == "pass" else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
