from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multi_view_world_dataset.diagnostics import (
    run_navigation_sweep,
    run_sampling_diagnostics,
)
from multi_view_world_dataset.dataset_diagnostics import summarize_generated_dataset
from multi_view_world_dataset.generator import generate_dataset
from multi_view_world_dataset.pipeline import inspect_simulator_runtime, run_simulator_probe
from multi_view_world_dataset.pilot_report import generate_pilot_report
from multi_view_world_dataset.production import (
    finalize_dataset,
    launch_scene_shards,
    run_scene_worker,
)
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

    diagnostics = subparsers.add_parser(
        "sampling-diagnostics",
        help="Sample placement/path distributions without dense episode rendering",
    )
    diagnostics.add_argument("--config", required=True)
    diagnostics.add_argument("--scene")
    diagnostics.add_argument("--samples", type=int)
    diagnostics.add_argument(
        "--with-overlap-preflight", action="store_true",
        help="Also render sparse GT-depth keyframes and report temporal overlap topology",
    )
    _machine_arguments(diagnostics, output=True)
    navigation_sweep = subparsers.add_parser(
        "navigation-sweep",
        help="Run final-robot SE(2) feasibility over discovered scenes without RGB rollouts",
    )
    navigation_sweep.add_argument("--config", required=True)
    navigation_sweep.add_argument(
        "--scene", help="Optional single-scene microtest; omit for the full installed sweep"
    )
    _machine_arguments(navigation_sweep, output=True)
    dataset_diagnostics = subparsers.add_parser(
        "dataset-diagnostics",
        help="Aggregate finalized Dataset-v1.1 episodes without the simulator",
    )
    dataset_diagnostics.add_argument("--dataset-root", required=True)
    dataset_diagnostics.add_argument(
        "--output", help="Optional JSON path; defaults inside the dataset root"
    )

    generate = subparsers.add_parser(
        "generate",
        help="Generate or safely resume an accepted smoke/integration dataset",
    )
    generate.add_argument("--config", required=True)
    generate.add_argument("--scene", help="Installed scene ID; defaults to dynamically discovered scenes")
    generate.add_argument(
        "--allow-large",
        action="store_true",
        help="Explicitly allow pilot/default generation (never enabled implicitly)",
    )
    _machine_arguments(generate, output=True)

    worker = subparsers.add_parser(
        "scene-worker", help="Generate exactly one isolated per-scene shard"
    )
    worker.add_argument("--config", required=True)
    worker.add_argument("--scene", required=True)
    worker.add_argument("--allow-large", action="store_true")
    _machine_arguments(worker, output=True)

    launch = subparsers.add_parser(
        "production-launch",
        help="Launch one fresh scene-worker process per scene on a resumable GPU queue",
    )
    launch.add_argument("--config", required=True)
    launch.add_argument("--gpus", required=True, help="Comma-separated physical GPU IDs")
    launch.add_argument("--max-workers", type=int)
    launch.add_argument("--retry-failed", action="store_true")
    launch.add_argument("--allow-large", action="store_true")
    _machine_arguments(launch, output=True)

    finalize = subparsers.add_parser(
        "finalize-dataset", help="Deterministically merge completed shard metadata"
    )
    finalize.add_argument("--dataset-root", required=True)
    pilot_report = subparsers.add_parser(
        "pilot-report", help="Generate production-readiness diagnostics from finalized shards"
    )
    pilot_report.add_argument("--dataset-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_yaml_config(args.config)
        print(json.dumps({"status": "ok", "profile": config["profile"]}, indent=2))
        return 0
    if args.command == "pilot-report":
        output, report = generate_pilot_report(args.dataset_root)
        print(json.dumps(
            {"status": "pass", "output": str(output), **report},
            indent=2, sort_keys=True,
        ))
        return 0
    if args.command == "finalize-dataset":
        output, report = finalize_dataset(args.dataset_root)
        print(json.dumps(
            {"status": "pass", "output": str(output), **report},
            indent=2, sort_keys=True,
        ))
        return 0
    if args.command == "dataset-diagnostics":
        output, report = summarize_generated_dataset(
            args.dataset_root, output_path=args.output
        )
        print(json.dumps(
            {"status": "pass", "output": str(output), **report},
            indent=2, sort_keys=True,
        ))
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
    if args.command == "sampling-diagnostics":
        output, report = run_sampling_diagnostics(
            runtime, config, scene_id=args.scene, samples=args.samples,
            include_overlap_preflight=bool(args.with_overlap_preflight),
        )
        print(json.dumps({"status": "pass", "output": str(output), **report}, indent=2, sort_keys=True))
        return 0
    if args.command == "navigation-sweep":
        output, report = run_navigation_sweep(
            runtime, config, scene_id=args.scene
        )
        print(json.dumps({
            "status": "pass", "output": str(output),
            "scene_count": report["scene_count"],
            "classification_counts": report["classification_counts"],
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "scene-worker":
        output, result = run_scene_worker(
            runtime, config, args.scene, allow_large=bool(args.allow_large)
        )
        print(json.dumps(
            {"status": "pass", "output": str(output), **result},
            indent=2, sort_keys=True,
        ))
        return 0
    if args.command == "production-launch":
        gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
        output, result = launch_scene_shards(
            runtime,
            config,
            args.config,
            gpus=gpus,
            max_workers=args.max_workers or len(gpus),
            allow_large=bool(args.allow_large),
            retry_failed=bool(args.retry_failed),
        )
        print(json.dumps(
            {"output": str(output), **result}, indent=2, sort_keys=True
        ))
        return 0 if result["status"] == "pass" else 2
    if args.command == "generate":
        output, result = generate_dataset(
            runtime,
            config,
            scene_id=args.scene,
            allow_large=bool(args.allow_large),
        )
        print(
            json.dumps(
                {"status": result["status"], "output": str(output), **result},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "pass" else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
