from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from multi_view_world_dataset.errors import ConfigurationError, SimulatorUnavailableError


@dataclass(frozen=True)
class RuntimePaths:
    behavior_root: Path
    output_root: Path | None
    cache_root: Path | None

    def require_output(self) -> Path:
        if self.output_root is None:
            raise ConfigurationError("Output root is required: pass --output-root or set DATASET_DEV_OUTPUT")
        self.output_root.mkdir(parents=True, exist_ok=True)
        return self.output_root


def _resolve_path(explicit: str | Path | None, env_name: str | None, environ: Mapping[str, str]) -> Path | None:
    value = explicit if explicit is not None else (environ.get(env_name) if env_name else None)
    return Path(value).expanduser().resolve() if value else None


def resolve_runtime_paths(
    config: Mapping[str, Any],
    *,
    behavior_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimePaths:
    env = os.environ if environ is None else environ
    machine = config.get("machine", {})
    resolved_behavior = _resolve_path(behavior_root, machine.get("behavior_root_env"), env)
    if resolved_behavior is None:
        raise SimulatorUnavailableError("BEHAVIOR root is required: pass --behavior-root or set BEHAVIOR_ROOT")
    if not (resolved_behavior / "OmniGibson" / "omnigibson").is_dir():
        raise SimulatorUnavailableError(f"Not a BEHAVIOR-1K root with OmniGibson: {resolved_behavior}")
    return RuntimePaths(
        behavior_root=resolved_behavior,
        output_root=_resolve_path(output_root, machine.get("output_root_env"), env),
        cache_root=_resolve_path(cache_root, machine.get("cache_root_env"), env),
    )


def installed_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("omnigibson", "isaacsim", "bddl"):
        try:
            result[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def generator_git_commit(repository_root: Path) -> str | None:
    """Read HEAD without executing git; supports a normal non-packed branch ref."""
    head = repository_root / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref: "):
        return value or None
    ref = repository_root / ".git" / value[5:]
    return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None

