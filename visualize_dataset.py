"""Batch-convert every image modality in MVWD episodes to videos and/or PNGs.

The input may be one episode_XXX directory or a complete generated dataset
root. Temporal world / robot observations become videos and contact sheets;
static environment BEVs become PNGs. A manifest and an HTML index are written
for every episode.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_MODALITIES = (
    "rgb",
    "depth_linear",
    "height",
    "semantic",
    "instance",
    "instance_id",
    "occupancy",
    "normal",
)
CATEGORICAL_MODALITIES = {"semantic", "instance", "instance_id"}
SCALAR_MODALITIES = {"depth_linear", "height"}


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    temporal: bool
    validate_bev: bool = False


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def discover_episodes(input_path: Path) -> list[Path]:
    """Return finalized episode directories beneath input_path."""
    root = input_path.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_dir() and (root / "bev" / "world_before.npz").is_file():
        return [root]
    episodes = sorted({path.parent.parent for path in root.rglob("bev/world_before.npz")})
    if not episodes:
        raise ValueError(
            f"No episode found under {root}; expected an episode_XXX directory "
            "or a dataset root containing episodes/*/*/episode_XXX"
        )
    return episodes


def _dataset_root(episode: Path) -> Path | None:
    for ancestor in episode.parents:
        if (ancestor / "episodes").is_dir() and (ancestor / "configurations").is_dir():
            return ancestor
    return None


def discover_sources(episode: Path) -> list[Source]:
    """Discover all canonical image-bearing NPZ groups for one episode."""
    sources: list[Source] = []
    dataset_root = _dataset_root(episode)
    scene_id = episode.parent.parent.name
    configuration_id = episode.parent.name
    if dataset_root is not None:
        environment_base = (
            dataset_root / "configurations" / scene_id / configuration_id / "bev" / "environment_base.npz"
        )
        if environment_base.is_file():
            sources.append(Source("environment_base", environment_base, temporal=False))

    environment_after = episode / "bev" / "environment_after.npz"
    if environment_after.is_file():
        sources.append(Source("environment_after", environment_after, temporal=False))

    for state in ("before", "after"):
        world = episode / "bev" / f"world_{state}.npz"
        if world.is_file():
            sources.append(Source(f"world_{state}", world, temporal=True, validate_bev=True))
        robot_root = episode / "robot_views" / state
        for robot in sorted(robot_root.glob("robot_*.npz")):
            sources.append(Source(f"robot_{state}_{robot.stem}", robot, temporal=True))
    return sources


def _rgb(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)[..., :3]
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image = image.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size and float(finite.max()) <= 1.0:
        image = image * 255.0
    return np.nan_to_num(np.clip(image, 0, 255), nan=0.0).astype(np.uint8)


def _sample_finite(array: np.ndarray, maximum: int = 1_000_000) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    if flat.size > maximum:
        flat = flat[:: math.ceil(flat.size / maximum)]
    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]
    return flat


def scalar_bounds(array: np.ndarray) -> tuple[float, float]:
    sample = _sample_finite(array)
    if not sample.size:
        return 0.0, 1.0
    low, high = np.percentile(sample.astype(np.float64, copy=False), (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _categorical(array: np.ndarray) -> np.ndarray:
    """Map integer labels to stable colors without assuming contiguous IDs."""
    labels = np.asarray(array).astype(np.uint64, copy=False)
    values = labels.copy()
    values ^= values >> np.uint64(16)
    values *= np.uint64(0x7FEB352D)
    values ^= values >> np.uint64(15)
    values *= np.uint64(0x846CA68B)
    values ^= values >> np.uint64(16)
    colors = np.stack(
        (
            ((values >> np.uint64(0)) & np.uint64(255)).astype(np.uint8),
            ((values >> np.uint64(8)) & np.uint64(255)).astype(np.uint8),
            ((values >> np.uint64(16)) & np.uint64(255)).astype(np.uint8),
        ),
        axis=-1,
    )
    colors = np.maximum(colors, np.uint8(35))
    colors[labels == 0] = 0
    return colors


def _normal(array: np.ndarray) -> np.ndarray:
    xyz = np.asarray(array)[..., :3].astype(np.float32, copy=False)
    finite = xyz[np.isfinite(xyz)]
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        mapped = xyz * 255.0
    else:
        mapped = (xyz + 1.0) * 127.5
    return np.nan_to_num(np.clip(mapped, 0, 255), nan=0.0).astype(np.uint8)


def visualize_frame(
    modality: str,
    array: np.ndarray,
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert one raw HxW[xC] modality frame to RGB uint8."""
    frame = np.asarray(array)
    if modality == "rgb":
        return _rgb(frame)
    if modality == "normal":
        return _normal(frame)
    if modality in CATEGORICAL_MODALITIES:
        return _categorical(frame)
    if modality == "occupancy":
        mask = np.asarray(frame).squeeze() > 0
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=-1)
    if modality in SCALAR_MODALITIES:
        low, high = bounds if bounds is not None else scalar_bounds(frame)
        values = np.asarray(frame).squeeze().astype(np.float32, copy=False)
        valid = np.isfinite(values)
        normalized = np.zeros(values.shape, dtype=np.uint8)
        normalized[valid] = np.clip((values[valid] - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        if modality == "depth_linear":
            normalized[valid] = 255 - normalized[valid]
        colored = cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        colored[~valid] = 0
        return colored
    raise ValueError(f"Unsupported image modality: {modality}")


def _modality_name(key: str) -> str | None:
    leaf = key.rsplit("/", 1)[-1]
    return leaf if leaf in IMAGE_MODALITIES else None


def _validate_world_bev(data: Any, maximum_fraction: float) -> None:
    if "occupancy" not in data:
        raise ValueError("world BEV has no occupancy array")
    occupancy = np.asarray(data["occupancy"])
    if occupancy.ndim < 3:
        raise ValueError(f"world BEV occupancy must be TxHxW, got {occupancy.shape}")
    fractions = np.mean(occupancy > 0, axis=tuple(range(1, occupancy.ndim)))
    bad = np.flatnonzero(fractions >= maximum_fraction)
    if bad.size:
        raise ValueError(
            "suspicious world BEV: nearly full-frame occupancy at frames "
            f"{bad[:10].tolist()} (maximum fraction {float(fractions.max()):.4f}); "
            "this matches the stale perspective-camera capture bug"
        )


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def _write_opencv_video(
    path: Path,
    array: np.ndarray,
    modality: str,
    bounds: tuple[float, float] | None,
    fps: float,
    codec: str,
) -> None:
    first = visualize_frame(modality, array[0], bounds)
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for index in range(array.shape[0]):
            frame = visualize_frame(modality, array[index], bounds)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "H.264 MP4 output requires ffmpeg or imageio-ffmpeg; use "
            "--video-extension .avi for the OpenCV MJPG fallback"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def _write_h264_video(
    path: Path,
    array: np.ndarray,
    modality: str,
    bounds: tuple[float, float] | None,
    fps: float,
) -> None:
    first = visualize_frame(modality, array[0], bounds)
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in range(array.shape[0]):
            frame = np.ascontiguousarray(visualize_frame(modality, array[index], bounds))
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code:
        details = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed for {path}: {details}")


def _write_video(
    path: Path,
    array: np.ndarray,
    modality: str,
    bounds: tuple[float, float] | None,
    fps: float,
    codec: str,
) -> None:
    if path.suffix.lower() == ".mp4":
        _write_h264_video(path, array, modality, bounds, fps)
    else:
        _write_opencv_video(path, array, modality, bounds, fps, codec)


def _contact_sheet(
    array: np.ndarray,
    modality: str,
    bounds: tuple[float, float] | None,
    count: int,
    thumbnail_width: int,
) -> np.ndarray:
    indices = np.unique(np.rint(np.linspace(0, array.shape[0] - 1, min(count, array.shape[0]))).astype(int))
    tiles = []
    for index in indices:
        frame = visualize_frame(modality, array[int(index)], bounds)
        target_height = max(1, round(frame.shape[0] * thumbnail_width / frame.shape[1]))
        tile = cv2.resize(frame, (thumbnail_width, target_height), interpolation=cv2.INTER_AREA)
        banner = np.full((28, thumbnail_width, 3), 245, dtype=np.uint8)
        label = f"t={int(index):03d}"
        if bounds is not None:
            label += f"  range=[{bounds[0]:.3g}, {bounds[1]:.3g}]"
        cv2.putText(banner, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(np.concatenate((banner, tile), axis=0))
    return np.concatenate(tiles, axis=1)


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    sample = _sample_finite(array)
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if sample.size:
        result.update(min=float(sample.min()), max=float(sample.max()))
    if np.issubdtype(array.dtype, np.floating):
        result["finite_fraction"] = float(np.mean(np.isfinite(array)))
    return result


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def convert_source(
    source: Source,
    output_root: Path,
    *,
    output_kind: str,
    fps: float,
    codec: str,
    video_extension: str,
    contact_frames: int,
    thumbnail_width: int,
    frame_stride: int,
    maximum_occupancy_fraction: float,
    allow_suspicious_bev: bool,
) -> dict[str, Any]:
    source_output = output_root / source.name
    report: dict[str, Any] = {
        "name": source.name,
        "input": str(source.path),
        "temporal": source.temporal,
        "items": [],
    }
    with np.load(source.path, allow_pickle=False) as data:
        if source.validate_bev and not allow_suspicious_bev:
            _validate_world_bev(data, maximum_occupancy_fraction)
        keys = [key for key in data.files if _modality_name(key) is not None]
        for key in keys:
            modality = _modality_name(key)
            assert modality is not None
            array = np.asarray(data[key])
            key_name = _safe_name(key)
            item = {"key": key, "modality": modality, **_array_stats(array), "outputs": []}
            bounds = scalar_bounds(array) if modality in SCALAR_MODALITIES else None
            if bounds is not None:
                item["visualization_range"] = list(bounds)

            if source.temporal:
                if array.ndim not in (3, 4) or array.shape[0] == 0:
                    raise ValueError(f"Temporal modality {source.path}:{key} has invalid shape {array.shape}")
                if output_kind in {"video", "both"}:
                    video = source_output / f"{key_name}{video_extension}"
                    _write_video(video, array, modality, bounds, fps, codec)
                    item["outputs"].append(_relative(video, output_root))
                sheet = source_output / f"{key_name}_contact.png"
                _write_png(sheet, _contact_sheet(array, modality, bounds, contact_frames, thumbnail_width))
                item["outputs"].append(_relative(sheet, output_root))
                if output_kind in {"frames", "both"}:
                    frames_root = source_output / f"{key_name}_frames"
                    for index in range(0, array.shape[0], frame_stride):
                        target = frames_root / f"frame_{index:06d}.png"
                        _write_png(target, visualize_frame(modality, array[index], bounds))
                    item["outputs"].append(_relative(frames_root, output_root) + "/")
            else:
                if array.ndim not in (2, 3):
                    raise ValueError(f"Static modality {source.path}:{key} has invalid shape {array.shape}")
                image_path = source_output / f"{key_name}.png"
                _write_png(image_path, visualize_frame(modality, array, bounds))
                item["outputs"].append(_relative(image_path, output_root))
            report["items"].append(item)
    return report


def _write_html(output_root: Path, episode: Path, reports: list[dict[str, Any]], errors: list[str]) -> Path:
    sections = []
    for report in reports:
        cards = []
        for item in report["items"]:
            media = []
            for relative in item["outputs"]:
                escaped = html.escape(relative, quote=True)
                if relative.endswith(".png"):
                    media.append(f'<a href="{escaped}"><img loading="lazy" src="{escaped}"></a>')
                elif relative.endswith((".avi", ".mp4")):
                    media.append(
                        f'<video controls preload="metadata" src="{escaped}"></video>'
                        f'<a href="{escaped}">download video</a>'
                    )
                else:
                    media.append(f'<a href="{escaped}">{escaped}</a>')
            details = html.escape(
                json.dumps({key: value for key, value in item.items() if key != "outputs"}, ensure_ascii=False)
            )
            cards.append(
                f"<article><h3>{html.escape(item['key'])}</h3>{''.join(media)}<pre>{details}</pre></article>"
            )
        sections.append(
            f"<section><h2>{html.escape(report['name'])}</h2><p>{html.escape(report['input'])}</p>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )
    error_html = "".join(f"<li>{html.escape(error)}</li>" for error in errors)
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>MVWD all-modality visualization</title><style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#17191c}}h1,h2{{margin-top:28px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}
article{{background:white;border:1px solid #d9dde3;border-radius:8px;padding:12px;overflow:auto}}
img,video{{display:block;max-width:100%;max-height:520px;margin:8px 0}}pre{{white-space:pre-wrap;font-size:11px}}
</style></head><body><h1>MVWD all-modality visualization</h1>
<p>Episode: {html.escape(str(episode))}</p><p><a href="manifest.json">manifest.json</a></p>
<ul>{error_html}</ul>{''.join(sections)}</body></html>"""
    target = output_root / "index.html"
    target.write_text(document, encoding="utf-8")
    return target


def _write_markdown(output_root: Path, episode: Path, reports: list[dict[str, Any]], errors: list[str]) -> Path:
    lines = ["# MVWD all-modality visualization", "", f"Episode: {episode}", ""]
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    for report in reports:
        lines.extend([f"## {report['name']}", "", f"Input: {report['input']}", ""])
        for item in report["items"]:
            lines.extend([f"### {item['key']}", ""])
            for relative in item["outputs"]:
                if relative.endswith(".png"):
                    lines.extend([f"![{item['key']}]({relative})", ""])
                elif relative.endswith((".avi", ".mp4")):
                    lines.extend([f"[Open {relative}]({relative})", ""])
                else:
                    lines.extend([f"[Open frames]({relative})", ""])
            details = {key: value for key, value in item.items() if key != "outputs"}
            lines.extend(["    " + json.dumps(details, ensure_ascii=False), ""])
    target = output_root / "index.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def convert_episode(episode: Path, output_root: Path, args: argparse.Namespace) -> tuple[Path, list[str]]:
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in discover_sources(episode):
        print(f"  [{source.name}] {source.path}", flush=True)
        try:
            reports.append(
                convert_source(
                    source,
                    output_root,
                    output_kind=args.output_kind,
                    fps=args.fps,
                    codec=args.codec,
                    video_extension=args.video_extension,
                    contact_frames=args.contact_frames,
                    thumbnail_width=args.thumbnail_width,
                    frame_stride=args.frame_stride,
                    maximum_occupancy_fraction=args.maximum_occupancy_fraction,
                    allow_suspicious_bev=args.allow_suspicious_bev,
                )
            )
        except Exception as error:  # Continue the batch and make every failed source explicit.
            message = f"{source.name}: {type(error).__name__}: {error}"
            errors.append(message)
            print(f"    ERROR: {message}", file=sys.stderr, flush=True)
    manifest = {"episode": str(episode), "sources": reports, "errors": errors}
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_html(output_root, episode, reports, errors)
    return _write_markdown(output_root, episode, reports, errors), errors


def _default_episode_output(episode: Path) -> Path:
    return episode / "inspection" / "all_modalities"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="An episode_XXX directory or complete generated dataset root")
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        help="Output for a single episode; batch input always writes inside each episode",
    )
    parser.add_argument("--output-kind", choices=("video", "frames", "both"), default="video")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", default="MJPG", help="FourCC used only for AVI output (default: MJPG)")
    parser.add_argument(
        "--video-extension",
        choices=(".mp4", ".avi"),
        default=".mp4",
        help="MP4 uses browser-compatible H.264; AVI uses the OpenCV codec (default: .mp4)",
    )
    parser.add_argument("--contact-frames", type=int, default=6)
    parser.add_argument("--thumbnail-width", type=int, default=320)
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride when --output-kind includes frames")
    parser.add_argument("--max-episodes", type=int, help="Only process the first N discovered episodes")
    parser.add_argument("--maximum-occupancy-fraction", type=float, default=0.98)
    parser.add_argument(
        "--allow-suspicious-bev",
        action="store_true",
        help="Convert world BEVs even when occupancy QA matches the known perspective-camera corruption",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.contact_frames <= 0 or args.thumbnail_width <= 0 or args.frame_stride <= 0:
        parser.error("fps, contact-frames, thumbnail-width, and frame-stride must be positive")
    if len(args.codec) != 4:
        parser.error("codec must contain exactly four characters")
    if args.output_root is not None and not (args.input / "bev" / "world_before.npz").is_file():
        parser.error("--output-root is only valid when input is one episode directory")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = discover_episodes(args.input)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    print(f"Discovered {len(episodes)} episode(s)")
    failed = 0
    for index, episode in enumerate(episodes, start=1):
        output = args.output_root.expanduser().resolve() if args.output_root else _default_episode_output(episode)
        print(f"[{index}/{len(episodes)}] {episode}")
        index_path, errors = convert_episode(episode, output, args)
        failed += len(errors)
        print(f"  Index: {index_path}")
    if failed:
        print(f"Finished with {failed} failed source(s); inspect each manifest.json", file=sys.stderr)
        return 1
    print("All modalities converted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
