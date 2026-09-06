"""Convert a generated world-BEV NPZ rollout to a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_rgb_frames(
    npz_path: Path,
    *,
    validate_bev: bool = True,
    maximum_occupancy_fraction: float = 0.98,
) -> np.ndarray:
    """Load RGB(A) frames and reject the known stale perspective-view failure."""
    with np.load(npz_path) as data:
        if "rgb" not in data:
            raise ValueError(f"{npz_path} does not contain an 'rgb' array")
        rgb = np.asarray(data["rgb"])
        if validate_bev:
            if "occupancy" not in data:
                raise ValueError(
                    f"{npz_path} has no 'occupancy' array; pass --skip-bev-validation "
                    "only if this is intentionally not a dataset world-BEV file"
                )
            occupancy = np.asarray(data["occupancy"])
            if occupancy.ndim < 3 or occupancy.shape[0] != rgb.shape[0]:
                raise ValueError(
                    "occupancy must have the same frame count as rgb and at least "
                    f"three dimensions, got rgb={rgb.shape}, occupancy={occupancy.shape}"
                )
            fractions = np.mean(occupancy > 0, axis=tuple(range(1, occupancy.ndim)))
            saturated = np.flatnonzero(fractions >= maximum_occupancy_fraction)
            if saturated.size:
                preview = ", ".join(str(int(index)) for index in saturated[:10])
                raise ValueError(
                    "BEV validation failed: occupancy is nearly full-frame in "
                    f"frame(s) {preview} (maximum={float(fractions.max()):.4f}). "
                    "This matches the stale robot-perspective capture bug; regenerate "
                    "the rollout instead of converting it. Use --skip-bev-validation "
                    "only for deliberate non-BEV input."
                )

    if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4) or rgb.shape[0] == 0:
        raise ValueError(f"rgb must have shape (T, H, W, 3|4), got {rgb.shape}")
    rgb = rgb[..., :3]
    if not np.all(np.isfinite(rgb)):
        raise ValueError("rgb contains non-finite values")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        if rgb.size and float(rgb.max()) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def write_video(
    frames: np.ndarray,
    output_path: Path,
    *,
    fps: float = 10.0,
    codec: str = "MJPG",
) -> None:
    if len(codec) != 4:
        raise ValueError("codec must be a four-character OpenCV codec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, height, width, _ = frames.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, help="Path to bev/world_before.npz or world_after.npz")
    parser.add_argument("-o", "--output", type=Path, help="Output video (default: NPZ path with .avi)")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", default="MJPG")
    parser.add_argument("--maximum-occupancy-fraction", type=float, default=0.98)
    parser.add_argument(
        "--skip-bev-validation",
        action="store_true",
        help="Allow input without valid BEV occupancy QA (unsafe for generated world rollouts)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.npz_path.with_suffix(".avi")
    frames = load_rgb_frames(
        args.npz_path,
        validate_bev=not args.skip_bev_validation,
        maximum_occupancy_fraction=args.maximum_occupancy_fraction,
    )
    write_video(frames, output_path, fps=args.fps, codec=args.codec)
    print(f"Saved: {output_path.resolve()}")
    print(f"Frames: {frames.shape[0]}, resolution: {frames.shape[2]}x{frames.shape[1]}")


if __name__ == "__main__":
    main()
