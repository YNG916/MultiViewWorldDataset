"""Convert a generated per-robot NPZ rollout to a video."""

from __future__ import annotations

import argparse
from pathlib import Path

from npz2video_bev import load_rgb_frames, write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, help="Path to a robot_XX.npz rollout")
    parser.add_argument("-o", "--output", type=Path, help="Output video (default: NPZ path with .avi)")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", default="MJPG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.npz_path.with_suffix(".avi")
    frames = load_rgb_frames(args.npz_path, validate_bev=False)
    write_video(frames, output_path, fps=args.fps, codec=args.codec)
    print(f"Saved: {output_path.resolve()}")
    print(f"Frames: {frames.shape[0]}, resolution: {frames.shape[2]}x{frames.shape[1]}")


if __name__ == "__main__":
    main()
