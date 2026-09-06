from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.schema.records import Trajectory
from multi_view_world_dataset.utils.serialization import dump_json


def save_rgb(path: Path, array: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Inspection images require the 'inspection' extra with Pillow") from error
    image = np.asarray(array)
    if image.dtype != np.uint8:
        image = np.clip(image * (255.0 if image.max(initial=0) <= 1 else 1.0), 0, 255).astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_trajectory_inspection(
    path: Path,
    traversability: dict[str, Any],
    trajectories: tuple[Trajectory, ...],
    temporal_overlap: dict[str, Any],
) -> None:
    """Draw the exact eroded planning map, paths, waypoints, headings, and overlap QA."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Inspection images require the 'inspection' extra with Pillow") from error
    mask = np.asarray(traversability["traversable"], dtype=bool)
    raster = np.full((*mask.shape, 3), 35, dtype=np.uint8)
    raster[mask] = (225, 225, 225)
    # OmniGibson traversability rows increase with world Y, while image rows
    # increase downwards. Flip the native map to match RGB BEV: +X right, +Y up.
    raster = np.flipud(raster).copy()
    scale = max(1, int(np.ceil(700 / max(mask.shape))))
    base = Image.fromarray(raster).resize(
        (mask.shape[1] * scale, mask.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    )
    banner_height = 103
    canvas = Image.new("RGB", (base.width, base.height + banner_height), (248, 248, 248))
    canvas.paste(base, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    resolution = float(traversability["map_resolution_m"])
    map_height, map_width = mask.shape

    def pixel(point: np.ndarray) -> tuple[float, float]:
        native_column = float(point[0]) / resolution + map_width / 2.0
        native_row = float(point[1]) / resolution + map_height / 2.0
        column = native_column * scale
        row = (map_height - 1.0 - native_row) * scale + banner_height
        return column, row

    colors = ((220, 45, 45), (35, 105, 220), (25, 155, 80))
    for trajectory, color in zip(sorted(trajectories, key=lambda item: item.robot_id), colors, strict=True):
        xy = trajectory.base_to_world[:, :2, 3]
        pixels = [pixel(point) for point in xy]
        draw.line(pixels, fill=color, width=max(2, 2 * scale), joint="curve")
        radius = max(4, 2 * scale)
        start_u, start_v = pixels[0]
        end_u, end_v = pixels[-1]
        draw.ellipse((start_u - radius, start_v - radius, start_u + radius, start_v + radius), fill=color, outline=(0, 0, 0))
        draw.rectangle((end_u - radius, end_v - radius, end_u + radius, end_v + radius), fill=color, outline=(0, 0, 0))
        for waypoint in trajectory.control_waypoints_xy[1:-1]:
            u, v = pixel(waypoint)
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 205, 35), outline=color, width=2)
        for frame_index in np.unique(np.rint(np.linspace(0, trajectory.frames - 1, 6)).astype(int)):
            u, v = pixels[int(frame_index)]
            yaw = float(np.arctan2(
                trajectory.base_to_world[frame_index, 1, 0],
                trajectory.base_to_world[frame_index, 0, 0],
            ))
            arrow_length = max(10, 6 * scale)
            # Positive world Y points upward, i.e. toward decreasing image row.
            tip = (u + arrow_length * np.cos(yaw), v - arrow_length * np.sin(yaw))
            draw.line((u, v, *tip), fill=(0, 0, 0), width=max(1, scale))
            draw.ellipse((tip[0] - 2, tip[1] - 2, tip[0] + 2, tip[1] + 2), fill=(0, 0, 0))
        draw.text((start_u + radius + 2, start_v - radius), f"{trajectory.robot_id} {trajectory.path_family}", fill=color)

    connected_fraction = float(temporal_overlap["connected_fraction"])
    maximum_isolation = temporal_overlap["maximum_consecutive_isolated_keyframes"]
    draw.text((10, 8), "Robot-eroded traversability (not RGB BEV) | +X right, +Y up", fill=(0, 0, 0))
    draw.text((10, 29), "circle=start square=end yellow=intermediate waypoint arrows=planned heading", fill=(0, 0, 0))
    draw.text(
        (10, 50),
        f"temporal overlap connected: {temporal_overlap['connected_keyframe_count']}/{temporal_overlap['keyframe_count']} ({connected_fraction:.3f})",
        fill=(0, 0, 0),
    )
    draw.text((10, 71), f"maximum consecutive isolated keyframes: {maximum_isolation}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_html_summary(output_dir: Path, title: str, findings: dict[str, Any], image_names: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "summary.json", findings)
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><pre>{html.escape(str(value))}</pre></td></tr>"
        for key, value in findings.items()
    )
    images = "".join(f'<figure><img src="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in image_names)
    document = f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><table>{rows}</table>{images}"
    target = output_dir / "index.html"
    target.write_text(document, encoding="utf-8")
    return target
