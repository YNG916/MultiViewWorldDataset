from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from multi_view_world_dataset.assets import ROBOT_APPEARANCE_VARIANTS

def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _floor_array(bev: Mapping[str, Any], floor_id: str, leaf: str) -> np.ndarray:
    preferred = f"{floor_id}/{leaf}"
    if preferred in bev:
        return np.asarray(bev[preferred])
    matches = [key for key in bev if str(key).rsplit("/", 1)[-1] == leaf]
    if len(matches) != 1:
        raise KeyError(f"Expected one BEV array named {leaf!r}, found {matches}")
    return np.asarray(bev[matches[0]])


def save_environment_room_inspection(
    path: Path,
    environment_bev: Mapping[str, Any],
    objects: Sequence[Any],
    event: Any,
    *,
    floor_id: str,
) -> None:
    """Annotate B_env with room anchors and the intervention target poses."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            "Inspection images require the 'inspection' extra with Pillow"
        ) from error
    rgb = np.asarray(_floor_array(environment_bev, floor_id, "rgb"))[..., :3]
    image = Image.fromarray(rgb.astype(np.uint8, copy=False)).convert("RGB")
    banner_height = 44
    canvas = Image.new(
        "RGB", (image.width, image.height + banner_height), (248, 248, 248)
    )
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    world_to_pixel = _floor_array(
        environment_bev, floor_id, "calibration_world_to_pixel"
    )

    def pixel(xy: Sequence[float]) -> tuple[float, float]:
        homogeneous = np.asarray([float(xy[0]), float(xy[1]), 0.0, 1.0])
        projected = np.asarray(world_to_pixel, dtype=np.float64) @ homogeneous
        return float(projected[0]), float(projected[1] + banner_height)

    room_centers: dict[str, list[np.ndarray]] = {}
    for obj in objects:
        if _record_value(obj, "floor_id") != floor_id:
            continue
        room_id = str(_record_value(obj, "room_id") or "unknown")
        transform = np.asarray(
            _record_value(obj, "object_to_world"), dtype=np.float64
        )
        if transform.shape == (4, 4):
            room_centers.setdefault(room_id, []).append(transform[:2, 3])
    for room_id, centers in sorted(room_centers.items()):
        u, v = pixel(np.median(np.stack(centers), axis=0))
        left, top, right, bottom = draw.textbbox((u, v), room_id, anchor="mm")
        draw.rounded_rectangle(
            (left - 3, top - 2, right + 3, bottom + 2),
            radius=3,
            fill=(255, 255, 255),
            outline=(40, 40, 40),
        )
        draw.text((u, v), room_id, fill=(20, 20, 20), anchor="mm")

    target_states = (
        (
            "target before",
            _record_value(event, "before_object_state"),
            (225, 45, 45),
        ),
        (
            "target after",
            _record_value(event, "after_object_state"),
            (40, 110, 225),
        ),
    )
    for label, state, color in target_states:
        transform = np.asarray(
            _record_value(state, "object_to_world"), dtype=np.float64
        )
        if transform.shape != (4, 4):
            continue
        u, v = pixel(transform[:2, 3])
        draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline=color, width=3)
        draw.line((u - 9, v, u + 9, v), fill=color, width=2)
        draw.line((u, v - 9, u, v + 9), fill=color, width=2)
        draw.text((u + 10, v - 7), label, fill=color)
    draw.text(
        (10, 7),
        f"B_env RGB + room labels | {floor_id} | +X right, +Y up",
        fill=(0, 0, 0),
    )
    draw.text(
        (10, 24),
        "red=intervention target before, blue=target after",
        fill=(0, 0, 0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_overlap_graph_inspection(
    path: Path, temporal_overlap: Mapping[str, Any]
) -> None:
    """Draw every persisted keyframe graph and its pairwise overlap values."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            "Inspection images require the 'inspection' extra with Pillow"
        ) from error
    keyframes = list(temporal_overlap["keyframes"])
    panel_width, panel_height, banner_height = 154, 190, 62
    canvas = Image.new(
        "RGB",
        (max(1, len(keyframes)) * panel_width, panel_height + banner_height),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 7),
        (
            f"GT-depth overlap keyframes | G_union={temporal_overlap['union_edges']} | "
            f"requested={temporal_overlap['requested_regime']} -> "
            f"realized={temporal_overlap['realized_regime']}"
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (10, 27),
        (
            f"connected G_t={temporal_overlap['connected_keyframe_count']}/"
            f"{temporal_overlap['keyframe_count']} (soft target "
            f"{float(temporal_overlap['connected_fraction_target']):.2f}); "
            "colored edges passed the useful-overlap threshold"
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (10, 45),
        f"maximum isolated runs={temporal_overlap['maximum_consecutive_isolated_keyframes']}",
        fill=(0, 0, 0),
    )
    node_offsets = {
        "robot_00": (77, 28),
        "robot_01": (34, 105),
        "robot_02": (120, 105),
    }
    pairs = (
        ("robot_00", "robot_01"),
        ("robot_00", "robot_02"),
        ("robot_01", "robot_02"),
    )
    colors = {
        robot_id: values["rgb8"]
        for robot_id, values in ROBOT_APPEARANCE_VARIANTS.items()
    }
    for panel_index, keyframe in enumerate(keyframes):
        origin_x = panel_index * panel_width
        draw.rectangle(
            (
                origin_x,
                banner_height,
                origin_x + panel_width - 1,
                banner_height + panel_height - 1,
            ),
            outline=(205, 205, 205),
        )
        draw.text(
            (origin_x + 7, banner_height + 5),
            f"t={int(keyframe['frame_index']):03d} connected={bool(keyframe['connected'])}",
            fill=(0, 0, 0),
        )
        edge_set = {tuple(edge) for edge in keyframe["edges"]}
        overlaps = keyframe["overlaps"]
        for left, right in pairs:
            x0, y0 = node_offsets[left]
            x1, y1 = node_offsets[right]
            useful = (left, right) in edge_set or (right, left) in edge_set
            draw.line(
                (
                    origin_x + x0,
                    banner_height + y0,
                    origin_x + x1,
                    banner_height + y1,
                ),
                fill=(35, 35, 35) if useful else (205, 205, 205),
                width=4 if useful else 1,
            )
        for robot_id, (x, y) in node_offsets.items():
            color = colors[robot_id]
            draw.ellipse(
                (
                    origin_x + x - 13,
                    banner_height + y - 13,
                    origin_x + x + 13,
                    banner_height + y + 13,
                ),
                fill=color,
                outline=(0, 0, 0),
            )
            draw.text(
                (origin_x + x, banner_height + y),
                robot_id[-2:],
                fill=(255, 255, 255),
                anchor="mm",
            )
        for row, (left, right) in enumerate(pairs):
            value = overlaps.get(
                f"{left}|{right}", overlaps.get(f"{right}|{left}", 0.0)
            )
            draw.text(
                (origin_x + 7, banner_height + 132 + row * 16),
                f"{left[-2:]}-{right[-2:]}: {float(value):.3f}",
                fill=(0, 0, 0),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_robot_appearance_summary(
    path: Path,
    world_rgb: np.ndarray,
    world_instance: np.ndarray,
) -> None:
    """Save a world overview and instance-derived close-up for every robot."""
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Inspection images require the 'inspection' extra with Pillow"
        ) from error
    rgb = np.asarray(world_rgb)[..., :3].astype(np.uint8, copy=False)
    instances = np.asarray(world_instance)
    if rgb.ndim != 3 or instances.ndim != 2:
        raise ValueError(
            "robot appearance summary expects one RGB and instance frame"
        )
    overview = ImageOps.contain(
        Image.fromarray(rgb).convert("RGB"), (720, 430)
    )
    canvas = Image.new("RGB", (960, 540), (245, 245, 245))
    canvas.paste(overview, ((720 - overview.width) // 2, 58))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        "Final-robot appearance QA | BEV RGB and public-instance crops",
        fill=(0, 0, 0),
    )
    draw.text(
        (12, 31),
        "visual colors only; collision geometry, footprint, and camera unchanged",
        fill=(45, 45, 45),
    )
    panel_x = 730
    for index, (robot_id, specification) in enumerate(
        ROBOT_APPEARANCE_VARIANTS.items(), start=1
    ):
        y = 58 + (index - 1) * 157
        color = tuple(int(value) for value in specification["rgb8"])
        draw.rounded_rectangle(
            (panel_x, y, panel_x + 218, y + 145),
            radius=8,
            fill=(255, 255, 255),
            outline=(175, 175, 175),
        )
        draw.rectangle(
            (panel_x + 8, y + 8, panel_x + 38, y + 32), fill=color
        )
        draw.text(
            (panel_x + 46, y + 11),
            f"{robot_id} | {specification['display_name']}",
            fill=(0, 0, 0),
        )
        mask = instances == index
        if bool(mask.any()):
            rows, columns = np.nonzero(mask)
            pad = 18
            crop = Image.fromarray(rgb).crop(
                (
                    max(0, int(columns.min()) - pad),
                    max(0, int(rows.min()) - pad),
                    min(rgb.shape[1], int(columns.max()) + pad + 1),
                    min(rgb.shape[0], int(rows.max()) + pad + 1),
                )
            )
            crop = ImageOps.contain(crop, (202, 98))
            canvas.paste(
                crop,
                (
                    panel_x + 8 + (202 - crop.width) // 2,
                    y + 40 + (98 - crop.height) // 2,
                ),
            )
        else:
            draw.text(
                (panel_x + 109, y + 91),
                "not visible in t=000 BEV",
                anchor="mm",
                fill=(90, 90, 90),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_intervention_target_crops(
    path: Path,
    before_views: Mapping[str, Mapping[str, np.ndarray]],
    after_views: Mapping[str, Mapping[str, np.ndarray]],
    target_public_instance_id: int,
) -> bool:
    """Save the strongest per-robot before/after target evidence as RGB crops."""
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Inspection images require the 'inspection' extra with Pillow"
        ) from error
    rows: list[tuple[str, int, list[tuple[str, Any, int]]]] = []
    for robot_id in sorted(set(before_views) & set(after_views)):
        before_instance = np.asarray(before_views[robot_id]["instance"])
        after_instance = np.asarray(after_views[robot_id]["instance"])
        frame_scores = np.sum(
            before_instance == target_public_instance_id, axis=(1, 2)
        )
        frame_scores += np.sum(
            after_instance == target_public_instance_id, axis=(1, 2)
        )
        if int(frame_scores.max(initial=0)) == 0:
            continue
        frame_index = int(np.argmax(frame_scores))
        tiles: list[tuple[str, Any, int]] = []
        for phase, views, instance in (
            ("before", before_views, before_instance),
            ("after", after_views, after_instance),
        ):
            rgb = np.asarray(views[robot_id]["rgb"][frame_index])[..., :3]
            mask = instance[frame_index] == target_public_instance_id
            pixel_count = int(mask.sum())
            if pixel_count:
                rows_y, columns_x = np.nonzero(mask)
                scale_x = rgb.shape[1] / mask.shape[1]
                scale_y = rgb.shape[0] / mask.shape[0]
                left = max(
                    0, int(np.floor(columns_x.min() * scale_x)) - 28
                )
                right = min(
                    rgb.shape[1],
                    int(np.ceil((columns_x.max() + 1) * scale_x)) + 28,
                )
                top = max(0, int(np.floor(rows_y.min() * scale_y)) - 28)
                bottom = min(
                    rgb.shape[0],
                    int(np.ceil((rows_y.max() + 1) * scale_y)) + 28,
                )
                crop = Image.fromarray(
                    rgb.astype(np.uint8, copy=False)
                ).crop((left, top, right, bottom))
            else:
                crop = Image.new("RGB", (320, 180), (25, 25, 25))
                ImageDraw.Draw(crop).text(
                    (160, 90),
                    "target not visible",
                    fill=(235, 235, 235),
                    anchor="mm",
                )
            crop = ImageOps.contain(crop.convert("RGB"), (360, 190))
            tile = Image.new("RGB", (360, 190), (20, 20, 20))
            tile.paste(
                crop, ((360 - crop.width) // 2, (190 - crop.height) // 2)
            )
            tiles.append((phase, tile, pixel_count))
        rows.append((robot_id, frame_index, tiles))
    if not rows:
        return False
    banner_height, row_height = 42, 226
    canvas = Image.new(
        "RGB",
        (728, banner_height + row_height * len(rows)),
        (246, 246, 246),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8),
        (
            f"Intervention target public instance={target_public_instance_id}: "
            "strongest before/after evidence"
        ),
        fill=(0, 0, 0),
    )
    for row_index, (robot_id, frame_index, tiles) in enumerate(rows):
        y = banner_height + row_index * row_height
        for column, (phase, tile, pixel_count) in enumerate(tiles):
            x = 4 + column * 362
            canvas.paste(tile, (x, y + 28))
            draw.text(
                (x + 5, y + 7),
                (
                    f"{robot_id} t={frame_index:03d} {phase}: "
                    f"{pixel_count} geometry pixels"
                ),
                fill=(0, 0, 0),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return True
