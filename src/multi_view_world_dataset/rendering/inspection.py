from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

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

