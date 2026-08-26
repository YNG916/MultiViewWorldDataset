from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np


def infer_scene_family(scene_id: str) -> str:
    """Group numbered variants and explicit upper/lower floors without assuming a scene catalog size."""
    family = re.sub(r"_(?:int|garden)$", "", scene_id)
    family = re.sub(r"_(?:lower|upper)$", "", family)
    family = re.sub(r"_\d+$", "", family)
    return family


def assign_scene_family_splits(
    scene_ids: list[str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    if not np.isclose(sum(ratios.values()), 1.0) or any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative and sum to one")
    by_family: dict[str, list[str]] = defaultdict(list)
    for scene_id in sorted(scene_ids):
        by_family[infer_scene_family(scene_id)].append(scene_id)
    families = sorted(by_family, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    split_names = list(ratios)
    cumulative = np.cumsum([ratios[name] for name in split_names])
    result: dict[str, str] = {}
    for index, family in enumerate(families):
        fraction = (index + 0.5) / max(1, len(families))
        split = split_names[min(int(np.searchsorted(cumulative, fraction, side="right")), len(split_names) - 1)]
        for scene_id in by_family[family]:
            result[scene_id] = split
    return result

