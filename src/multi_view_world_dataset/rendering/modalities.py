from __future__ import annotations

import numpy as np


THREE_CHANNEL_MODALITIES = frozenset({"rgb", "normal"})


def canonicalize_public_modality(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize public RGB / normal arrays to Dataset-v1.1 channel semantics.

    Isaac render products may expose an alpha / padding component. Alpha is
    renderer transport metadata, not a dataset channel, so both RGB and XYZ
    camera-space normals are stored with exactly three channel-last values.
    Other modalities pass through unchanged.
    """
    array = np.asarray(values)
    modality = str(name).rsplit("/", 1)[-1]
    if modality not in THREE_CHANNEL_MODALITIES:
        return array
    if array.ndim < 3 or array.shape[-1] < 3:
        raise ValueError(
            f"{name} must be channel-last with at least 3 channels; got {array.shape}"
        )
    return np.ascontiguousarray(array[..., :3])


def canonicalize_public_modalities(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Canonicalize all public arrays while preserving names and alignment."""
    return {
        name: canonicalize_public_modality(name, values)
        for name, values in arrays.items()
    }
