"""Sample <-> image space conversion and nearest-wavelength lookup.

This is the single source of truth for the axial-space convention. The JSON
distances and :func:`refocus` work in **image space**; the UI works in
**sample space**. ``dz_image = dz_sample * M**2``.
"""
from __future__ import annotations

from typing import Mapping


def sample_to_image(dz_sample: float, magnification: float) -> float:
    """Convert a sample-space axial displacement to image space."""
    return dz_sample * magnification ** 2


def image_to_sample(dz_image: float, magnification: float) -> float:
    """Convert an image-space axial displacement to sample space."""
    return dz_image / magnification ** 2


def nearest_wavelength_key(wl_map: Mapping[str, float], wavelength_nm: float) -> str:
    """Return the key of ``wl_map`` whose nm value is nearest ``wavelength_nm``.

    Keys are strings such as ``"660.000"``. ``wavelength_nm`` is in nanometres
    (convert a metres value with ``wavelength * 1e9`` before calling).
    """
    if not wl_map:
        raise ValueError("empty wavelength map")
    return min(wl_map, key=lambda k: abs(float(k) - wavelength_nm))


def base_dz_image_for_frame(
    json_entry: Mapping[str, float], wavelength_nm: float
) -> float:
    """Absolute image-space distance for a frame at the nearest wavelength."""
    key = nearest_wavelength_key(json_entry, wavelength_nm)
    return float(json_entry[key])
