"""Read/write the Reconstruction_Distances.json (absolute, image space).

Schema: ``{ "<frame_index>": { "<wavelength_nm>": distance_m, ... }, ... }``.
"""
from __future__ import annotations

import json
import warnings
from typing import Dict, List

from ..core.model import Frame, OpticalParams
from ..core.optics import base_dz_image_for_frame, sample_to_image


def load_distances(path: str) -> Dict[str, Dict[str, float]]:
    """Load the raw distances JSON."""
    with open(path, "r") as fh:
        return json.load(fh)


def base_dz_for_frame(
    raw: Dict[str, Dict[str, float]], frame_index: int, wavelength_nm: float
) -> float:
    """Absolute image-space distance for a frame at the nearest wavelength.

    Returns 0.0 (with a warning) if the frame index is absent from the JSON.
    """
    entry = raw.get(str(frame_index))
    if entry is None:
        warnings.warn(
            f"frame {frame_index} missing from distances JSON; using 0.0",
            stacklevel=2,
        )
        return 0.0
    return base_dz_image_for_frame(entry, wavelength_nm)


def apply_base_distances(
    frames: List[Frame],
    raw: Dict[str, Dict[str, float]],
    wavelength_nm: float,
) -> None:
    """Fill each frame's ``base_dz_image`` from the JSON (re-bindable on WL change)."""
    for f in frames:
        f.base_dz_image = base_dz_for_frame(raw, f.index, wavelength_nm)


def write_distances(
    path: str,
    frames: List[Frame],
    optics: OpticalParams,
) -> None:
    """Write final absolute image-space distances at the reference wavelength.

    For each frame the value is ``base_dz_image + sample_to_image(final_dz)``;
    frames without a chosen/interpolated position keep their base distance.
    """
    wl_key = f"{optics.wavelength_nm:.3f}"
    out: Dict[str, Dict[str, float]] = {}
    for f in frames:
        dz_sample = f.final_dz_sample or 0.0
        abs_image = f.base_dz_image + sample_to_image(dz_sample, optics.magnification)
        out[str(f.index)] = {wl_key: float(abs_image)}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=4)
