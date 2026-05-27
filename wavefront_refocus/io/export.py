"""Export a refocused sequence in any of the four output formats.

For each frame the wavefront is loaded, propagated by its final sample-space
displacement (:func:`refocus_one`), and the refocused complex field is written
out. An updated distances JSON is written alongside.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import tifffile
from skimage.restoration import unwrap_phase

from ..config import (
    FMT_INTENSITY_OPD,
    FMT_INTENSITY_PHASE,
    FMT_STACK,
    FMT_STACK_FOLDER,
)
from ..core.engine import refocus_one
from ..core.model import Session
from ..vendor.utils_propag import save_stack
from .distances import write_distances


def _frame_field(session: Session, frame) -> np.ndarray:
    """Load and refocus one frame; return the complex field."""
    phase, amp = session.loader(frame.source)
    dz = frame.final_dz_sample or 0.0
    return refocus_one(phase, amp, dz, session.optics)


def _opd_nm(field: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """Unwrapped, zero-mean OPD in nm from a complex field."""
    unwrapped = unwrap_phase(np.angle(field))
    opd = unwrapped * wavelength_nm / (2.0 * np.pi)
    return (opd - opd.mean()).astype(np.float32)


def export_folder(
    session: Session,
    out_format: str,
    out_dir: str,
    *,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    distances_filename: str = "Reconstruction_Distances.json",
) -> None:
    """Refocus every frame and write it in ``out_format`` under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    frames = session.frames
    total = len(frames)
    wl_nm = session.optics.wavelength_nm

    if out_format in (FMT_INTENSITY_OPD, FMT_INTENSITY_PHASE):
        intens_dir = os.path.join(out_dir, "Intensity")
        second_dir = os.path.join(
            out_dir, "OPD" if out_format == FMT_INTENSITY_OPD else "Phase"
        )
        os.makedirs(intens_dir, exist_ok=True)
        os.makedirs(second_dir, exist_ok=True)

        for i, f in enumerate(frames):
            field = _frame_field(session, f)
            amp = np.abs(field).astype(np.float32)
            tifffile.imwrite(
                os.path.join(intens_dir, f"frame_{f.index:04d}.tif"), amp
            )
            if out_format == FMT_INTENSITY_OPD:
                data = _opd_nm(field, wl_nm)
            else:
                data = unwrap_phase(np.angle(field)).astype(np.float32)
            tifffile.imwrite(
                os.path.join(second_dir, f"frame_{f.index:04d}.tif"), data
            )
            if progress_cb:
                progress_cb(i + 1, total)

    elif out_format == FMT_STACK_FOLDER:
        for i, f in enumerate(frames):
            field = _frame_field(session, f)
            phase = np.angle(field).astype(np.float32)
            amp = np.abs(field).astype(np.float32)
            # Per-frame 2-channel (phase, intensity) stack: (1, H, W, 2).
            stack = np.stack([phase, amp], axis=-1)[None, ...]
            save_stack(
                os.path.join(out_dir, f"frame_{f.index:04d}.tif"),
                stack,
                wavelengths=(wl_nm,),
            )
            if progress_cb:
                progress_cb(i + 1, total)

    elif out_format == FMT_STACK:
        # One multi-frame stack: (N, H, W, 2) with channels (phase, intensity).
        planes = []
        for i, f in enumerate(frames):
            field = _frame_field(session, f)
            phase = np.angle(field).astype(np.float32)
            amp = np.abs(field).astype(np.float32)
            planes.append(np.stack([phase, amp], axis=-1))
            if progress_cb:
                progress_cb(i + 1, total)
        stack = np.stack(planes, axis=0)  # (N, H, W, 2)
        save_stack(os.path.join(out_dir, "wavefront_stack.tif"), stack,
                   wavelengths=(wl_nm,))

    else:
        raise ValueError(f"unknown output format: {out_format}")

    write_distances(os.path.join(out_dir, distances_filename),
                    session.frames, session.optics)
