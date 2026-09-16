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


def _source_paths(frame) -> tuple:
    """The input path(s) behind a frame, in (primary, secondary) order.

    Frame sources differ per layout: a bare path (folder of stacks), a
    ``(path, time_index)`` pair (single multi-frame stack), or an
    ``(intensity, other)`` pair of paths (the two-folder layouts).
    """
    src = frame.source
    if isinstance(src, str):
        return src, None
    if isinstance(src, (tuple, list)) and src:
        first = src[0]
        second = src[1] if len(src) > 1 else None
        if isinstance(second, str):
            return first, second
        return first, None      # (path, time_index)
    return None, None


def frame_basename(frame, keep_original_names: bool, which: str = "primary") -> str:
    """Stem to write a frame under (no extension).

    With ``keep_original_names`` the input file's stem is reused; ``which``
    picks between the two source files of the paired layouts, so an
    intensity/phase pair keeps each side's own name. Frames whose source has no
    usable filename — a slice of a multi-frame stack, say — fall back to the
    indexed ``frame_NNNN`` form so an export never silently loses a frame.
    """
    if keep_original_names:
        primary, secondary = _source_paths(frame)
        path = secondary if (which == "secondary" and secondary) else primary
        if path:
            return os.path.splitext(os.path.basename(path))[0]
    return f"frame_{frame.index:04d}"


def _unique(name: str, used: set) -> str:
    """Disambiguate a repeated stem by suffixing ``_2``, ``_3``, …

    Original names can collide once they leave their folders (two layouts may
    both contain ``frame_1.tif``), and silently overwriting an exported frame
    would lose data.
    """
    if name not in used:
        used.add(name)
        return name
    n = 2
    while f"{name}_{n}" in used:
        n += 1
    out = f"{name}_{n}"
    used.add(out)
    return out


def _opd_nm(field: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """Unwrapped, zero-mean OPD in nm from a complex field."""
    unwrapped = unwrap_phase(np.angle(field))
    opd = unwrapped * wavelength_nm / (2.0 * np.pi)
    return (opd - opd.mean()).astype(np.float32)


def _phase_rad(field: np.ndarray) -> np.ndarray:
    """Unwrapped phase in radians from a complex field.

    ``np.angle`` alone returns phase wrapped into [-π, π], so any structure
    taller than π folds back on itself — a bright bead reads as a dark speck at
    its own peak. The app displays unwrapped phase, so exports must unwrap too
    or the written file will not match what was on screen.
    """
    return unwrap_phase(np.angle(field)).astype(np.float32)


def export_folder(
    session: Session,
    out_format: str,
    out_dir: str,
    *,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    distances_filename: str = "Reconstruction_Distances.json",
    keep_original_names: bool = False,
) -> None:
    """Refocus every frame and write it in ``out_format`` under ``out_dir``.

    With ``keep_original_names`` each output reuses its input file's name
    instead of the indexed ``frame_NNNN`` form. Frames with no usable source
    filename keep the indexed form, and repeated names are suffixed ``_2``,
    ``_3``, … so nothing is overwritten.
    """
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
        used_i: set = set()
        used_s: set = set()

        for i, f in enumerate(frames):
            field = _frame_field(session, f)
            amp = np.abs(field).astype(np.float32)
            name_i = _unique(frame_basename(f, keep_original_names), used_i)
            name_s = _unique(
                frame_basename(f, keep_original_names, "secondary"), used_s
            )
            tifffile.imwrite(os.path.join(intens_dir, f"{name_i}.tif"), amp)
            if out_format == FMT_INTENSITY_OPD:
                data = _opd_nm(field, wl_nm)
            else:
                data = _phase_rad(field)
            tifffile.imwrite(os.path.join(second_dir, f"{name_s}.tif"), data)
            if progress_cb:
                progress_cb(i + 1, total)

    elif out_format == FMT_STACK_FOLDER:
        used: set = set()
        for i, f in enumerate(frames):
            field = _frame_field(session, f)
            phase = _phase_rad(field)
            amp = np.abs(field).astype(np.float32)
            # Per-frame 2-channel (phase, intensity) stack: (1, H, W, 2).
            stack = np.stack([phase, amp], axis=-1)[None, ...]
            name = _unique(frame_basename(f, keep_original_names), used)
            save_stack(
                os.path.join(out_dir, f"{name}.tif"),
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
            phase = _phase_rad(field)
            amp = np.abs(field).astype(np.float32)
            planes.append(np.stack([phase, amp], axis=-1))
            if progress_cb:
                progress_cb(i + 1, total)
        stack = np.stack(planes, axis=0)  # (N, H, W, 2)
        # A single output file: reuse the source stack's own name when asked.
        name = "wavefront_stack"
        if keep_original_names and frames:
            primary, _ = _source_paths(frames[0])
            if primary:
                name = os.path.splitext(os.path.basename(primary))[0]
        save_stack(os.path.join(out_dir, f"{name}.tif"), stack,
                   wavelengths=(wl_nm,))

    else:
        raise ValueError(f"unknown output format: {out_format}")

    write_distances(os.path.join(out_dir, distances_filename),
                    session.frames, session.optics)
