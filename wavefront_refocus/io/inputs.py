"""Loaders for the four supported input layouts.

Each ``open_*`` function returns a :class:`LoadedInput` carrying the frame
metadata and a **lazy** ``loader(frame) -> (phase, amp)`` callable. Only the
current frame's wavefront is ever held in memory; everything else is read from
disk on demand.

Phase is always in radians and amplitude is linear; both are square float32
arrays (``compute_opd_stack`` requires square 2-D input — non-square inputs are
centre-cropped to the largest centred square).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from ..config import (
    FMT_INTENSITY_OPD,
    FMT_INTENSITY_PHASE,
    FMT_STACK,
    FMT_STACK_FOLDER,
)
from ..vendor.utils_images import load_imgfile, load_wavefront_tif, make_PHY

_TIF_EXTS = (".tif", ".tiff")


@dataclass
class LoadedInput:
    """Result of opening a dataset."""

    input_format: str
    # Per-frame source descriptor (path or (path, time_index)).
    sources: List[object]
    loader: Callable[[object], Tuple[np.ndarray, np.ndarray]]

    @property
    def n_frames(self) -> int:
        return len(self.sources)


# ── helpers ───────────────────────────────────────────────────────────────

def _natural_key(name: str):
    """Sort key that orders ``frame2`` before ``frame10``."""
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", os.path.basename(name))
    ]


def list_tifs(folder: str) -> List[str]:
    """Natural-sorted absolute paths of TIFF files in ``folder``."""
    if not os.path.isdir(folder):
        raise ValueError(f"not a folder: {folder}")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(_TIF_EXTS)
    ]
    if not files:
        raise ValueError(f"no .tif files in {folder}")
    return sorted(files, key=_natural_key)


def to_square(img: np.ndarray) -> np.ndarray:
    """Centre-crop a 2-D array to its largest centred square."""
    if img.ndim != 2:
        raise ValueError(f"expected 2-D image, got shape {img.shape}")
    h, w = img.shape
    if h == w:
        return img
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    return img[top : top + s, left : left + s]


# ── layout (a): intensity folder + OPD folder ──────────────────────────────

def open_intensity_opd(
    intensity_dir: str, opd_dir: str, wavelength_nm: float
) -> LoadedInput:
    """Pair intensity (linear) and OPD (nm) folders, frame by frame.

    OPD is converted to phase via the WF relation
    ``phase = opd / wavelength * 2π`` (``make_PHY`` with ``from_phase=False``);
    the wavelength must be in the **same unit as the OPD**, i.e. nanometres.
    """
    intens = list_tifs(intensity_dir)
    opds = list_tifs(opd_dir)
    if len(intens) != len(opds):
        raise ValueError(
            f"intensity ({len(intens)}) and OPD ({len(opds)}) frame counts differ"
        )
    sources = list(zip(intens, opds))

    def loader(src):
        ipath, opath = src
        amp = to_square(np.asarray(load_imgfile(ipath), dtype=np.float32))
        opd = to_square(np.asarray(load_imgfile(opath), dtype=np.float32))
        # make_PHY returns (amplitude, phase) stacked on axis 0.
        _, phase = make_PHY(amp, opd, wavelenght=wavelength_nm, from_phase=False)
        return phase.astype(np.float32), amp.astype(np.float32)

    return LoadedInput(FMT_INTENSITY_OPD, sources, loader)


# ── layout (b): intensity folder + phase folder ────────────────────────────

def open_intensity_phase(intensity_dir: str, phase_dir: str) -> LoadedInput:
    """Pair intensity (linear) and phase (radians) folders, frame by frame."""
    intens = list_tifs(intensity_dir)
    phases = list_tifs(phase_dir)
    if len(intens) != len(phases):
        raise ValueError(
            f"intensity ({len(intens)}) and phase ({len(phases)}) counts differ"
        )
    sources = list(zip(intens, phases))

    def loader(src):
        ipath, ppath = src
        amp = to_square(np.asarray(load_imgfile(ipath), dtype=np.float32))
        phase = to_square(np.asarray(load_imgfile(ppath), dtype=np.float32))
        return phase.astype(np.float32), amp.astype(np.float32)

    return LoadedInput(FMT_INTENSITY_PHASE, sources, loader)


# ── layout (c): single multi-frame wavefront stack ─────────────────────────

def open_stack(path: str) -> LoadedInput:
    """One TIFF with 2 channels (phase, intensity) and a time axis."""
    _, _, n_frames = load_wavefront_tif(path, 0)
    sources = [(path, t) for t in range(n_frames)]

    def loader(src):
        p, t = src
        phase, amp, _ = load_wavefront_tif(p, t)
        return to_square(phase), to_square(amp)

    return LoadedInput(FMT_STACK, sources, loader)


# ── layout (d): folder of wavefront stacks ─────────────────────────────────

def open_stack_folder(folder: str) -> LoadedInput:
    """Folder where each TIFF is one frame's phase+intensity stack."""
    files = list_tifs(folder)
    sources = list(files)

    def loader(src):
        phase, amp, _ = load_wavefront_tif(src, 0)
        return to_square(phase), to_square(amp)

    return LoadedInput(FMT_STACK_FOLDER, sources, loader)
