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
from typing import Callable, List, Optional, Tuple

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
    """Result of opening a dataset.

    When only a slice of the sequence was loaded, ``sources`` holds just that
    chunk while ``indices`` carries each frame's position in the **full**
    sequence. Keeping the original numbering is what lets a dataset be worked
    through in chunks: distances-JSON lookups and exported filenames stay
    aligned with the whole sequence, so frames 51–150 export as
    ``frame_0051…frame_0150`` no matter which chunk they were loaded in.
    """

    input_format: str
    # Per-frame source descriptor (path or (path, time_index)).
    sources: List[object]
    loader: Callable[[object], Tuple[np.ndarray, np.ndarray]]
    # Original index of each source in the full sequence; defaults to 0..n-1.
    indices: List[int] = None  # type: ignore[assignment]
    # Length of the full sequence the chunk was taken from.
    total_frames: int = 0

    def __post_init__(self) -> None:
        if self.indices is None:
            self.indices = list(range(len(self.sources)))
        elif len(self.indices) != len(self.sources):
            raise ValueError(
                f"indices ({len(self.indices)}) and sources "
                f"({len(self.sources)}) counts differ"
            )
        if not self.total_frames:
            self.total_frames = (
                max(self.indices) + 1 if self.indices else 0
            )

    @property
    def n_frames(self) -> int:
        return len(self.sources)

    @property
    def is_partial(self) -> bool:
        return self.n_frames != self.total_frames


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


def resolve_range(
    n_total: int, first: Optional[int] = None, last: Optional[int] = None
) -> Tuple[int, int]:
    """Validate an inclusive ``first..last`` frame range against ``n_total``.

    ``None`` means "from the start" / "to the end". Returns the resolved
    ``(first, last)`` pair, both inclusive.
    """
    if n_total <= 0:
        raise ValueError("sequence is empty")
    lo = 0 if first is None else int(first)
    hi = n_total - 1 if last is None else int(last)
    if lo < 0 or hi < 0:
        raise ValueError("frame range must be non-negative")
    if lo > hi:
        raise ValueError(f"first frame ({lo}) is after last frame ({hi})")
    if lo >= n_total:
        raise ValueError(
            f"first frame {lo} is past the end of the sequence "
            f"({n_total} frames, 0–{n_total - 1})"
        )
    if hi >= n_total:
        raise ValueError(
            f"last frame {hi} is past the end of the sequence "
            f"({n_total} frames, 0–{n_total - 1})"
        )
    return lo, hi


def _slice(seq: List, first: Optional[int], last: Optional[int]):
    """Return ``(chunk, indices, n_total)`` for an inclusive frame range."""
    n_total = len(seq)
    lo, hi = resolve_range(n_total, first, last)
    return seq[lo : hi + 1], list(range(lo, hi + 1)), n_total


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
    intensity_dir: str,
    opd_dir: str,
    wavelength_nm: float,
    first: Optional[int] = None,
    last: Optional[int] = None,
) -> LoadedInput:
    """Pair intensity (linear) and OPD (nm) folders, frame by frame.

    OPD is converted to phase via the WF relation
    ``phase = opd / wavelength * 2π`` (``make_PHY`` with ``from_phase=False``);
    the wavelength must be in the **same unit as the OPD**, i.e. nanometres.

    ``first``/``last`` select an inclusive slice of the sequence; the frames
    keep their original indices.
    """
    intens = list_tifs(intensity_dir)
    opds = list_tifs(opd_dir)
    if len(intens) != len(opds):
        raise ValueError(
            f"intensity ({len(intens)}) and OPD ({len(opds)}) frame counts differ"
        )
    sources, indices, n_total = _slice(list(zip(intens, opds)), first, last)

    def loader(src):
        ipath, opath = src
        amp = to_square(np.asarray(load_imgfile(ipath), dtype=np.float32))
        opd = to_square(np.asarray(load_imgfile(opath), dtype=np.float32))
        # make_PHY returns (amplitude, phase) stacked on axis 0.
        _, phase = make_PHY(amp, opd, wavelenght=wavelength_nm, from_phase=False)
        return phase.astype(np.float32), amp.astype(np.float32)

    return LoadedInput(FMT_INTENSITY_OPD, sources, loader, indices, n_total)


# ── layout (b): intensity folder + phase folder ────────────────────────────

def open_intensity_phase(
    intensity_dir: str,
    phase_dir: str,
    first: Optional[int] = None,
    last: Optional[int] = None,
) -> LoadedInput:
    """Pair intensity (linear) and phase (radians) folders, frame by frame.

    ``first``/``last`` select an inclusive slice of the sequence; the frames
    keep their original indices.
    """
    intens = list_tifs(intensity_dir)
    phases = list_tifs(phase_dir)
    if len(intens) != len(phases):
        raise ValueError(
            f"intensity ({len(intens)}) and phase ({len(phases)}) counts differ"
        )
    sources, indices, n_total = _slice(list(zip(intens, phases)), first, last)

    def loader(src):
        ipath, ppath = src
        amp = to_square(np.asarray(load_imgfile(ipath), dtype=np.float32))
        phase = to_square(np.asarray(load_imgfile(ppath), dtype=np.float32))
        return phase.astype(np.float32), amp.astype(np.float32)

    return LoadedInput(FMT_INTENSITY_PHASE, sources, loader, indices, n_total)


# ── layout (c): single multi-frame wavefront stack ─────────────────────────

def open_stack(
    path: str, first: Optional[int] = None, last: Optional[int] = None
) -> LoadedInput:
    """One TIFF with 2 channels (phase, intensity) and a time axis.

    ``first``/``last`` select an inclusive slice of the time axis; the frames
    keep their original indices.
    """
    _, _, n_frames = load_wavefront_tif(path, 0)
    sources, indices, n_total = _slice(
        [(path, t) for t in range(n_frames)], first, last
    )

    def loader(src):
        p, t = src
        phase, amp, _ = load_wavefront_tif(p, t)
        return to_square(phase), to_square(amp)

    return LoadedInput(FMT_STACK, sources, loader, indices, n_total)


# ── layout (d): folder of wavefront stacks ─────────────────────────────────

def open_stack_folder(
    folder: str, first: Optional[int] = None, last: Optional[int] = None
) -> LoadedInput:
    """Folder where each TIFF is one frame's phase+intensity stack.

    ``first``/``last`` select an inclusive slice of the sequence; the frames
    keep their original indices.
    """
    sources, indices, n_total = _slice(list_tifs(folder), first, last)

    def loader(src):
        phase, amp, _ = load_wavefront_tif(src, 0)
        return to_square(phase), to_square(amp)

    return LoadedInput(FMT_STACK_FOLDER, sources, loader, indices, n_total)
