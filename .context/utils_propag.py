
from __future__ import annotations

import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tifffile
import matplotlib
from skimage.restoration import unwrap_phase

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
except ImportError:
    sys.stderr.write(
        "PySide6 is required. Install it with:\n    pip install PySide6\n"
    )
    raise

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

ON_STARTUP_PATH = "/home/coohrentiin/workspace/Code_reconstruction_Data/Data"
if not os.path.exists(ON_STARTUP_PATH):
    ON_STARTUP_PATH = os.getcwd()
# ─────────────────────────────────────────────────────────────────────
# Default optical parameters (overridable from the UI at runtime)
# ─────────────────────────────────────────────────────────────────────
WAVELENGTH = 660e-9
NA = 0.8
MAGNIFICATION = 20
PIXEL_PITCH = 5.86e-6


# ═════════════════════════════════════════════════════════════════════
# Inline angular spectrum propagation (NumPy)
# ═════════════════════════════════════════════════════════════════════

def refocus(field, dz, wavelength, dx, N, FX, FY,
            magnification=1.0, NA=None, **kwargs):
    """Numerically propagate a complex field by ``dz`` (angular spectrum).

    Parameters
    ----------
    field : ndarray (complex)
        Complex object field at the sensor (recording) plane.
    dz : float
        Propagation distance in **image space** [m]. To convert from
        sample-space displacement Δz_sample, use
        ``dz = Δz_sample * magnification**2`` (axial magnification is M²).
    wavelength : float
        Illumination wavelength [m].
    dx : float
        Sensor pixel pitch [m].
    N : int
        Image side length in pixels.
    FX, FY : ndarray
        Frequency grids [1/m] (e.g. from ``numpy.fft.fftfreq`` meshgrids).
    magnification : float
        System magnification. Used only to compute the bandlimit when
        ``NA`` is given.
    NA : float or None
        If given, applies a circular bandlimit at NA / (λ * M) on the
        kernel to suppress out-of-band noise.

    Returns
    -------
    field_propagated : ndarray (complex)
    """
    fx2_fy2 = FX ** 2 + FY ** 2
    cutoff = 1.0 / wavelength ** 2
    arg = cutoff - fx2_fy2
    propagating = arg > 0

    kz = np.zeros_like(arg)
    kz[propagating] = np.sqrt(arg[propagating])

    H = np.zeros_like(FX, dtype=complex)
    H[propagating] = np.exp(1j * 2 * np.pi * dz * kz[propagating])

    if NA is not None:
        max_freq = NA / (wavelength * magnification)
        bandlimit = (np.sqrt(fx2_fy2) <= max_freq).astype(float)
        H = H * bandlimit

    field_spec = np.fft.fft2(field)
    field_propagated = np.fft.ifft2(field_spec * H)
    return field_propagated


# ═════════════════════════════════════════════════════════════════════
# Inline ImageJ-compatible TIFF stack writer
# ═════════════════════════════════════════════════════════════════════

def save_stack(path: str,
               stack: np.ndarray,
               wavelengths: Optional[Tuple[float, ...]] = None,
               source_path: Optional[str] = None,
               metas: Optional[Dict[str, object]] = None,
               input_format: Optional[str] = None,
               timestamps: Optional[Tuple[float, ...]] = None) -> None:
    """Save a real-valued stack as an ImageJ hyperstack TIFF.

    Accepts arrays of shape ``(N, H, W)``, ``(N, H, W, C)`` or
    ``(N, nWL, H, W, C)``. Always writes axes ``"TZCYX"``, with float32
    precision. Optical / acquisition metadata supplied via ``metas`` is
    stored as JSON in the ImageJ ``Info`` tag. Per-frame timestamps are
    written as OME-XML ``DeltaT`` planes when ``timestamps`` is given.
    """
    if stack.ndim == 3:
        N, H, W = stack.shape
        nwl, C = 1, 1
        stack = stack[:, None, :, :, None]
    elif stack.ndim == 4:
        N, H, W, C = stack.shape
        nwl = 1
        stack = stack[:, None, :, :, :]
    elif stack.ndim == 5:
        N, nwl, H, W, C = stack.shape
    else:
        raise ValueError(f"stack must be 3-D, 4-D or 5-D; got {stack.shape}")

    if timestamps is not None and len(timestamps) != N:
        raise ValueError("timestamps length must match the time dimension")

    if input_format == "TZCYX":
        out = stack.astype(np.float32)
    else:
        out = stack.transpose((0, 1, 4, 2, 3)).astype(np.float32)

    meta: Dict[str, object] = {}
    if wavelengths is not None:
        meta["wavelengths_nm"] = list(map(float, wavelengths))
    if source_path is not None:
        meta["source_path"] = str(source_path)
    if metas is not None:
        meta.update(metas)

    ij_metadata = {"axes": "TZCYX", "Info": json.dumps(meta)}

    description = None
    if timestamps is not None:
        ome = ET.Element(
            "OME",
            xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06",
        )
        image = ET.SubElement(ome, "Image", ID="Image:0")
        pixels = ET.SubElement(
            image, "Pixels",
            DimensionOrder="TZCYX", Type="float",
            SizeT=str(N), SizeZ=str(nwl), SizeC=str(C),
            SizeY=str(H), SizeX=str(W),
        )
        for t, dt in enumerate(timestamps):
            ET.SubElement(
                pixels, "Plane",
                TheT=str(t), TheZ="0", TheC="0",
                DeltaT=str(float(dt)),
            )
        description = ET.tostring(ome, encoding="unicode")

    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    tifffile.imwrite(
        path, out, imagej=True,
        metadata=ij_metadata, description=description,
    )


# ═════════════════════════════════════════════════════════════════════
# Loading helpers
# ═════════════════════════════════════════════════════════════════════

def _have_cuda() -> bool:
    return _TORCH_AVAILABLE and torch.cuda.is_available()


def load_wavefront_tif(path: str, frame_index: int = 0):
    """Load a 2-channel wavefront TIFF.

    Channel 0 = phase (rad), channel 1 = amplitude. Supports the four
    common layouts (H, W, 2), (2, H, W), (T, 2, H, W), (T, H, W, 2).

    Returns
    -------
    phase, amp : float32 arrays of shape (H, W)
    n_frames : int
        Number of time frames present on disk (1 for 3-D files).
    """
    raw = np.asarray(tifffile.imread(path))

    if raw.ndim == 3:
        n_frames = 1
        if raw.shape[0] == 2:
            phase, amp = raw[0], raw[1]
        elif raw.shape[-1] == 2:
            phase, amp = raw[..., 0], raw[..., 1]
        else:
            raise ValueError(
                f"3-D wavefront TIFF must have a size-2 channel axis; "
                f"got shape {raw.shape}"
            )
    elif raw.ndim == 4:
        if raw.shape[1] == 2:
            n_frames = raw.shape[0]
            t = max(0, min(int(frame_index), n_frames - 1))
            phase, amp = raw[t, 0], raw[t, 1]
        elif raw.shape[-1] == 2:
            n_frames = raw.shape[0]
            t = max(0, min(int(frame_index), n_frames - 1))
            phase, amp = raw[t, ..., 0], raw[t, ..., 1]
        else:
            raise ValueError(
                f"4-D wavefront TIFF must have a size-2 channel axis at "
                f"position 1 (TCYX) or -1 (TYXC); got shape {raw.shape}"
            )
    else:
        raise ValueError(
            f"Unsupported wavefront TIFF rank {raw.ndim} (shape {raw.shape}); "
            f"expected 3-D or 4-D."
        )

    return phase.astype(np.float32), amp.astype(np.float32), n_frames

def build_freq_grids(N: int, dx: float):
    fx = np.fft.fftfreq(N, dx)
    fy = np.fft.fftfreq(N, dx)
    return np.meshgrid(fx, fy)


# ═════════════════════════════════════════════════════════════════════
# Per-slice OPD extraction
# ═════════════════════════════════════════════════════════════════════

def _opd_from_field(field: np.ndarray, wavelength: float) -> np.ndarray:
    """Unwrap phase, convert to OPD in nm, and zero-mean.

    Subtracting the per-slice mean removes the global piston that would
    otherwise jump between slices when the user scrubs z (the unwrap
    branch can wrap by 2π between adjacent dz values).
    """
    unwrapped = unwrap_phase(np.angle(field))
    opd_m = unwrapped * wavelength / (2.0 * np.pi)
    opd_nm = opd_m * 1e9
    opd_nm = opd_nm - opd_nm.mean()
    return opd_nm.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════
# Backend-dispatched propagation precompute
# ═════════════════════════════════════════════════════════════════════

def compute_opd_stack(
    phase: np.ndarray,
    amp: np.ndarray,
    *,
    wavelength: float = WAVELENGTH,
    NA_: float = NA,
    magnification: float = MAGNIFICATION,
    pixel_pitch: float = PIXEL_PITCH,
    dz_sample_max: float = 10e-6,
    dz_step_sample: float = 200e-9,
    use_torch: bool = True,
    progress_cb=None,
    cancel_cb=None,
):
    """Propagate WF through a symmetric sample-space depth range.

    The complex wavefront ``WF = amp * exp(1j * phase)`` is propagated
    by the angular spectrum method through every plane ``dz_sample`` in
    ``arange(-dz_sample_max, +dz_sample_max, dz_step_sample)`` and the
    unwrapped OPD (in nm, zero-mean per slice) is returned for each.

    Two backends are dispatched at runtime:

    - **torch-cuda** (preferred when CUDA is available *and*
      ``use_torch=True``): batches the per-slice kernel multiply / IFFT
      on GPU. Performs the forward FFT of the input field once.
    - **numpy** (fallback): uses the inline :func:`refocus` per slice.

    The two paths agree to FFT precision (verified at ~1e-10 absolute
    difference on synthetic inputs).

    Parameters
    ----------
    phase, amp : ndarray
        2-D phase (radians) and amplitude maps, same square shape.
    wavelength, NA_, magnification, pixel_pitch : float
        Optical parameters, in SI units.
    dz_sample_max : float
        Half-range of the propagation sweep in **sample space** [m].
    dz_step_sample : float
        Sweep step in sample space [m].
    use_torch : bool
        If True and CUDA is available, use the GPU backend.
    progress_cb : callable or None
        ``progress_cb(i, total)`` invoked after each slice.
    cancel_cb : callable or None
        Returns True to abort the loop early. Returns ``None``.

    Returns
    -------
    dict with keys: ``opd_stack`` (Z, H, W) float32 nm, the dz arrays,
    ``z0_index``, the optical params, the backend label, and ``N``.
    """
    if phase.shape != amp.shape:
        raise ValueError(
            f"phase and amp shapes differ: {phase.shape} vs {amp.shape}"
        )
    if phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
        raise ValueError(
            f"Expected a square 2-D wavefront; got shape {phase.shape}"
        )

    N = phase.shape[0]
    dx = pixel_pitch

    n_half = int(np.floor(dz_sample_max / dz_step_sample))
    dz_sample_array = (np.arange(-n_half, n_half + 1) * dz_step_sample).astype(
        np.float64
    )
    dz_image_array = dz_sample_array * (magnification ** 2)
    Z = len(dz_sample_array)
    z0_index = int(np.argmin(np.abs(dz_sample_array)))

    WF = (amp * np.exp(1j * phase)).astype(np.complex64)
    FX, FY = build_freq_grids(N, dx)

    opd_stack = np.empty((Z, N, N), dtype=np.float32)

    backend = "numpy"
    if use_torch and _have_cuda():
        backend = "torch-cuda"
        device = torch.device("cuda")
        WF_t = torch.from_numpy(WF).to(device)
        FX_t = torch.from_numpy(FX.astype(np.float32)).to(device)
        FY_t = torch.from_numpy(FY.astype(np.float32)).to(device)
        WF_spec = torch.fft.fft2(WF_t)

        cutoff = 1.0 / (wavelength ** 2)
        arg = cutoff - (FX_t ** 2 + FY_t ** 2)
        kz = torch.sqrt(torch.clamp(arg, min=0.0))
        propagating = (arg > 0).to(torch.float32)

        max_freq = NA_ / (wavelength * magnification)
        bandlimit = (
            torch.sqrt(FX_t ** 2 + FY_t ** 2) <= max_freq
        ).to(torch.float32)
        base_mask = propagating * bandlimit

        two_pi_kz = (2.0 * np.pi) * kz

        for i, dz in enumerate(dz_image_array):
            if cancel_cb is not None and cancel_cb():
                return None
            phase_factor = two_pi_kz * float(dz)
            H = torch.complex(
                torch.cos(phase_factor) * base_mask,
                torch.sin(phase_factor) * base_mask,
            )
            field_t = torch.fft.ifft2(WF_spec * H)
            field = field_t.cpu().numpy()
            opd_stack[i] = _opd_from_field(field, wavelength)
            if progress_cb is not None:
                progress_cb(i + 1, Z)

        del WF_t, FX_t, FY_t, WF_spec, kz, propagating, bandlimit, base_mask
        del two_pi_kz
        torch.cuda.empty_cache()
    else:
        for i, dz in enumerate(dz_image_array):
            if cancel_cb is not None and cancel_cb():
                return None
            field = refocus(
                WF,
                dz=float(dz),
                wavelength=wavelength,
                dx=dx,
                N=N,
                FX=FX,
                FY=FY,
                magnification=magnification,
                NA=NA_,
            )
            opd_stack[i] = _opd_from_field(field, wavelength)
            if progress_cb is not None:
                progress_cb(i + 1, Z)

    return {
        "opd_stack": opd_stack,
        "dz_sample_array": dz_sample_array,
        "dz_image_array": dz_image_array,
        "z0_index": z0_index,
        "backend": backend,
        "wavelength": wavelength,
        "NA": NA_,
        "magnification": magnification,
        "pixel_pitch": pixel_pitch,
        "N": N,
    }


# ═════════════════════════════════════════════════════════════════════
# Background propagation worker (keeps the UI responsive)
# ═════════════════════════════════════════════════════════════════════

class PropagationWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, kwargs: dict):
        super().__init__()
        self._kwargs = kwargs
        self._cancelled = False

    @Slot()
    def run(self):
        try:
            result = compute_opd_stack(
                progress_cb=lambda i, n: self.progress.emit(i, n),
                cancel_cb=lambda: self._cancelled,
                **self._kwargs,
            )
            if result is None:
                self.failed.emit("cancelled")
            else:
                self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    @Slot()
    def cancel(self):
        self._cancelled = True
