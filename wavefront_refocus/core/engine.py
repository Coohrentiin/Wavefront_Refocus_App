"""Thin wrappers around the vendored angular-spectrum propagation engine."""
from __future__ import annotations

import numpy as np

from ..vendor.utils_propag import build_freq_grids, refocus
from .model import OpticalParams, SweepParams
from .optics import sample_to_image
from .sweep import resolve_sweep


def build_compute_kwargs(
    phase: np.ndarray,
    amp: np.ndarray,
    optics: OpticalParams,
    sweep: SweepParams,
    *,
    use_torch: bool = True,
    center_dz_sample: float = 0.0,
) -> dict:
    """Assemble keyword arguments for ``compute_opd_stack``.

    ``compute_opd_stack`` always sweeps symmetrically about the plane of the
    field it is handed. To sweep about ``center_dz_sample`` instead (sample
    space, relative to the frame's base plane) we first propagate the field to
    that plane and sweep about *it* — so a ±2 µm sweep recentred on +4 µm
    covers +2…+6 µm without recomputing the planes already known to be bad.
    """
    if center_dz_sample:
        field = refocus_one(phase, amp, float(center_dz_sample), optics)
        phase, amp = np.angle(field), np.abs(field)

    sw = resolve_sweep(sweep.half_range_sample, sweep.n_planes)
    return dict(
        phase=phase,
        amp=amp,
        wavelength=optics.wavelength,
        NA_=optics.NA,
        magnification=optics.magnification,
        pixel_pitch=optics.pixel_pitch,
        use_torch=use_torch,
        **sw,
    )


def refocus_one(
    phase: np.ndarray,
    amp: np.ndarray,
    dz_sample: float,
    optics: OpticalParams,
) -> np.ndarray:
    """Propagate one wavefront by ``dz_sample`` and return the complex field.

    Used for single-plane export and previews. ``dz_sample`` is in sample
    space; it is converted to image space (``* M**2``) before propagation,
    matching :func:`compute_opd_stack`.
    """
    if phase.shape != amp.shape:
        raise ValueError(
            f"phase and amp shapes differ: {phase.shape} vs {amp.shape}"
        )
    N = phase.shape[0]
    FX, FY = build_freq_grids(N, optics.pixel_pitch)
    WF = (amp * np.exp(1j * phase)).astype(np.complex64)
    dz_image = sample_to_image(dz_sample, optics.magnification)
    return refocus(
        WF,
        dz=float(dz_image),
        wavelength=optics.wavelength,
        dx=optics.pixel_pitch,
        N=N,
        FX=FX,
        FY=FY,
        magnification=optics.magnification,
        NA=optics.NA,
    )
