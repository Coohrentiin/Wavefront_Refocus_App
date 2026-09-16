"""Central data model: optical params, sweep params, frames and session.

Distances live in two spaces:

- **image space** — what the JSON stores and what :func:`refocus` propagates.
- **sample space** — what the UI displays and edits.

``dz_image = dz_sample * M**2`` (axial magnification is M²).

``Frame.chosen_dz_sample`` / ``interpolated_dz_sample`` are *relative*
excursions about the frame's own base plane, because the propagation sweep is
centred on 0 (the file/base plane). The absolute distance only re-enters at
export and JSON write time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class OpticalParams:
    """System optics. All lengths in SI metres; ``wavelength`` in metres."""

    pixel_pitch: float
    magnification: float
    wavelength: float
    NA: float = 0.8

    @property
    def wavelength_nm(self) -> float:
        return self.wavelength * 1e9


@dataclass
class SweepParams:
    """Propagation sweep, in **sample space**.

    ``half_range_sample`` is the maximum excursion each side ("minimal
    propagation distance" in the UI); ``n_planes`` is the requested total
    number of planes (rounded up to odd so the centre plane is included).
    """

    half_range_sample: float
    n_planes: int


@dataclass
class PropagationResult:
    """Mirror of the dict returned by ``compute_opd_stack``.

    ``dz_sample_array`` is relative to the frame's **base** plane: the engine
    sweeps symmetrically about 0, and ``center_dz_sample`` — the position the
    sweep was centred on — is added back here. So the centre plane of a sweep
    recentred on +4 µm reads 4 µm, not 0, and choosing a plane stores a
    base-relative displacement exactly as before.
    """

    opd_stack: np.ndarray        # (Z, H, W) float32 nm, zero-mean per slice
    dz_sample_array: np.ndarray  # (Z,) m, sample space, rel. to the base plane
    dz_image_array: np.ndarray   # (Z,) m, image space
    z0_index: int                # index of the sweep's centre plane
    backend: str
    N: int
    center_dz_sample: float = 0.0  # m, sample space, what the sweep centres on

    @classmethod
    def from_dict(cls, d: dict, center_dz_sample: float = 0.0,
                  magnification: float = 1.0) -> "PropagationResult":
        center = float(center_dz_sample)
        dz_sample = np.asarray(d["dz_sample_array"]) + center
        # Keep the image-space array the exact counterpart of the shifted
        # sample-space one (dz_image = dz_sample * M²).
        dz_image = np.asarray(d["dz_image_array"]) + center * magnification ** 2
        return cls(
            opd_stack=d["opd_stack"],
            dz_sample_array=dz_sample,
            dz_image_array=dz_image,
            z0_index=int(d["z0_index"]),
            backend=str(d["backend"]),
            N=int(d["N"]),
            center_dz_sample=center,
        )


@dataclass
class Frame:
    """One wavefront in the sequence.

    ``source`` is whatever the loader needs to lazily reload the frame
    (a path, or ``(path, time_index)`` for a multi-frame stack).
    """

    index: int
    source: Any
    base_dz_image: float = 0.0
    chosen_dz_sample: Optional[float] = None
    interpolated_dz_sample: Optional[float] = None
    is_chosen: bool = False
    result: Optional[PropagationResult] = None

    @property
    def final_dz_sample(self) -> Optional[float]:
        """The displacement to apply on export (sample space, relative)."""
        if self.is_chosen and self.chosen_dz_sample is not None:
            return self.chosen_dz_sample
        return self.interpolated_dz_sample

    def has_position(self) -> bool:
        return self.final_dz_sample is not None


@dataclass
class Session:
    """Whole-app state owned by the main window."""

    frames: list[Frame] = field(default_factory=list)
    optics: Optional[OpticalParams] = None
    sweep: Optional[SweepParams] = None
    input_format: Optional[str] = None
    distances_path: Optional[str] = None
    # Position of the current frame within ``frames`` — NOT its frame index.
    # The two differ on a partial load, where ``frames[0].index`` may be 51.
    current_index: int = 0
    smoothing_factor: float = 0.0
    interp_method: str = "spline"
    # Opaque loader produced by io.inputs; loader(frame) -> (phase, amp).
    loader: Any = None

    @property
    def current_frame(self) -> Optional[Frame]:
        if 0 <= self.current_index < len(self.frames):
            return self.frames[self.current_index]
        return None

    def chosen_frames(self) -> list[Frame]:
        return [f for f in self.frames if f.is_chosen]

    def position_of(self, frame_index: int) -> Optional[int]:
        """List position of the frame numbered ``frame_index``, if loaded."""
        for pos, f in enumerate(self.frames):
            if f.index == frame_index:
                return pos
        return None

    @property
    def frame_indices(self) -> list[int]:
        return [f.index for f in self.frames]
