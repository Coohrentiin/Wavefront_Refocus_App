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
    """Mirror of the dict returned by ``compute_opd_stack``."""

    opd_stack: np.ndarray        # (Z, H, W) float32 nm, zero-mean per slice
    dz_sample_array: np.ndarray  # (Z,) m, sample space, centred on 0
    dz_image_array: np.ndarray   # (Z,) m, image space
    z0_index: int                # index of the centre (dz == 0) plane
    backend: str
    N: int

    @classmethod
    def from_dict(cls, d: dict) -> "PropagationResult":
        return cls(
            opd_stack=d["opd_stack"],
            dz_sample_array=np.asarray(d["dz_sample_array"]),
            dz_image_array=np.asarray(d["dz_image_array"]),
            z0_index=int(d["z0_index"]),
            backend=str(d["backend"]),
            N=int(d["N"]),
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
    current_index: int = 0
    smoothing_factor: float = 0.0
    # Opaque loader produced by io.inputs; loader(frame) -> (phase, amp).
    loader: Any = None

    @property
    def current_frame(self) -> Optional[Frame]:
        if 0 <= self.current_index < len(self.frames):
            return self.frames[self.current_index]
        return None

    def chosen_frames(self) -> list[Frame]:
        return [f for f in self.frames if f.is_chosen]
