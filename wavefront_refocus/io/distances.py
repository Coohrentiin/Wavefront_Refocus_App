"""Read/write the Reconstruction_Distances.json (absolute, image space).

Schema: ``{ "<frame_index>": { "<wavelength_nm>": distance_m, ... }, ... }``.

A long acquisition is often refocused in chunks (frames 0–35, 36–70, …), each
exporting its own JSON. :func:`merge_distances` reassembles those per-chunk
files into one table covering the whole sequence, so the full range can then be
loaded and processed together.

:func:`transfer_distances` goes the other way: it takes refocusing recorded in
one pair of JSONs (an input and a target) and replays it onto the frames
currently loaded.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from ..core.model import Frame, OpticalParams
from ..core.optics import base_dz_image_for_frame, image_to_sample, sample_to_image


def load_distances(path: str) -> Dict[str, Dict[str, float]]:
    """Load the raw distances JSON."""
    with open(path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"{os.path.basename(path)}: expected a JSON object mapping frame "
            f"index -> wavelength table, got {type(data).__name__}"
        )
    return data


@dataclass
class MergeReport:
    """What :func:`merge_distances` did, for reporting back to the user."""

    merged: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Frames present in more than one file, with identical values.
    duplicates: List[int] = field(default_factory=list)
    # Frames present in more than one file with DIFFERENT values; the last
    # file listed wins. These are the ones worth warning about.
    conflicts: List[int] = field(default_factory=list)
    # (path, n_frames) per source file, in the order merged.
    sources: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.merged)

    @property
    def frame_indices(self) -> List[int]:
        return sorted(int(k) for k in self.merged)

    def missing(self, expected: Iterable[int]) -> List[int]:
        """Frames in ``expected`` that no source file covered."""
        have = set(self.merged)
        return [i for i in expected if str(i) not in have]

    def gaps(self) -> List[Tuple[int, int]]:
        """Inclusive ``(lo, hi)`` runs missing between the merged extremes.

        A gap means some chunk of the acquisition was never exported — those
        frames would silently fall back to a 0.0 distance when loaded.
        """
        idx = self.frame_indices
        if len(idx) < 2:
            return []
        out = []
        for a, b in zip(idx, idx[1:]):
            if b > a + 1:
                out.append((a + 1, b - 1))
        return out


def merge_distances(paths: Sequence[str]) -> MergeReport:
    """Combine several per-chunk distances JSONs into one table.

    Files are merged in the order given; where two files carry the same frame,
    the later one wins. Identical repeats are recorded as ``duplicates`` and
    genuine disagreements as ``conflicts`` so the caller can surface them —
    overlapping chunks are normal, silently picking one of two different
    answers is not.
    """
    if not paths:
        raise ValueError("no distances files to merge")

    report = MergeReport()
    for path in paths:
        raw = load_distances(path)
        report.sources.append((path, len(raw)))
        for key, entry in raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{os.path.basename(path)}: frame key {key!r} is not an integer"
                )
            if key in report.merged:
                if report.merged[key] == entry:
                    report.duplicates.append(idx)
                else:
                    report.conflicts.append(idx)
            report.merged[key] = entry

    report.duplicates = sorted(set(report.duplicates))
    report.conflicts = sorted(set(report.conflicts))
    return report


def write_raw_distances(path: str, raw: Dict[str, Dict[str, float]]) -> None:
    """Write a raw distances table, frame keys in numeric order."""
    ordered = {k: raw[k] for k in sorted(raw, key=int)}
    with open(path, "w") as fh:
        json.dump(ordered, fh, indent=4)


# ── distance transfer ─────────────────────────────────────────────────────


@dataclass
class TransferReport:
    """What :func:`transfer_distances` changed, for reporting to the user."""

    # frame index -> applied sample-space displacement (relative to base).
    applied: Dict[int, float] = field(default_factory=dict)
    # Loaded frames the target JSON did not cover; left untouched.
    uncovered: List[int] = field(default_factory=list)
    # Target frames that aren't in the loaded sequence; ignored.
    unused: List[int] = field(default_factory=list)
    # Frames the input JSON lacked, so the delta used input = 0.
    assumed_zero_input: List[int] = field(default_factory=list)
    # Frames whose chosen/interpolated position was overwritten.
    overwritten: List[int] = field(default_factory=list)

    @property
    def n_applied(self) -> int:
        return len(self.applied)

    def shift_range_m(self) -> Tuple[float, float]:
        """Smallest and largest applied displacement (sample space, m)."""
        if not self.applied:
            return 0.0, 0.0
        vals = list(self.applied.values())
        return min(vals), max(vals)


def transfer_distances(
    frames: List[Frame],
    target: Dict[str, Dict[str, float]],
    optics: OpticalParams,
    input_raw: Dict[str, Dict[str, float]] | None = None,
    *,
    apply_to_frames: bool = True,
) -> TransferReport:
    """Carry distances from a ``target`` JSON onto the loaded ``frames``.

    The shift applied to each frame is the **difference** between the target
    and input tables, ``target[i] - input[i]`` (image space), converted to a
    sample-space displacement and stored as that frame's *chosen* plane. A
    frame missing from ``input_raw`` — or an absent/empty input table, which is
    the usual case when the dataset was opened without distances — contributes
    an input of 0, so the transfer reduces to applying the target outright.

    Working in deltas rather than absolutes means the refocusing encoded in the
    target survives even when the current sequence sits on a different base:
    what moves across is how far each frame was displaced, not where it landed.

    Frames the target does not mention are left exactly as they are and listed
    in the report. Pass ``apply_to_frames=False`` to preview without mutating.
    """
    if not frames:
        raise ValueError("no frames loaded to transfer onto")
    if not target:
        raise ValueError("the target distances table is empty")

    input_raw = input_raw or {}
    wl_nm = optics.wavelength_nm
    M = optics.magnification
    report = TransferReport()

    loaded = {f.index for f in frames}
    report.unused = sorted(
        int(k) for k in target if int(k) not in loaded
    )

    for f in frames:
        key = str(f.index)
        entry = target.get(key)
        if entry is None:
            report.uncovered.append(f.index)
            continue

        target_image = base_dz_image_for_frame(entry, wl_nm)
        in_entry = input_raw.get(key)
        if in_entry is None:
            if input_raw:
                # Input table present but silent on this frame: treat as 0 and
                # say so, since it changes what the delta means.
                report.assumed_zero_input.append(f.index)
            input_image = 0.0
        else:
            input_image = base_dz_image_for_frame(in_entry, wl_nm)

        dz_sample = image_to_sample(target_image - input_image, M)
        report.applied[f.index] = dz_sample

        if f.has_position():
            report.overwritten.append(f.index)
        if apply_to_frames:
            f.chosen_dz_sample = dz_sample
            f.interpolated_dz_sample = None
            f.is_chosen = True

    return report


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
