import os

import numpy as np
import pytest
import tifffile

from wavefront_refocus.config import (
    FMT_INTENSITY_OPD,
    FMT_INTENSITY_PHASE,
    FMT_STACK,
    FMT_STACK_FOLDER,
)
from wavefront_refocus.core.model import Frame, OpticalParams, Session
from wavefront_refocus.io import inputs
from wavefront_refocus.io.export import export_folder, frame_basename
from wavefront_refocus.vendor.utils_images import save_stack


def _optics():
    return OpticalParams(pixel_pitch=5.86e-6, magnification=20.0,
                         wavelength=660e-9, NA=0.8)


def _session(li):
    frames = [Frame(index=i, source=s) for i, s in zip(li.indices, li.sources)]
    return Session(frames=frames, optics=_optics(),
                   input_format=li.input_format, loader=li.loader)


@pytest.fixture
def named_pair_dirs(tmp_path, synthetic_wavefront):
    """Intensity/phase folders with distinctive, non-indexed file names."""
    phase, amp = synthetic_wavefront
    idir, pdir = tmp_path / "I", tmp_path / "P"
    idir.mkdir(); pdir.mkdir()
    names = ["acqA_t000", "acqA_t001", "acqA_t002"]
    for n in names:
        tifffile.imwrite(str(idir / f"{n}_int.tif"), amp)
        tifffile.imwrite(str(pdir / f"{n}_ph.tif"), phase)
    return str(idir), str(pdir), names


# ── frame_basename ────────────────────────────────────────────────────────


def test_basename_indexed_by_default():
    f = Frame(index=7, source=("/data/I/foo.tif", "/data/P/bar.tif"))
    assert frame_basename(f, False) == "frame_0007"


def test_basename_keeps_original_per_side():
    f = Frame(index=7, source=("/data/I/foo.tif", "/data/P/bar.tif"))
    assert frame_basename(f, True) == "foo"
    assert frame_basename(f, True, "secondary") == "bar"


def test_basename_plain_path_source():
    f = Frame(index=3, source="/data/stacks/cell_04.tif")
    assert frame_basename(f, True) == "cell_04"
    # no secondary file -> falls back to the primary name
    assert frame_basename(f, True, "secondary") == "cell_04"


def test_basename_stack_slice_falls_back_to_index():
    """A (path, time_index) source has no per-frame filename."""
    f = Frame(index=5, source=("/data/movie.tif", 5))
    assert frame_basename(f, True) == "movie"
    assert frame_basename(f, True, "secondary") == "movie"


def test_basename_handles_missing_source():
    assert frame_basename(Frame(index=2, source=None), True) == "frame_0002"


# ── exporting with original names ─────────────────────────────────────────


def test_export_pairs_keeps_both_sides_names(named_pair_dirs, tmp_path):
    idir, pdir, names = named_pair_dirs
    li = inputs.open_intensity_phase(idir, pdir)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_INTENSITY_PHASE, str(out),
                  keep_original_names=True)
    assert sorted(os.listdir(out / "Intensity")) == [f"{n}_int.tif" for n in names]
    assert sorted(os.listdir(out / "Phase")) == [f"{n}_ph.tif" for n in names]


def test_export_pairs_indexed_by_default(named_pair_dirs, tmp_path):
    idir, pdir, _ = named_pair_dirs
    li = inputs.open_intensity_phase(idir, pdir)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_INTENSITY_PHASE, str(out))
    assert sorted(os.listdir(out / "Intensity")) == [
        "frame_0000.tif", "frame_0001.tif", "frame_0002.tif"
    ]


def test_export_opd_format_keeps_names(named_pair_dirs, tmp_path):
    idir, pdir, names = named_pair_dirs
    li = inputs.open_intensity_phase(idir, pdir)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_INTENSITY_OPD, str(out),
                  keep_original_names=True)
    assert sorted(os.listdir(out / "OPD")) == [f"{n}_ph.tif" for n in names]


def test_export_stack_folder_keeps_names(tmp_path, synthetic_wavefront):
    phase, amp = synthetic_wavefront
    src = tmp_path / "src"
    src.mkdir()
    names = ["sampleX", "sampleY"]
    for n in names:
        save_stack(str(src / f"{n}.tif"),
                   np.stack([phase, amp], axis=-1)[None, ...],
                   wavelengths=(660.0,))
    li = inputs.open_stack_folder(str(src))
    out = tmp_path / "out"
    export_folder(_session(li), FMT_STACK_FOLDER, str(out),
                  keep_original_names=True)
    written = sorted(f for f in os.listdir(out) if f.endswith(".tif"))
    assert written == ["sampleX.tif", "sampleY.tif"]


def test_export_single_stack_keeps_source_name(stack_file, tmp_path):
    path, _ = stack_file          # conftest writes "wf.tif"
    li = inputs.open_stack(path)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_STACK, str(out), keep_original_names=True)
    assert "wf.tif" in os.listdir(out)


def test_export_single_stack_default_name(stack_file, tmp_path):
    path, _ = stack_file
    li = inputs.open_stack(path)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_STACK, str(out))
    assert "wavefront_stack.tif" in os.listdir(out)


def test_stack_slices_fall_back_to_indexed_names(stack_file, tmp_path):
    """Every slice of one stack shares a source path; names must not collide."""
    path, T = stack_file
    li = inputs.open_stack(path)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_STACK_FOLDER, str(out),
                  keep_original_names=True)
    written = sorted(f for f in os.listdir(out) if f.endswith(".tif"))
    # 3 slices of "wf.tif" -> wf, wf_2, wf_3 (no overwrite, nothing lost)
    assert len(written) == T
    assert written == ["wf.tif", "wf_2.tif", "wf_3.tif"]


def test_duplicate_names_are_disambiguated(tmp_path, synthetic_wavefront):
    """Two frames whose sources share a basename must not overwrite."""
    phase, amp = synthetic_wavefront
    out = tmp_path / "out"
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for folder in (a, b):
        save_stack(str(folder / "same.tif"),
                   np.stack([phase, amp], axis=-1)[None, ...],
                   wavelengths=(660.0,))

    def loader(src):
        from wavefront_refocus.vendor.utils_images import load_wavefront_tif
        p, am, _ = load_wavefront_tif(src, 0)
        return p, am

    sess = Session(
        frames=[Frame(index=0, source=str(a / "same.tif")),
                Frame(index=1, source=str(b / "same.tif"))],
        optics=_optics(), input_format=FMT_STACK_FOLDER, loader=loader,
    )
    export_folder(sess, FMT_STACK_FOLDER, str(out), keep_original_names=True)
    written = sorted(f for f in os.listdir(out) if f.endswith(".tif"))
    assert written == ["same.tif", "same_2.tif"]


def test_partial_load_keeps_names_of_its_own_frames(named_pair_dirs, tmp_path):
    """A chunk exports the names of the frames it actually holds."""
    idir, pdir, names = named_pair_dirs
    li = inputs.open_intensity_phase(idir, pdir, 1, 2)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_INTENSITY_PHASE, str(out),
                  keep_original_names=True)
    assert sorted(os.listdir(out / "Intensity")) == [
        f"{names[1]}_int.tif", f"{names[2]}_int.tif"
    ]


# ── exported phase must be unwrapped ──────────────────────────────────────


@pytest.fixture
def steep_wavefront():
    """A wavefront with peaks well above pi, so wrapping is visible."""
    N = 96
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    phase = np.zeros((N, N))
    for cy, cx, h, s in [(40, 44, 9.0, 4), (44, 56, 7.0, 3), (60, 40, 8.0, 4)]:
        phase += h * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2)))
    return phase.astype(np.float32), np.ones((N, N), np.float32)


def _stack_session(tmp_path, phase, amp, name="cell"):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    save_stack(str(src / f"{name}.tif"),
               np.stack([phase, amp], axis=-1)[None, ...], wavelengths=(660.0,))
    li = inputs.open_stack_folder(str(src))
    return _session(li), li


def test_exported_stack_phase_is_unwrapped(tmp_path, steep_wavefront):
    """Regression: stack exports wrote np.angle(), clamping phase to [-pi, pi].

    A bead taller than pi then folded back on itself and showed as a dark
    speck at its own peak, unlike the (unwrapped) view in the app.
    """
    from wavefront_refocus.vendor.utils_images import load_wavefront_tif

    phase, amp = steep_wavefront
    sess, _ = _stack_session(tmp_path, phase, amp)
    out = tmp_path / "out"
    export_folder(sess, FMT_STACK_FOLDER, str(out))

    written, _, _ = load_wavefront_tif(str(out / "frame_0000.tif"), 0)
    assert np.ptp(written) > 2 * np.pi, "phase is still wrapped into [-pi, pi]"
    # no pixel at a bright peak sits at the global minimum (the dark specks)
    bright = written > np.percentile(written, 99.0)
    assert not (bright & (written < written.min() + 0.6)).any()


def test_exported_single_stack_phase_is_unwrapped(tmp_path, steep_wavefront):
    from wavefront_refocus.vendor.utils_images import load_wavefront_tif

    phase, amp = steep_wavefront
    sess, _ = _stack_session(tmp_path, phase, amp)
    out = tmp_path / "out"
    export_folder(sess, FMT_STACK, str(out))
    written, _, _ = load_wavefront_tif(str(out / "wavefront_stack.tif"), 0)
    assert np.ptp(written) > 2 * np.pi


def test_exported_phase_folder_is_unwrapped(tmp_path, steep_wavefront):
    """The intensity+phase layout writes phase too; it must also be unwrapped."""
    import tifffile as tf

    phase, amp = steep_wavefront
    idir, pdir = tmp_path / "I", tmp_path / "P"
    idir.mkdir(); pdir.mkdir()
    tf.imwrite(str(idir / "f0.tif"), amp)
    tf.imwrite(str(pdir / "f0.tif"), phase)
    li = inputs.open_intensity_phase(str(idir), str(pdir))
    out = tmp_path / "out"
    export_folder(_session(li), FMT_INTENSITY_PHASE, str(out))
    written = tf.imread(str(out / "Phase" / "frame_0000.tif"))
    assert np.ptp(written) > 2 * np.pi


def test_exported_phase_matches_what_the_app_displays(tmp_path, steep_wavefront):
    """The written phase must describe the same OPD the phase view showed."""
    from wavefront_refocus.vendor.utils_images import load_wavefront_tif
    from wavefront_refocus.vendor.utils_propag import _opd_from_field
    from wavefront_refocus.core.engine import refocus_one

    phase, amp = steep_wavefront
    sess, _ = _stack_session(tmp_path, phase, amp)
    sess.frames[0].is_chosen = True
    sess.frames[0].chosen_dz_sample = 2e-6
    out = tmp_path / "out"
    export_folder(sess, FMT_STACK_FOLDER, str(out))

    written, _, _ = load_wavefront_tif(str(out / "frame_0000.tif"), 0)
    opd_written = written * sess.optics.wavelength_nm / (2 * np.pi)
    opd_written -= opd_written.mean()

    shown = _opd_from_field(
        refocus_one(phase, amp, 2e-6, sess.optics), sess.optics.wavelength
    )
    assert np.allclose(opd_written, shown, atol=1e-3)


def test_exported_phase_still_reloads(tmp_path, steep_wavefront):
    """Unwrapped phase must round-trip to the field the exporter produced.

    Writing unwrapped rather than wrapped phase must not change what reloading
    reconstructs: the two differ by multiples of 2*pi, so exp(1j*phase) — and
    therefore every downstream propagation — is identical.
    """
    from wavefront_refocus.core.engine import refocus_one

    phase, amp = steep_wavefront
    sess, _ = _stack_session(tmp_path, phase, amp)
    sess.frames[0].is_chosen = True
    sess.frames[0].chosen_dz_sample = 2e-6
    out = tmp_path / "out"
    export_folder(sess, FMT_STACK_FOLDER, str(out))

    li2 = inputs.open_stack_folder(str(out))
    ph2, am2 = li2.loader(li2.sources[0])

    exported_field = refocus_one(phase, amp, 2e-6, sess.optics)
    # Reloaded phase is unwrapped, so compare on the unit circle (mod 2*pi).
    assert np.allclose(np.exp(1j * ph2), np.exp(1j * np.angle(exported_field)),
                       atol=1e-4)
    assert np.allclose(am2, np.abs(exported_field), atol=1e-4)


def test_distances_json_still_written_with_original_names(
    named_pair_dirs, tmp_path
):
    import json

    idir, pdir, _ = named_pair_dirs
    li = inputs.open_intensity_phase(idir, pdir)
    out = tmp_path / "out"
    export_folder(_session(li), FMT_INTENSITY_PHASE, str(out),
                  keep_original_names=True)
    data = json.load(open(out / "Reconstruction_Distances.json"))
    # keyed by frame index regardless of the file naming scheme
    assert sorted(data, key=int) == ["0", "1", "2"]
