import numpy as np
import pytest
import tifffile

from wavefront_refocus.io import inputs


def test_natural_sort(tmp_path):
    for name in ["frame_10.tif", "frame_2.tif", "frame_1.tif"]:
        tifffile.imwrite(str(tmp_path / name), np.zeros((4, 4), np.float32))
    files = inputs.list_tifs(str(tmp_path))
    assert [f.split("_")[-1] for f in files] == ["1.tif", "2.tif", "10.tif"]


def test_to_square_center_crop():
    img = np.arange(6 * 10, dtype=np.float32).reshape(6, 10)
    sq = inputs.to_square(img)
    assert sq.shape == (6, 6)


def test_open_stack(stack_file):
    path, T = stack_file
    li = inputs.open_stack(path)
    assert li.n_frames == T
    phase, amp = li.loader(li.sources[0])
    assert phase.shape == amp.shape
    assert phase.ndim == 2 and phase.shape[0] == phase.shape[1]


def test_open_intensity_phase(intensity_phase_dirs):
    idir, pdir = intensity_phase_dirs
    li = inputs.open_intensity_phase(idir, pdir)
    assert li.n_frames == 3
    phase, amp = li.loader(li.sources[2])
    # frame index 2 was scaled by 3
    assert np.allclose(phase, phase)  # loads without error, square
    assert phase.shape == amp.shape


def test_open_intensity_opd(tmp_path):
    N = 32
    amp = np.ones((N, N), np.float32)
    opd_nm = np.linspace(0, 100, N * N, dtype=np.float32).reshape(N, N)
    idir = tmp_path / "I"
    odir = tmp_path / "O"
    idir.mkdir(); odir.mkdir()
    tifffile.imwrite(str(idir / "f0.tif"), amp)
    tifffile.imwrite(str(odir / "f0.tif"), opd_nm)
    li = inputs.open_intensity_opd(str(idir), str(odir), wavelength_nm=660.0)
    phase, a = li.loader(li.sources[0])
    # phase = opd / wavelength * 2pi
    expected = opd_nm / 660.0 * 2 * np.pi
    assert np.allclose(phase, expected, atol=1e-4)


# ── partial loading (frame ranges) ────────────────────────────────────────


def test_resolve_range_defaults_to_everything():
    assert inputs.resolve_range(480) == (0, 479)
    assert inputs.resolve_range(480, 51, None) == (51, 479)
    assert inputs.resolve_range(480, None, 50) == (0, 50)


def test_resolve_range_single_frame():
    assert inputs.resolve_range(480, 7, 7) == (7, 7)


@pytest.mark.parametrize(
    "first,last,match",
    [
        (100, 50, "after last frame"),
        (480, 500, "past the end"),
        (0, 480, "past the end"),
        (-1, 5, "non-negative"),
    ],
)
def test_resolve_range_rejects_bad_input(first, last, match):
    with pytest.raises(ValueError, match=match):
        inputs.resolve_range(480, first, last)


def test_resolve_range_empty_sequence():
    with pytest.raises(ValueError, match="empty"):
        inputs.resolve_range(0)


def test_partial_load_keeps_original_indices(intensity_phase_dirs):
    idir, pdir = intensity_phase_dirs           # 3 frames: 0, 1, 2
    li = inputs.open_intensity_phase(idir, pdir, 1, 2)
    assert li.n_frames == 2
    assert li.indices == [1, 2]                 # not renumbered to 0, 1
    assert li.total_frames == 3
    assert li.is_partial


def test_full_load_is_not_partial(intensity_phase_dirs):
    idir, pdir = intensity_phase_dirs
    li = inputs.open_intensity_phase(idir, pdir)
    assert li.indices == [0, 1, 2]
    assert li.total_frames == 3
    assert not li.is_partial


def test_chunks_tile_the_sequence_without_gaps(intensity_phase_dirs):
    """0–0 then 1–2 must cover every frame exactly once, in order."""
    idir, pdir = intensity_phase_dirs
    a = inputs.open_intensity_phase(idir, pdir, 0, 0)
    b = inputs.open_intensity_phase(idir, pdir, 1, 2)
    assert a.indices + b.indices == [0, 1, 2]
    assert a.sources + b.sources == inputs.open_intensity_phase(idir, pdir).sources


def test_partial_load_reads_the_right_frame(intensity_phase_dirs):
    """The chunk's source must be the same file the full load has there."""
    idir, pdir = intensity_phase_dirs
    full = inputs.open_intensity_phase(idir, pdir)
    chunk = inputs.open_intensity_phase(idir, pdir, 2, 2)
    assert chunk.sources[0] == full.sources[2]
    assert np.allclose(chunk.loader(chunk.sources[0])[0],
                       full.loader(full.sources[2])[0])


def test_partial_stack_load(stack_file):
    path, T = stack_file
    li = inputs.open_stack(path, 1, 2)
    assert li.n_frames == 2
    assert li.indices == [1, 2]
    assert li.total_frames == T
    # sources carry the original time-axis positions
    assert [t for _, t in li.sources] == [1, 2]


def test_partial_stack_folder_load(tmp_path):
    from wavefront_refocus.vendor.utils_images import save_stack

    N = 16
    for i in range(5):
        frame = np.stack(
            [np.full((N, N), float(i), np.float32), np.ones((N, N), np.float32)],
            axis=-1,
        )[None, ...]
        save_stack(str(tmp_path / f"f{i}.tif"), frame, wavelengths=(660.0,))
    li = inputs.open_stack_folder(str(tmp_path), 2, 3)
    assert li.indices == [2, 3]
    assert li.total_frames == 5
    phase, _ = li.loader(li.sources[0])
    assert np.allclose(phase, 2.0)   # really frame 2, not frame 0


def test_out_of_range_chunk_rejected(intensity_phase_dirs):
    idir, pdir = intensity_phase_dirs
    with pytest.raises(ValueError, match="past the end"):
        inputs.open_intensity_phase(idir, pdir, 0, 99)


def test_loaded_input_rejects_mismatched_indices():
    with pytest.raises(ValueError, match="counts differ"):
        inputs.LoadedInput("fmt", ["a", "b"], lambda s: s, [0])
