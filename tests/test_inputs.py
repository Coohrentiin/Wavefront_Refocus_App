import numpy as np
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
