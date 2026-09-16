import json

import pytest

from wavefront_refocus.core.model import Frame, OpticalParams
from wavefront_refocus.io import distances as dist_io


def _optics():
    return OpticalParams(pixel_pitch=5.86e-6, magnification=20.0, wavelength=660e-9, NA=0.8)


def test_apply_and_write_roundtrip(tmp_path):
    raw = {"0": {"660.000": -0.0155}, "1": {"660.000": -0.0150}}
    frames = [Frame(index=0, source=None), Frame(index=1, source=None)]
    optics = _optics()
    dist_io.apply_base_distances(frames, raw, optics.wavelength_nm)
    assert frames[0].base_dz_image == -0.0155

    # Choose a plane on frame 0: +2 um sample -> +2e-6 * 400 image.
    frames[0].chosen_dz_sample = 2e-6
    frames[0].is_chosen = True

    out = tmp_path / "out.json"
    dist_io.write_distances(str(out), frames, optics)
    data = json.load(open(out))
    assert data["0"]["660.000"] == -0.0155 + 2e-6 * 400
    # frame 1 had no position -> base distance preserved
    assert data["1"]["660.000"] == -0.0150


def test_missing_frame_defaults_zero(recwarn):
    raw = {"0": {"660.000": -0.01}}
    val = dist_io.base_dz_for_frame(raw, 5, 660.0)
    assert val == 0.0
    assert len(recwarn) >= 1


def test_nearest_wavelength_rebind():
    raw = {"0": {"650.000": -0.01, "660.000": -0.02}}
    frames = [Frame(index=0, source=None)]
    dist_io.apply_base_distances(frames, raw, 659.0)
    assert frames[0].base_dz_image == -0.02
    dist_io.apply_base_distances(frames, raw, 651.0)
    assert frames[0].base_dz_image == -0.01


# ── merging per-chunk exports ─────────────────────────────────────────────


def _write(tmp_path, name, mapping):
    """Write a chunk JSON: {frame_index: distance_m} at 660 nm."""
    p = tmp_path / name
    p.write_text(json.dumps({str(k): {"660.000": v} for k, v in mapping.items()}))
    return str(p)


def test_merge_consecutive_chunks(tmp_path):
    a = _write(tmp_path, "a.json", {i: i * 1e-3 for i in range(0, 36)})
    b = _write(tmp_path, "b.json", {i: i * 1e-3 for i in range(36, 71)})
    rep = dist_io.merge_distances([a, b])
    assert rep.n_frames == 71
    assert rep.frame_indices == list(range(71))
    assert not rep.conflicts and not rep.duplicates
    assert rep.gaps() == []
    assert rep.merged["70"]["660.000"] == pytest.approx(70e-3)


def test_merge_single_file_is_identity(tmp_path):
    a = _write(tmp_path, "a.json", {0: 1e-3, 1: 2e-3})
    rep = dist_io.merge_distances([a])
    assert rep.merged == dist_io.load_distances(a)


def test_merge_detects_gap(tmp_path):
    a = _write(tmp_path, "a.json", {0: 0.0, 1: 1e-3})
    b = _write(tmp_path, "b.json", {5: 5e-3, 8: 8e-3})
    rep = dist_io.merge_distances([a, b])
    assert rep.gaps() == [(2, 4), (6, 7)]


def test_merge_identical_overlap_is_a_duplicate_not_a_conflict(tmp_path):
    a = _write(tmp_path, "a.json", {0: 0.0, 1: 1e-3})
    b = _write(tmp_path, "b.json", {1: 1e-3, 2: 2e-3})
    rep = dist_io.merge_distances([a, b])
    assert rep.duplicates == [1]
    assert rep.conflicts == []


def test_merge_conflicting_overlap_last_file_wins(tmp_path):
    a = _write(tmp_path, "a.json", {1: 1e-3})
    b = _write(tmp_path, "b.json", {1: 9e-3})
    rep = dist_io.merge_distances([a, b])
    assert rep.conflicts == [1]
    assert rep.merged["1"]["660.000"] == pytest.approx(9e-3)
    # order matters: reversing the list flips the winner
    rep2 = dist_io.merge_distances([b, a])
    assert rep2.merged["1"]["660.000"] == pytest.approx(1e-3)


def test_merge_out_of_order_files(tmp_path):
    """Chunks may be selected in any order; the result is still sorted."""
    a = _write(tmp_path, "a.json", {i: float(i) for i in range(0, 3)})
    b = _write(tmp_path, "b.json", {i: float(i) for i in range(3, 6)})
    rep = dist_io.merge_distances([b, a])
    assert rep.frame_indices == [0, 1, 2, 3, 4, 5]


def test_merge_missing_reports_uncovered_frames(tmp_path):
    a = _write(tmp_path, "a.json", {0: 0.0, 2: 2e-3})
    rep = dist_io.merge_distances([a])
    assert rep.missing([0, 1, 2, 3]) == [1, 3]


def test_merge_requires_files():
    with pytest.raises(ValueError, match="no distances files"):
        dist_io.merge_distances([])


def test_merge_rejects_non_object_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        dist_io.merge_distances([str(p)])


def test_merge_rejects_non_integer_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"frame_one": {"660.000": 1.0}}))
    with pytest.raises(ValueError, match="not an integer"):
        dist_io.merge_distances([str(p)])


def test_write_raw_distances_sorts_numerically(tmp_path):
    raw = {"10": {"660.000": 1.0}, "2": {"660.000": 2.0}, "1": {"660.000": 3.0}}
    out = tmp_path / "merged.json"
    dist_io.write_raw_distances(str(out), raw)
    # keys must be in numeric, not lexicographic, order
    assert list(json.load(open(out))) == ["1", "2", "10"]


# ── distance transfer ─────────────────────────────────────────────────────


def _tbl(mapping, wl="660.000"):
    """A raw distances table from {frame_index: distance_m}."""
    return {str(k): {wl: v} for k, v in mapping.items()}


def _frames(indices, bases=None):
    fs = [Frame(index=i, source=None) for i in indices]
    if bases:
        for f, b in zip(fs, bases):
            f.base_dz_image = b
    return fs


def test_transfer_applies_delta_as_chosen_plane():
    # M=20 -> M^2 = 400, so an image-space delta of 400e-6 is +1 um sample.
    fs = _frames([0, 1])
    rep = dist_io.transfer_distances(
        fs, _tbl({0: 400e-6, 1: 800e-6}), _optics(), _tbl({0: 0.0, 1: 400e-6})
    )
    assert rep.n_applied == 2
    assert fs[0].chosen_dz_sample == pytest.approx(1e-6)
    assert fs[1].chosen_dz_sample == pytest.approx(1e-6)
    assert all(f.is_chosen for f in fs)
    assert all(f.interpolated_dz_sample is None for f in fs)


def test_transfer_without_input_applies_target_outright():
    fs = _frames([0])
    dist_io.transfer_distances(fs, _tbl({0: 400e-6}), _optics(), None)
    assert fs[0].chosen_dz_sample == pytest.approx(1e-6)


def test_empty_input_table_is_same_as_none():
    a, b = _frames([0]), _frames([0])
    dist_io.transfer_distances(a, _tbl({0: 400e-6}), _optics(), None)
    dist_io.transfer_distances(b, _tbl({0: 400e-6}), _optics(), {})
    assert a[0].chosen_dz_sample == pytest.approx(b[0].chosen_dz_sample)


def test_transfer_is_independent_of_current_base():
    """The delta carries over even when the frames sit on a different base."""
    fs = _frames([0, 1], bases=[-0.5, 0.25])
    dist_io.transfer_distances(
        fs, _tbl({0: 400e-6, 1: 400e-6}), _optics(), _tbl({0: 0.0, 1: 0.0})
    )
    assert fs[0].chosen_dz_sample == pytest.approx(1e-6)
    assert fs[1].chosen_dz_sample == pytest.approx(1e-6)
    # bases untouched; only the relative displacement was set
    assert fs[0].base_dz_image == -0.5 and fs[1].base_dz_image == 0.25


def test_identical_input_and_target_is_a_no_op_shift():
    fs = _frames([0, 1])
    rep = dist_io.transfer_distances(
        fs, _tbl({0: 1e-3, 1: 2e-3}), _optics(), _tbl({0: 1e-3, 1: 2e-3})
    )
    assert rep.n_applied == 2
    assert fs[0].chosen_dz_sample == pytest.approx(0.0)
    assert rep.shift_range_m() == (pytest.approx(0.0), pytest.approx(0.0))


def test_negative_delta():
    fs = _frames([0])
    dist_io.transfer_distances(fs, _tbl({0: 0.0}), _optics(), _tbl({0: 400e-6}))
    assert fs[0].chosen_dz_sample == pytest.approx(-1e-6)


def test_uncovered_frames_left_untouched():
    fs = _frames([0, 1, 2])
    fs[2].interpolated_dz_sample = 7e-6      # pre-existing position
    rep = dist_io.transfer_distances(fs, _tbl({0: 400e-6}), _optics())
    assert rep.uncovered == [1, 2]
    assert fs[1].chosen_dz_sample is None and not fs[1].is_chosen
    assert fs[2].interpolated_dz_sample == 7e-6   # preserved
    assert not fs[2].is_chosen


def test_target_frames_not_loaded_are_reported():
    fs = _frames([0, 1])
    rep = dist_io.transfer_distances(fs, _tbl({0: 0.0, 1: 0.0, 99: 5e-3}), _optics())
    assert rep.unused == [99]


def test_frames_missing_from_input_are_reported():
    fs = _frames([0, 1])
    rep = dist_io.transfer_distances(
        fs, _tbl({0: 400e-6, 1: 400e-6}), _optics(), _tbl({0: 0.0})
    )
    assert rep.assumed_zero_input == [1]
    assert fs[1].chosen_dz_sample == pytest.approx(1e-6)


def test_overwritten_positions_are_reported():
    fs = _frames([0, 1])
    fs[0].is_chosen = True
    fs[0].chosen_dz_sample = 3e-6
    rep = dist_io.transfer_distances(fs, _tbl({0: 0.0, 1: 0.0}), _optics())
    assert rep.overwritten == [0]
    assert fs[0].chosen_dz_sample == pytest.approx(0.0)   # replaced


def test_preview_mode_does_not_mutate():
    fs = _frames([0])
    rep = dist_io.transfer_distances(
        fs, _tbl({0: 400e-6}), _optics(), apply_to_frames=False
    )
    assert rep.applied[0] == pytest.approx(1e-6)
    assert fs[0].chosen_dz_sample is None and not fs[0].is_chosen


def test_transfer_uses_nearest_wavelength():
    fs = _frames([0])
    target = {"0": {"650.000": 400e-6, "660.000": 800e-6}}
    dist_io.transfer_distances(fs, target, _optics())   # optics is 660 nm
    assert fs[0].chosen_dz_sample == pytest.approx(2e-6)


def test_transfer_requires_frames_and_target():
    with pytest.raises(ValueError, match="no frames"):
        dist_io.transfer_distances([], _tbl({0: 0.0}), _optics())
    with pytest.raises(ValueError, match="target .* is empty"):
        dist_io.transfer_distances(_frames([0]), {}, _optics())


def test_transfer_then_export_round_trip(tmp_path):
    """Transferred planes must survive a write/reload as absolute distances."""
    optics = _optics()
    fs = _frames([0, 1], bases=[1e-3, 2e-3])
    dist_io.transfer_distances(fs, _tbl({0: 400e-6, 1: 800e-6}), optics)
    out = tmp_path / "after.json"
    dist_io.write_distances(str(out), fs, optics)
    data = json.load(open(out))
    # base + delta(image space) == base + target, since input was 0
    assert data["0"]["660.000"] == pytest.approx(1e-3 + 400e-6)
    assert data["1"]["660.000"] == pytest.approx(2e-3 + 800e-6)


def test_merged_file_round_trips_into_a_session(tmp_path):
    """The end-to-end point: merge chunk exports, reload as one sequence."""
    a = _write(tmp_path, "a.json", {i: i * 1e-3 for i in range(0, 3)})
    b = _write(tmp_path, "b.json", {i: i * 1e-3 for i in range(3, 6)})
    rep = dist_io.merge_distances([a, b])
    out = tmp_path / "merged.json"
    dist_io.write_raw_distances(str(out), rep.merged)

    raw = dist_io.load_distances(str(out))
    frames = [Frame(index=i, source=None) for i in range(6)]
    dist_io.apply_base_distances(frames, raw, 660.0)
    assert [f.base_dz_image for f in frames] == pytest.approx(
        [i * 1e-3 for i in range(6)]
    )
