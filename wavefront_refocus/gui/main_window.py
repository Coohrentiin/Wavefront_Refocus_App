"""Main window: owns the Session and wires the panels together."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from skimage.restoration import unwrap_phase

from ..core.engine import build_compute_kwargs
from ..core.frame_spec import parse_frame_spec
from ..core.interpolation import build_evaluator, fit_and_eval
from ..core.model import Frame, Session
from ..core.optics import image_to_sample
from ..core.worker import PropagationController
from ..io import distances as dist_io
from ..io.export import export_folder
from .controls_panel import ControlsPanel
from .display_range_dialog import DisplayRangeDialog
from .distance_graph import DistanceGraph
from .export_dialog import ExportDialog
from .input_dialog import InputDialog
from .interpolation_panel import InterpolationPanel
from .phase_view import PhaseView
from .transfer_dialog import TransferDialog


def _summarise(items: list, limit: int = 6) -> str:
    """Comma-joined items, elided after ``limit`` so the message stays short."""
    head = ", ".join(str(i) for i in items[:limit])
    return head if len(items) <= limit else f"{head}, … (+{len(items) - limit})"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wavefront Refocus App")
        self.resize(1200, 800)

        self.session = Session()
        self._raw_distances: dict = {}
        self.controller = PropagationController(self)
        self._display_dialog = None

        # ── widgets ──
        self.controls = ControlsPanel()
        self.interp_panel = InterpolationPanel()
        self.graph = DistanceGraph()
        self.phase_view = PhaseView()

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self.controls, 1)
        left_lay.addWidget(self.interp_panel)
        left_lay.addWidget(self.graph, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.phase_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 780])

        central = QWidget()
        cl = QHBoxLayout(central)
        cl.addWidget(splitter)
        self.setCentralWidget(central)

        self._build_menu()
        self._connect()

    # ── setup ──
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        open_act = m.addAction("Open dataset…")
        open_act.triggered.connect(self.open_dataset)
        m.addSeparator()
        merge_act = m.addAction("Merge distances JSONs…")
        merge_act.setToolTip(
            "Combine per-chunk distances files into one covering the whole "
            "acquisition."
        )
        merge_act.triggered.connect(self.merge_distances_files)
        transfer_act = m.addAction("Transfer distances…")
        transfer_act.setToolTip(
            "Apply refocusing recorded in an input/target JSON pair to the "
            "frames loaded here."
        )
        transfer_act.triggered.connect(self.transfer_distances)
        m.addSeparator()
        quit_act = m.addAction("Quit")
        quit_act.triggered.connect(self.close)

    def _connect(self):
        self.controls.params_changed.connect(self._on_params_changed)
        self.controls.compute_requested.connect(self.compute_current)
        self.controls.cancel_requested.connect(self.controller.cancel)
        self.controls.export_requested.connect(self.export_folder)
        self.controls.export_json_requested.connect(self.export_json)

        self.phase_view.frame_changed.connect(self.select_frame)
        self.phase_view.z_changed.connect(self._on_z_changed)
        self.phase_view.choose_requested.connect(self.choose_plane)
        self.phase_view.display_range_requested.connect(self.open_display_range)

        # The graph emits a frame index; the slider emits a list position.
        self.graph.point_clicked.connect(self.select_frame_index)
        self.interp_panel.run_requested.connect(self.run_interpolation)

        self.controller.progress.connect(self.controls.set_progress)
        self.controller.finished.connect(self._on_compute_finished)
        self.controller.failed.connect(self._on_compute_failed)

    # ── dataset ──
    def open_dataset(self):
        dlg = InputDialog(self)
        # seed the dialog wavelength from the controls
        dlg.wl_spin.setValue(self.controls.optical_params().wavelength * 1e9)
        if dlg.exec() != InputDialog.Accepted or dlg.loaded is None:
            return

        loaded = dlg.loaded
        optics = self.controls.optical_params()
        optics.wavelength = dlg.wavelength_nm * 1e-9
        self.controls.set_optical_params(optics)

        # Frames keep their index in the FULL sequence, so a partial load still
        # lines up with the distances JSON and exports under the right numbers.
        frames = [
            Frame(index=i, source=src)
            for i, src in zip(loaded.indices, loaded.sources)
        ]

        self._raw_distances = {}
        if dlg.distances_paths:
            self._raw_distances = self._load_distances(
                dlg.distances_paths, frames, optics
            )

        self.session = Session(
            frames=frames,
            optics=optics,
            sweep=self.controls.sweep_params(),
            input_format=loaded.input_format,
            distances_path=dlg.distances_path or None,
            current_index=0,
            loader=loaded.loader,
        )

        self.phase_view.set_frame_count(len(frames), self.session.frame_indices)
        self.graph.set_session(self.session)
        self.select_frame(0)
        if loaded.is_partial:
            self.setWindowTitle(
                f"Wavefront Refocus App — frames {frames[0].index}"
                f"–{frames[-1].index} of {loaded.total_frames}"
            )
        else:
            self.setWindowTitle("Wavefront Refocus App")

    def _load_distances(self, paths: list, frames: list, optics) -> dict:
        """Load one or more distances JSONs, merging per-chunk exports.

        Reports anything the user should know before working on the merged
        sequence: conflicting values for a frame, and frames left uncovered
        (which would silently fall back to a 0.0 distance).
        """
        try:
            report = dist_io.merge_distances(paths)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Distances", f"Could not load distances:\n{exc}")
            return {}

        dist_io.apply_base_distances(frames, report.merged, optics.wavelength_nm)

        notes = []
        if len(paths) > 1:
            notes.append(
                f"Merged {len(paths)} files into {report.n_frames} frames "
                f"({min(report.frame_indices)}–{max(report.frame_indices)})."
            )
        if report.conflicts:
            notes.append(
                "Frames defined differently in more than one file — the last "
                f"file selected wins: {_summarise(report.conflicts)}."
            )
        if report.duplicates:
            notes.append(
                f"Frames repeated identically across files: "
                f"{_summarise(report.duplicates)}."
            )
        # Only frames we actually loaded matter; a merged table may legitimately
        # cover more of the acquisition than this session does.
        uncovered = report.missing(f.index for f in frames)
        if uncovered:
            notes.append(
                f"No distance for {len(uncovered)} loaded frame(s), using 0.0: "
                f"{_summarise(uncovered)}."
            )

        if notes and (report.conflicts or uncovered or len(paths) > 1):
            QMessageBox.information(self, "Distances", "\n\n".join(notes))
        return report.merged

    # ── parameters ──
    def _on_params_changed(self):
        if not self.session.frames:
            return
        optics = self.controls.optical_params()
        self.session.optics = optics
        self.session.sweep = self.controls.sweep_params()
        # Re-bind base distances to the (possibly new) reference wavelength.
        if self._raw_distances:
            dist_io.apply_base_distances(
                self.session.frames, self._raw_distances, optics.wavelength_nm
            )
        self.graph.redraw()

    # ── frame selection ──
    def select_frame(self, position: int):
        """Show the frame at list ``position`` (not its frame index)."""
        if not (0 <= position < len(self.session.frames)):
            return
        self.session.current_index = position
        frame = self.session.frames[position]
        self.phase_view.set_current_frame(position, frame.index)
        self.phase_view.disable_planes()
        self.graph.set_current_index(frame.index)
        self.controls.show_frame_center(frame.final_dz_sample or 0.0)
        self._show_base_or_position(frame)

    def select_frame_index(self, frame_index: int):
        """Show the frame numbered ``frame_index`` (as clicked on the graph)."""
        pos = self.session.position_of(frame_index)
        if pos is not None:
            self.select_frame(pos)

    def _show_base_or_position(self, frame: Frame):
        """Show the frame at its base plane, or at its chosen/interp plane."""
        try:
            phase, amp = self.session.loader(frame.source)
        except Exception as exc:  # noqa: BLE001
            self.phase_view.set_status(f"Load error: {exc}")
            return

        dz = frame.final_dz_sample
        if dz is None or dz == 0.0:
            disp = self._zero_mean_opd(np.exp(1j * phase), amp)
            label = f"frame {frame.index} — base plane"
        else:
            from ..core.engine import refocus_one
            field = refocus_one(phase, amp, dz, self.session.optics)
            disp = self._zero_mean_opd(field, None)
            label = f"frame {frame.index} — {dz * 1e6:.3f} µm"
        self.phase_view.show_image(disp, label)
        self.phase_view.set_status(label)

    def _zero_mean_opd(self, field_or_phase, amp) -> np.ndarray:
        """Unwrapped, zero-mean OPD (nm) for display consistency across planes."""
        if amp is not None:
            field = amp * field_or_phase
        else:
            field = field_or_phase
        unwrapped = unwrap_phase(np.angle(field))
        opd = unwrapped * self.session.optics.wavelength_nm / (2.0 * np.pi)
        return (opd - opd.mean()).astype(np.float32)

    # ── compute current ──
    def compute_current(self):
        frame = self.session.current_frame
        if frame is None:
            return
        if self.controller.is_running():
            return
        self.session.optics = self.controls.optical_params()
        self.session.sweep = self.controls.sweep_params()
        try:
            phase, amp = self.session.loader(frame.source)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        # Sweep around wherever the frame currently sits (a chosen or
        # interpolated plane), not always around the base plane — so a second
        # pass can explore ±range about e.g. +4 µm instead of re-covering
        # planes already known to be out of focus.
        center = self._compute_center(frame)
        kwargs = build_compute_kwargs(
            phase, amp, self.session.optics, self.session.sweep,
            center_dz_sample=center,
        )
        self.controls.set_computing(True)
        if center:
            self.phase_view.set_status(
                f"Computing around {center * 1e6:+.3f} µm…"
            )
        else:
            self.phase_view.set_status("Computing…")
        if not self.controller.start(
            kwargs, center, self.session.optics.magnification
        ):
            self.controls.set_computing(False)

    def _compute_center(self, frame: Frame) -> float:
        """The sample-space plane the next sweep should be centred on.

        Honours an explicit override from the controls; otherwise it is the
        frame's current position (chosen or interpolated), or 0 for a frame
        that has none yet.
        """
        if self.controls.use_custom_center():
            return self.controls.center_dz_sample()
        return frame.final_dz_sample or 0.0

    def _on_compute_finished(self, result):
        self.controls.set_computing(False)
        frame = self.session.current_frame
        if frame is None:
            return
        frame.result = result
        self.phase_view.enable_planes(len(result.dz_sample_array), result.z0_index)
        self._show_plane(result.z0_index)
        self.phase_view.set_status(
            f"Computed {len(result.dz_sample_array)} planes ({result.backend})"
        )

    def _on_compute_failed(self, message: str):
        self.controls.set_computing(False)
        if message == "cancelled":
            self.phase_view.set_status("Cancelled")
        else:
            QMessageBox.critical(self, "Propagation failed", message)
            self.phase_view.set_status("Failed")

    # ── display range floating window ──
    def open_display_range(self):
        if self._display_dialog is None:
            low, high, margin = self.phase_view.display_range()
            # parent=None -> a detached, independent top-level window. We keep
            # a reference so it isn't garbage-collected.
            self._display_dialog = DisplayRangeDialog(low, high, margin, parent=None)
            self._display_dialog.range_changed.connect(
                self.phase_view.set_display_range
            )
        self._display_dialog.show()
        self._display_dialog.raise_()
        self._display_dialog.activateWindow()

    def _on_z_changed(self, z: int):
        self._show_plane(z)

    def _show_plane(self, z: int):
        frame = self.session.current_frame
        if frame is None or frame.result is None:
            return
        res = frame.result
        z = max(0, min(z, len(res.dz_sample_array) - 1))
        dz_um = res.dz_sample_array[z] * 1e6
        self.phase_view.show_image(
            res.opd_stack[z], f"frame {frame.index} — plane dz = {dz_um:.3f} µm"
        )
        self.phase_view.set_status(f"plane {z}/{len(res.dz_sample_array) - 1}, dz = {dz_um:.3f} µm")

    # ── choose plane ──
    def choose_plane(self):
        frame = self.session.current_frame
        if frame is None or frame.result is None:
            return
        z = self.phase_view.current_z()
        frame.chosen_dz_sample = float(frame.result.dz_sample_array[z])
        frame.interpolated_dz_sample = None
        frame.is_chosen = True
        self.graph.redraw()
        self.controls.show_frame_center(frame.chosen_dz_sample)
        self.phase_view.set_status(
            f"Chose plane for frame {frame.index}: dz = {frame.chosen_dz_sample * 1e6:.3f} µm"
        )

    # ── interpolation ──
    def run_interpolation(self, method: str, smoothing: float, frames_text: str):
        chosen = self.session.chosen_frames()
        if not chosen:
            QMessageBox.information(self, "Interpolation",
                                    "Choose a plane for at least one frame first.")
            return

        # An explicit frame list restricts which frames get written; empty
        # means "every frame without a chosen plane", as before.
        try:
            targets = parse_frame_spec(
                frames_text, [f.index for f in self.session.frames]
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Interpolation", str(exc))
            return
        target_set = set(targets)

        self.session.smoothing_factor = smoothing
        self.session.interp_method = method
        M = self.session.optics.magnification

        # Fit the spline in ABSOLUTE sample space (the distance shown on the
        # graph), so interpolated points lie on the drawn curve. Each chosen
        # frame's absolute distance is base + chosen dz.
        def absolute_sample(frame) -> float:
            return image_to_sample(frame.base_dz_image, M) + frame.chosen_dz_sample

        # The fit always uses every chosen frame, except any the user asked to
        # redefine — those are being thrown back to the curve, so they must not
        # also anchor it.
        fit_frames = [f for f in chosen if f.index not in target_set]
        if not fit_frames:
            QMessageBox.information(
                self, "Interpolation",
                "Every chosen frame is in the list to redefine, so there is "
                "nothing left to fit a curve through.\nLeave at least one "
                "chosen frame out of the list.",
            )
            return

        chosen_idx = [f.index for f in fit_frames]
        chosen_abs = [absolute_sample(f) for f in fit_frames]
        all_idx = [f.index for f in self.session.frames]

        abs_by_idx = fit_and_eval(chosen_idx, chosen_abs, all_idx, smoothing, method)

        # Store back as a RELATIVE dz per frame (subtract that frame's base),
        # keeping export/JSON math unchanged.
        n_written = 0
        reverted = []
        for f in self.session.frames:
            wanted = f.index in target_set if target_set else not f.is_chosen
            if not wanted:
                continue
            if f.is_chosen:
                # Explicitly listed: drop the manual pick and follow the curve.
                f.is_chosen = False
                f.chosen_dz_sample = None
                reverted.append(f.index)
            base_sample = image_to_sample(f.base_dz_image, M)
            f.interpolated_dz_sample = abs_by_idx[f.index] - base_sample
            n_written += 1

        # Dense fitted curve for overlay (display µm), spanning all frames.
        evaluator = build_evaluator(chosen_idx, chosen_abs, smoothing, method)
        lo, hi = min(all_idx), max(all_idx)
        xs = np.linspace(lo, hi, max(2, (hi - lo) * 8 + 1))
        ys_um = np.array([evaluator(x) for x in xs]) * 1e6
        self.graph.set_interp_curve(xs, ys_um)

        # The current frame may have just moved; keep the sweep-centre box in
        # step with its new position.
        cur = self.session.current_frame
        if cur is not None:
            self.controls.show_frame_center(cur.final_dz_sample or 0.0)

        scope = f"{n_written} listed frame(s)" if target_set else "unchosen frames"
        msg = f"Interpolated planes for {scope} ({method})."
        if reverted:
            msg += f" Reverted chosen frame(s): {_summarise(reverted)}."
        self.phase_view.set_status(msg)

    # ── export ──
    def export_folder(self):
        if not self.session.frames:
            return
        dlg = ExportDialog(default_format=self.session.input_format, parent=self)
        if dlg.exec() != ExportDialog.Accepted:
            return
        self.session.optics = self.controls.optical_params()

        total = len(self.session.frames)
        progress = QProgressDialog("Exporting refocused frames…", None, 0, total, self)
        progress.setWindowTitle("Export")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def on_progress(done: int, n: int):
            progress.setMaximum(n)
            progress.setValue(done)
            QApplication.processEvents()

        try:
            export_folder(self.session, dlg.out_format, dlg.out_dir,
                          progress_cb=on_progress,
                          keep_original_names=dlg.keep_original_names)
        except Exception as exc:  # noqa: BLE001
            progress.close()
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        progress.close()
        QMessageBox.information(self, "Export", f"Exported to:\n{dlg.out_dir}")

    def transfer_distances(self):
        """Replay refocusing from an input/target JSON pair onto these frames."""
        if not self.session.frames:
            QMessageBox.information(
                self, "Transfer distances", "Open a dataset first."
            )
            return

        dlg = TransferDialog(self.session, self)
        if dlg.exec() != TransferDialog.Accepted or dlg.report is None:
            return

        # Re-run for real: the dialog's report was a preview and did not mutate.
        report = dist_io.transfer_distances(
            self.session.frames, dlg.target_raw, self.session.optics,
            dlg.input_raw,
        )

        # Transferred planes are chosen planes, so any previous fitted curve no
        # longer describes the points on the graph.
        self.graph.clear_interp_curve()
        self.graph.redraw()
        frame = self.session.current_frame
        if frame is not None:
            self.controls.show_frame_center(frame.final_dz_sample or 0.0)
            self._show_base_or_position(frame)

        lo, hi = report.shift_range_m()
        msg = [
            f"Transferred {report.n_applied} frame(s), "
            f"{lo * 1e6:+.3f} to {hi * 1e6:+.3f} µm (sample space)."
        ]
        if report.uncovered:
            msg.append(
                f"Left unchanged (not in target): {_summarise(report.uncovered)}."
            )
        if report.unused:
            msg.append(
                f"Ignored (not loaded here): {_summarise(report.unused)}."
            )
        if report.assumed_zero_input:
            msg.append(
                "Input taken as 0 for: "
                f"{_summarise(report.assumed_zero_input)}."
            )
        QMessageBox.information(self, "Transfer distances", "\n\n".join(msg))
        self.phase_view.set_status(
            f"Transferred distances onto {report.n_applied} frame(s)."
        )

    def merge_distances_files(self):
        """Combine per-chunk distances JSONs into one file for the sequence."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select distances JSONs to merge", filter="JSON (*.json)"
        )
        if not paths:
            return
        try:
            report = dist_io.merge_distances(paths)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Merge failed", str(exc))
            return

        idx = report.frame_indices
        summary = [
            f"Merged {len(paths)} files → {report.n_frames} frames "
            f"({idx[0]}–{idx[-1]})."
        ]
        gaps = report.gaps()
        if gaps:
            summary.append(
                "Missing frames in the merged range: "
                + _summarise([f"{a}–{b}" if b > a else str(a) for a, b in gaps])
                + "."
            )
        if report.conflicts:
            summary.append(
                "Frames defined differently in more than one file — the last "
                f"file selected wins: {_summarise(report.conflicts)}."
            )
        if report.duplicates:
            summary.append(
                f"Frames repeated identically: {_summarise(report.duplicates)}."
            )

        out, _ = QFileDialog.getSaveFileName(
            self, "Save merged distances JSON", "Reconstruction_Distances.json",
            filter="JSON (*.json)",
        )
        if not out:
            return
        try:
            dist_io.write_raw_distances(out, report.merged)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Merge failed", str(exc))
            return
        summary.append(f"\nWritten to:\n{out}")
        QMessageBox.information(self, "Merge distances", "\n\n".join(summary))

    def export_json(self):
        if not self.session.frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export distances JSON", "Reconstruction_Distances.json",
            filter="JSON (*.json)",
        )
        if not path:
            return
        self.session.optics = self.controls.optical_params()
        dist_io.write_distances(path, self.session.frames, self.session.optics)
        QMessageBox.information(self, "Export", f"Distances written to:\n{path}")
