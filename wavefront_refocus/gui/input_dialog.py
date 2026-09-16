"""Dialog to choose an input layout and its files, plus reference wavelength."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_WAVELENGTH,
    FMT_INTENSITY_OPD,
    FMT_INTENSITY_PHASE,
    FMT_STACK,
    FMT_STACK_FOLDER,
    FORMAT_LABELS,
)
from ..io import inputs


class _PathRow(QWidget):
    """A read-only line edit with a Browse button (folder or file)."""

    path_changed = Signal()

    def __init__(self, is_dir: bool, parent=None):
        super().__init__(parent)
        self._is_dir = is_dir
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.edit.textChanged.connect(self.path_changed)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit)
        lay.addWidget(btn)

    def _browse(self):
        if self._is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select file", filter="TIFF (*.tif *.tiff)"
            )
        if path:
            self.edit.setText(path)

    def path(self) -> str:
        return self.edit.text()


class InputDialog(QDialog):
    """Collects an input layout, paths, a distances JSON and a wavelength."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open dataset")
        self.loaded: Optional[inputs.LoadedInput] = None
        self.distances_path: str = ""
        # All selected distances files, merged on accept (chunked acquisitions).
        self.distances_paths: list = []
        self._json_paths: list = []
        self.wavelength_nm: float = DEFAULT_WAVELENGTH * 1e9

        self.combo = QComboBox()
        for fmt in (FMT_INTENSITY_OPD, FMT_INTENSITY_PHASE, FMT_STACK, FMT_STACK_FOLDER):
            self.combo.addItem(FORMAT_LABELS[fmt], fmt)
        self.combo.currentIndexChanged.connect(self._update_rows)

        # Two path rows; meaning depends on the selected format.
        self.row1 = _PathRow(is_dir=True)
        self.row2 = _PathRow(is_dir=True)
        self.label1 = QLabel()
        self.label2 = QLabel()

        self.wl_spin = QDoubleSpinBox()
        self.wl_spin.setRange(100.0, 5000.0)
        self.wl_spin.setDecimals(3)
        self.wl_spin.setSuffix(" nm")
        self.wl_spin.setValue(self.wavelength_nm)

        self.dist_row = _PathRow(is_dir=False)
        # distances JSON browse should accept .json
        self.dist_row.edit.setPlaceholderText("Reconstruction_Distances.json")

        # ── partial load ──
        # Long sequences can be worked through in chunks; frames keep their
        # original indices so each chunk exports under the right frame numbers.
        self.partial_check = QCheckBox("Load only frames")
        self.partial_check.setToolTip(
            "Load a slice of the sequence instead of all of it, so a long\n"
            "dataset can be refocused and exported in chunks.\n"
            "Frames keep their original indices, so chunk 51–150 exports as\n"
            "frame_0051…frame_0150 and matches the distances JSON."
        )
        self.first_spin = QSpinBox()
        self.first_spin.setRange(0, 999999)
        self.last_spin = QSpinBox()
        self.last_spin.setRange(0, 999999)
        for s in (self.first_spin, self.last_spin):
            s.setEnabled(False)
        self.partial_check.toggled.connect(self._on_partial_toggled)
        self.first_spin.valueChanged.connect(self._on_first_changed)
        self.range_label = QLabel("")
        self.range_label.setStyleSheet("color: palette(mid);")

        range_row = QHBoxLayout()
        range_row.addWidget(self.partial_check)
        range_row.addWidget(self.first_spin, 1)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.last_spin, 1)

        # Re-probe the frame count whenever the inputs change.
        self.row1.path_changed.connect(self._refresh_count)
        self.row2.path_changed.connect(self._refresh_count)
        self.combo.currentIndexChanged.connect(self._refresh_count)
        self._n_total = 0

        form = QFormLayout()
        form.addRow("Input layout", self.combo)
        form.addRow(self.label1, self.row1)
        form.addRow(self.label2, self.row2)
        form.addRow("Frame range", range_row)
        form.addRow("", self.range_label)
        form.addRow("Reference wavelength", self.wl_spin)
        form.addRow("Distances JSON", self._json_row())

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

        self._update_rows()

    def _json_row(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.json_edit = QLineEdit()
        self.json_edit.setReadOnly(True)
        self.json_edit.setPlaceholderText("Reconstruction_Distances.json")
        self.json_edit.setToolTip(
            "One distances JSON, or several per-chunk files merged together.\n"
            "Select multiple files to reassemble an acquisition that was\n"
            "refocused in chunks (0–35, 36–70, …) into one sequence."
        )
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_json)
        lay.addWidget(self.json_edit)
        lay.addWidget(btn)
        return w

    def _browse_json(self):
        # Multi-select so per-chunk exports can be merged in one go.
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select distances JSON(s)", filter="JSON (*.json)"
        )
        if paths:
            self._json_paths = list(paths)
            self.json_edit.setText(self._describe_json(paths))

    @staticmethod
    def _describe_json(paths: list) -> str:
        import os

        if len(paths) == 1:
            return paths[0]
        names = ", ".join(os.path.basename(p) for p in paths)
        return f"{len(paths)} files: {names}"

    def _fmt(self) -> str:
        return self.combo.currentData()

    def _update_rows(self):
        fmt = self._fmt()
        if fmt == FMT_INTENSITY_OPD:
            self.label1.setText("Intensity folder")
            self.label2.setText("OPD folder (nm)")
            self.row2.setVisible(True)
            self.label2.setVisible(True)
            self._set_dir(self.row1, True)
            self._set_dir(self.row2, True)
        elif fmt == FMT_INTENSITY_PHASE:
            self.label1.setText("Intensity folder")
            self.label2.setText("Phase folder (rad)")
            self.row2.setVisible(True)
            self.label2.setVisible(True)
            self._set_dir(self.row1, True)
            self._set_dir(self.row2, True)
        elif fmt == FMT_STACK:
            self.label1.setText("Wavefront stack (.tif)")
            self.label2.setVisible(False)
            self.row2.setVisible(False)
            self._set_dir(self.row1, False)
        elif fmt == FMT_STACK_FOLDER:
            self.label1.setText("Folder of stacks")
            self.label2.setVisible(False)
            self.row2.setVisible(False)
            self._set_dir(self.row1, True)

    @staticmethod
    def _set_dir(row: _PathRow, is_dir: bool):
        row._is_dir = is_dir

    # ── partial load ──
    def _count_frames(self) -> int:
        """Frames available for the current selection, or 0 if unknown.

        Cheap probes only — listing a folder, or reading the stack's time-axis
        length — so it can run on every path change without loading images.
        """
        fmt = self._fmt()
        p1 = self.row1.path()
        if not p1:
            return 0
        try:
            if fmt == FMT_STACK:
                from ..vendor.utils_images import load_wavefront_tif

                _, _, n = load_wavefront_tif(p1, 0)
                return int(n)
            if fmt in (FMT_INTENSITY_OPD, FMT_INTENSITY_PHASE):
                p2 = self.row2.path()
                if not p2:
                    return 0
                return min(len(inputs.list_tifs(p1)), len(inputs.list_tifs(p2)))
            return len(inputs.list_tifs(p1))
        except Exception:  # noqa: BLE001 — a bad path just means "unknown"
            return 0

    def _refresh_count(self) -> None:
        """Re-probe the sequence length and rescale the range spin boxes."""
        n = self._count_frames()
        if n == self._n_total:
            return
        self._n_total = n
        if n <= 0:
            self.range_label.setText("")
            return
        self.range_label.setText(f"{n} frames available (0–{n - 1})")
        for s in (self.first_spin, self.last_spin):
            s.blockSignals(True)
            s.setMaximum(n - 1)
            s.blockSignals(False)
        if not self.partial_check.isChecked():
            self.first_spin.setValue(0)
            self.last_spin.setValue(n - 1)

    def _on_partial_toggled(self, on: bool) -> None:
        self.first_spin.setEnabled(on)
        self.last_spin.setEnabled(on)
        if on and self._n_total > 0 and self.last_spin.value() == 0:
            self.last_spin.setValue(self._n_total - 1)

    def _on_first_changed(self, value: int) -> None:
        # Keep the range non-empty: last never precedes first.
        if self.last_spin.value() < value:
            self.last_spin.setValue(value)

    def frame_range(self) -> tuple:
        """The requested ``(first, last)``, or ``(None, None)`` for all."""
        if not self.partial_check.isChecked():
            return None, None
        return self.first_spin.value(), self.last_spin.value()

    def _on_accept(self):
        from PySide6.QtWidgets import QMessageBox

        fmt = self._fmt()
        wl = self.wl_spin.value()
        first, last = self.frame_range()
        try:
            if fmt == FMT_INTENSITY_OPD:
                loaded = inputs.open_intensity_opd(
                    self.row1.path(), self.row2.path(), wl, first, last
                )
            elif fmt == FMT_INTENSITY_PHASE:
                loaded = inputs.open_intensity_phase(
                    self.row1.path(), self.row2.path(), first, last
                )
            elif fmt == FMT_STACK:
                loaded = inputs.open_stack(self.row1.path(), first, last)
            else:
                loaded = inputs.open_stack_folder(self.row1.path(), first, last)
        except Exception as exc:  # noqa: BLE001 — surface load errors to user
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.loaded = loaded
        self.distances_paths = list(self._json_paths)
        # Kept for callers that just want "the" file (single-selection case).
        self.distances_path = self._json_paths[0] if self._json_paths else ""
        self.wavelength_nm = wl
        self.accept()
