"""Dialog to pick an output format and target directory at export time."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    FMT_INTENSITY_OPD,
    FMT_INTENSITY_PHASE,
    FMT_STACK,
    FMT_STACK_FOLDER,
    FORMAT_LABELS,
)


class ExportDialog(QDialog):
    """Collects the output format (defaulting to the input format) and dir."""

    def __init__(self, default_format: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export refocused folder")
        self.out_format: str = default_format or FMT_STACK
        self.out_dir: str = ""

        self.combo = QComboBox()
        for fmt in (FMT_INTENSITY_OPD, FMT_INTENSITY_PHASE, FMT_STACK, FMT_STACK_FOLDER):
            self.combo.addItem(FORMAT_LABELS[fmt], fmt)
        if default_format:
            idx = self.combo.findData(default_format)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)

        self.dir_edit = QLineEdit()
        self.dir_edit.setReadOnly(True)
        dir_row = QWidget()
        dlay = QHBoxLayout(dir_row)
        dlay.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        dlay.addWidget(self.dir_edit)
        dlay.addWidget(btn)

        form = QFormLayout()
        form.addRow("Output format", self.combo)
        form.addRow("Target directory", dir_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select target directory")
        if path:
            self.dir_edit.setText(path)

    def _on_accept(self):
        from PySide6.QtWidgets import QMessageBox

        if not self.dir_edit.text():
            QMessageBox.warning(self, "No directory", "Choose a target directory.")
            return
        self.out_format = self.combo.currentData()
        self.out_dir = self.dir_edit.text()
        self.accept()
