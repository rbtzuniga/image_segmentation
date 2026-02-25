"""Settings / controls panel (right side): offset, prefix, format, output folder, actions."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QPushButton,
    QFileDialog,
)


class SettingsPanel(QWidget):
    """Right-hand panel with output settings and action buttons."""

    save_all_clicked = pyqtSignal()
    delete_segment_clicked = pyqtSignal()
    tool_changed = pyqtSignal(str)  # "select" | "segment"
    offset_changed = pyqtSignal(int)
    output_folder_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_folder: str = ""
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Tool selection ───────────────────────────────────────────────
        tool_group = QGroupBox("Tools")
        tool_layout = QHBoxLayout(tool_group)
        self._btn_select = QPushButton("Select")
        self._btn_select.setCheckable(True)
        self._btn_select.setChecked(True)
        self._btn_segment = QPushButton("Segment")
        self._btn_segment.setCheckable(True)
        tool_layout.addWidget(self._btn_select)
        tool_layout.addWidget(self._btn_segment)
        layout.addWidget(tool_group)

        self._btn_select.clicked.connect(lambda: self._set_tool("select"))
        self._btn_segment.clicked.connect(lambda: self._set_tool("segment"))

        # ── Segment settings ────────────────────────────────────────────
        seg_group = QGroupBox("Segment Settings")
        form = QFormLayout(seg_group)

        self._spin_offset = QSpinBox()
        self._spin_offset.setRange(0, 999999)
        self._spin_offset.setValue(0)
        self._spin_offset.valueChanged.connect(self.offset_changed.emit)
        form.addRow("Label offset:", self._spin_offset)

        self._edit_label = QLineEdit()
        self._edit_label.setPlaceholderText("(select a segment)")
        self._edit_label.setEnabled(False)
        form.addRow("Edit label:", self._edit_label)

        layout.addWidget(seg_group)

        # ── Output settings ─────────────────────────────────────────────
        out_group = QGroupBox("Output")
        out_form = QFormLayout(out_group)

        self._btn_output = QPushButton("Select Output Folder…")
        self._btn_output.clicked.connect(self._on_select_output)
        out_form.addRow(self._btn_output)

        self._lbl_output = QLabel("Not set")
        self._lbl_output.setWordWrap(True)
        out_form.addRow("Folder:", self._lbl_output)

        self._edit_prefix = QLineEdit("segment")
        out_form.addRow("Prefix:", self._edit_prefix)

        self._combo_format = QComboBox()
        self._combo_format.addItems(["png", "jpg", "tif"])
        out_form.addRow("Format:", self._combo_format)

        layout.addWidget(out_group)

        # ── Action buttons ──────────────────────────────────────────────
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        self._btn_delete = QPushButton("Delete Selected Segment")
        self._btn_delete.clicked.connect(self.delete_segment_clicked.emit)
        action_layout.addWidget(self._btn_delete)

        self._btn_fit = QPushButton("Fit View")
        action_layout.addWidget(self._btn_fit)

        self._btn_save = QPushButton("Save All")
        self._btn_save.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; padding: 8px; font-weight: bold; }"
        )
        self._btn_save.clicked.connect(self.save_all_clicked.emit)
        action_layout.addWidget(self._btn_save)

        layout.addWidget(action_group)
        layout.addStretch(1)

        self.setMinimumWidth(200)
        self.setMaximumWidth(280)

    # ── public API ───────────────────────────────────────────────────────

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @property
    def prefix(self) -> str:
        return self._edit_prefix.text().strip() or "segment"

    @property
    def image_format(self) -> str:
        return self._combo_format.currentText()

    @property
    def offset(self) -> int:
        return self._spin_offset.value()

    @property
    def fit_button(self) -> QPushButton:
        return self._btn_fit

    def set_segment_label(self, label: str | None) -> None:
        """Update the label editor with the currently selected segment's label."""
        if label is None:
            self._edit_label.setText("")
            self._edit_label.setEnabled(False)
            self._edit_label.setPlaceholderText("(select a segment)")
        else:
            self._edit_label.setEnabled(True)
            self._edit_label.setText(label)

    @property
    def label_edit(self) -> QLineEdit:
        return self._edit_label

    # ── slots ────────────────────────────────────────────────────────────

    def _set_tool(self, tool: str) -> None:
        self._btn_select.setChecked(tool == "select")
        self._btn_segment.setChecked(tool == "segment")
        self.tool_changed.emit(tool)

    def _on_select_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_folder = folder
            self._lbl_output.setText(folder)
            self.output_folder_changed.emit(folder)
