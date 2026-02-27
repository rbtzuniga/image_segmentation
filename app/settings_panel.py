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
    QSlider,
)


class SettingsPanel(QWidget):
    """Right-hand panel with output settings and action buttons."""

    save_all_clicked = pyqtSignal()
    delete_segment_clicked = pyqtSignal()
    split_segment_clicked = pyqtSignal()
    combine_clicked = pyqtSignal()
    uncombine_clicked = pyqtSignal()
    delete_all_page_clicked = pyqtSignal()
    delete_all_all_clicked = pyqtSignal()
    auto_segment_page_clicked = pyqtSignal()
    auto_segment_all_clicked = pyqtSignal()
    relabel_page_clicked = pyqtSignal()
    relabel_all_clicked = pyqtSignal()
    add_segment_grid_clicked = pyqtSignal(int)  # number of rows per column
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

        self._btn_split = QPushButton("Split Selected Segment  (press S)")
        self._btn_split.setToolTip(
            "Split the selected segment horizontally in half,\n"
            "creating a top and bottom segment."
        )
        self._btn_split.clicked.connect(self.split_segment_clicked.emit)
        form.addRow(self._btn_split)

        self._btn_combine = QPushButton("Combine Selected  (press C)")
        self._btn_combine.setToolTip(
            "Combine two selected segments into a paired\n"
            "top/bottom unit that exports as one image."
        )
        self._btn_combine.clicked.connect(self.combine_clicked.emit)
        form.addRow(self._btn_combine)

        self._btn_uncombine = QPushButton("Uncombine Selected  (press U)")
        self._btn_uncombine.setToolTip(
            "Uncombine the selected segment from its paired\n"
            "top/bottom partner."
        )
        self._btn_uncombine.clicked.connect(self.uncombine_clicked.emit)
        form.addRow(self._btn_uncombine)

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

        # ── Auto-segment tuning ─────────────────────────────────────────
        auto_group = QGroupBox("Auto-Segment Tuning")
        auto_form = QFormLayout(auto_group)

        # Merge sensitivity: how aggressively lines merge vertically
        self._slider_merge = QSlider(Qt.Orientation.Horizontal)
        self._slider_merge.setRange(10, 150)  # percentage of default
        self._slider_merge.setValue(100)
        self._slider_merge.setTickInterval(10)
        self._slider_merge.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._lbl_merge = QLabel("100%")
        self._slider_merge.valueChanged.connect(
            lambda v: self._lbl_merge.setText(f"{v}%")
        )
        merge_row = QHBoxLayout()
        merge_row.addWidget(self._slider_merge)
        merge_row.addWidget(self._lbl_merge)
        auto_form.addRow("Merge sens.:", merge_row)

        # Horizontal reach: how far characters merge into words
        self._slider_hreach = QSlider(Qt.Orientation.Horizontal)
        self._slider_hreach.setRange(10, 150)
        self._slider_hreach.setValue(100)
        self._slider_hreach.setTickInterval(10)
        self._slider_hreach.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._lbl_hreach = QLabel("100%")
        self._slider_hreach.valueChanged.connect(
            lambda v: self._lbl_hreach.setText(f"{v}%")
        )
        hreach_row = QHBoxLayout()
        hreach_row.addWidget(self._slider_hreach)
        hreach_row.addWidget(self._lbl_hreach)
        auto_form.addRow("H. reach:", hreach_row)

        # Minimum lines to keep a block
        self._slider_minlines = QSlider(Qt.Orientation.Horizontal)
        self._slider_minlines.setRange(1, 6)
        self._slider_minlines.setValue(2)
        self._slider_minlines.setTickInterval(1)
        self._slider_minlines.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._lbl_minlines = QLabel("2")
        self._slider_minlines.valueChanged.connect(
            lambda v: self._lbl_minlines.setText(str(v))
        )
        minlines_row = QHBoxLayout()
        minlines_row.addWidget(self._slider_minlines)
        minlines_row.addWidget(self._lbl_minlines)
        auto_form.addRow("Min lines:", minlines_row)

        # Column count (per-page expected columns)
        self._spin_columns = QSpinBox()
        self._spin_columns.setRange(1, 10)
        self._spin_columns.setValue(1)
        self._spin_columns.setToolTip(
            "Expected number of text columns per page.\n"
            "Auto-estimated when a folder is loaded."
        )
        auto_form.addRow("Columns:", self._spin_columns)

        self._btn_reset_tuning = QPushButton("Reset to Defaults")
        self._btn_reset_tuning.clicked.connect(self._reset_tuning)
        auto_form.addRow(self._btn_reset_tuning)

        layout.addWidget(auto_group)

        # ── Auto-segment actions ───────────────────────────────────────
        auto_action_group = QGroupBox()
        auto_action_group.setFlat(True)
        auto_action_layout = QVBoxLayout(auto_action_group)
        auto_action_layout.setContentsMargins(0, 4, 0, 4)

        _auto_btn_style = (
            "QPushButton { background-color: #CCE5FF; }"
        )

        self._btn_auto_page = QPushButton("Auto Segment Page")
        self._btn_auto_page.setStyleSheet(_auto_btn_style)
        self._btn_auto_page.clicked.connect(self.auto_segment_page_clicked.emit)
        auto_action_layout.addWidget(self._btn_auto_page)

        self._btn_auto_all = QPushButton("Auto Segment All Pages")
        self._btn_auto_all.setStyleSheet(_auto_btn_style)
        self._btn_auto_all.clicked.connect(self.auto_segment_all_clicked.emit)
        auto_action_layout.addWidget(self._btn_auto_all)

        self._btn_relabel = QPushButton("Relabel Page  (press R)")
        self._btn_relabel.setStyleSheet(_auto_btn_style)
        self._btn_relabel.setToolTip(
            "Re-label all segments on the current page\n"
            "in column-aware reading order."
        )
        self._btn_relabel.clicked.connect(self.relabel_page_clicked.emit)
        auto_action_layout.addWidget(self._btn_relabel)

        self._btn_relabel_all = QPushButton("Relabel All Pages")
        self._btn_relabel_all.setStyleSheet(_auto_btn_style)
        self._btn_relabel_all.setToolTip(
            "Re-label all segments on all pages\n"
            "in column-aware reading order."
        )
        self._btn_relabel_all.clicked.connect(self.relabel_all_clicked.emit)
        auto_action_layout.addWidget(self._btn_relabel_all)

        # Segment grid row
        grid_row = QHBoxLayout()
        self._spin_grid_rows = QSpinBox()
        self._spin_grid_rows.setRange(1, 100)
        self._spin_grid_rows.setValue(10)
        self._spin_grid_rows.setToolTip("Number of rows per column for segment grid")
        grid_row.addWidget(self._spin_grid_rows)
        self._btn_add_grid = QPushButton("Add Segment Grid")
        self._btn_add_grid.setStyleSheet(_auto_btn_style)
        self._btn_add_grid.setToolTip(
            "Delete all segments and create a grid of segments.\n"
            "Creates this many rows per column."
        )
        self._btn_add_grid.clicked.connect(self._on_add_segment_grid)
        grid_row.addWidget(self._btn_add_grid)
        auto_action_layout.addLayout(grid_row)

        layout.addWidget(auto_action_group)

        # ── Delete actions ───────────────────────────────────────────
        delete_group = QGroupBox()
        delete_group.setFlat(True)
        delete_layout = QVBoxLayout(delete_group)
        delete_layout.setContentsMargins(0, 4, 0, 4)

        _del_btn_style = (
            "QPushButton { background-color: #FFCCCC; }"
        )

        self._btn_delete = QPushButton("Delete Selected Segment(s)")
        self._btn_delete.setStyleSheet(_del_btn_style)
        self._btn_delete.clicked.connect(self.delete_segment_clicked.emit)
        delete_layout.addWidget(self._btn_delete)

        self._btn_delete_all = QPushButton("Delete All Segments (Page)")
        self._btn_delete_all.setStyleSheet(_del_btn_style)
        self._btn_delete_all.clicked.connect(self.delete_all_page_clicked.emit)
        delete_layout.addWidget(self._btn_delete_all)

        self._btn_delete_all_all = QPushButton("Delete All Segments (All Pages)")
        self._btn_delete_all_all.setStyleSheet(_del_btn_style)
        self._btn_delete_all_all.clicked.connect(self.delete_all_all_clicked.emit)
        delete_layout.addWidget(self._btn_delete_all_all)

        layout.addWidget(delete_group)

        # ── Other actions ────────────────────────────────────────────
        other_group = QGroupBox()
        other_group.setFlat(True)
        other_layout = QVBoxLayout(other_group)
        other_layout.setContentsMargins(0, 4, 0, 4)

        self._btn_fit = QPushButton("Fit View")
        other_layout.addWidget(self._btn_fit)

        self._btn_save = QPushButton("Save All")
        self._btn_save.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; padding: 8px; font-weight: bold; }"
        )
        self._btn_save.clicked.connect(self.save_all_clicked.emit)
        other_layout.addWidget(self._btn_save)

        layout.addWidget(other_group)
        layout.addStretch(1)

        self.setMinimumWidth(200)
        self.setMaximumWidth(280)

    # ── public API ───────────────────────────────────────────────────────

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value: str) -> None:
        self._output_folder = value
        self._lbl_output.setText(value if value else "Not set")

    @property
    def prefix(self) -> str:
        return self._edit_prefix.text().strip() or "segment"

    @prefix.setter
    def prefix(self, value: str) -> None:
        self._edit_prefix.setText(value)

    @property
    def image_format(self) -> str:
        return self._combo_format.currentText()

    @image_format.setter
    def image_format(self, value: str) -> None:
        idx = self._combo_format.findText(value)
        if idx >= 0:
            self._combo_format.setCurrentIndex(idx)

    @property
    def offset(self) -> int:
        return self._spin_offset.value()

    @property
    def fit_button(self) -> QPushButton:
        return self._btn_fit

    @property
    def merge_sensitivity(self) -> float:
        """Vertical merge sensitivity as a multiplier (1.0 = default)."""
        return self._slider_merge.value() / 100.0

    @property
    def horizontal_reach(self) -> float:
        """Horizontal dilation multiplier (1.0 = default)."""
        return self._slider_hreach.value() / 100.0

    @property
    def min_lines(self) -> int:
        """Minimum text lines for a block to be kept."""
        return self._slider_minlines.value()

    @property
    def n_columns(self) -> int:
        """Expected number of text columns per page."""
        return self._spin_columns.value()

    @n_columns.setter
    def n_columns(self, value: int) -> None:
        self._spin_columns.setValue(max(1, min(value, 10)))

    @property
    def columns_spin(self) -> QSpinBox:
        """Expose the columns spinner for signal connections."""
        return self._spin_columns

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

    def _reset_tuning(self) -> None:
        self._slider_merge.setValue(100)
        self._slider_hreach.setValue(100)
        self._slider_minlines.setValue(2)

    def _on_select_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_folder = folder
            self._lbl_output.setText(folder)
            self.output_folder_changed.emit(folder)

    def _on_add_segment_grid(self) -> None:
        """Emit signal with the number of rows per column."""
        self.add_segment_grid_clicked.emit(self._spin_grid_rows.value())
