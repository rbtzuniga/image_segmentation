"""Main application window – assembles all panels and coordinates behaviour."""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QHBoxLayout,
    QWidget,
    QMessageBox,
    QStatusBar,
    QSplitter,
)

from app.canvas import CanvasView
from app.export import export_all
from app.segment import PageData
from app.settings_panel import SettingsPanel
from app.thumbnail_panel import ThumbnailPanel


class MainWindow(QMainWindow):
    """Top-level window for the Image Segmentation Tool."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Segmentation Tool")
        self.resize(1200, 800)

        # Page data keyed by file path
        self._pages: Dict[str, PageData] = {}
        self._ordered_paths: List[str] = []
        self._current_page_idx: int = -1

        self._build_ui()
        self._connect_signals()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Central widget with horizontal splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        self._thumb_panel = ThumbnailPanel()
        self._canvas = CanvasView()
        self._settings = SettingsPanel()

        self._splitter.addWidget(self._thumb_panel)
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._settings)
        self._splitter.setStretchFactor(0, 0)  # thumbnail panel fixed-ish
        self._splitter.setStretchFactor(1, 1)  # canvas stretches
        self._splitter.setStretchFactor(2, 0)  # settings panel fixed-ish

        self.setCentralWidget(self._splitter)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready – select an input folder to begin.")

    # ── signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Thumbnail → canvas
        self._thumb_panel.folder_loaded.connect(self._on_folder_loaded)
        self._thumb_panel.page_selected.connect(self._on_page_selected)

        # Settings → canvas / export
        self._settings.tool_changed.connect(self._canvas.set_tool)
        self._settings.offset_changed.connect(self._canvas.set_offset)
        self._settings.save_all_clicked.connect(self._on_save_all)
        self._settings.delete_segment_clicked.connect(self._canvas.delete_selected_segment)
        self._settings.fit_button.clicked.connect(self._canvas.fit_view)

        # Canvas → settings (label editing)
        self._canvas.segment_selected.connect(self._on_segment_selected)
        self._canvas.segments_changed.connect(self._on_segments_changed)
        self._canvas.tool_switched.connect(self._settings._set_tool)

        # Label edit committed
        self._settings.label_edit.editingFinished.connect(self._on_label_edited)

    # ── slots ────────────────────────────────────────────────────────────

    def _on_folder_loaded(self, file_paths: list) -> None:
        self._ordered_paths = file_paths
        self._pages.clear()
        for fp in file_paths:
            self._pages[fp] = PageData(file_path=fp)
        total = len(file_paths)
        self._status.showMessage(f"Loaded {total} page{'s' if total != 1 else ''}.")

    def _on_page_selected(self, index: int) -> None:
        if 0 <= index < len(self._ordered_paths):
            self._current_page_idx = index
            path = self._ordered_paths[index]
            page = self._pages[path]
            self._canvas.load_page(page)
            self._settings.set_segment_label(None)
            self._status.showMessage(
                f"Page {index + 1}/{len(self._ordered_paths)}: {path}"
            )

    def _on_segment_selected(self, seg_idx: int) -> None:
        page = self._canvas.current_page_data()
        if page and 0 <= seg_idx < len(page.segments):
            self._settings.set_segment_label(page.segments[seg_idx].label)
        else:
            self._settings.set_segment_label(None)

    def _on_segments_changed(self) -> None:
        page = self._canvas.current_page_data()
        if page:
            count = len(page.segments)
            self._status.showMessage(
                f"Page {self._current_page_idx + 1} – {count} segment{'s' if count != 1 else ''}"
            )

    def _on_label_edited(self) -> None:
        page = self._canvas.current_page_data()
        seg_idx = self._canvas.selected_segment_index()
        if page and 0 <= seg_idx < len(page.segments):
            new_label = self._settings.label_edit.text().strip()
            if new_label:
                page.segments[seg_idx].label = new_label
                self._canvas.viewport().update()

    def _on_save_all(self) -> None:
        output = self._settings.output_folder
        if not output:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder first.")
            return

        pages_with_segments = [
            self._pages[p] for p in self._ordered_paths if self._pages[p].segments
        ]
        if not pages_with_segments:
            QMessageBox.information(self, "Nothing to Save", "No segments have been created yet.")
            return

        try:
            saved = export_all(
                pages=[self._pages[p] for p in self._ordered_paths],
                output_folder=output,
                prefix=self._settings.prefix,
                ext=self._settings.image_format,
            )
            QMessageBox.information(
                self,
                "Export Complete",
                f"Saved {len(saved)} segment image{'s' if len(saved) != 1 else ''}.\n\n"
                f"Output folder:\n{output}",
            )
            self._status.showMessage(f"Exported {len(saved)} segments to {output}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"An error occurred:\n{exc}")
