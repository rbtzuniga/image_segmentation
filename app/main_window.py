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

from app.auto_segment import (
    auto_segment_page,
    estimate_columns_sample,
    estimate_column_separators,
    relabel_page,
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
        self._settings.auto_segment_page_clicked.connect(self._on_auto_segment_page)
        self._settings.auto_segment_all_clicked.connect(self._on_auto_segment_all)
        self._settings.relabel_page_clicked.connect(self._on_relabel_page)
        self._settings.delete_segment_clicked.connect(self._canvas.delete_selected_segment)
        self._settings.split_segment_clicked.connect(self._canvas.split_selected_segment)
        self._canvas.relabel_requested.connect(self._on_relabel_page)
        self._settings.delete_all_page_clicked.connect(self._on_delete_all_page)
        self._settings.delete_all_all_clicked.connect(self._on_delete_all_all)
        self._settings.fit_button.clicked.connect(self._canvas.fit_view)

        # Canvas → settings (label editing)
        self._canvas.segment_selected.connect(self._on_segment_selected)
        self._canvas.segments_changed.connect(self._on_segments_changed)
        self._canvas.tool_switched.connect(self._settings._set_tool)

        # Label edit committed
        self._settings.label_edit.editingFinished.connect(self._on_label_edited)

        # Re-estimate separators when column count changes
        self._settings.columns_spin.valueChanged.connect(self._on_columns_changed)

    # ── slots ────────────────────────────────────────────────────────────

    def _on_folder_loaded(self, file_paths: list) -> None:
        self._ordered_paths = file_paths
        self._pages.clear()
        for fp in file_paths:
            self._pages[fp] = PageData(file_path=fp)
        total = len(file_paths)

        # Estimate column count from a sample of pages
        if file_paths:
            est = estimate_columns_sample(file_paths)
            self._settings.n_columns = est
            # Estimate separator positions for every page
            self._estimate_all_separators(est)
            self._status.showMessage(
                f"Loaded {total} page{'s' if total != 1 else ''} "
                f"(estimated {est} column{'s' if est != 1 else ''})."
            )
        else:
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

    def _auto_params(self) -> dict:
        """Collect auto-segment tuning parameters from the settings panel."""
        return dict(
            min_lines=self._settings.min_lines,
            merge_sensitivity=self._settings.merge_sensitivity,
            horizontal_reach=self._settings.horizontal_reach,
            n_columns=self._settings.n_columns,
        )

    def _current_separators(self) -> list:
        """Return column separators for the current page."""
        page = self._canvas.current_page_data()
        if page and page.column_separators:
            return page.column_separators
        return []

    def _estimate_all_separators(self, n_columns: int) -> None:
        """Estimate separator positions for all loaded pages."""
        for path in self._ordered_paths:
            page = self._pages[path]
            page.column_separators = estimate_column_separators(path, n_columns)

    def _on_columns_changed(self, n_columns: int) -> None:
        """Re-estimate separators for all pages when the column spinner changes."""
        self._estimate_all_separators(n_columns)
        self._canvas.viewport().update()

    def _on_auto_segment_page(self) -> None:
        page = self._canvas.current_page_data()
        if not page:
            QMessageBox.warning(self, "No Page", "Please select a page first.")
            return
        offset = self._settings.offset
        added = auto_segment_page(
            page, offset, **self._auto_params(),
            separators=page.column_separators or None,
        )
        self._canvas.viewport().update()
        self._canvas.segments_changed.emit()
        self._status.showMessage(
            f"Auto-detected {added} segment{'s' if added != 1 else ''} on page {self._current_page_idx + 1}."
        )

    def _on_auto_segment_all(self) -> None:
        if not self._ordered_paths:
            QMessageBox.warning(self, "No Pages", "Please load a folder first.")
            return
        offset = self._settings.offset
        params = self._auto_params()
        total = 0
        for path in self._ordered_paths:
            page = self._pages[path]
            total += auto_segment_page(
                page, offset, **params,
                separators=page.column_separators or None,
            )
        self._canvas.viewport().update()
        self._canvas.segments_changed.emit()
        self._status.showMessage(
            f"Auto-detected {total} segment{'s' if total != 1 else ''} across {len(self._ordered_paths)} pages."
        )

    def _on_delete_all_page(self) -> None:
        page = self._canvas.current_page_data()
        if not page:
            return
        page.segments.clear()
        self._canvas.select_segment(-1)
        self._canvas.segments_changed.emit()
        self._canvas.viewport().update()
        self._status.showMessage(
            f"Deleted all segments on page {self._current_page_idx + 1}."
        )

    def _on_relabel_page(self) -> None:
        page = self._canvas.current_page_data()
        if not page:
            QMessageBox.warning(self, "No Page", "Please select a page first.")
            return
        if not page.segments:
            QMessageBox.information(self, "No Segments", "There are no segments to relabel.")
            return
        offset = self._settings.offset
        n_columns = self._settings.n_columns
        relabel_page(page, offset, n_columns,
                     separators=page.column_separators or None)
        self._canvas.select_segment(-1)
        self._canvas.viewport().update()
        self._canvas.segments_changed.emit()
        self._status.showMessage(
            f"Relabelled {len(page.segments)} segment{'s' if len(page.segments) != 1 else ''} "
            f"on page {self._current_page_idx + 1}."
        )

    def _on_delete_all_all(self) -> None:
        total = 0
        for page in self._pages.values():
            total += len(page.segments)
            page.segments.clear()
        self._canvas.select_segment(-1)
        self._canvas.segments_changed.emit()
        self._canvas.viewport().update()
        self._status.showMessage(
            f"Deleted {total} segment{'s' if total != 1 else ''} across all pages."
        )

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
