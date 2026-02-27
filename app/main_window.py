"""Main application window – assembles all panels and coordinates behaviour."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow,
    QHBoxLayout,
    QWidget,
    QMessageBox,
    QStatusBar,
    QSplitter,
    QProgressDialog,
    QFileDialog,
)

from app.auto_segment import (
    auto_segment_page,
    estimate_columns_sample,
    estimate_column_separators,
    estimate_content_bounds,
    relabel_page,
)
from app.canvas import CanvasView
from app.export import export_all
from app.segment import PageData, Segment
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
        self._next_combined_id: int = 0
        self._last_seg_file: Optional[str] = None

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()

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
        self._thumb_panel.page_removed.connect(self._on_page_removed)
        self._thumb_panel.images_added.connect(self._on_images_added)
        self._thumb_panel.order_changed.connect(self._on_order_changed)
        self._thumb_panel.save_segmentation_clicked.connect(self._on_save_segmentation)
        self._thumb_panel.load_segmentation_clicked.connect(self._on_load_segmentation)

        # Settings → canvas / export
        self._settings.tool_changed.connect(self._canvas.set_tool)
        self._settings.offset_changed.connect(self._canvas.set_offset)
        self._settings.save_all_clicked.connect(self._on_save_all)
        self._settings.auto_segment_page_clicked.connect(self._on_auto_segment_page)
        self._settings.auto_segment_all_clicked.connect(self._on_auto_segment_all)
        self._settings.relabel_page_clicked.connect(self._on_relabel_page)
        self._settings.add_segment_grid_clicked.connect(self._on_add_segment_grid)
        self._settings.delete_segment_clicked.connect(self._canvas.delete_selected_segment)
        self._settings.split_segment_clicked.connect(self._canvas.split_selected_segment)
        self._settings.combine_clicked.connect(self._canvas._request_combine)
        self._settings.uncombine_clicked.connect(self._canvas._request_uncombine)
        self._canvas.relabel_requested.connect(self._on_relabel_page)
        self._settings.delete_all_page_clicked.connect(self._on_delete_all_page)
        self._settings.delete_all_all_clicked.connect(self._on_delete_all_all)
        self._settings.fit_button.clicked.connect(self._canvas.fit_view)

        # Canvas → settings (label editing)
        self._canvas.segment_selected.connect(self._on_segment_selected)
        self._canvas.segments_changed.connect(self._on_segments_changed)
        self._canvas.tool_switched.connect(self._settings._set_tool)
        self._canvas.multi_delete_requested.connect(self._on_multi_delete)
        self._canvas.combine_requested.connect(self._on_combine)
        self._canvas.uncombine_requested.connect(self._on_uncombine)

        # Label edit committed
        self._settings.label_edit.editingFinished.connect(self._on_label_edited)

        # Re-estimate separators when column count changes
        self._settings.columns_spin.valueChanged.connect(self._on_columns_changed)

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts for the main window."""
        # Ctrl+S to quick-save segmentation
        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self._on_quick_save)

    # ── slots ────────────────────────────────────────────────────────────

    def _on_folder_loaded(self, file_paths: list) -> None:
        self._ordered_paths = file_paths
        self._pages.clear()
        for fp in file_paths:
            self._pages[fp] = PageData(file_path=fp)
        total = len(file_paths)

        if not file_paths:
            self._status.showMessage(f"Loaded {total} page{'s' if total != 1 else ''}.")
            return

        # Show progress dialog for processing
        progress = QProgressDialog(
            "Processing pages...", "Cancel", 0, total + 1, self
        )
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)  # Show after 300ms
        progress.setValue(0)
        QCoreApplication.processEvents()

        # Estimate column count from a sample of pages
        progress.setLabelText("Estimating column layout...")
        est = estimate_columns_sample(file_paths)
        self._settings.n_columns = est
        progress.setValue(1)
        QCoreApplication.processEvents()

        if progress.wasCanceled():
            return

        # Estimate separator positions for every page with progress
        self._estimate_all_separators_with_progress(est, progress)

        progress.setValue(total + 1)
        progress.close()

        self._status.showMessage(
            f"Loaded {total} page{'s' if total != 1 else ''} "
            f"(estimated {est} column{'s' if est != 1 else ''})."
        )

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

    def _on_page_removed(self, index: int) -> None:
        """Remove a page from the project (does not delete the file on disk)."""
        if 0 <= index < len(self._ordered_paths):
            path = self._ordered_paths.pop(index)
            self._pages.pop(path, None)
            self._canvas.clear_multi_selection(path)

            remaining = len(self._ordered_paths)
            if remaining == 0:
                self._current_page_idx = -1
                self._canvas._scene.clear()
                self._canvas._pixmap_item = None
                self._canvas._page_data = None
                self._status.showMessage("All pages removed.")
            else:
                new_idx = min(index, remaining - 1)
                self._thumb_panel.select_page(new_idx)
                self._status.showMessage(
                    f"Removed page. {remaining} page{'s' if remaining != 1 else ''} remaining."
                )

    def _on_images_added(self, file_paths: list) -> None:
        """Handle images added via 'Add Images' button."""
        for fp in file_paths:
            if fp not in self._pages:
                self._ordered_paths.append(fp)
                self._pages[fp] = PageData(file_path=fp)

        count = len(self._ordered_paths)
        self._status.showMessage(
            f"Added {len(file_paths)} image{'s' if len(file_paths) != 1 else ''}. "
            f"Total: {count} page{'s' if count != 1 else ''}."
        )

    def _on_order_changed(self, new_order: list) -> None:
        """Handle reordering of pages via drag-and-drop."""
        self._ordered_paths = new_order
        self._status.showMessage("Page order updated.")

    def _on_save_segmentation(self) -> None:
        """Save current session to a .seg file (with file dialog)."""
        if not self._ordered_paths:
            QMessageBox.warning(self, "No Data", "No pages loaded to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation",
            self._last_seg_file or "",
            "Segmentation Files (*.seg);;All Files (*)"
        )
        if not file_path:
            return

        # Ensure .seg extension
        if not file_path.lower().endswith(".seg"):
            file_path += ".seg"

        if self._save_segmentation_to_file(file_path):
            self._last_seg_file = file_path

    def _on_quick_save(self) -> None:
        """Quick save (Ctrl+S) – save to last file or prompt if none."""
        if not self._ordered_paths:
            self._status.showMessage("No data to save.")
            return

        if self._last_seg_file:
            if self._save_segmentation_to_file(self._last_seg_file):
                pass  # Status message already shown
        else:
            # No previous file – open Save As dialog
            self._on_save_segmentation()

    def _save_segmentation_to_file(self, file_path: str) -> bool:
        """Save segmentation data to the specified file. Returns True on success."""
        # Build data structure
        data = {
            "version": 1,
            "n_columns": self._settings.n_columns,
            "output_folder": self._settings.output_folder,
            "prefix": self._settings.prefix,
            "image_format": self._settings.image_format,
            "pages": []
        }

        for path in self._ordered_paths:
            page = self._pages.get(path)
            if not page:
                continue

            page_data = {
                "file_path": page.file_path,
                "segments": [],
                "column_separators": list(page.column_separators) if page.column_separators else [],
                "content_bounds": None,
                "_counter": page._counter,
            }

            # Serialize content bounds
            if page.content_bounds:
                left, right = page.content_bounds
                page_data["content_bounds"] = [list(left), list(right)]

            # Serialize segments
            for seg in page.segments:
                seg_data = {
                    "label": seg.label,
                    "vertices": [list(v) for v in seg.vertices],
                    "combined_id": seg.combined_id,
                    "combined_role": seg.combined_role,
                }
                page_data["segments"].append(seg_data)

            data["pages"].append(page_data)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            from pathlib import Path
            filename = Path(file_path).name
            self._status.showMessage(f"Saved: {filename}", 5000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save: {e}")
            return False

    def _on_load_segmentation(self) -> None:
        """Load a session from a .seg file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Segmentation",
            "",
            "Segmentation Files (*.seg);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load: {e}")
            return

        # Validate version
        version = data.get("version", 0)
        if version != 1:
            QMessageBox.warning(
                self, "Version Mismatch",
                f"Unsupported file version: {version}. Expected version 1."
            )
            return

        pages_data = data.get("pages", [])
        if not pages_data:
            QMessageBox.warning(self, "Empty File", "No pages found in the file.")
            return

        from pathlib import Path

        # Collect file paths and check which exist
        file_paths = []
        missing = []
        path_mapping = {}  # old_path -> new_path

        for pd in pages_data:
            fp = pd.get("file_path", "")
            if fp:
                if Path(fp).is_file():
                    file_paths.append(fp)
                    path_mapping[fp] = fp
                else:
                    missing.append(fp)

        # If some files are missing, ask user to locate the input folder
        if missing:
            reply = QMessageBox.question(
                self, "Missing Files",
                f"{len(missing)} image(s) not found at their original location.\n\n"
                f"Would you like to select a new input folder to locate them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                new_folder = QFileDialog.getExistingDirectory(
                    self, "Select Input Folder with Images"
                )
                if new_folder:
                    new_folder_path = Path(new_folder)
                    still_missing = []
                    for old_path in missing:
                        # Try to find by filename
                        filename = Path(old_path).name
                        new_path = new_folder_path / filename
                        if new_path.is_file():
                            new_path_str = str(new_path)
                            file_paths.append(new_path_str)
                            path_mapping[old_path] = new_path_str
                        else:
                            still_missing.append(old_path)
                    if still_missing:
                        QMessageBox.warning(
                            self, "Still Missing",
                            f"{len(still_missing)} image(s) could not be found:\n" +
                            "\n".join(Path(p).name for p in still_missing[:5]) +
                            ("\n..." if len(still_missing) > 5 else "")
                        )
            else:
                # User declined, show what's missing
                QMessageBox.warning(
                    self, "Missing Files",
                    f"{len(missing)} image(s) will be skipped:\n" +
                    "\n".join(Path(p).name for p in missing[:5]) +
                    ("\n..." if len(missing) > 5 else "")
                )

        if not file_paths:
            QMessageBox.critical(self, "No Valid Files", "No valid image files found.")
            return

        # Check output folder
        output_folder = data.get("output_folder", "")
        if output_folder and not Path(output_folder).is_dir():
            reply = QMessageBox.question(
                self, "Output Folder Not Found",
                f"The saved output folder no longer exists:\n{output_folder}\n\n"
                f"Would you like to select a new output folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                new_output = QFileDialog.getExistingDirectory(
                    self, "Select Output Folder"
                )
                if new_output:
                    output_folder = new_output
                else:
                    output_folder = ""
            else:
                output_folder = ""

        # Load thumbnails
        self._thumb_panel.load_files(file_paths)

        # Restore pages data
        self._ordered_paths = file_paths
        self._pages.clear()

        for pd in pages_data:
            old_fp = pd.get("file_path", "")
            # Use mapped path (could be relocated)
            fp = path_mapping.get(old_fp)
            if not fp or fp not in file_paths:
                continue

            page = PageData(file_path=fp)
            page._counter = pd.get("_counter", 0)

            # Restore column separators
            seps = pd.get("column_separators", [])
            page.column_separators = [(s[0], s[1]) for s in seps if len(s) >= 2]

            # Restore content bounds
            cb = pd.get("content_bounds")
            if cb and len(cb) == 2:
                left, right = cb
                if len(left) >= 2 and len(right) >= 2:
                    page.content_bounds = ((left[0], left[1]), (right[0], right[1]))

            # Restore segments
            for seg_d in pd.get("segments", []):
                verts = seg_d.get("vertices", [])
                if len(verts) != 4:
                    continue
                seg = Segment(
                    label=seg_d.get("label", ""),
                    vertices=[(v[0], v[1]) for v in verts],
                    combined_id=seg_d.get("combined_id"),
                    combined_role=seg_d.get("combined_role"),
                )
                page.segments.append(seg)

            self._pages[fp] = page

        # Restore column count setting
        n_cols = data.get("n_columns", 1)
        self._settings.n_columns = n_cols

        # Restore output settings
        if output_folder:
            self._settings.output_folder = output_folder
        prefix = data.get("prefix", "")
        if prefix:
            self._settings.prefix = prefix
        img_format = data.get("image_format", "")
        if img_format:
            self._settings.image_format = img_format

        # Select first page
        if file_paths:
            self._thumb_panel.select_page(0)
            self._current_page_idx = 0
            self._canvas.load_page(self._pages[file_paths[0]])

        total = len(file_paths)
        total_segs = sum(len(p.segments) for p in self._pages.values())
        self._status.showMessage(
            f"Loaded {total} page{'s' if total != 1 else ''} with "
            f"{total_segs} segment{'s' if total_segs != 1 else ''} from {file_path}"
        )
        self._last_seg_file = file_path

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
        """Estimate separator positions and content bounds for all loaded pages."""
        for path in self._ordered_paths:
            page = self._pages[path]
            # Estimate content bounds first
            page.content_bounds = estimate_content_bounds(path)
            # Then estimate separators using content bounds for better placement
            page.column_separators = estimate_column_separators(
                path, n_columns, content_bounds=page.content_bounds
            )

    def _estimate_all_separators_with_progress(
        self, n_columns: int, progress: QProgressDialog
    ) -> None:
        """Estimate separator positions with progress updates."""
        for i, path in enumerate(self._ordered_paths):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"Processing page {i + 1} of {len(self._ordered_paths)}...")
            page = self._pages[path]
            page.content_bounds = estimate_content_bounds(path)
            page.column_separators = estimate_column_separators(
                path, n_columns, content_bounds=page.content_bounds
            )
            progress.setValue(i + 2)  # +1 for column estimation step
            QCoreApplication.processEvents()

    def _on_columns_changed(self, n_columns: int) -> None:
        """Re-estimate separators for all pages when the column spinner changes."""
        total = len(self._ordered_paths)
        if total > 5:
            # Show progress for larger datasets
            progress = QProgressDialog(
                "Recalculating separators...", "Cancel", 0, total, self
            )
            progress.setWindowTitle("Processing")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(200)
            for i, path in enumerate(self._ordered_paths):
                if progress.wasCanceled():
                    break
                page = self._pages[path]
                page.column_separators = estimate_column_separators(
                    path, n_columns, content_bounds=page.content_bounds
                )
                progress.setValue(i + 1)
                QCoreApplication.processEvents()
            progress.close()
        else:
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
            content_bounds=page.content_bounds or None,
        )
        self._canvas.clear_multi_selection()
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
                content_bounds=page.content_bounds or None,
            )
        self._canvas.clear_multi_selection()
        self._canvas.viewport().update()
        self._canvas.segments_changed.emit()
        self._status.showMessage(
            f"Auto-detected {total} segment{'s' if total != 1 else ''} across {len(self._ordered_paths)} pages."
        )

    def _on_add_segment_grid(self, n_rows: int) -> None:
        """Create a grid of segments on the current page.
        
        Deletes all existing segments and creates n_rows segments per column,
        where column width is determined by separators/content bounds.
        """
        page = self._canvas.current_page_data()
        if not page:
            QMessageBox.warning(self, "No Page", "Please select a page first.")
            return
        
        # Get image dimensions
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap(page.file_path)
        if pixmap.isNull():
            return
        img_w = pixmap.width()
        img_h = pixmap.height()
        
        # Determine column boundaries
        n_columns = self._settings.n_columns
        
        # Get content bounds or use full image width
        if page.content_bounds:
            left_bound, right_bound = page.content_bounds
            # Use average of top/bottom for each bound
            left_x = (left_bound[0] + left_bound[1]) / 2
            right_x = (right_bound[0] + right_bound[1]) / 2
        else:
            left_x = 0
            right_x = img_w
        
        # Build column x-positions from separators
        column_edges = [left_x]
        if page.column_separators:
            for x_top, x_bot in page.column_separators:
                sep_x = (x_top + x_bot) / 2
                if left_x < sep_x < right_x:
                    column_edges.append(sep_x)
        column_edges.append(right_x)
        column_edges.sort()
        
        # Clear existing segments
        page.segments.clear()
        
        # Calculate row height
        row_height = img_h / n_rows
        
        # Create segments
        offset = self._settings.offset
        for col_idx in range(len(column_edges) - 1):
            col_left = column_edges[col_idx]
            col_right = column_edges[col_idx + 1]
            
            for row_idx in range(n_rows):
                y_top = row_idx * row_height
                y_bot = (row_idx + 1) * row_height
                
                label = page.next_label(offset)
                seg = Segment(
                    label=label,
                    vertices=[
                        (col_left, y_top),   # top-left
                        (col_right, y_top),  # top-right
                        (col_right, y_bot),  # bottom-right
                        (col_left, y_bot),   # bottom-left
                    ]
                )
                page.segments.append(seg)
        
        self._canvas.clear_multi_selection()
        self._canvas.viewport().update()
        self._canvas.segments_changed.emit()
        n_segs = len(page.segments)
        self._status.showMessage(
            f"Created {n_segs} segment{'s' if n_segs != 1 else ''} "
            f"({n_rows} rows × {len(column_edges) - 1} column{'s' if len(column_edges) > 2 else ''})."
        )

    def _on_multi_delete(self, to_delete: dict) -> None:
        """Delete segments across multiple pages based on multi-selection."""
        # First, uncombine partners of any combined segments being deleted
        for fp, indices in to_delete.items():
            if fp in self._pages:
                page = self._pages[fp]
                for idx in indices:
                    if 0 <= idx < len(page.segments):
                        seg = page.segments[idx]
                        if seg.combined_id:
                            self._uncombine_partner(seg.combined_id, fp, idx)

        # Then delete (reverse order to preserve indices)
        total = 0
        for fp, indices in to_delete.items():
            if fp in self._pages:
                page = self._pages[fp]
                for idx in sorted(indices, reverse=True):
                    if 0 <= idx < len(page.segments):
                        del page.segments[idx]
                        total += 1
        self._canvas.segments_changed.emit()
        self._canvas.viewport().update()
        self._status.showMessage(
            f"Deleted {total} segment{'s' if total != 1 else ''}."
        )

    def _uncombine_partner(self, combined_id: str, exclude_fp: str, exclude_idx: int) -> None:
        """Find and uncombine the partner of a combined segment being deleted."""
        for fp in self._ordered_paths:
            page = self._pages[fp]
            for i, seg in enumerate(page.segments):
                if seg.combined_id == combined_id and not (fp == exclude_fp and i == exclude_idx):
                    if seg.combined_role == "top":
                        seg.label = seg.label.removesuffix("_top")
                    elif seg.combined_role == "bottom":
                        seg.label = seg.label.removesuffix("_bottom")
                    seg.combined_id = None
                    seg.combined_role = None
                    return

    def _on_combine(self, selection: dict) -> None:
        """Combine exactly two selected segments as a top/bottom pair."""
        entries = []
        for fp, indices in selection.items():
            if fp not in self._pages:
                continue
            page = self._pages[fp]
            page_idx = self._ordered_paths.index(fp)
            for seg_idx in indices:
                if 0 <= seg_idx < len(page.segments):
                    entries.append((page_idx, fp, seg_idx, page.segments[seg_idx]))

        if len(entries) != 2:
            QMessageBox.warning(self, "Combine", "Select exactly 2 segments to combine.")
            return

        for _, _, _, seg in entries:
            if seg.combined_id:
                QMessageBox.warning(
                    self, "Combine",
                    "One of the selected segments is already combined.\n"
                    "Delete or uncombine it first.",
                )
                return

        # Determine top and bottom:
        #   different pages → smaller page number is top
        #   same page → leftmost (smaller centroid-X) is top
        #   same column → topmost (smaller centroid-Y) is top
        def _sort_key(entry):
            page_idx, fp, seg_idx, seg = entry
            cx = sum(v[0] for v in seg.vertices) / len(seg.vertices)
            cy = sum(v[1] for v in seg.vertices) / len(seg.vertices)
            return (page_idx, cx, cy)

        entries.sort(key=_sort_key)
        top_seg = entries[0][3]
        bottom_seg = entries[1][3]

        self._next_combined_id += 1
        cid = str(self._next_combined_id)
        base_label = top_seg.label

        top_seg.combined_id = cid
        top_seg.combined_role = "top"
        top_seg.label = f"{base_label}_top"

        bottom_seg.combined_id = cid
        bottom_seg.combined_role = "bottom"
        bottom_seg.label = f"{base_label}_bottom"

        self._canvas.clear_multi_selection()
        self._canvas.segments_changed.emit()
        self._canvas.viewport().update()
        self._status.showMessage(
            f"Combined segments as '{base_label}_top' and '{base_label}_bottom'."
        )

    def _on_uncombine(self, file_path: str, seg_idx: int) -> None:
        """Uncombine a segment and its partner."""
        if file_path not in self._pages:
            return
        page = self._pages[file_path]
        if not (0 <= seg_idx < len(page.segments)):
            return
        seg = page.segments[seg_idx]
        if not seg.combined_id:
            return

        combined_id = seg.combined_id

        # Restore this segment
        if seg.combined_role == "top":
            seg.label = seg.label.removesuffix("_top")
        elif seg.combined_role == "bottom":
            seg.label = seg.label.removesuffix("_bottom")
        seg.combined_id = None
        seg.combined_role = None

        # Find and restore the partner
        self._uncombine_partner(combined_id, file_path, seg_idx)

        self._canvas.segments_changed.emit()
        self._canvas.viewport().update()
        self._status.showMessage("Uncombined segment pair.")

    def _on_delete_all_page(self) -> None:
        page = self._canvas.current_page_data()
        if not page:
            return
        page.segments.clear()
        self._canvas.clear_multi_selection(page.file_path)
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
        # Update bottom partners (may be on other pages)
        self._update_bottom_partners(page)
        # Clear stale multi-selection (segment indices changed after sort)
        self._canvas.clear_multi_selection()
        self._canvas.select_segment(-1)
        self._canvas.viewport().update()
        self._canvas.segments_changed.emit()
        self._status.showMessage(
            f"Relabelled {len(page.segments)} segment{'s' if len(page.segments) != 1 else ''} "
            f"on page {self._current_page_idx + 1}."
        )

    def _update_bottom_partners(self, page: PageData) -> None:
        """After relabeling, update every bottom partner whose top lives on *page*."""
        for seg in page.segments:
            if seg.combined_role == "top" and seg.combined_id:
                base_label = seg.label.removesuffix("_top")
                # Find its bottom partner (could be on any page)
                for fp in self._ordered_paths:
                    other_page = self._pages[fp]
                    for other_seg in other_page.segments:
                        if (other_seg.combined_id == seg.combined_id
                                and other_seg.combined_role == "bottom"):
                            other_seg.label = f"{base_label}_bottom"

    def _on_delete_all_all(self) -> None:
        total = 0
        for page in self._pages.values():
            total += len(page.segments)
            page.segments.clear()
        self._canvas.clear_multi_selection()
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
            saved, warnings = export_all(
                pages=[self._pages[p] for p in self._ordered_paths],
                output_folder=output,
                prefix=self._settings.prefix,
                ext=self._settings.image_format,
            )
            msg = f"Saved {len(saved)} segment image{'s' if len(saved) != 1 else ''}."
            if warnings:
                msg += f"\n\n{len(warnings)} segment(s) could not be exported:\n"
                msg += "\n".join(warnings[:20])  # Limit display to first 20
                if len(warnings) > 20:
                    msg += f"\n... and {len(warnings) - 20} more"
            msg += f"\n\nOutput folder:\n{output}"
            if warnings:
                QMessageBox.warning(self, "Export Complete with Warnings", msg)
            else:
                QMessageBox.information(self, "Export Complete", msg)
            self._status.showMessage(f"Exported {len(saved)} segments to {output}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"An error occurred:\n{exc}")
