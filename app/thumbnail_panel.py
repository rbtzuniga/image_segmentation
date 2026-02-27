"""Side panel showing page thumbnails (miniatures) for quick navigation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon, QKeyEvent
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
    QFileDialog,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
THUMB_SIZE = QSize(140, 180)


def _natural_sort_key(path: Path):
    """Sort key that handles embedded numbers naturally (1, 2, 10 not 1, 10, 2)."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', path.name)
    ]


class ThumbnailPanel(QWidget):
    """Vertical panel listing page thumbnails."""

    # Emitted when the user clicks a thumbnail.  Carries the 0-based page index.
    page_selected = pyqtSignal(int)
    # Emitted when the user presses Delete on a thumbnail.  Carries the 0-based page index.
    page_removed = pyqtSignal(int)
    # Emitted when a new input folder is loaded.  Carries the list of file paths.
    folder_loaded = pyqtSignal(list)
    # Emitted when images are added. Carries the list of new file paths.
    images_added = pyqtSignal(list)
    # Emitted when order changes due to drag/drop. Carries the new ordered list of file paths.
    order_changed = pyqtSignal(list)
    # Emitted when the user clicks Save Segmentation
    save_segmentation_clicked = pyqtSignal()
    # Emitted when the user clicks Load Segmentation
    load_segmentation_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file_paths: List[str] = []
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._btn_open = QPushButton("Select Input Folder…")
        self._btn_open.clicked.connect(self._on_open_folder)
        layout.addWidget(self._btn_open)

        # Save/Load segmentation buttons
        self._btn_save_seg = QPushButton("Save Segmentation…")
        self._btn_save_seg.clicked.connect(self.save_segmentation_clicked.emit)
        layout.addWidget(self._btn_save_seg)

        self._btn_load_seg = QPushButton("Load Segmentation…")
        self._btn_load_seg.clicked.connect(self.load_segmentation_clicked.emit)
        layout.addWidget(self._btn_load_seg)

        self._btn_add_images = QPushButton("Add Images…")
        self._btn_add_images.clicked.connect(self._on_add_images)
        layout.addWidget(self._btn_add_images)

        self._label = QLabel("No folder loaded")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        self._list = QListWidget()
        self._list.setIconSize(THUMB_SIZE)
        self._list.setSpacing(4)
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list, stretch=1)

        self.setMinimumWidth(170)
        self.setMaximumWidth(220)

    # ── public API ───────────────────────────────────────────────────────

    @property
    def file_paths(self) -> List[str]:
        return list(self._file_paths)

    def load_folder(self, folder: str) -> None:
        """Load all images from *folder* and populate the list."""
        self._list.clear()
        self._file_paths.clear()

        files = sorted(
            (
                f
                for f in Path(folder).iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=_natural_sort_key,
        )

        for f in files:
            path_str = str(f)
            self._file_paths.append(path_str)

            pixmap = QPixmap(path_str)
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(
                THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            item = QListWidgetItem(QIcon(thumb), f.name)
            item.setSizeHint(QSize(THUMB_SIZE.width() + 10, THUMB_SIZE.height() + 24))
            item.setToolTip(path_str)
            self._list.addItem(item)

        count = len(self._file_paths)
        self._label.setText(f"{count} page{'s' if count != 1 else ''} loaded")
        self.folder_loaded.emit(list(self._file_paths))

        if count:
            self._list.setCurrentRow(0)

    def load_files(self, file_paths: List[str]) -> None:
        """Load specific image files and populate the list.
        
        Used when restoring a session from a .seg file.
        """
        self._list.clear()
        self._file_paths.clear()

        for path_str in file_paths:
            path = Path(path_str)
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            self._file_paths.append(path_str)

            pixmap = QPixmap(path_str)
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(
                THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            item = QListWidgetItem(QIcon(thumb), path.name)
            item.setSizeHint(QSize(THUMB_SIZE.width() + 10, THUMB_SIZE.height() + 24))
            item.setToolTip(path_str)
            self._list.addItem(item)

        count = len(self._file_paths)
        self._label.setText(f"{count} page{'s' if count != 1 else ''} loaded")
        # Note: we don't emit folder_loaded here since main_window handles restoration

    def select_page(self, index: int) -> None:
        """Programmatically select a page by index."""
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    # ── slots ────────────────────────────────────────────────────────────

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.load_folder(folder)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.page_selected.emit(row)

    def _on_add_images(self) -> None:
        """Open file dialog to add individual images."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )
        if not files:
            return

        added = []
        for path_str in files:
            path = Path(path_str)
            if path_str in self._file_paths:
                continue  # Already loaded
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            self._file_paths.append(path_str)
            added.append(path_str)

            pixmap = QPixmap(path_str)
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(
                THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            item = QListWidgetItem(QIcon(thumb), path.name)
            item.setSizeHint(QSize(THUMB_SIZE.width() + 10, THUMB_SIZE.height() + 24))
            item.setToolTip(path_str)
            self._list.addItem(item)

        if added:
            count = len(self._file_paths)
            self._label.setText(f"{count} page{'s' if count != 1 else ''} loaded")
            self.images_added.emit(added)

    def _on_rows_moved(self) -> None:
        """Handle drag-drop reordering: rebuild _file_paths from list order."""
        new_order = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item:
                new_order.append(item.toolTip())
        self._file_paths = new_order
        self.order_changed.emit(list(new_order))

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            row = self._list.currentRow()
            if 0 <= row < len(self._file_paths):
                self._file_paths.pop(row)
                self._list.takeItem(row)
                count = len(self._file_paths)
                self._label.setText(f"{count} page{'s' if count != 1 else ''} loaded")
                self.page_removed.emit(row)
        else:
            super().keyPressEvent(event)
