"""Interactive canvas for displaying a page image and drawing/editing segments."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QWheelEvent,
    QPainterPath,
    QTransform,
)
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

from app.segment import Segment, PageData

# Visual constants
VERTEX_RADIUS = 6
EDGE_COLOR = QColor(0, 120, 215)
EDGE_COLOR_SELECTED = QColor(255, 80, 80)
VERTEX_COLOR = QColor(255, 255, 255)
FILL_COLOR = QColor(0, 120, 215, 40)
FILL_COLOR_SELECTED = QColor(255, 80, 80, 50)
LABEL_COLOR = QColor(255, 255, 255)
LABEL_BG = QColor(0, 120, 215, 180)


class CanvasView(QGraphicsView):
    """Zoomable / pannable graphics view that hosts the canvas scene."""

    segments_changed = pyqtSignal()  # Emitted whenever segments are added / moved / deleted
    segment_selected = pyqtSignal(int)  # index of selected segment (-1 = none)
    tool_switched = pyqtSignal(str)  # Emitted when tool changes via canvas interaction (e.g. right-click)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._page_data: Optional[PageData] = None
        self._zoom: float = 1.0

        # Interaction state
        self._tool: str = "select"  # "select" | "segment"
        self._drawing: bool = False
        self._draw_origin: Optional[QPointF] = None
        self._active_segment_idx: int = -1
        self._dragging_vertex: Optional[int] = None  # vertex index being dragged
        self._drag_segment_idx: int = -1
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None
        self._offset: int = 0

    # ── public API ───────────────────────────────────────────────────────

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        if tool == "segment":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_offset(self, offset: int) -> None:
        self._offset = offset

    def load_page(self, page_data: PageData) -> None:
        """Display a page image and its segments."""
        self._page_data = page_data
        self._scene.clear()
        self._pixmap_item = None
        self._active_segment_idx = -1

        pixmap = QPixmap(page_data.file_path)
        if pixmap.isNull():
            return
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.viewport().update()

    def current_page_data(self) -> Optional[PageData]:
        return self._page_data

    def selected_segment_index(self) -> int:
        return self._active_segment_idx

    def select_segment(self, idx: int) -> None:
        self._active_segment_idx = idx
        self.segment_selected.emit(idx)
        self.viewport().update()

    def delete_selected_segment(self) -> None:
        if self._page_data and 0 <= self._active_segment_idx < len(self._page_data.segments):
            del self._page_data.segments[self._active_segment_idx]
            self._active_segment_idx = -1
            self.segments_changed.emit()
            self.segment_selected.emit(-1)
            self.viewport().update()

    # ── zoom ─────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom *= factor

    def fit_view(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()

    # ── mouse handling ───────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = self.mapToScene(event.pos())

        # Middle-button pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # Right-click → toggle between select and segment
        if event.button() == Qt.MouseButton.RightButton:
            new_tool = "segment" if self._tool == "select" else "select"
            self.set_tool(new_tool)
            self.tool_switched.emit(new_tool)
            return

        if event.button() != Qt.MouseButton.LeftButton or not self._page_data:
            return

        img_x, img_y = pos.x(), pos.y()

        if self._tool == "select":
            # Check vertex hit first (on active segment)
            if self._active_segment_idx >= 0:
                seg = self._page_data.segments[self._active_segment_idx]
                vi = seg.vertex_at(img_x, img_y, radius=VERTEX_RADIUS / self._zoom * 2)
                if vi is not None:
                    self._dragging_vertex = vi
                    self._drag_segment_idx = self._active_segment_idx
                    return

            # Check all segments for vertex hit
            for i, seg in enumerate(self._page_data.segments):
                vi = seg.vertex_at(img_x, img_y, radius=VERTEX_RADIUS / self._zoom * 2)
                if vi is not None:
                    self._active_segment_idx = i
                    self._dragging_vertex = vi
                    self._drag_segment_idx = i
                    self.segment_selected.emit(i)
                    self.viewport().update()
                    return

            # Check segment body hit
            hit = -1
            for i, seg in enumerate(self._page_data.segments):
                if seg.contains_point(img_x, img_y):
                    hit = i
                    break
            self._active_segment_idx = hit
            self.segment_selected.emit(hit)
            self.viewport().update()

            # If clicked on empty space, start panning
            if hit < 0:
                self._panning = True
                self._pan_start = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

        elif self._tool == "segment":
            self._drawing = True
            self._draw_origin = pos
            # Create a new segment with a rectangle at the origin point
            label = self._page_data.next_label(self._offset)
            seg = Segment(
                label=label,
                vertices=[
                    (img_x, img_y),
                    (img_x, img_y),
                    (img_x, img_y),
                    (img_x, img_y),
                ],
            )
            self._page_data.segments.append(seg)
            self._active_segment_idx = len(self._page_data.segments) - 1
            self.segment_selected.emit(self._active_segment_idx)
            self.segments_changed.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return

        if not self._page_data:
            return

        pos = self.mapToScene(event.pos())
        img_x, img_y = pos.x(), pos.y()

        if self._drawing and self._draw_origin is not None:
            # Update the rectangle being drawn (vertices 1,2,3)
            seg = self._page_data.segments[self._active_segment_idx]
            ox, oy = self._draw_origin.x(), self._draw_origin.y()
            seg.vertices[0] = (ox, oy)
            seg.vertices[1] = (img_x, oy)
            seg.vertices[2] = (img_x, img_y)
            seg.vertices[3] = (ox, img_y)
            self.viewport().update()

        elif self._dragging_vertex is not None and self._drag_segment_idx >= 0:
            seg = self._page_data.segments[self._drag_segment_idx]
            seg.vertices[self._dragging_vertex] = (img_x, img_y)
            self.viewport().update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            if self._panning:
                self._panning = False
                self._pan_start = None
                self.setCursor(
                    Qt.CursorShape.CrossCursor if self._tool == "segment" else Qt.CursorShape.ArrowCursor
                )

        if event.button() == Qt.MouseButton.LeftButton:
            if self._drawing:
                self._drawing = False
                self._draw_origin = None
                self.segments_changed.emit()
            if self._dragging_vertex is not None:
                self._dragging_vertex = None
                self._drag_segment_idx = -1
                self.segments_changed.emit()

    # ── keyboard handling ────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_segment()
        else:
            super().keyPressEvent(event)

    # ── painting overlay ─────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._page_data:
            return

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for i, seg in enumerate(self._page_data.segments):
            if len(seg.vertices) != 4:
                continue
            is_active = i == self._active_segment_idx
            self._paint_segment(painter, seg, is_active)

        painter.end()

    def _paint_segment(self, painter: QPainter, seg: Segment, active: bool) -> None:
        """Draw one segment overlay."""
        pts = [QPointF(self.mapFromScene(QPointF(x, y))) for x, y in seg.vertices]

        # Fill
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()
        painter.fillPath(path, QBrush(FILL_COLOR_SELECTED if active else FILL_COLOR))

        # Edges
        pen = QPen(EDGE_COLOR_SELECTED if active else EDGE_COLOR, 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPath(path)

        # Vertices
        if active:
            for p in pts:
                painter.setPen(QPen(EDGE_COLOR_SELECTED, 1.5))
                painter.setBrush(QBrush(VERTEX_COLOR))
                painter.drawEllipse(p, VERTEX_RADIUS, VERTEX_RADIUS)

        # Label
        cx = sum(p.x() for p in pts) / 4
        cy = sum(p.y() for p in pts) / 4
        text = seg.label
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 8
        th = fm.height() + 4
        label_rect = QRectF(cx - tw / 2, cy - th / 2, tw, th)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(LABEL_BG))
        painter.drawRoundedRect(label_rect, 3, 3)
        painter.setPen(QPen(LABEL_COLOR))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)
