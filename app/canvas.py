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
SEPARATOR_COLOR = QColor(0, 180, 60, 160)
SEPARATOR_COLOR_ACTIVE = QColor(0, 220, 80, 220)
SEPARATOR_HIT_TOLERANCE = 8  # pixels (screen) for grabbing a separator


class CanvasView(QGraphicsView):
    """Zoomable / pannable graphics view that hosts the canvas scene."""

    segments_changed = pyqtSignal()  # Emitted whenever segments are added / moved / deleted
    segment_selected = pyqtSignal(int)  # index of selected segment (-1 = none)
    tool_switched = pyqtSignal(str)  # Emitted when tool changes via canvas interaction (e.g. right-click)
    separators_changed = pyqtSignal()  # Emitted when column separators are moved

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
        self._dragging_separator: int = -1  # index of separator being dragged (-1 = none)
        self._dragging_sep_endpoint: str = ""  # "top" or "bottom"

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

    def split_selected_segment(self) -> None:
        """Split the selected segment horizontally at its vertical midpoint."""
        if not self._page_data or not (0 <= self._active_segment_idx < len(self._page_data.segments)):
            return
        seg = self._page_data.segments[self._active_segment_idx]
        if len(seg.vertices) != 4:
            return

        # Vertices: TL(0), TR(1), BR(2), BL(3)
        tl, tr, br, bl = seg.vertices

        # Midpoints of left and right edges
        ml = ((tl[0] + bl[0]) / 2.0, (tl[1] + bl[1]) / 2.0)
        mr = ((tr[0] + br[0]) / 2.0, (tr[1] + br[1]) / 2.0)

        # Top half: TL, TR, MR, ML
        top_seg = Segment(
            label=seg.label,
            vertices=[tl, tr, mr, ml],
        )
        # Bottom half: ML, MR, BR, BL
        bot_seg = Segment(
            label=self._page_data.next_label(self._offset),
            vertices=[ml, mr, br, bl],
        )

        # Replace original with top, insert bottom after
        idx = self._active_segment_idx
        self._page_data.segments[idx] = top_seg
        self._page_data.segments.insert(idx + 1, bot_seg)

        self._active_segment_idx = idx  # keep top selected
        self.segments_changed.emit()
        self.segment_selected.emit(idx)
        self.viewport().update()

    def _separator_at(self, img_x: float, img_y: float) -> tuple:
        """Return (separator_index, endpoint) near (img_x, img_y), or (-1, '').

        endpoint is 'top', 'bottom', or 'line' (grab anywhere along the line).
        """
        if not self._page_data or not self._page_data.column_separators:
            return -1, ""
        if not self._pixmap_item:
            return -1, ""
        tol = SEPARATOR_HIT_TOLERANCE / max(self._zoom, 0.01)
        img_h = self._pixmap_item.pixmap().height()
        handle_radius = tol * 1.5  # slightly larger for endpoint handles

        for i, (x_top, x_bot) in enumerate(self._page_data.column_separators):
            # Check top endpoint
            if (img_x - x_top) ** 2 + img_y ** 2 <= handle_radius ** 2:
                return i, "top"
            # Check bottom endpoint
            if (img_x - x_bot) ** 2 + (img_y - img_h) ** 2 <= handle_radius ** 2:
                return i, "bottom"
            # Check along the line body
            if img_h > 0:
                t = img_y / img_h
                line_x = x_top + (x_bot - x_top) * t
                if abs(img_x - line_x) <= tol:
                    return i, "line"  # move entire separator horizontally
        return -1, ""

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
            # Check separator hit first
            sep_idx, sep_ep = self._separator_at(img_x, img_y)
            if sep_idx >= 0:
                self._dragging_separator = sep_idx
                self._dragging_sep_endpoint = sep_ep
                self.setCursor(Qt.CursorShape.SplitHCursor)
                return

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
        if self._dragging_separator >= 0 and self._page_data:
            pos = self.mapToScene(event.pos())
            new_x = pos.x()
            # Clamp to image bounds
            if self._pixmap_item:
                new_x = max(0.0, min(new_x, self._pixmap_item.pixmap().width()))
            sep = self._page_data.column_separators[self._dragging_separator]
            if self._dragging_sep_endpoint == "bottom":
                self._page_data.column_separators[self._dragging_separator] = (sep[0], new_x)
            elif self._dragging_sep_endpoint == "top":
                self._page_data.column_separators[self._dragging_separator] = (new_x, sep[1])
            else:  # "line" — shift both endpoints by the same delta
                mid_x = (sep[0] + sep[1]) / 2.0
                if self._pixmap_item:
                    img_h = self._pixmap_item.pixmap().height()
                    t = max(0.0, min(pos.y() / img_h, 1.0)) if img_h > 0 else 0.5
                else:
                    t = 0.5
                current_line_x = sep[0] + (sep[1] - sep[0]) * t
                dx = new_x - current_line_x
                new_top = sep[0] + dx
                new_bot = sep[1] + dx
                if self._pixmap_item:
                    w = self._pixmap_item.pixmap().width()
                    new_top = max(0.0, min(new_top, w))
                    new_bot = max(0.0, min(new_bot, w))
                self._page_data.column_separators[self._dragging_separator] = (new_top, new_bot)
            self.viewport().update()
            return

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
            if self._dragging_separator >= 0:
                self._dragging_separator = -1
                self._dragging_sep_endpoint = ""
                # Re-sort separators by midpoint after drag
                if self._page_data:
                    self._page_data.column_separators.sort(
                        key=lambda s: (s[0] + s[1]) / 2.0
                    )
                    self.separators_changed.emit()
                self.setCursor(
                    Qt.CursorShape.CrossCursor if self._tool == "segment" else Qt.CursorShape.ArrowCursor
                )
                self.viewport().update()
                return
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

        # Draw column separators
        if self._page_data.column_separators and self._pixmap_item:
            img_h = self._pixmap_item.pixmap().height()
            for i, (x_top, x_bot) in enumerate(self._page_data.column_separators):
                top = QPointF(self.mapFromScene(QPointF(x_top, 0)))
                bot = QPointF(self.mapFromScene(QPointF(x_bot, img_h)))
                is_active = i == self._dragging_separator
                pen = QPen(SEPARATOR_COLOR_ACTIVE if is_active else SEPARATOR_COLOR, 2)
                pen.setCosmetic(True)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(top, bot)
                # Handles at both endpoints
                painter.setBrush(QBrush(SEPARATOR_COLOR_ACTIVE if is_active else SEPARATOR_COLOR))
                painter.drawEllipse(top, 5, 5)
                painter.drawEllipse(bot, 5, 5)

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
