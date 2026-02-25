"""Data model for a single segment (quadrilateral region on a page)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# A point is (x, y) in image-pixel coordinates.
Point = Tuple[float, float]


@dataclass
class Segment:
    """Represents one quadrilateral segment on a page.

    Vertices are stored in order: top-left, top-right, bottom-right, bottom-left
    (though the user can drag them into any convex/concave shape).
    """

    label: str
    vertices: List[Point] = field(default_factory=list)  # exactly 4 points

    # ── helpers ──────────────────────────────────────────────────────────

    def bounding_rect(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) axis-aligned bounding box."""
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return min(xs), min(ys), max(xs), max(ys)

    def contains_point(self, px: float, py: float, margin: float = 0) -> bool:
        """Simple ray-casting point-in-polygon test."""
        n = len(self.vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            # expand by margin
            if margin:
                cx = sum(v[0] for v in self.vertices) / n
                cy = sum(v[1] for v in self.vertices) / n
                xi += (xi - cx) / max(abs(xi - cx), 1e-9) * margin if abs(xi - cx) > 1e-9 else 0
                yi += (yi - cy) / max(abs(yi - cy), 1e-9) * margin if abs(yi - cy) > 1e-9 else 0
                xj += (xj - cx) / max(abs(xj - cx), 1e-9) * margin if abs(xj - cx) > 1e-9 else 0
                yj += (yj - cy) / max(abs(yj - cy), 1e-9) * margin if abs(yj - cy) > 1e-9 else 0
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def vertex_at(self, px: float, py: float, radius: float = 8.0) -> int | None:
        """Return the index of the vertex near (px, py), or None."""
        for i, (vx, vy) in enumerate(self.vertices):
            if (vx - px) ** 2 + (vy - py) ** 2 <= radius ** 2:
                return i
        return None


@dataclass
class PageData:
    """All data associated with one input image / page."""

    file_path: str  # absolute path to the image file
    segments: List[Segment] = field(default_factory=list)
    column_separators: List[float] = field(default_factory=list)  # x-positions of column dividers
    _counter: int = 0  # internal counter for auto-labeling

    def next_label(self, offset: int) -> str:
        """Generate the next automatic label using the current offset."""
        label = str(offset + self._counter)
        self._counter += 1
        return label
