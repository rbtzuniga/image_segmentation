"""Automatic paragraph-like segment detection.

Bottom-up approach:
1. Binarise → find connected components (blobs of ink).
2. Merge nearby blobs into word-level boxes using a small horizontal gap.
3. Cluster word-boxes into paragraph blocks using spatial proximity:
   - horizontally overlapping boxes that are vertically close → same paragraph
   - uses Union-Find for efficient clustering.
4. Filter out tiny or page-spanning blocks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np

from app.segment import Segment, PageData

BBox = Tuple[int, int, int, int]


# ── Union-Find ───────────────────────────────────────────────────────────

class _UF:
    """Minimal Union-Find (disjoint set) structure."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ── Helpers ──────────────────────────────────────────────────────────────

def _estimate_line_height(binary: np.ndarray) -> Tuple[int, int]:
    """Estimate median text-line height and median inter-line gap.

    Returns (median_line_h, median_gap).
    """
    h, w = binary.shape
    row_sums = np.sum(binary, axis=1) / 255
    threshold = w * 0.003
    text_rows = row_sums > threshold

    line_heights: List[int] = []
    gap_heights: List[int] = []
    in_run = False
    run_start = 0
    last_run_end = 0
    for i, is_text in enumerate(text_rows):
        if is_text and not in_run:
            in_run = True
            run_start = i
            if last_run_end > 0 and (i - last_run_end) > 0:
                gap_heights.append(i - last_run_end)
        elif not is_text and in_run:
            in_run = False
            line_heights.append(i - run_start)
            last_run_end = i
    if in_run:
        line_heights.append(h - run_start)

    if not line_heights:
        return max(h // 50, 8), 4
    med_line = max(int(np.median(line_heights)), 5)

    if gap_heights:
        # Sort gaps to find the natural break between
        # "within-paragraph" spacing and "between-paragraph" spacing.
        sorted_gaps = sorted(gap_heights)
        # Use median as the baseline within-paragraph gap
        med_gap = max(int(np.median(sorted_gaps)), 2)

        # Look for a natural jump in gap sizes to find the paragraph break point.
        # If there's a gap that's ≥ 1.8× the median, that's a paragraph separator.
        # Set threshold just below that jump.
        big_gaps = [g for g in sorted_gaps if g > med_gap * 1.8]
        if big_gaps:
            # Threshold = midpoint between median gap and smallest "big" gap
            para_gap = min(big_gaps)
            threshold_gap = (med_gap + para_gap) // 2
        else:
            threshold_gap = med_gap
    else:
        med_gap = med_line // 3
        threshold_gap = med_gap

    return med_line, threshold_gap


def _horizontal_gap(a: BBox, b: BBox) -> float:
    """Horizontal gap between two boxes (negative = overlap)."""
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    a_right = ax + aw
    b_right = bx + bw
    # gap from right edge of left box to left edge of right box
    if ax < bx:
        return bx - a_right
    else:
        return ax - b_right


def _horizontal_overlap(a: BBox, b: BBox) -> float:
    """Return the horizontal overlap ratio between two boxes (0–1)."""
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    a_left, a_right = ax, ax + aw
    b_left, b_right = bx, bx + bw
    overlap = max(0, min(a_right, b_right) - max(a_left, b_left))
    min_width = min(aw, bw)
    if min_width <= 0:
        return 0
    return overlap / min_width


def _vertical_gap(a: BBox, b: BBox) -> float:
    """Signed vertical gap between two boxes (negative = overlap)."""
    _, ay, _, ah = a
    _, by, _, bh = b
    a_bot = ay + ah
    b_top = by
    # Ensure a is the upper box
    if ay > by:
        a_bot = by + bh
        b_top = ay
    return b_top - a_bot


# ── Column estimation ────────────────────────────────────────────────────

def estimate_columns(image_path: str) -> int:
    """Estimate the number of text columns in a scanned document image.

    Uses a vertical projection profile: columns are separated by gutters
    (vertical bands with very little ink).  The number of distinct content
    bands equals the number of columns.

    Returns 1 if estimation fails or the image has a single column.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 1

    h, w = img.shape
    # Binarise – invert so text = white
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Vertical projection: sum each column of pixels
    col_sums = np.sum(binary, axis=0) / 255  # number of text pixels per column

    # Smooth the profile to ignore small noise
    kernel_size = max(w // 80, 5) | 1  # odd number
    smoothed = np.convolve(col_sums, np.ones(kernel_size) / kernel_size, mode="same")

    # Threshold: columns with ink density below a fraction of the mean are "gutter"
    mean_val = np.mean(smoothed)
    if mean_val < 1:
        return 1
    gutter_thresh = mean_val * 0.15

    is_content = smoothed > gutter_thresh

    # Trim margins: ignore the outer 5% on each side
    margin = int(w * 0.05)
    is_content[:margin] = False
    is_content[-margin:] = False

    # Count content runs (each run = one column)
    n_columns = 0
    in_run = False
    run_width = 0
    min_run_width = max(w // 30, 10)  # columns must be at least this wide

    for val in is_content:
        if val and not in_run:
            in_run = True
            run_width = 1
        elif val and in_run:
            run_width += 1
        elif not val and in_run:
            if run_width >= min_run_width:
                n_columns += 1
            in_run = False
            run_width = 0
    if in_run and run_width >= min_run_width:
        n_columns += 1

    return max(n_columns, 1)


def estimate_column_separators(
    image_path: str,
    n_columns: int,
    content_bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
) -> List[Tuple[float, float]]:
    """Estimate the x-positions of column separators.

    Uses a vertical projection profile to find the deepest gutters
    between content bands.  Returns a list of (n_columns - 1) pairs
    of (x_top, x_bottom) — initially vertical (same x for both endpoints).
    Falls back to evenly-spaced positions within content bounds if detection
    has low confidence.
    """
    if n_columns <= 1:
        return []

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    col_sums = np.sum(binary, axis=0) / 255
    kernel_size = max(w // 80, 5) | 1
    smoothed = np.convolve(col_sums, np.ones(kernel_size) / kernel_size, mode="same")

    n_seps = n_columns - 1

    # Determine content area bounds
    if content_bounds:
        left_bound, right_bound = content_bounds
        # Use average of top/bottom for vertical lines
        left_x = (left_bound[0] + left_bound[1]) / 2.0
        right_x = (right_bound[0] + right_bound[1]) / 2.0
    else:
        # Fallback: use 5% margins
        left_x = w * 0.05
        right_x = w * 0.95

    content_width = right_x - left_x

    def _equidistant_separators() -> List[Tuple[float, float]]:
        """Return separators equally spaced within content bounds."""
        return [
            (left_x + content_width * (i + 1) / n_columns,
             left_x + content_width * (i + 1) / n_columns)
            for i in range(n_seps)
        ]

    # Find gutter regions (low ink density)
    mean_val = np.mean(smoothed)
    if mean_val < 1:
        return _equidistant_separators()

    gutter_thresh = mean_val * 0.25
    margin = int(left_x)  # Use content bound as margin
    right_margin = int(w - right_x)

    # Build a "gutter score" — lower ink = deeper gutter
    gutter_window = max(w // 40, 10)
    gutter_score = np.convolve(
        smoothed, np.ones(gutter_window) / gutter_window, mode="same"
    )

    # Mask out areas outside content bounds
    gutter_score[:max(margin, 1)] = np.inf
    gutter_score[-max(right_margin, 1):] = np.inf

    # Find the n_seps best (lowest-score) gutter positions
    separators: List[float] = []
    min_sep_distance = int(content_width // (n_columns * 2))

    for _ in range(n_seps):
        idx = int(np.argmin(gutter_score))
        if gutter_score[idx] == np.inf:
            break
        separators.append(float(idx))
        lo = max(0, idx - min_sep_distance)
        hi = min(len(gutter_score), idx + min_sep_distance)
        gutter_score[lo:hi] = np.inf

    # Confidence check: verify separators are reasonable
    if len(separators) == n_seps:
        separators.sort()
        expected_spacing = content_width / n_columns
        low_confidence = False

        for i, sep_x in enumerate(separators):
            # Check if separator is within content bounds
            if sep_x < left_x or sep_x > right_x:
                low_confidence = True
                break
            # Check if separator is roughly where expected (within 40% tolerance)
            expected_pos = left_x + expected_spacing * (i + 1)
            if abs(sep_x - expected_pos) > expected_spacing * 0.4:
                low_confidence = True
                break

        # Check spacing between separators is roughly equal
        if not low_confidence and n_seps > 1:
            gaps = [separators[i+1] - separators[i] for i in range(n_seps - 1)]
            avg_gap = sum(gaps) / len(gaps)
            for gap in gaps:
                if abs(gap - avg_gap) > avg_gap * 0.5:
                    low_confidence = True
                    break

        if low_confidence:
            return _equidistant_separators()

        return [(x, x) for x in separators]

    # Didn't find enough separators — use equidistant
    return _equidistant_separators()


def estimate_columns_sample(file_paths: List[str], max_sample: int = 5) -> int:
    """Estimate column count from a sample of pages (uses the mode)."""
    if not file_paths:
        return 1
    import random
    sample = file_paths if len(file_paths) <= max_sample else random.sample(file_paths, max_sample)
    counts = [estimate_columns(p) for p in sample]
    if not counts:
        return 1
    # Return the mode (most common value)
    from collections import Counter
    return Counter(counts).most_common(1)[0][0]


def estimate_content_bounds(image_path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Estimate the left and right content boundaries of a scanned document.

    Uses vertical projection profile to find where the content starts and ends.
    Returns ((left_top, left_bottom), (right_top, right_bottom)) — initially vertical.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ((0.0, 0.0), (100.0, 100.0))

    h, w = img.shape
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    col_sums = np.sum(binary, axis=0) / 255
    kernel_size = max(w // 80, 5) | 1
    smoothed = np.convolve(col_sums, np.ones(kernel_size) / kernel_size, mode="same")

    mean_val = np.mean(smoothed)
    if mean_val < 1:
        # No content detected, return small margins
        return ((w * 0.05, w * 0.05), (w * 0.95, w * 0.95))

    # Threshold for detecting content
    content_thresh = mean_val * 0.1

    # Find left boundary: first column with significant content
    left_x = 0
    for i in range(w):
        if smoothed[i] > content_thresh:
            left_x = max(0, i - 5)  # Small padding
            break

    # Find right boundary: last column with significant content
    right_x = w
    for i in range(w - 1, -1, -1):
        if smoothed[i] > content_thresh:
            right_x = min(w, i + 5)  # Small padding
            break

    return ((float(left_x), float(left_x)), (float(right_x), float(right_x)))


def _sep_x_at_y(
    sep: Tuple[float, float],
    y: float,
    img_height: int,
) -> float:
    """Interpolate separator x-position at a given y.

    sep is (x_top, x_bottom).  y=0 → x_top, y=img_height → x_bottom.
    """
    x_top, x_bot = sep
    if img_height <= 0:
        return (x_top + x_bot) / 2.0
    t = y / img_height
    return x_top + (x_bot - x_top) * t


def _column_sort_key(
    box: Tuple[float, ...],
    n_columns: int,
    img_width: int,
    separators: List[Tuple[float, float]] | None = None,
    img_height: int = 0,
) -> Tuple[int, float]:
    """Return (column_index, centroid_y) for column-aware reading order.

    Uses explicit separator positions when available, otherwise divides
    the image width equally into *n_columns* bins.
    """
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    if n_columns <= 1:
        return (0, cy)
    col_idx = _column_for_x(cx, cy, n_columns, img_width, separators, img_height)
    return (col_idx, cy)


def _column_for_x(
    x: float,
    y: float,
    n_columns: int,
    img_width: int,
    separators: List[Tuple[float, float]] | None = None,
    img_height: int = 0,
) -> int:
    """Return the 0-based column index for a point (x, y)."""
    if n_columns <= 1:
        return 0
    if separators and len(separators) == n_columns - 1:
        for i, sep in enumerate(separators):
            sx = _sep_x_at_y(sep, y, img_height)
            if x < sx:
                return i
        return n_columns - 1
    # Fallback: equal width columns
    col_width = img_width / n_columns
    return min(int(x / col_width), n_columns - 1)


def _boxes_cross_separator(
    box: BBox,
    separators: List[Tuple[float, float]],
    img_height: int = 0,
) -> bool:
    """Return True if the box spans across any separator line."""
    bx, by, bw, bh = box
    left, right = bx, bx + bw
    cy = by + bh / 2.0
    for sep in separators:
        sx = _sep_x_at_y(sep, cy, img_height)
        if left < sx < right:
            return True
    return False


# ── Main detection ───────────────────────────────────────────────────────

def detect_paragraphs(
    image_path: str,
    min_lines: int = 2,
    merge_sensitivity: float = 1.0,
    horizontal_reach: float = 1.0,
    n_columns: int = 1,
    separators: List[float] | None = None,
    content_bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
) -> List[BBox]:
    """Detect paragraph-like text blocks in a scanned document image.

    Parameters
    ----------
    min_lines : int
        Minimum text lines for a block to be kept.
    merge_sensitivity : float
        Multiplier for vertical merge gap (>1 = merge more, <1 = split more).
    horizontal_reach : float
        Multiplier for horizontal dilation (>1 = merge wider, <1 = tighter).
    n_columns : int
        Expected number of text columns.  Used to cap horizontal dilation
        and set a tighter cross-column gap threshold.
    separators : list of float, optional
        Explicit x-positions of column separator lines.  When provided,
        boxes that span across a separator are never merged, and the
        separator positions are used for column-aware sorting.
    content_bounds : tuple, optional
        ((left_top, left_bottom), (right_top, right_bottom)) content area bounds.
        Boxes that fall mostly outside these bounds will be filtered out.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape

    # 1. Binarise (Otsu) – invert so text = white
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Estimate line height and inter-line gap
    line_h, line_gap = _estimate_line_height(binary)

    # 3. Light horizontal dilation to merge characters into words/short phrases
    #    Use half the line height — enough to connect characters within a word
    #    but NOT enough to bridge column gutters.
    #    When column count is known, cap the kernel to stay within a single column.
    h_kern_w = max(int(line_h * 0.5 * horizontal_reach), 5)
    if n_columns > 1:
        col_width = w // n_columns
        h_kern_w = min(h_kern_w, max(col_width // 6, 5))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kern_w, 1))
    word_mask = cv2.dilate(binary, h_kernel, iterations=1)

    # 4. Find word-level connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(word_mask, connectivity=8)

    # Collect word boxes (skip background label 0)
    word_boxes: List[BBox] = []
    for i in range(1, num_labels):
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # Skip tiny noise
        if area < line_h * 2:
            continue
        if bw < 3 or bh < 3:
            continue
        word_boxes.append((bx, by, bw, bh))

    if not word_boxes:
        return []

    # 5. Cluster word-boxes into paragraph blocks via Union-Find.
    #    Two boxes belong to the same paragraph if:
    #    - They overlap horizontally by ≥ 30% (or horizontal gap is small)
    #    - Their vertical gap is small (within normal line spacing)
    #    - They are NOT separated by a wide horizontal gap (column gutter)
    n = len(word_boxes)
    uf = _UF(n)

    # Allow merging if vertical gap ≤ threshold_gap
    max_v_gap = int((line_gap + max(line_gap // 3, 2)) * merge_sensitivity)

    # Maximum horizontal gap to allow merging — if boxes are farther apart
    # horizontally than this, they're in different columns.
    # When column count is known, use a fraction of the expected column width
    # so we never merge across gutters.
    if n_columns > 1:
        col_width = w // n_columns
        max_h_gap = min(int(line_h * 2 * horizontal_reach),
                        int(col_width * 0.3))
    else:
        max_h_gap = int(line_h * 2 * horizontal_reach)

    # Sort by y for efficient pairwise comparison
    sorted_indices = sorted(range(n), key=lambda i: word_boxes[i][1])

    for idx_a in range(len(sorted_indices)):
        i = sorted_indices[idx_a]
        box_a = word_boxes[i]
        _, ay, _, ah = box_a
        a_bottom = ay + ah

        # Only compare with boxes that start within max_v_gap below box_a's bottom
        for idx_b in range(idx_a + 1, len(sorted_indices)):
            j = sorted_indices[idx_b]
            box_b = word_boxes[j]
            _, by, _, _ = box_b

            # If box_b starts too far below, stop (remaining are even further)
            if by > a_bottom + max_v_gap:
                break

            v_gap = _vertical_gap(box_a, box_b)
            if v_gap > max_v_gap:
                continue

            # Check horizontal proximity — reject if boxes are in different columns
            h_gap = _horizontal_gap(box_a, box_b)
            if h_gap > max_h_gap:
                continue

            # Never merge boxes that sit in different columns (across a separator)
            if separators:
                cx_a = box_a[0] + box_a[2] / 2.0
                cy_a = box_a[1] + box_a[3] / 2.0
                cx_b = box_b[0] + box_b[2] / 2.0
                cy_b = box_b[1] + box_b[3] / 2.0
                col_a = _column_for_x(cx_a, cy_a, n_columns, w, separators, h)
                col_b = _column_for_x(cx_b, cy_b, n_columns, w, separators, h)
                if col_a != col_b:
                    continue

            h_overlap = _horizontal_overlap(box_a, box_b)

            # Merge if: good horizontal overlap (same column, stacked lines)
            if h_overlap >= 0.3:
                uf.union(i, j)
            # OR: boxes vertically overlap/touch AND are horizontally close
            # (catches small labels like "R", "S" sitting just left of an entry)
            elif v_gap <= 0 and h_gap <= line_h * 1.5:
                uf.union(i, j)

    # 6. Build merged bounding boxes for each cluster
    clusters: dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    result_boxes: List[BBox] = []
    min_block_h = line_h * min_lines * 0.6

    for members in clusters.values():
        if not members:
            continue
        xs = [word_boxes[i][0] for i in members]
        ys = [word_boxes[i][1] for i in members]
        x2s = [word_boxes[i][0] + word_boxes[i][2] for i in members]
        y2s = [word_boxes[i][1] + word_boxes[i][3] for i in members]

        bx = min(xs)
        by = min(ys)
        bw = max(x2s) - bx
        bh = max(y2s) - by

        # Filter
        if bh < min_block_h:
            continue
        if bw < line_h:
            continue
        if bw * bh > w * h * 0.8:
            continue  # skip near-full-page

        # Filter by content bounds if provided
        if content_bounds:
            left_bound, right_bound = content_bounds
            box_cy = by + bh / 2.0
            # Interpolate left and right bounds at the box's vertical center
            left_x = left_bound[0] + (left_bound[1] - left_bound[0]) * (box_cy / h) if h > 0 else left_bound[0]
            right_x = right_bound[0] + (right_bound[1] - right_bound[0]) * (box_cy / h) if h > 0 else right_bound[0]
            box_cx = bx + bw / 2.0
            # Skip if the box center is outside the content area
            if box_cx < left_x or box_cx > right_x:
                continue

        # Small padding
        pad = 3
        bx = max(0, bx - pad)
        by = max(0, by - pad)
        bw = min(w - bx, bw + 2 * pad)
        bh = min(h - by, bh + 2 * pad)
        result_boxes.append((bx, by, bw, bh))

    # Sort in reading order: column by column (left-to-right),
    # then top-to-bottom within each column.
    result_boxes.sort(key=lambda b: _column_sort_key(b, n_columns, w, separators, h))
    return result_boxes


def auto_segment_page(
    page: PageData,
    offset: int,
    min_lines: int = 2,
    merge_sensitivity: float = 1.0,
    horizontal_reach: float = 1.0,
    n_columns: int = 1,
    separators: List[Tuple[float, float]] | None = None,
    content_bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
) -> int:
    """Run auto-detection on a single page and add segments.

    Clears any existing segments on the page before detection.
    Returns the number of new segments added.
    """
    # Clear existing segments and reset counter
    page.segments.clear()
    page._counter = 0

    boxes = detect_paragraphs(
        page.file_path,
        min_lines=min_lines,
        merge_sensitivity=merge_sensitivity,
        horizontal_reach=horizontal_reach,
        n_columns=n_columns,
        separators=separators,
        content_bounds=content_bounds,
    )
    added = 0
    for x, y, bw, bh in boxes:
        label = page.next_label(offset)
        seg = Segment(
            label=label,
            vertices=[
                (float(x), float(y)),
                (float(x + bw), float(y)),
                (float(x + bw), float(y + bh)),
                (float(x), float(y + bh)),
            ],
        )
        page.segments.append(seg)
        added += 1
    return added


def relabel_page(
    page: PageData, offset: int, n_columns: int = 1,
    separators: List[Tuple[float, float]] | None = None,
) -> None:
    """Re-label all segments on a page in column-aware reading order.

    Assigns each segment to a column bin based on its centroid-x,
    then sorts by (column_index, centroid_y) and assigns sequential labels.
    """
    if not page.segments:
        return

    # Determine image dimensions from the image header
    try:
        img = cv2.imread(page.file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_h, img_w = img.shape
        else:
            img_h, img_w = 1, 1
    except Exception:
        img_h, img_w = 1, 1

    def _seg_sort_key(seg: Segment) -> Tuple[int, float]:
        xs = [v[0] for v in seg.vertices]
        ys = [v[1] for v in seg.vertices]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        col_idx = _column_for_x(cx, cy, n_columns, img_w, separators, img_h)
        return (col_idx, cy)

    page.segments.sort(key=_seg_sort_key)

    # Reassign labels:
    #   - Normal segments and "top" combined segments get the next label in sequence
    #   - "bottom" combined segments are skipped here (handled by caller)
    page._counter = 0
    for seg in page.segments:
        if seg.combined_role == "bottom":
            continue  # bottom inherits from its top partner – handled externally
        base = page.next_label(offset)
        if seg.combined_role == "top":
            seg.label = f"{base}_top"
        else:
            seg.label = base
