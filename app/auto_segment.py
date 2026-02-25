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
        # The inter-line gap within a paragraph is the smaller gaps.
        # Paragraph-separating gaps are the bigger ones.
        # Use the 25th percentile as "within-paragraph" gap estimate.
        med_gap = max(int(np.percentile(gap_heights, 25)), 2)
    else:
        med_gap = med_line // 3

    return med_line, med_gap


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


# ── Main detection ───────────────────────────────────────────────────────

def detect_paragraphs(
    image_path: str,
    min_lines: int = 2,
) -> List[BBox]:
    """Detect paragraph-like text blocks in a scanned document image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape

    # 1. Binarise (Otsu) – invert so text = white
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Estimate line height and inter-line gap
    line_h, line_gap = _estimate_line_height(binary)

    # 3. Light horizontal dilation to merge characters into words/short phrases
    #    Keep it small so it doesn't bridge columns.
    h_kern_w = max(line_h, 8)
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
    #    - They overlap horizontally by ≥ 30%
    #    - Their vertical gap is small (within normal line spacing)
    #
    #    The key threshold: use the measured inter-line gap (within paragraphs)
    #    plus a small margin — NOT the full line height.  This ensures that
    #    the blank line separating entries (≈ 1 line_h) breaks the cluster.
    n = len(word_boxes)
    uf = _UF(n)

    # Allow merging if vertical gap ≤ line_gap + small margin
    # This bridges normal line spacing but NOT paragraph/entry gaps
    max_v_gap = line_gap + max(line_gap // 2, 2)

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

            h_overlap = _horizontal_overlap(box_a, box_b)
            if h_overlap >= 0.3:
                uf.union(i, j)

    # 6. Build merged bounding boxes for each cluster
    from collections import defaultdict
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

        # Small padding
        pad = 3
        bx = max(0, bx - pad)
        by = max(0, by - pad)
        bw = min(w - bx, bw + 2 * pad)
        bh = min(h - by, bh + 2 * pad)
        result_boxes.append((bx, by, bw, bh))

    # Sort top-to-bottom, then left-to-right
    result_boxes.sort(key=lambda b: (b[1], b[0]))
    return result_boxes


def auto_segment_page(page: PageData, offset: int) -> int:
    """Run auto-detection on a single page and add segments.

    Returns the number of new segments added.
    """
    boxes = detect_paragraphs(page.file_path)
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
