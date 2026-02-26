"""Export logic – crop segments from source images and save to the output folder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from app.segment import PageData, Segment


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def crop_quadrilateral(image_path: str, vertices: List[Tuple[float, float]]) -> np.ndarray:
    """Crop & perspective-warp a quadrilateral region from an image.

    Returns the warped region as a BGR numpy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    pts = np.array(vertices, dtype=np.float32)
    ordered = _order_points(pts)

    # Compute destination size
    tl, tr, br, bl = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(max(height_left, height_right))

    if max_w < 1 or max_h < 1:
        raise ValueError("Segment area is too small to crop.")

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, matrix, (max_w, max_h))
    return warped


def save_segment_image(
    warped: np.ndarray,
    output_folder: str,
    prefix: str,
    page_num: int,
    label: str,
    ext: str,
) -> str:
    """Save a warped segment image. Returns the saved file path."""
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{prefix}_{page_num}_{label}.{ext}"
    filepath = os.path.join(output_folder, filename)

    # Convert BGR → RGB for Pillow (better format support)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    save_kwargs = {}
    if ext == "jpg":
        save_kwargs["quality"] = 95
    elif ext == "tif":
        save_kwargs["compression"] = "tiff_lzw"

    pil_img.save(filepath, **save_kwargs)
    return filepath


def _find_combined_pair(
    pages: List[PageData],
    combined_id: str,
) -> tuple:
    """Locate both halves of a combined pair across all pages.

    Returns (top_seg, top_page_idx, bottom_seg, bottom_page_idx).
    Any element may be ``None`` if the partner is missing.
    """
    top_seg = top_idx = bottom_seg = bottom_idx = None
    for pi, page in enumerate(pages):
        for seg in page.segments:
            if seg.combined_id == combined_id:
                if seg.combined_role == "top":
                    top_seg, top_idx = seg, pi
                elif seg.combined_role == "bottom":
                    bottom_seg, bottom_idx = seg, pi
    return top_seg, top_idx, bottom_seg, bottom_idx


def _vstack_with_padding(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Vertically stack two images, padding the narrower one with white."""
    h1, w1 = top.shape[:2]
    h2, w2 = bottom.shape[:2]
    max_w = max(w1, w2)
    channels = top.shape[2] if top.ndim == 3 else 1

    if w1 < max_w:
        pad = np.ones((h1, max_w - w1, channels), dtype=top.dtype) * 255
        top = np.hstack([top, pad])
    if w2 < max_w:
        pad = np.ones((h2, max_w - w2, channels), dtype=bottom.dtype) * 255
        bottom = np.hstack([bottom, pad])
    return np.vstack([top, bottom])


def export_all(
    pages: List[PageData],
    output_folder: str,
    prefix: str = "segment",
    ext: str = "png",
) -> List[str]:
    """Export every segment from every page. Returns list of saved file paths.

    Combined segment pairs are exported as a single vertically-concatenated
    image using the top segment's page number and base label.
    """
    saved: List[str] = []
    exported_combined: set = set()  # combined_ids already exported

    for page_idx, page in enumerate(pages):
        page_num = page_idx + 1  # 1-based
        for seg in page.segments:
            if len(seg.vertices) != 4:
                continue

            if seg.combined_id:
                if seg.combined_id in exported_combined:
                    continue  # already exported this pair
                exported_combined.add(seg.combined_id)

                top_seg, top_pi, bottom_seg, bottom_pi = _find_combined_pair(
                    pages, seg.combined_id,
                )
                if top_seg and bottom_seg and top_pi is not None and bottom_pi is not None:
                    top_img = crop_quadrilateral(
                        pages[top_pi].file_path, top_seg.vertices,
                    )
                    bottom_img = crop_quadrilateral(
                        pages[bottom_pi].file_path, bottom_seg.vertices,
                    )
                    combined_img = _vstack_with_padding(top_img, bottom_img)
                    base_label = top_seg.label.removesuffix("_top")
                    combined_page_num = top_pi + 1
                    path = save_segment_image(
                        combined_img, output_folder, prefix,
                        combined_page_num, base_label, ext,
                    )
                    saved.append(path)
                continue

            # Regular (non-combined) segment
            warped = crop_quadrilateral(page.file_path, seg.vertices)
            path = save_segment_image(warped, output_folder, prefix, page_num, seg.label, ext)
            saved.append(path)
    return saved
