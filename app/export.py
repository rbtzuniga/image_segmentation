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


def export_all(
    pages: List[PageData],
    output_folder: str,
    prefix: str = "segment",
    ext: str = "png",
) -> List[str]:
    """Export every segment from every page. Returns list of saved file paths."""
    saved: List[str] = []
    for page_idx, page in enumerate(pages):
        page_num = page_idx + 1  # 1-based
        for seg in page.segments:
            if len(seg.vertices) != 4:
                continue
            warped = crop_quadrilateral(page.file_path, seg.vertices)
            path = save_segment_image(warped, output_folder, prefix, page_num, seg.label, ext)
            saved.append(path)
    return saved
