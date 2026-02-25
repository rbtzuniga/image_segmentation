# Image Segmentation Tool

A Python GUI application for segmenting scanned document images into labeled regions.

## Features

- Load all images from a folder (PNG, JPG, TIFF, BMP)
- Thumbnail side panel for quick page navigation (ScanTailor-style)
- Draw rectangular segments on each page, then adjust vertices to form arbitrary quadrilaterals
- Add multiple segments per page with automatic ID labeling
- Configurable label offset – new segments get labels `offset + i`
- Manually edit segment labels at any time
- Export all segments as cropped images with configurable format (PNG / JPG / TIFF) and filename prefix

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. **Select input folder** – loads all images as pages in the thumbnail panel.
2. **Select output folder** – where exported segments will be saved.
3. Click a thumbnail to display the page on the canvas.
4. Use the **Segment** tool to draw a rectangle on the canvas.
5. Drag any vertex to reshape the rectangle into an arbitrary quadrilateral.
6. Add as many segments as needed; each gets an auto-incremented label.
7. Adjust the **offset**, **prefix**, or **format** in the settings panel.
8. Click **Save All** to export every segment as `prefix_page_label.ext`.

## Output naming

Each segment is saved as:

```
{prefix}_{page}_{label}.{ext}
```

- `prefix` – configurable, default `segment`
- `page` – 1-based page number (from thumbnail order)
- `label` – segment label (offset + counter, or manually set)
- `ext` – `png` (default), `jpg`, or `tif`
