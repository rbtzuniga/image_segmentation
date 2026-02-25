# Image Segmentation Tool

A Python GUI application for segmenting scanned document images into labeled regions. Supports automatic paragraph detection, column-aware labeling, and manual editing of quadrilateral segments.

## Features

- Load all images from a folder (PNG, JPG, TIFF, BMP)
- Thumbnail side panel for quick page navigation (natural sort order)
- Draw rectangular segments on each page, then adjust vertices to form arbitrary quadrilaterals
- **Auto-segment** – automatic paragraph/block detection with tunable parameters
- **Column-aware labeling** – segments are labeled in reading order (column by column, top to bottom)
- **Relabel Page** – reassign labels to existing segments in reading order
- Configurable label offset – new segments get labels `offset + i`
- Manually edit segment labels at any time
- Right-click to toggle between Select and Segment tools
- Delete/Backspace to remove selected segment
- Middle-click or left-click on empty space to pan the canvas
- Export all segments as cropped images with configurable format (PNG / JPG / TIFF) and filename prefix

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+.

## Usage

```bash
python main.py
```

### Basic workflow

1. **Select input folder** – loads all images as pages in the thumbnail panel.
2. **Select output folder** – where exported segments will be saved.
3. Click a thumbnail to display the page on the canvas.
4. Use the **Segment** tool (or right-click to toggle) to draw a rectangle on the canvas.
5. Drag any vertex to reshape the rectangle into an arbitrary quadrilateral.
6. Add as many segments as needed; each gets an auto-incremented label.
7. Adjust the **offset**, **prefix**, or **format** in the settings panel.
8. Click **Save All** to export every segment as `prefix_page_label.ext`.

### Auto-segment

The tool can automatically detect paragraph-like text blocks on scanned document pages.

1. Set the **Columns** spinner to the number of text columns on your pages (auto-estimated on folder load).
2. Click **Auto Segment Page** (current page) or **Auto Segment All Pages**.
3. Existing segments on the affected page(s) are cleared before detection runs.
4. Detected segments are labeled in reading order: all segments in column 1 (top to bottom), then column 2, etc.

#### Tuning parameters

| Parameter | Default | Description |
|---|---|---|
| **Merge sens.** | 100% | How aggressively lines merge vertically into blocks. Higher = larger blocks. |
| **H. reach** | 100% | How far characters merge horizontally. Higher = wider word grouping. |
| **Min lines** | 2 | Minimum text lines for a block to be kept. |
| **Columns** | auto | Expected number of text columns. Constrains horizontal merging. |

> **Tip:** If auto-segment results are poor (segments are too large, merge across columns, or capture the whole page), try setting both **Merge sens.** and **H. reach** to **30–40%**. This produces tighter, more conservative segments that you can then fine-tune manually.

Click **Reset to Defaults** to restore all sliders to their original values.

### Relabeling

After manually adding, deleting, or rearranging segments, click **Relabel Page** to reassign labels in column-aware reading order (left-to-right by column, top-to-bottom within each column), starting from the current label offset.

### Keyboard & mouse shortcuts

| Action | Shortcut |
|---|---|
| Toggle Select / Segment tool | Right-click |
| Delete selected segment | Delete or Backspace |
| Pan canvas | Middle-click drag, or left-click drag on empty space (Select mode) |
| Zoom | Scroll wheel |

## Output naming

Each segment is saved as:

```
{prefix}_{page}_{label}.{ext}
```

- `prefix` – configurable, default `segment`
- `page` – 1-based page number (from thumbnail order)
- `label` – segment label (offset + counter, or manually set)
- `ext` – `png` (default), `jpg`, or `tif`
