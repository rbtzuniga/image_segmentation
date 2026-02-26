# Image Segmentation Tool

A Python GUI application for segmenting scanned document images into labeled regions. Supports automatic paragraph detection, column-aware labeling, and manual editing of quadrilateral segments.

## Demo
Here's a quick demo of the tool in action: 
[Demo](https://www.dropbox.com/scl/fi/b1ss1fh12wutqwujjpmpn/img_seg_demo.mp4?rlkey=oyyyp8puvguri440uq3c5aqp0&st=zqtcoqdx&dl=0)

## Features

- Load all images from a folder (PNG, JPG, TIFF, BMP)
- Thumbnail side panel for quick page navigation (natural sort order)
- Draw rectangular segments on each page, then adjust vertices to form arbitrary quadrilaterals
- **Auto-segment** – automatic paragraph/block detection with tunable parameters
- **Column-aware labeling** – segments are labeled in reading order (column by column, top to bottom)
- **Column separators** – visual draggable separator lines between columns; supports tilted separators for skewed scans
- **Relabel Page** – reassign labels to existing segments in reading order
- **Multi-select** – Ctrl+click to select multiple segments, even across pages
- **Combine segments** – pair two selected segments into a top/bottom unit that exports as one vertically-concatenated image
- **Split segment** – split a selected segment horizontally into top/bottom halves
- **Edge dragging** – drag any edge of a selected segment to resize it with parallel movement
- Configurable label offset – new segments get labels `offset + i`
- Manually edit segment labels at any time
- Right-click to toggle between Select and Segment tools
- Delete/Backspace to remove selected segment
- Press **S** to split the selected segment
- Press **C** to combine two multi-selected segments
- Press **U** to uncombine a combined segment
- Press **R** to relabel the current page
- Middle-click or left-click on empty space to pan the canvas
- Export all segments as cropped images with configurable format (PNG / JPG / TIFF) and filename prefix

## Installation

### Prerequisites

- **Python 3.11 or newer** – Download from [python.org](https://www.python.org/downloads/). During installation on Windows, make sure to check **"Add Python to PATH"**.
- **pip** – comes bundled with Python. You can verify it's available by running `pip --version` in a terminal.

### Steps

1. **Clone or download** this repository and open a terminal in the project folder.

2. *(Recommended)* **Create a virtual environment** to keep dependencies isolated:

   ```bash
   python -m venv venv
   ```

   Then activate it:

   - **Windows (cmd):** `venv\Scripts\activate`
   - **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
   - **macOS / Linux:** `source venv/bin/activate`

   You should see `(venv)` at the start of your terminal prompt.

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This installs PyQt6, Pillow, OpenCV, and NumPy.

4. **Run the application:**

   ```bash
   python main.py
   ```

> **Troubleshooting:** If `python` is not recognized, try `python3` instead. On Windows, you can also use `py -3`.

## Usage

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

After manually adding, deleting, or rearranging segments, click **Relabel Page (press R)** or press **R** to reassign labels in column-aware reading order (left-to-right by column, top-to-bottom within each column), starting from the current label offset.

### Multi-select

Hold **Ctrl** and click segments to toggle them in and out of a multi-selection. Multi-selected segments can span different pages. Actions that operate on the selection (Delete, Combine) will apply to all multi-selected segments.

- **Ctrl+click** a segment to add/remove it from the selection.
- A normal click (without Ctrl) clears the multi-selection.

### Combining segments

Two segments can be paired into a **combined** top/bottom unit so that they export as a single vertically-concatenated image.

1. **Ctrl+click** two segments (on the same page or different pages) to multi-select them.
2. Click **Combine Selected (press C)** or press **C**.
3. The segments are linked: the one with the smaller page number (or leftmost / topmost on the same page) becomes the **top** half, the other becomes the **bottom** half.
4. Labels are automatically suffixed with `_top` and `_bottom`.
5. Combined segments are drawn in **purple** to visually distinguish them from regular segments.
6. On export, the pair is cropped separately, then stacked vertically (with white padding if widths differ) and saved as a single image under the top segment's page and base label.

To **uncombine**, select either segment of a combined pair and click **Uncombine Selected (press U)** or press **U**. Both segments revert to independent segments with their original labels.

### Splitting segments

Select a segment and click **Split Selected Segment (press S)** or press **S** to split it horizontally at its vertical midpoint, creating a top half and a bottom half.

### Column separators

When a folder is loaded, column separators are auto-estimated from the vertical projection profile. You can:

- **Drag the line body** to shift a separator horizontally.
- **Drag an endpoint handle** (top or bottom) to tilt the separator for skewed scans.
- Separators prevent auto-segment from merging blocks across columns.

### Edge & vertex dragging

In **Select** mode with a segment selected:

- **Drag a vertex** (corner) to reshape the quadrilateral.
- **Hover near an edge** to see a directional resize cursor; drag to move that edge with parallel movement.

### UI color coding

- **Light blue** buttons – auto-segment actions (Auto Segment Page, Auto Segment All Pages, Relabel Page)
- **Purple** segments – combined segment pairs
- **Light red** buttons – destructive actions (Delete Selected, Delete All Page, Delete All Pages)

### Keyboard & mouse shortcuts

| Action | Shortcut |
|---|---|
| Toggle Select / Segment tool | Right-click |
| Delete selected segment(s) | Delete or Backspace |
| Split selected segment | S |
| Combine two selected segments | C |
| Uncombine selected segment | U |
| Multi-select toggle | Ctrl+click |
| Relabel current page | R |
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
