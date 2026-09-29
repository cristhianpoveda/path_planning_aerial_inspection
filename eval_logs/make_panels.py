#!/usr/bin/env python3
"""
Resolution target panel generator.

Emits print-exact vector PDFs for the line-pair panels used to measure w_min,
and for the static depth-of-field dwells that fit k_dof, b_0, d_f and the
[d_min, d_max] band.

Usage:     python make_panels.py                 # 5 panels -> panels.pdf
           python make_panels.py -n 8 -o out.pdf
           python make_panels.py --separate      # one file per panel
"""

import argparse

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Panel specification. Every dimension is in millimetres on the printed sheet.
# ---------------------------------------------------------------------------

BAR_MIN_MM = 0.75          # floor group: unresolvable everywhere in the band
BAR_MAX_MM = 4.00          # ceiling group: resolvable everywhere in the band
N_GROUPS = 12              # geometric ladder between the two, inclusive
BARS_PER_GROUP = 6         # bar width equals gap width
BAR_LENGTH_MM = 20.0

N_EDGES = 4                # slanted-edge squares
EDGE_ANGLE_DEG = 5.0
EDGE_W_MM = 34.0
EDGE_H_MM = 54.0

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_SIZE_MM = 40.0       # marker side, excluding the white quiet zone
ARUCO_QUIET_MODULES = 1    # quiet zone in marker modules, per the standard
MARKERS_PER_PANEL = 4

SCALE_BAR_MM = 100.0
SCALE_TICK_MM = 10.0

MARGIN_MM = 8.0

PAGE_W_MM = A4[0] / mm
PAGE_H_MM = A4[1] / mm


def bar_ladder(lo=BAR_MIN_MM, hi=BAR_MAX_MM, n=N_GROUPS):
    """Geometric ladder of bar widths, so each rung is an equal ratio apart.

    A linear ladder wastes rungs at the coarse end, where every viewpoint
    resolves the bars, and has too few where w_min actually falls.
    """
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def aruco_bits(marker_id, dictionary=ARUCO_DICT):
    """Module grid for a marker, including its black border. 1 = white."""
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    side = int(np.sqrt(d.markerSize ** 2)) + 2      # payload + 1 border module
    img = cv2.aruco.generateImageMarker(d, marker_id, side)
    return (np.asarray(img) // 255).astype(int)


# ---------------------------------------------------------------------------
# Drawing primitives. All take and return millimetres.
# ---------------------------------------------------------------------------

def draw_marker(c, marker_id, x, y, size=ARUCO_SIZE_MM):
    """Draw one ArUco marker with its quiet zone, bottom-left at (x, y)."""
    bits = aruco_bits(marker_id)
    n = bits.shape[0]
    module = size / n
    quiet = ARUCO_QUIET_MODULES * module

    c.setFillGray(1.0)
    c.rect((x - quiet) * mm, (y - quiet) * mm,
           (size + 2 * quiet) * mm, (size + 2 * quiet) * mm,
           stroke=0, fill=1)

    c.setFillGray(0.0)
    for r in range(n):
        for col in range(n):
            if bits[r, col] == 0:                    # black module
                # image rows run top-down; PDF y runs bottom-up
                mx = x + col * module
                my = y + (n - 1 - r) * module
                c.rect(mx * mm, my * mm, module * mm, module * mm,
                       stroke=0, fill=1)


def draw_bar_group(c, bar_w, cx, cy, length=BAR_LENGTH_MM,
                   n_bars=BARS_PER_GROUP):
    """Vertical bars, width equal to gap, centred on (cx, cy)."""
    span = (2 * n_bars - 1) * bar_w
    x0 = cx - span / 2
    y0 = cy - length / 2

    c.setFillGray(0.0)
    for i in range(n_bars):
        c.rect((x0 + 2 * i * bar_w) * mm, y0 * mm,
               bar_w * mm, length * mm, stroke=0, fill=1)

    c.setFont("Helvetica", 6)
    c.setFillGray(0.0)
    c.drawCentredString(cx * mm, (y0 - 4.5) * mm, f"{bar_w:.2f} mm")


def draw_slanted_square(c, cx, cy, w=EDGE_W_MM, h=EDGE_H_MM,
                        angle=EDGE_ANGLE_DEG):
    """Black rectangle rotated off-axis, giving four slanted edges for MTF50."""
    c.saveState()
    c.translate(cx * mm, cy * mm)
    c.rotate(angle)
    c.setFillGray(0.0)
    c.rect((-w / 2) * mm, (-h / 2) * mm, w * mm, h * mm, stroke=0, fill=1)
    c.restoreState()


def draw_scale_bar(c, cx, cy, length=SCALE_BAR_MM, tick=SCALE_TICK_MM):
    """Ruler for verifying that the print was not scaled."""
    x0 = cx - length / 2
    c.setLineWidth(0.4)
    c.setStrokeGray(0.0)
    c.line(x0 * mm, cy * mm, (x0 + length) * mm, cy * mm)

    n = int(round(length / tick))
    for i in range(n + 1):
        x = x0 + i * tick
        h = 3.0 if i % 5 == 0 else 1.8
        c.line(x * mm, cy * mm, x * mm, (cy + h) * mm)

    c.setFont("Helvetica", 6)
    c.drawCentredString(cx * mm, (cy - 4.0) * mm,
                        f"{length:.0f} mm - verify with calipers; "
                        f"print at 100 %, scaling off")


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def draw_panel(c, panel_index, label=None):
    """One panel on the current page. Marker IDs are 4*index .. 4*index+3."""
    label = label or f"P{panel_index + 1}"
    ids = [MARKERS_PER_PANEL * panel_index + k
           for k in range(MARKERS_PER_PANEL)]

    m = MARGIN_MM
    a = ARUCO_SIZE_MM

    # corner markers: bottom-left, bottom-right, top-left, top-right
    positions = [(m, m),
                 (PAGE_W_MM - m - a, m),
                 (m, PAGE_H_MM - m - a),
                 (PAGE_W_MM - m - a, PAGE_H_MM - m - a)]
    for marker_id, (x, y) in zip(ids, positions):
        draw_marker(c, marker_id, x, y)

    # panel label, top centre, between the upper markers
    c.setFont("Helvetica-Bold", 13)
    c.setFillGray(0.0)
    c.drawCentredString(PAGE_W_MM / 2 * mm, (PAGE_H_MM - m - 18) * mm, label)
    c.setFont("Helvetica", 6)
    c.drawCentredString(PAGE_W_MM / 2 * mm, (PAGE_H_MM - m - 25) * mm,
                        f"ArUco DICT_4X4_50, IDs "
                        f"{ids[0]}-{ids[-1]}")

    # line-pair groups, 3 columns x 4 rows
    widths = bar_ladder()
    cols, rows = 3, 4
    # grid_top clears the upper markers plus their quiet zone and the label
    grid_top, grid_bottom = 243.0, 128.0
    cell_w = (PAGE_W_MM - 2 * m) / cols
    cell_h = (grid_top - grid_bottom) / rows

    for i, bw in enumerate(widths):
        r, col = divmod(i, cols)
        cx = m + (col + 0.5) * cell_w
        cy = grid_top - (r + 0.5) * cell_h
        draw_bar_group(c, bw, cx, cy)

    # slanted-edge squares, one row
    edge_y = 88.0
    for i in range(N_EDGES):
        cx = m + (i + 0.5) * (PAGE_W_MM - 2 * m) / N_EDGES
        draw_slanted_square(c, cx, edge_y)

    c.setFont("Helvetica", 6)
    c.drawCentredString(PAGE_W_MM / 2 * mm, (edge_y - EDGE_H_MM / 2 - 6) * mm,
                        f"slanted edges, {EDGE_ANGLE_DEG:.0f} deg, for MTF50")

    # scale bar, between the lower markers
    draw_scale_bar(c, PAGE_W_MM / 2, 34.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--panels", type=int, default=5,
                    help="number of panels to generate (default 5)")
    ap.add_argument("-o", "--output", default="panels.pdf",
                    help="output PDF (default panels.pdf)")
    ap.add_argument("--separate", action="store_true",
                    help="write one file per panel instead of one document")
    args = ap.parse_args()

    if args.separate:
        for i in range(args.panels):
            name = args.output.replace(".pdf", f"_P{i + 1}.pdf")
            c = canvas.Canvas(name, pagesize=A4)
            c.setTitle(f"Resolution panel P{i + 1}")
            draw_panel(c, i)
            c.showPage()
            c.save()
            print(f"wrote {name}")
    else:
        c = canvas.Canvas(args.output, pagesize=A4)
        c.setTitle("Resolution panels")
        for i in range(args.panels):
            draw_panel(c, i)
            c.showPage()
        c.save()
        print(f"wrote {args.output}, {args.panels} panels")

    print("\nbar widths (mm): " +
          ", ".join(f"{w:.2f}" for w in bar_ladder()))
    print(f"marker IDs: panel k uses {MARKERS_PER_PANEL}k .. "
          f"{MARKERS_PER_PANEL}k+{MARKERS_PER_PANEL - 1}")


if __name__ == "__main__":
    main()