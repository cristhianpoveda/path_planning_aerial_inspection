"""Panel geometry, extracted from panels.pdf. All millimetres, page frame.
"""

import numpy as np

PAGE_W_MM, PAGE_H_MM = 210.0, 297.0
ARUCO_DICT = 'DICT_4X4_50'
ARUCO_SIZE_MM = 53.33
SCALE_BAR = dict(x0=55.0, x1=155.0, y=268.0, length_mm=100.0)

# Four corner markers. IDs are 4*(panel-1) + k, k in 0..3, in this order.
ARUCO = [
    dict(x=1.33, y=1.33, size=53.33),
    dict(x=155.33, y=1.33, size=53.33),
    dict(x=1.33, y=242.33, size=53.33),
    dict(x=155.33, y=242.33, size=53.33),
]

# Twelve six-bar groups. Bar width equals gap, so one line pair is 2*w.
BAR_GROUPS = [
    dict(w_mm=0.75, x0=36.21, x1=44.46, y0=57.38, y1=77.38, n_bars=6),
    dict(w_mm=0.87, x0=100.20, x1=109.80, y0=57.38, y1=77.38, n_bars=6),
    dict(w_mm=1.02, x0=164.07, x1=175.26, y0=57.38, y1=77.38, n_bars=6),
    dict(w_mm=1.18, x0=33.82, x1=46.84, y0=84.12, y1=104.12, n_bars=6),
    dict(w_mm=1.38, x0=97.42, x1=112.58, y0=84.12, y1=104.12, n_bars=6),
    dict(w_mm=1.61, x0=160.84, x1=178.50, y0=84.12, y1=104.12, n_bars=6),
    dict(w_mm=1.87, x0=30.05, x1=50.61, y0=110.88, y1=130.88, n_bars=6),
    dict(w_mm=2.18, x0=93.03, x1=116.97, y0=110.88, y1=130.88, n_bars=6),
    dict(w_mm=2.53, x0=155.73, x1=183.60, y0=110.88, y1=130.88, n_bars=6),
    dict(w_mm=2.95, x0=24.11, x1=56.56, y0=137.62, y1=157.62, n_bars=6),
    dict(w_mm=3.44, x0=86.11, x1=123.89, y0=137.62, y1=157.62, n_bars=6),
    dict(w_mm=4.00, x0=147.67, x1=191.67, y0=137.62, y1=157.62, n_bars=6),
]

# Four slanted-edge patches, nominally 5 deg, for MTF50.
SLANTED_EDGES = [
    [[17.668, 226.379], [51.539, 223.416], [46.832, 169.621], [12.961, 172.584]],
    [[66.168, 226.379], [100.039, 223.416], [95.332, 169.621], [61.461, 172.584]],
    [[114.668, 226.379], [148.539, 223.416], [143.832, 169.621], [109.961, 172.584]],
    [[163.168, 226.379], [197.039, 223.416], [192.332, 169.621], [158.461, 172.584]],
]

# Six grey steps, 0/25/50/75/90/100 per cent, for the clipping check.
GREY_PCT = [0, 25, 50, 75, 90, 100]
GREY_STEPS = [
    dict(x=49.00, y=240.00, w=17.00, h=17.00),
    dict(x=68.00, y=240.00, w=17.00, h=17.00),
    dict(x=87.00, y=240.00, w=17.00, h=17.00),
    dict(x=106.00, y=240.00, w=17.00, h=17.00),
    dict(x=125.00, y=240.00, w=17.00, h=17.00),
    dict(x=144.00, y=240.00, w=17.00, h=17.00),
]


def aruco_ids(panel):
    """Marker ids on panel P<n>, in the order of ARUCO."""
    return [4 * (int(panel) - 1) + k for k in range(4)]


def marker_corners_mm(i):
    """The four corners of ARUCO[i], in page millimetres, TL TR BR BL."""
    a = ARUCO[i]
    x, y, s = a["x"], a["y"], a["size"]
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], float)


def slant_angle_deg(i=0):
    """Actual slant of edge i, from the extracted polygon."""
    p = np.asarray(SLANTED_EDGES[i], float)
    v = p[1] - p[0]
    return float(np.degrees(np.arctan2(v[1], v[0])))
