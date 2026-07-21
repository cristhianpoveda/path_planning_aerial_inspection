#!/usr/bin/env python3
"""Camera calibration from AprilGrid (Kalibr-style) or ChArUco images.

Board convention (AprilGrid):
    tag size    t  [m]   -- side of the black tag square
    spacing     g  [m]   -- white gap between adjacent tags
    pitch       p = t + g
    tag (col c, row r) top-left corner at (c*p, r*p)

Board frame: X right, Y down, Z out of the board. This matches OpenCV's
marker-corner ordering (top-left, top-right, bottom-right, bottom-left), so
object points and image points correspond directly.

Usage
-----
  # 1. Identify the marker dictionary (run this first):
  python3 calibrate_board.py --images "imgs/*.png" --probe

  # 2. Calibrate:
  python3 calibrate_board.py --images "imgs/*.png" \
      --cols 6 --rows 8 --tag-size 0.088 --spacing 0.0264 \
      --dict DICT_APRILTAG_36h11 --out camera_calibration.yaml
"""
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

# Dictionaries worth probing, most likely first.
CANDIDATE_DICTS = [
    "DICT_APRILTAG_36h11",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_16h5",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_4X4_250",
    "DICT_ARUCO_ORIGINAL",
]


# ---------------------------------------------------------------- aruco shim
REFINE = {"none": "CORNER_REFINE_NONE",
          "subpix": "CORNER_REFINE_SUBPIX",
          "contour": "CORNER_REFINE_CONTOUR"}


def make_detector(dict_name, refine="subpix"):
    """Return a callable(gray) -> (corners, ids), across OpenCV versions."""
    dict_id = getattr(cv2.aruco, dict_name)
    refine_id = getattr(cv2.aruco, REFINE[refine])

    if hasattr(cv2.aruco, "ArucoDetector"):          # OpenCV >= 4.7
        adict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        # Sub-pixel refinement matters: corner accuracy drives calibration.
        params.cornerRefinementMethod = refine_id
        # The default adaptive-threshold window range is too narrow for large
        # tags under uneven lighting (measured: 15/36 -> 36/36 with this range).
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        det = cv2.aruco.ArucoDetector(adict, params)

        def _detect(gray):
            corners, ids, _ = det.detectMarkers(gray)
            return corners, ids
    else:                                            # OpenCV < 4.7
        adict = cv2.aruco.Dictionary_get(dict_id)
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = refine_id
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4

        def _detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
            return corners, ids

    return _detect


# ------------------------------------------------------------ board geometry
def build_object_points(cols, rows, tag_size, spacing,
                        row_origin="bottom", corner_rot=2):
    """Map tag id -> (4,3) float32 object points, in board coordinates.

    Two conventions are NOT safe to assume and are searched by the caller:

    `row_origin`  -- whether tag id 0 is the top-left or bottom-left tag.
                     Wrong choice mirrors the board; the solve diverges.
    `corner_rot`  -- which physical corner OpenCV reports first. For AprilTag
                     36h11 the first reported corner is the BOTTOM-RIGHT, not
                     the top-left, i.e. corner_rot=2 (measured, not assumed).
    """
    pitch = tag_size + spacing
    table = {}
    for r in range(rows):
        for c in range(cols):
            tid = r * cols + c
            rr = r if row_origin == "top" else (rows - 1 - r)
            x0, y0 = c * pitch, rr * pitch
            base = [
                [x0,            y0,            0.0],   # top-left
                [x0 + tag_size, y0,            0.0],   # top-right
                [x0 + tag_size, y0 + tag_size, 0.0],   # bottom-right
                [x0,            y0 + tag_size, 0.0],   # bottom-left
            ]
            base = base[corner_rot:] + base[:corner_rot]
            table[tid] = np.array(base, dtype=np.float32)
    return table


def detect_all(paths, detect, min_tags=3):
    """Detect once per image -> [(path, ids, corners)], img_size."""
    dets, img_size = [], None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ! unreadable: {p}")
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != img_size:
            print(f"  ! size mismatch, skipping: {p}")
            continue

        corners, ids = detect(img)
        n = 0 if ids is None else len(ids)
        if n < min_tags:
            print(f"  - {os.path.basename(p):>12}: {n} tags (skipped)")
            continue
        dets.append((p, ids.flatten(), corners))
        print(f"  + {os.path.basename(p):>12}: {n} tags, {n*4} corners")
    return dets, img_size


def build_correspondences(dets, obj_table, min_tags=3):
    """Apply a mapping hypothesis to cached detections."""
    obj_points, img_points, used = [], [], []
    for p, ids, corners in dets:
        o, i = [], []
        for tid, quad in zip(ids, corners):
            if tid in obj_table:
                o.append(obj_table[tid])
                i.append(quad.reshape(4, 2))
        if len(o) < min_tags:
            continue
        obj_points.append(np.concatenate(o).astype(np.float32))
        img_points.append(np.concatenate(i).astype(np.float32))
        used.append(p)
    return obj_points, img_points, used


# -------------------------------------------------------------------- output
def coverage_plot(img_points, img_size, path):
    """Scatter every detected corner on a blank frame: reveals gaps."""
    w, h = img_size
    canvas = np.full((h, w, 3), 255, np.uint8)
    # Thirds, to judge periphery coverage by eye.
    for f in (1 / 3, 2 / 3):
        cv2.line(canvas, (int(w * f), 0), (int(w * f), h), (220, 220, 220), 1)
        cv2.line(canvas, (0, int(h * f)), (w, int(h * f)), (220, 220, 220), 1)
    for pts in img_points:
        for x, y in pts:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (200, 60, 60), -1)
    cv2.imwrite(path, canvas)


def write_orbslam_yaml(path, K, dist, img_size, fps, n_features):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d = dist.flatten().tolist() + [0.0] * 5
    k1, k2, p1, p2, k3 = d[0], d[1], d[2], d[3], d[4]

    with open(path, "w") as f:
        f.write(f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters (generated by calibrate_board.py)
#--------------------------------------------------------------------------------------------
File.version: "1.0"

Camera.type: "PinHole"

Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}

Camera1.k1: {k1:.8f}
Camera1.k2: {k2:.8f}
Camera1.p1: {p1:.8f}
Camera1.p2: {p2:.8f}
Camera1.k3: {k3:.8f}

Camera.width: {img_size[0]}
Camera.height: {img_size[1]}

Camera.fps: {fps}

# 0: BGR, 1: RGB. Ignored if grayscale.
Camera.RGB: 0

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------
ORBextractor.nFeatures: {n_features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
""")


# ---------------------------------------------------------------------- main
def natural_sort(paths):
    def key(p):
        m = re.findall(r"\d+", os.path.basename(p))
        return (int(m[0]) if m else 0, p)
    return sorted(paths, key=key)


def run_calibration(obj_points, img_points, img_size, fix_k3=True):
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None, flags=flags)
    return rms, K, dist, rvecs, tvecs


def per_view_errors(obj_points, img_points, rvecs, tvecs, K, dist):
    errs = []
    for o, i, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        proj = proj.reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=1)))))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help='glob, e.g. "imgs/*.png"')
    ap.add_argument("--cols", type=int, default=6, help="tags across")
    ap.add_argument("--rows", type=int, default=6, help="tags down")
    ap.add_argument("--tag-size", type=float, default=0.088, help="metres")
    ap.add_argument("--spacing", type=float, default=0.0264, help="metres")
    ap.add_argument("--dict", default="DICT_APRILTAG_36h11")
    ap.add_argument("--out", default="camera_calibration.yaml")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--features", type=int, default=1250)
    ap.add_argument("--estimate-k3", action="store_true")
    ap.add_argument("--refine", default="subpix", choices=list(REFINE),
                    help="corner refinement; 'contour' often fits AprilTags better")
    ap.add_argument("--scan-tag-size", action="store_true",
                    help="sweep effective tag size (pitch held fixed) and report RMS")
    ap.add_argument("--probe", action="store_true",
                    help="try dictionaries and report detections, then exit")
    args = ap.parse_args()

    paths = natural_sort(glob.glob(args.images))
    if not paths:
        sys.exit(f"no images matched {args.images!r}")
    print(f"{len(paths)} images\n")

    # ---- probe mode ----
    if args.probe:
        sample = paths[: min(5, len(paths))]
        print("probing dictionaries on", len(sample), "images:")
        for name in CANDIDATE_DICTS:
            if not hasattr(cv2.aruco, name):
                continue
            det = make_detector(name, args.refine)
            total, ids_seen = 0, set()
            for p in sample:
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                _, ids = det(g)
                if ids is not None:
                    total += len(ids)
                    ids_seen.update(ids.flatten().tolist())
            flag = "  <-- likely" if total > 0 else ""
            print(f"  {name:<24} {total:4d} markers, "
                  f"ids {min(ids_seen) if ids_seen else '-'}..{max(ids_seen) if ids_seen else '-'}{flag}")
        print("\nRe-run without --probe using the dictionary that detected markers.")
        return

    detect = make_detector(args.dict, args.refine)

    print("=== detecting ===")
    dets, size = detect_all(paths, detect)
    if len(dets) < 4:
        sys.exit(f"too few usable views ({len(dets)})")

    # ---- search id ordering x corner ordering; both are board/library
    # ---- specific and cannot be assumed. Detection is reused, so this is cheap.
    print("\n=== resolving board conventions ===")
    best = None
    for origin in ("top", "bottom"):
        for rot in range(4):
            table = build_object_points(args.cols, args.rows,
                                        args.tag_size, args.spacing, origin, rot)
            obj, imgp, used = build_correspondences(dets, table)
            if len(obj) < 4:
                continue
            rms, K, dist, rvecs, tvecs = run_calibration(
                obj, imgp, size, fix_k3=not args.estimate_k3)
            print(f"  row_origin={origin:<6} corner_rot={rot}  RMS={rms:8.4f} px")
            if best is None or rms < best[0]:
                best = (rms, K, dist, rvecs, tvecs, obj, imgp, used, size, origin, rot)

    if best is None:
        sys.exit("calibration failed: not enough detections")

    rms, K, dist, rvecs, tvecs, obj, imgp, used, size, origin, rot = best

    # ---- optional: the detected corners sit INSIDE the printed black square by
    # ---- an amount that is a detector artefact, so the best effective tag size
    # ---- is found empirically. Pitch is held at the measured value.
    if args.scan_tag_size:
        pitch = args.tag_size + args.spacing
        print(f"\n=== tag-size sweep (pitch fixed at {pitch*1000:.1f} mm) ===")
        sweep = []
        for frac in np.arange(0.80, 1.16, 0.02):
            t = args.tag_size * frac
            table = build_object_points(args.cols, args.rows, t, pitch - t, origin, rot)
            o, i, u = build_correspondences(dets, table)
            if len(o) < 4:
                continue
            r_, K_, d_, rv_, tv_ = run_calibration(
                o, i, size, fix_k3=not args.estimate_k3)
            sweep.append((r_, t, K_, d_, rv_, tv_, o, i, u))
            print(f"  tag={t*1000:6.2f} mm (ratio {t/pitch:.3f})  RMS={r_:7.4f} px")
        if sweep:
            sweep.sort(key=lambda s: s[0])
            if sweep[0][0] < rms:
                rms, t_best, K, dist, rvecs, tvecs, obj, imgp, used = sweep[0]
                print(f"  -> best tag size {t_best*1000:.2f} mm, RMS {rms:.4f} px")

    print(f"\n=== result (row_origin={origin}, corner_rot={rot}) ===")
    print(f"views used: {len(used)} / {len(paths)}")
    print(f"image size: {size[0]} x {size[1]}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"\nfx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print("dist:", np.round(dist.flatten(), 6).tolist())

    # Expected fx for a 24mm-equivalent lens ~ 0.66 * width. Sanity check.
    expected = 0.66 * size[0]
    ratio = K[0, 0] / expected
    note = "OK" if 0.7 < ratio < 1.4 else "SUSPICIOUS - check board params / units"
    print(f"\nfx vs 24mm-equiv expectation ({expected:.0f} px): ratio {ratio:.2f}  [{note}]")

    errs = per_view_errors(obj, imgp, rvecs, tvecs, K, dist)
    print("\nworst views (consider removing and re-running):")
    for e, p in sorted(zip(errs, used), reverse=True)[:5]:
        print(f"  {e:6.3f} px  {os.path.basename(p)}")

    cov = os.path.splitext(args.out)[0] + "_coverage.png"
    coverage_plot(imgp, size, cov)
    print(f"\ncoverage plot -> {cov}  (look for empty edges/corners)")

    write_orbslam_yaml(args.out, K, dist, size, args.fps, args.features)
    print(f"ORB-SLAM3 config -> {args.out}")


if __name__ == "__main__":
    main()
    