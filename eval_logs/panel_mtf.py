#!/usr/bin/env python3
"""MTF50 and line-pair contrast from the panel bags -> b_0, k_dof, band, k_r.

    python3 panel_mtf.py --bag dof_sweep_20260907_1349 --out mtf_1349.csv
    python3 panel_mtf.py --fit mtf_1349.csv mtf_1355.csv
"""

import argparse
import csv
import sys

import cv2
import numpy as np

import panel as PN

TOPIC_IMG = "/drone_1/camera/image/compressed"
TOPIC_DRONE = "/optitrack/rigid_bodies/dji_mini4"
TOPIC_PANEL = "/optitrack/rigid_bodies/panel"

# [M] camera_calibration.yaml
FX, FY, CX, CY = 1431.85, 1431.85, 970.57, 540.0
DIST = np.array([0.07182632, -0.08371309, 0.0, 0.0, 0.0])
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])


# ------------------------------------------------------------------ detection
def make_detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.01
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def panel_homography(gray, det, min_markers=3):
    """Homography panel-mm -> image, per visible panel. Returns {panel: H}."""
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    out = {}
    by_panel = {}
    for c, i in zip(corners, ids.flatten()):
        by_panel.setdefault(int(i) // 4 + 1, []).append((int(i) % 4, c[0]))
    for pnl, lst in by_panel.items():
        if len(lst) < min_markers:
            continue
        src, dst = [], []
        for k, quad in lst:
            src.append(PN.marker_corners_mm(k))
            dst.append(quad)
        H, _ = cv2.findHomography(np.vstack(src).astype(np.float32),
                                  np.vstack(dst).astype(np.float32), 0)
        if H is not None:
            out[pnl] = H
    return out


def to_image(H, pts_mm):
    p = np.hstack([np.asarray(pts_mm, float), np.ones((len(pts_mm), 1))]).T
    q = H @ p
    return (q[:2] / q[2]).T


# ----------------------------------------------------------------- MTF50
def mtf50_from_edge(gray, quad_img, bin_px=0.25, half_width=20.0):
    """Slanted-edge MTF50 in cycles per pixel, or None.

    quad_img: the four image-space corners of one dark edge patch. The edge
    used is the one between corner 0 and corner 3, which is the long left
    side of the patch as authored in the PDF.
    """
    q = np.asarray(quad_img, float)
    a, b = q[0], q[3]
    d = b - a
    L = np.linalg.norm(d)
    if L < 30:
        return None
    t = d / L
    n = np.array([-t[1], t[0]])              # normal to the edge

    # sample a band around the edge, in native pixels
    npts = int(L)
    s = np.linspace(0.12 * L, 0.88 * L, max(npts, 40))
    xs, ys, vs = [], [], []
    for si in s:
        base = a + t * si
        for w in np.arange(-half_width, half_width + 1e-9, 1.0):
            p = base + n * w
            xi, yi = int(round(p[0])), int(round(p[1]))
            if 0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]:
                xs.append(w)
                ys.append(si)
                vs.append(float(gray[yi, xi]))
    if len(vs) < 200:
        return None
    xs = np.array(xs)
    vs = np.array(vs)

    # bin the projected samples into a super-resolved ESF
    edges = np.arange(-half_width, half_width + bin_px, bin_px)
    idx = np.digitize(xs, edges) - 1
    esf = np.full(len(edges) - 1, np.nan)
    for k in range(len(esf)):
        m = idx == k
        if m.sum() >= 2:
            esf[k] = vs[m].mean()
    good = ~np.isnan(esf)
    if good.sum() < 20:
        return None
    esf = np.interp(np.arange(len(esf)), np.flatnonzero(good), esf[good])
    if abs(esf[-1] - esf[0]) < 15:           # no real edge here
        return None

    lsf = np.gradient(esf)
    lsf = lsf * np.hanning(len(lsf))
    M = np.abs(np.fft.rfft(lsf))
    if M[0] <= 0:
        return None
    M = M / M[0]
    f = np.fft.rfftfreq(len(lsf), d=bin_px)  # cycles per pixel
    below = np.flatnonzero(M < 0.5)
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return None
    f50 = np.interp(0.5, [M[i], M[i - 1]], [f[i], f[i - 1]])
    return float(f50)


def sigma_from_mtf50(f50):
    """Gaussian PSF width in pixels."""
    return 0.1874 / f50 if f50 and f50 > 1e-9 else np.nan


# ------------------------------------------------------------- line pairs
def group_contrast(gray, H, g):
    """Michelson contrast of one bar group, sampled along its centre line."""
    y = 0.5 * (g["y0"] + g["y1"])
    xs = np.linspace(g["x0"], g["x1"], 400)
    pts = to_image(H, np.column_stack([xs, np.full_like(xs, y)]))
    vals = []
    for x, yy in pts:
        xi, yi = int(round(x)), int(round(yy))
        if 0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]:
            vals.append(float(gray[yi, xi]))
    if len(vals) < 50:
        return np.nan
    v = np.array(vals)
    hi = np.percentile(v, 90)
    lo = np.percentile(v, 10)
    return float((hi - lo) / (hi + lo)) if (hi + lo) > 0 else np.nan


# ------------------------------------------------------------------ measure
def measure(args):
    from panel_triage import (read_bag, stamp, pose_array, quat_to_R,
                              runs_where, HOVER_SPEED, HOVER_MIN_S,
                              STANDOFF_TOL)
    from scipy.signal import savgol_filter

    d = read_bag(args.bag, [TOPIC_IMG, TOPIC_DRONE, TOPIC_PANEL])
    D = pose_array(d[TOPIC_DRONE])
    P = pose_array(d[TOPIC_PANEL])
    t = D[:, 0]
    p_panel = np.array([np.interp(t, P[:, 0], P[:, i]) for i in (1, 2, 3)]).T
    R0 = quat_to_R(P[len(P) // 2, 4:8])
    v_rel = D[:, 1:4] - p_panel
    standoff = np.linalg.norm(v_rel, axis=1)
    u = v_rel / standoff[:, None]
    cands = [(float(np.median(np.degrees(np.arccos(
        np.clip(u @ (sg * R0[:, ax]), -1, 1))))), ax, sg)
        for ax in (0, 1, 2) for sg in (1, -1)]
    med, ax, sg = min(cands)
    incid = np.degrees(np.arccos(np.clip(u @ (sg * R0[:, ax]), -1, 1)))
    sm = np.column_stack([savgol_filter(D[:, i], 21, 2) for i in (1, 2, 3)])
    speed = np.linalg.norm(np.gradient(sm, t, axis=0), axis=1)

    segs = []
    for i, j in runs_where((speed < HOVER_SPEED) & (incid < 35.0), t,
                           HOVER_MIN_S):
        if standoff[i:j].ptp() <= STANDOFF_TOL:
            segs.append((t[i], t[j - 1], float(standoff[i:j].mean()),
                         float(np.median(incid[i:j]))))
    print(f"{len(segs)} hover segments; sampling up to {args.per_segment} "
          "frames from each")

    det = make_detector()
    rows = []
    for (t0, t1, so, inc) in segs:
        picked = [m for m in d[TOPIC_IMG] if t0 <= stamp(m) <= t1]
        step = max(1, len(picked) // args.per_segment)
        n_ok = 0
        for m in picked[::step]:
            gray = cv2.imdecode(np.frombuffer(m.data, np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            Hs = panel_homography(gray, det)
            if not Hs:
                continue
            H = list(Hs.values())[0]
            f50s = []
            for e in PN.SLANTED_EDGES:
                q = to_image(H, np.asarray(e, float))
                f = mtf50_from_edge(gray, q)
                if f:
                    f50s.append(f)
            if not f50s:
                continue
            f50 = float(np.median(f50s))
            row = dict(t=stamp(m), standoff=so, incidence=inc,
                       n_edges=len(f50s), f50=f50,
                       sigma_px=sigma_from_mtf50(f50),
                       gsd_mm=1e3 * so / FX)
            for g in PN.BAR_GROUPS:
                row[f"c{g['w_mm']:.2f}"] = group_contrast(gray, H, g)
            for k, gs in enumerate(PN.GREY_STEPS):
                c = to_image(H, [[gs["x"] + gs["w"] / 2,
                                  gs["y"] + gs["h"] / 2]])[0]
                xi, yi = int(round(c[0])), int(round(c[1]))
                row[f"grey{PN.GREY_PCT[k]}"] = (
                    float(gray[yi, xi]) if 0 <= xi < gray.shape[1]
                    and 0 <= yi < gray.shape[0] else np.nan)
            rows.append(row)
            n_ok += 1
        print(f"  {t0 - t[0]:7.1f}-{t1 - t[0]:7.1f} s  d={so:.3f} m  "
              f"inc={inc:4.1f}  {n_ok} frames measured")

    if not rows:
        sys.exit("no frames yielded an edge measurement")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} frames)")


# --------------------------------------------------------------------- fit
def fit(paths, mtf_frac=0.7, contrast_thresh=0.10):
    from scipy.optimize import least_squares
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    d = np.array([float(r["standoff"]) for r in rows])
    sig = np.array([float(r["sigma_px"]) for r in rows])
    ok = np.isfinite(d) & np.isfinite(sig) & (sig > 0) & (sig < 20)
    d, sig = d[ok], sig[ok]
    print(f"{len(d)} frames, standoff {d.min():.2f}-{d.max():.2f} m")

    # pool per standoff so every hover weighs the same
    keys = np.round(d, 2)
    dd = np.array(sorted(set(keys)))
    ss = np.array([np.median(sig[keys == k]) for k in dd])
    ns = np.array([int((keys == k).sum()) for k in dd])
    print("\n  d (m)   n   sigma_px   MTF50 (cy/px)   GSD (mm/px)")
    for k, s, n in zip(dd, ss, ns):
        print(f"  {k:5.2f} {n:4d}   {s:7.3f}   {0.1874 / s:11.4f}   "
              f"{1e3 * k / FX:9.4f}")

    def model(p, x):
        b0, kd, df = p
        return np.sqrt(b0 ** 2 + (kd * np.abs(x - df) / x) ** 2)

    r = least_squares(lambda p: model(p, dd) - ss, [0.6, 2.0, 1.4],
                      bounds=([0.05, 0.01, 0.4], [10.0, 50.0, 4.0]))
    b0, kd, df = r.x
    resid = float(np.sqrt(np.mean(r.fun ** 2)))
    print(f"\n[M] b_0   = {b0:.4f} px")
    print(f"[M] k_dof = {kd:.4f} px")
    print(f"[M] d_f   = {df:.4f} m       (fit rms {resid:.4f} px)")

    grid = np.linspace(0.3, 4.0, 2000)
    m = 0.1874 / model(r.x, grid)
    peak = m.max()
    inb = grid[m >= mtf_frac * peak]
    print(f"[M] band  = {inb.min():.3f} - {inb.max():.3f} m  "
          f"(MTF50 within {mtf_frac:.0%} of its peak {peak:.4f} cy/px)")

    # k_r from the finest group still resolved, per standoff
    widths = [g["w_mm"] for g in PN.BAR_GROUPS]
    krs = []
    for k in dd:
        sub = [r_ for r_ in rows if abs(float(r_["standoff"]) - k) < 0.005]
        if not sub:
            continue
        finest = None
        for w in widths:
            c = np.nanmedian([float(r_.get(f"c{w:.2f}", "nan") or "nan")
                              for r_ in sub])
            if np.isfinite(c) and c >= contrast_thresh:
                finest = w
                break
        if finest is None:
            continue
        b_tot = model(r.x, np.array([k]))[0]
        gsd = 1e3 * k / FX
        krs.append(finest / (b_tot * gsd))
        print(f"  d={k:.2f} m: finest resolved {finest:.2f} mm, "
              f"b_tot {b_tot:.3f} px, GSD {gsd:.4f} mm/px -> k_r {krs[-1]:.3f}")
    if krs:
        print(f"\n[M] k_r   = {np.median(krs):.4f} "
              f"(median over {len(krs)} standoffs, "
              f"spread {np.min(krs):.2f}-{np.max(krs):.2f})")
    print("\nPaste these into QualityParams in model.py, replacing the [T]s.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag")
    ap.add_argument("--out", default="mtf.csv")
    ap.add_argument("--per-segment", type=int, default=25)
    ap.add_argument("--fit", nargs="+")
    ap.add_argument("--mtf-frac", type=float, default=0.7)
    ap.add_argument("--contrast", type=float, default=0.10)
    a = ap.parse_args()
    if a.fit:
        fit(a.fit, a.mtf_frac, a.contrast)
    elif a.bag:
        measure(a)
    else:
        ap.error("give --bag to measure or --fit to fit")


if __name__ == "__main__":
    main()
