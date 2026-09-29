#!/usr/bin/env python3
"""t_exp from motion smear. Needs b_0, k_dof and d_f already measured.

    python3 panel_texp.py --bag speed_1420 --out texp_1420.csv
    python3 panel_texp.py --fit texp_1420.csv

Per frame: measure the total blur sigma from the slanted edge, subtract the
static part in quadrature, and regress the remainder on the perpendicular
speed. b_m = f_x * t_exp * v_perp / d, so

    sigma_total^2 - sigma_static(d)^2  =  (f_x * t_exp / d)^2 * v_perp^2

is a straight line through the origin in v_perp^2 whose slope gives t_exp.
"""

import argparse
import csv
import sys

import numpy as np

FX = 1431.85

# [M] optics_constants.yaml
B0 = 1.461
K_DOF = 2.768
D_F = 1.567


def sigma_static(d):
    return np.sqrt(B0 ** 2 + (K_DOF * np.abs(d - D_F) / d) ** 2)


def measure(args):
    import cv2
    import panel as PN
    from panel_mtf import (make_detector, panel_homography, to_image,
                           mtf50_from_edge, sigma_from_mtf50)
    from panel_triage import (read_bag, stamp, pose_array, quat_to_R,
                              TOPIC_IMG, TOPIC_DRONE, TOPIC_PANEL)
    from scipy.signal import savgol_filter

    d = read_bag(args.bag, [TOPIC_IMG, TOPIC_DRONE, TOPIC_PANEL])
    D = pose_array(d[TOPIC_DRONE])
    P = pose_array(d[TOPIC_PANEL])
    t = D[:, 0]
    pp = np.array([np.interp(t, P[:, 0], P[:, i]) for i in (1, 2, 3)]).T
    v_rel = D[:, 1:4] - pp
    so = np.linalg.norm(v_rel, axis=1)
    u = v_rel / so[:, None]
    sm = np.column_stack([savgol_filter(D[:, i], 21, 2) for i in (1, 2, 3)])
    vel = np.gradient(sm, t, axis=0)
    # perpendicular component: remove the part along the viewing ray
    v_par = np.sum(vel * u, axis=1)
    v_perp = np.linalg.norm(vel - v_par[:, None] * u, axis=1)

    det = make_detector()
    rows = []
    imgs = d[TOPIC_IMG][::max(1, args.stride)]
    for k, m in enumerate(imgs):
        ts = stamp(m)
        if ts < t[0] or ts > t[-1]:
            continue
        dd = float(np.interp(ts, t, so))
        vp = float(np.interp(ts, t, v_perp))
        if not (args.min_standoff <= dd <= args.max_standoff):
            continue
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
            f = mtf50_from_edge(gray, to_image(H, np.asarray(e, float)))
            if f:
                f50s.append(f)
        if not f50s:
            continue
        sig = sigma_from_mtf50(float(np.median(f50s)))
        rows.append(dict(t=ts, standoff=dd, v_perp=vp, n_edges=len(f50s),
                         sigma_px=sig, sigma_static=float(sigma_static(dd))))
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} frames measured")
    if not rows:
        sys.exit("no frames yielded an edge measurement")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    v = np.array([r["v_perp"] for r in rows])
    print(f"wrote {args.out}: {len(rows)} frames, v_perp "
          f"{v.min():.2f}-{v.max():.2f} m/s, "
          f"{int((v > 0.3).sum())} above 0.3 m/s")


def fit(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    d = np.array([float(r["standoff"]) for r in rows])
    v = np.array([float(r["v_perp"]) for r in rows])
    s = np.array([float(r["sigma_px"]) for r in rows])
    s0 = sigma_static(d)
    ok = np.isfinite(s) & (s > 0) & (s < 30)
    d, v, s, s0 = d[ok], v[ok], s[ok], s0[ok]

    y = (s ** 2 - s0 ** 2) * d ** 2
    x = v ** 2
    print(f"{len(x)} frames, v_perp {v.min():.2f}-{v.max():.2f} m/s, "
          f"standoff {d.min():.2f}-{d.max():.2f} m")

    print("\n  v_perp bin   n   median excess blur (px, at 1 m)")
    bins = [0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.2, 5.0]
    for a_, b_ in zip(bins[:-1], bins[1:]):
        m = (v >= a_) & (v < b_)
        if m.sum() < 5:
            continue
        e = np.sqrt(np.maximum(y[m], 0.0))
        print(f"   {a_:.2f}-{b_:.2f}  {int(m.sum()):5d}   "
              f"{np.median(e):7.3f}")

    m = v > 0.02
    if m.sum() < 30:
        sys.exit("too few moving frames")
    A = np.column_stack([x[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    slope, icept = float(coef[0]), float(coef[1])
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    n = int(m.sum())
    # standard error of the slope
    resid = y[m] - A @ coef
    se = float(np.sqrt(np.sum(resid ** 2) / (n - 2)
                       / max(np.sum((x[m] - x[m].mean()) ** 2), 1e-12)))
    print(f"\nfree-intercept fit over {n} moving frames")
    print(f"  slope {slope:.2f} +- {se:.2f}   intercept {icept:.2f}   "
          f"r = {r:+.3f}")

    if slope <= 2 * se:
        
        vhi = float(np.percentile(v[m], 90))
        dmed = float(np.median(d[m]))
        spread = float(np.std(np.sqrt(np.maximum(y[m], 0.0))))
        b_max = spread / dmed
        t_hi = b_max * dmed / (FX * vhi)
        print("\n[M] MOTION SMEAR IS NOT SEPARABLE IN THIS DATA.")
        print(f"    The slope is within {2:.0f} standard errors of zero and "
              f"excess blur does not trend with speed (r = {r:+.3f}).")
        print(f"    Bound: any smear is hidden inside a scatter of "
              f"{spread:.2f} px at 1 m, so at the 90th-percentile speed of "
              f"{vhi:.2f} m/s and a median standoff of {dmed:.2f} m,")
        print(f"[M] t_exp < {t_hi * 1e3:.2f} ms")
        print(f"    and b_m < {FX * t_hi * vhi / dmed:.2f} px against "
              f"b_0 = {B0:.3f} px, i.e. motion smear is at most "
              f"{100 * FX * t_hi * vhi / dmed / B0:.0f} per cent of the "
              "static blur at the fastest speed flown.")
        return

    t_exp = np.sqrt(slope) / FX
    print(f"\n[M] t_exp = {t_exp * 1e3:.3f} ms "
          f"(+- {0.5 * se / slope * t_exp * 1e3:.3f})")
    for vv in (0.5, 0.9):
        print(f"    b_m at {vv} m/s and 1.4 m: {FX * t_exp * vv / 1.4:.3f} px"
              f"  against b_0 = {B0:.3f} px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag")
    ap.add_argument("--out", default="texp.csv")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-standoff", type=float, default=0.8)
    ap.add_argument("--max-standoff", type=float, default=3.0)
    ap.add_argument("--fit", nargs="+")
    a = ap.parse_args()
    if a.fit:
        fit(a.fit)
    elif a.bag:
        measure(a)
    else:
        ap.error("give --bag to measure or --fit to fit")


if __name__ == "__main__":
    main()
