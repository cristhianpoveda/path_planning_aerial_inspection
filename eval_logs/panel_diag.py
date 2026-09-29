#!/usr/bin/env python3
"""
    python3 panel_diag.py --csv mtf_1349.csv mtf_1355.csv
    python3 panel_diag.py --bag dof_sweep_20260907_1349 --contrast
"""

import argparse
import csv
import sys

import numpy as np

FX = 1431.85


# --------------------------------------------------------------------- Q1
def diag_csv(paths, gap_s=1.0):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                r["_src"] = p
                rows.append(r)
    t = np.array([float(r["t"]) for r in rows])
    o = np.argsort(t)
    rows = [rows[i] for i in o]
    t = t[o]

    # split back into segments on time gaps
    segs, start = [], 0
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > gap_s:
            segs.append((start, i))
            start = i
    segs.append((start, len(t)))

    print(f"{len(rows)} frames in {len(segs)} segments\n")
    print("   d(m)   n   sigma_px            within-segment")
    print("                med   sd   rel   lag1_r   drift(px)   n_edges")
    alld, allmed = [], []
    for a, b in segs:
        if b - a < 5:
            continue
        s = np.array([float(rows[i]["sigma_px"]) for i in range(a, b)])
        d = float(np.median([float(rows[i]["standoff"]) for i in range(a, b)]))
        ne = float(np.median([float(rows[i]["n_edges"]) for i in range(a, b)]))
        s = s[np.isfinite(s)]
        if len(s) < 5:
            continue
        z = s - s.mean()
        lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if len(z) > 3 else np.nan
        drift = float(np.polyfit(np.arange(len(s)), s, 1)[0] * len(s))
        print(f"  {d:5.2f} {len(s):4d}  {np.median(s):5.2f} {s.std():5.2f} "
              f"{s.std() / max(np.median(s), 1e-9):5.2f}   {lag1:+5.2f}   "
              f"{drift:+8.2f}   {ne:.1f}")
        alld.append(d)
        allmed.append(np.median(s))

    print("\ninterpretation")
    print("  lag1 near 0 and small drift -> blur is stationary within the")
    print("  segment, so the scatter is measurement noise and focus is fixed.")
    print("  lag1 near 1 with drift comparable to the spread -> the lens is")
    print("  hunting, and no single d_f describes the flight.")

    d = np.array(alld)
    s = np.array(allmed)
    k = np.argsort(d)
    d, s = d[k], s[k]
    far = d > 1.0
    if far.sum() >= 3:
        slope = np.polyfit(d[far], s[far], 1)[0]
        print(f"\n  slope of sigma against d beyond 1.0 m: {slope:+.3f} px/m")
        if slope < -0.1:
            print("  NEGATIVE: blur improves with distance, which a fixed")
            print("  focus at d_f < 2.3 m cannot produce. Consistent with")
            print("  autofocus, or with the near cluster being at the lens's")
            print("  minimum focus distance.")
        elif abs(slope) <= 0.1:
            print("  FLAT: blur is independent of distance over this range.")
            print("  Consistent with autofocus holding focus on the panel.")
        else:
            print("  POSITIVE: consistent with a fixed focus nearer than the")
            print("  far standoffs, i.e. the defocus model as written.")

    # what a flat model would cost
    rms_flat = float(np.sqrt(np.mean((s - s.mean()) ** 2)))
    print(f"\n  rms about a constant sigma = {s.mean():.3f} px: "
          f"{rms_flat:.3f} px")
    print("  Compare against the fitted defocus rms. If they are close, the")
    print("  defocus term is not earning its two extra parameters and the")
    print("  quality model should use a constant b_tot over the band.")


# --------------------------------------------------------------------- Q2
def diag_contrast(args):
    import cv2
    import panel as PN
    from panel_mtf import make_detector, panel_homography, to_image
    from panel_triage import (read_bag, stamp, pose_array, quat_to_R,
                              runs_where, HOVER_SPEED, HOVER_MIN_S,
                              STANDOFF_TOL, TOPIC_IMG, TOPIC_DRONE,
                              TOPIC_PANEL)
    from scipy.signal import savgol_filter

    d = read_bag(args.bag, [TOPIC_IMG, TOPIC_DRONE, TOPIC_PANEL])
    D = pose_array(d[TOPIC_DRONE])
    P = pose_array(d[TOPIC_PANEL])
    t = D[:, 0]
    pp = np.array([np.interp(t, P[:, 0], P[:, i]) for i in (1, 2, 3)]).T
    R0 = quat_to_R(P[len(P) // 2, 4:8])
    v = D[:, 1:4] - pp
    so = np.linalg.norm(v, axis=1)
    u = v / so[:, None]
    med, ax, sg = min((float(np.median(np.degrees(np.arccos(
        np.clip(u @ (s_ * R0[:, a_]), -1, 1))))), a_, s_)
        for a_ in (0, 1, 2) for s_ in (1, -1))
    inc = np.degrees(np.arccos(np.clip(u @ (sg * R0[:, ax]), -1, 1)))
    sm = np.column_stack([savgol_filter(D[:, i], 21, 2) for i in (1, 2, 3)])
    sp = np.linalg.norm(np.gradient(sm, t, axis=0), axis=1)

    segs = [(t[i], t[j - 1], float(so[i:j].mean()))
            for i, j in runs_where((sp < HOVER_SPEED) & (inc < 35.0), t,
                                   HOVER_MIN_S)
            if so[i:j].ptp() <= STANDOFF_TOL]
    det = make_detector()
    widths = [g["w_mm"] for g in PN.BAR_GROUPS]
    print("modulation at each group's own fundamental, floor from a blank")
    print("strip. '--' means the line pair falls under 3 px and is rejected")
    print("before measurement.\n")
    hdr = "  d(m)  " + "".join(f"{w:6.2f}" for w in widths)
    print(hdr)
    for (t0, t1, dd) in segs:
        picked = [m for m in d[TOPIC_IMG] if t0 <= stamp(m) <= t1]
        if not picked:
            continue
        step = max(1, len(picked) // args.per_segment)
        acc = {w: [] for w in widths}
        floor = []
        for m in picked[::step]:
            g = cv2.imdecode(np.frombuffer(m.data, np.uint8),
                             cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            Hs = panel_homography(g, det)
            if not Hs:
                continue
            H = list(Hs.values())[0]
            gsd = 1e3 * dd / FX
            for grp in PN.BAR_GROUPS:
                period_px = 2.0 * grp["w_mm"] / gsd
                if period_px < 3.0:
                    continue
                y = 0.5 * (grp["y0"] + grp["y1"])
                xs = np.linspace(grp["x0"], grp["x1"], 512)
                pts = to_image(H, np.column_stack([xs, np.full_like(xs, y)]))
                vals = [float(g[int(round(b)), int(round(a))])
                        for a, b in pts
                        if 0 <= int(round(a)) < g.shape[1]
                        and 0 <= int(round(b)) < g.shape[0]]
                if len(vals) < 128:
                    continue
                arr = np.array(vals) - np.mean(vals)
                n_cycles = (grp["x1"] - grp["x0"]) / (2.0 * grp["w_mm"])
                F = np.abs(np.fft.rfft(arr * np.hanning(len(arr))))
                k = int(round(n_cycles))
                if not (1 <= k < len(F) - 1):
                    continue
                amp = F[max(k - 1, 1):k + 2].max()
                acc[grp["w_mm"]].append(2.0 * amp / (len(arr) / 2)
                                        / max(np.mean(vals), 1e-9))
            # noise floor: a blank strip between the groups and the edges
            ys = np.linspace(150.0, 165.0, 64)
            pts = to_image(H, np.column_stack([np.full_like(ys, 105.0), ys]))
            bl = [float(g[int(round(b)), int(round(a))]) for a, b in pts
                  if 0 <= int(round(a)) < g.shape[1]
                  and 0 <= int(round(b)) < g.shape[0]]
            if len(bl) > 32:
                floor.append(np.std(bl) / max(np.mean(bl), 1e-9))
        f = float(np.median(floor)) if floor else 0.0
        cells = ""
        finest = None
        for w in widths:
            if not acc[w]:
                cells += "    --"
                continue
            c = float(np.median(acc[w]))
            cells += f"{c:6.3f}"
            if finest is None and c > max(3 * f, 0.05):
                finest = w
        print(cells.join([f"  {dd:5.2f} ", ""])
              + f"   floor {f:.3f}  finest {finest if finest else '--'}")
    print("\nRead the finest column that clears three times the floor. That,")
    print("with b_tot and the GSD at the same standoff, gives k_r.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+")
    ap.add_argument("--bag")
    ap.add_argument("--contrast", action="store_true")
    ap.add_argument("--per-segment", type=int, default=12)
    a = ap.parse_args()
    if a.csv:
        diag_csv(a.csv)
    elif a.bag and a.contrast:
        diag_contrast(a)
    else:
        ap.error("give --csv, or --bag with --contrast")


if __name__ == "__main__":
    main()
