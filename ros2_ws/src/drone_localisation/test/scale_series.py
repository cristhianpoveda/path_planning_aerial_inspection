#!/usr/bin/env python3
"""
scale_series.py -- did `s` hold, or did it walk?

Reads the innovation CSV (not the console log) and reports the scale trajectory
on a fixed grid, plus the per-channel accept rates that explain it.

Run:  python3 scale_series.py <innovation_log.csv> [t_max_rel] [grid_s]
"""
import sys

import numpy as np
import pandas as pd

CSV = sys.argv[1]
T_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 65.0
GRID = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
SPEED_MIN = 0.30
S_REF_ANGLE = 51.84            # [M] the misalignment measured in run A

d = pd.read_csv(CSV)
d["t"] -= d.t.min()
w = d[d.t < T_MAX]
if w.empty:
    sys.exit(f"no rows with t < {T_MAX}")

s0, s1 = float(w.s.iloc[0]), float(w.s.iloc[-1])
print(f"{CSV}\n  {len(w)} rows, t < {T_MAX:.0f} s")

print("\n1. SCALE TRAJECTORY")
print(f"   {'t':>7} {'s':>9} {'sigma_s':>9} {'b':>8} {'cov_s_b':>11}")
for lo in np.arange(0, T_MAX, GRID):
    g = w[(w.t >= lo) & (w.t < lo + GRID)]
    if len(g):
        print(f"   {lo:7.0f} {g.s.iloc[-1]:9.4f} {g.sigma_s.iloc[-1]:9.4f} "
              f"{g.b.iloc[-1]:8.3f} {g.cov_s_b.iloc[-1]:11.3e}")

drift = (s1 / s0 - 1.0) * 100 if s0 else float("nan")
print(f"\n2. VERDICT")
print(f"   s: {s0:.4f} -> {s1:.4f}   ({drift:+.1f} %)")
c = np.cos(np.radians(S_REF_ANGLE))
print(f"   walk floor predicted by the {S_REF_ANGLE:.1f} deg misalignment: "
      f"cos(phi) x s0 = {s0 * c:.4f}")
if s0:
    if abs(drift) < 8:
        v = "HELD -- the misalignment was the whole story"
    elif s1 < s0 * (c + 0.10):
        v = "WALKED TO THE cos(phi) FLOOR -- misalignment still present"
    else:
        v = "PARTIAL WALK -- a second contributor remains"
    print(f"   -> {v}")

print("\n3. ACCEPT RATES")
for k, g in w.groupby("kind"):
    print(f"   {k:9s} n={len(g):5d}  accepted {g.accepted.mean():5.1%}  "
          f"median NIS {g.nis.median():8.3f}")

v = w[w.kind == "velocity"].copy()
y = v[["y0", "y1", "y2"]].to_numpy(float)
h = v[["h0", "h1", "h2"]].to_numpy(float)
z = y + h
m = (np.linalg.norm(z, axis=1) > SPEED_MIN) & (np.linalg.norm(h, axis=1) > 1e-6)
if m.sum() > 5:
    zn, hn = np.linalg.norm(z[m], axis=1), np.linalg.norm(h[m], axis=1)
    ang = np.degrees(np.arccos(np.clip((z[m] * h[m]).sum(1) / (zn * hn), -1, 1)))
    print(f"\n4. RESIDUAL GEOMETRY  (n = {m.sum()})")
    print(f"   angle(z, h) median {np.median(ang):6.2f} deg")
    print(f"   |z|/|h|     median {np.median(zn / hn):6.3f}   "
          f"iqr {np.percentile(zn/hn, 25):.3f}..{np.percentile(zn/hn, 75):.3f}")
    print(f"   NOTE: with s ESTIMATED, |z|/|h| is driven back toward 1 by the "
          f"filter itself.\n         It is a scale statement only in a HELD run.")