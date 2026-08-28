#!/usr/bin/env python3
"""
bearing_offset.py -- rotation or reflection? and does it explain the s walk?

A constant ROTATION by phi maps bearing th -> th + phi, so bearing(z)-bearing(h)
is CONSTANT while bearing(h) varies.
A REFLECTION about a line at alpha maps th -> 2*alpha - th, so the difference
is 2*alpha - 2*th: it varies at TWICE the rate of the heading itself.

If the heading does not vary over the window the two are indistinguishable and
the script says so rather than guessing.

Run:  python3 bearing_offset.py <innovation_log.csv> [t_max_rel]
"""
import sys

import numpy as np
import pandas as pd

CSV = sys.argv[1]
T_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 65.0
SPEED_MIN = 0.30

d = pd.read_csv(CSV)
d["t"] -= d.t.min()
v = d[(d.kind == "velocity") & (d.t < T_MAX)].copy()
y = v[["y0", "y1", "y2"]].to_numpy(float)
h = v[["h0", "h1", "h2"]].to_numpy(float)
z = y + h
m = (np.linalg.norm(z, axis=1) > SPEED_MIN) & (np.linalg.norm(h, axis=1) > 1e-6)
z, h, tt = z[m], h[m], v.t.to_numpy()[m]

bz = np.degrees(np.arctan2(z[:, 1], z[:, 0]))
bh = np.degrees(np.arctan2(h[:, 1], h[:, 0]))
diff = (bz - bh + 180) % 360 - 180

print(f"  n = {len(z)}   s held at {v.s.median():.3f}")
print(f"\nHEADING COVERAGE (does the aircraft change direction here?)")
print(f"  bearing(h) spans {bh.min():7.1f} .. {bh.max():7.1f} deg   "
      f"sd {bh.std():.1f}")
print(f"  bearing(z) spans {bz.min():7.1f} .. {bz.max():7.1f} deg   "
      f"sd {bz.std():.1f}")
if bh.std() < 20:
    print("  *** heading barely varies -- rotation and reflection are NOT "
          "distinguishable on this window ***")

print(f"\nOFFSET  bearing(z) - bearing(h)")
print(f"  median {np.median(diff):+7.2f} deg   sd {diff.std():6.2f}   "
      f"range {diff.min():+.1f} .. {diff.max():+.1f}")

# rotation -> diff constant vs bh ; reflection -> diff = const - 2*bh
for lbl, x in (("rotation  (diff vs bearing(h), slope should be  0)", bh),
               ("reflection(diff vs bearing(h), slope should be -2)", bh)):
    if x.std() > 1e-6:
        A = np.polyfit(x, np.unwrap(np.radians(diff)) * 180 / np.pi, 1)
        print(f"  {lbl}: fitted slope {A[0]:+.3f}")
        break

print(f"\nS-WALK PREDICTION")
c = np.cos(np.radians(np.median(np.abs(diff))))
print(f"  a constant {np.median(np.abs(diff)):.1f} deg misalignment makes the "
      f"least-squares s equal cos(phi) = {c:.3f} x truth")
print(f"  s held here {v.s.median():.3f}  ->  predicted walk floor "
      f"{v.s.median() * c:.3f}")

print(f"\nPER-20s (is it drifting?)")
for lo in range(0, int(T_MAX), 20):
    sel = (tt >= lo) & (tt < lo + 20)
    if sel.sum() > 3:
        print(f"  {lo:3d}-{lo+20:3d}s n={sel.sum():3d}  offset "
              f"{np.median(diff[sel]):+7.2f}  bearing(h) sd {bh[sel].std():5.1f}")