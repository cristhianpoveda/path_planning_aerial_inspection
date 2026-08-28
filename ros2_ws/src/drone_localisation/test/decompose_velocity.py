#!/usr/bin/env python3
"""
decompose_velocity.py -- separate the two candidate causes of the scale walk.

The velocity update predicts h = s*u_w - dl/dt and measures z = v_enu / K_VEL.
The logged residual is y = z - h, so z is recoverable as y + h.

    frame error       -> z and h differ mainly in DIRECTION (perp >> parallel)
    scale/timing error-> z and h differ mainly in MAGNITUDE (parallel >> perp)

Run:  python3 decompose_velocity.py <innovation_log.csv> [t_max_rel]
"""
import sys

import numpy as np
import pandas as pd

CSV = sys.argv[1]
T_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 65.0
SPEED_MIN = 0.30                      # m/s, above the quantisation dead zone

d = pd.read_csv(CSV)
d["t"] -= d.t.min()
v = d[(d.kind == "velocity") & (d.t < T_MAX)].copy()

y = v[["y0", "y1", "y2"]].to_numpy(float)
h = v[["h0", "h1", "h2"]].to_numpy(float)
z = y + h                              # the measurement, reconstructed

m = (np.linalg.norm(z, axis=1) > SPEED_MIN) & (np.linalg.norm(h, axis=1) > 1e-6)
z, h = z[m], h[m]
n = len(z)
print(f"{CSV}\n  {len(v)} velocity rows, {n} usable (|z| > {SPEED_MIN} m/s), "
      f"t < {T_MAX:.0f} s")
if n < 10:
    sys.exit("  too few usable rows -- lower SPEED_MIN or raise T_MAX")
print(f"  s over this window: {v.s.min():.3f} .. {v.s.max():.3f}")

zn, hn = np.linalg.norm(z, axis=1), np.linalg.norm(h, axis=1)

# ---- 1. magnitude vs direction -------------------------------------------
cos = np.clip((z * h).sum(1) / (zn * hn), -1, 1)
ang = np.degrees(np.arccos(cos))
par = (z * h).sum(1) / hn                       # component of z along h
perp = np.linalg.norm(z - par[:, None] * h / hn[:, None], axis=1)

print("\n1. MAGNITUDE vs DIRECTION")
print(f"   |z|/|h|            median {np.median(zn / hn):7.3f}   "
      f"iqr {np.percentile(zn/hn, 25):.3f}..{np.percentile(zn/hn, 75):.3f}")
print(f"   angle(z, h) deg    median {np.median(ang):7.2f}   "
      f"iqr {np.percentile(ang, 25):.2f}..{np.percentile(ang, 75):.2f}")
print(f"   parallel  (z.h/|h|) median {np.median(par):7.4f}")
print(f"   perpendicular       median {np.median(perp):7.4f}")
verdict = ("DIRECTION -- frame/convention error"
           if np.median(perp) > abs(np.median(par - hn)) else
           "MAGNITUDE -- scale/timing error")
print(f"   -> dominated by {verdict}")

# ---- 2. best-fit linear map z ~ A h ---------------------------------------
A, *_ = np.linalg.lstsq(h, z, rcond=None)
A = A.T                                          # so that z ~ A @ h
res = np.linalg.norm(z - h @ A.T, axis=1)
print("\n2. BEST-FIT LINEAR MAP  z ~ A h")
np.set_printoptions(precision=3, suppress=True)
print("  ", str(A).replace("\n", "\n   "))
print(f"   det(A) = {np.linalg.det(A):+.3f}   "
      f"det(A[:2,:2]) = {np.linalg.det(A[:2, :2]):+.3f}")
print(f"   residual |z - Ah| median {np.median(res):.4f} "
      f"(vs {np.median(np.linalg.norm(z - h, axis=1)):.4f} for A = I)")

# polar decomposition: nearest rotation + stretch
U, S, Vt = np.linalg.svd(A)
R = U @ Vt
if np.linalg.det(R) < 0:
    U[:, -1] *= -1
    R = U @ Vt
    print("   NOTE: nearest orthogonal map is a REFLECTION -- axis sign/order bug")
angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
axis = axis / max(np.linalg.norm(axis), 1e-12)
print(f"   nearest rotation: {angle:6.2f} deg about axis {axis}")
print(f"   singular values (gains): {S}")

# ---- 3. candidate axis conventions ----------------------------------------
print("\n3. CANDIDATE CONVENTIONS  (median |z - M h|, lower is better)")
cands = {
    "identity":            np.eye(3),
    "swap xy":             np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.]]),
    "negate xy":           np.diag([-1, -1, 1.]),
    "negate z":            np.diag([1, 1, -1.]),
    "swap xy + negate xy": np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1.]]),
    "yaw +90":             np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]]),
    "yaw -90":             np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.]]),
    "yaw 180":             np.diag([-1, -1, 1.]),
}
rows = [(k, float(np.median(np.linalg.norm(z - h @ M.T, axis=1))))
        for k, M in cands.items()]
for k, r in sorted(rows, key=lambda kv: kv[1]):
    print(f"   {k:22s} {r:.4f}")

# ---- 4. is the disagreement constant, or drifting? ------------------------
print("\n4. ANGLE OVER TIME (median per 20 s)")
tt = v.t.to_numpy()[m]
for lo in range(0, int(T_MAX), 20):
    sel = (tt >= lo) & (tt < lo + 20)
    if sel.sum() > 3:
        print(f"   {lo:3d}-{lo+20:3d} s  n={sel.sum():4d}  "
              f"angle {np.median(ang[sel]):6.2f} deg  "
              f"|z|/|h| {np.median((zn/hn)[sel]):.3f}")
