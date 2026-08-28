#!/usr/bin/env python3
"""
gate_symmetry.py -- is the NIS gate culling velocity updates asymmetrically?

The velocity Jacobian is H[:, s] = u_w, so S grows with the state being
estimated: a residual that wants `s` LARGER produces a different S from one
that wants it smaller. A gate applied to NIS can therefore reject one sign
preferentially, and `s` stalls at whatever value makes the surviving residuals
look unbiased (params.py, NIS_GATE_VEL).

Projects each velocity residual onto h (parallel to u_w):
    par > 0  -> this sample wants s LARGER
    par < 0  -> this sample wants s SMALLER
and compares the accepted and rejected populations.

Run:  python3 gate_symmetry.py <innovation_log.csv> [t_max_rel]
"""
import sys

import numpy as np
import pandas as pd

CSV = sys.argv[1]
T_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 65.0

d = pd.read_csv(CSV)
d["t"] -= d.t.min()
v = d[(d.kind == "velocity") & (d.t < T_MAX)].copy()

y = v[["y0", "y1", "y2"]].to_numpy(float)
h = v[["h0", "h1", "h2"]].to_numpy(float)
hn = np.linalg.norm(h, axis=1)
m = hn > 1e-6
y, h, hn = y[m], h[m], hn[m]
acc = v.accepted.to_numpy().astype(bool)[m]
nis = v.nis.to_numpy(float)[m]
par = (y * h).sum(1) / hn                 # signed, m/s
ratio = np.linalg.norm(y + h, axis=1) / hn

print(f"{CSV}\n  {m.sum()} velocity updates with a usable prediction, "
      f"t < {T_MAX:.0f} s\n")
print(f"{'population':<12}{'n':>5}{'mean par':>11}{'median par':>12}"
      f"{'mean |z|/|h|':>14}{'median NIS':>12}")
for lbl, sel in (("accepted", acc), ("rejected", ~acc), ("ALL", np.ones_like(acc))):
    if sel.sum():
        print(f"{lbl:<12}{sel.sum():5d}{par[sel].mean():+11.4f}"
              f"{np.median(par[sel]):+12.4f}{ratio[sel].mean():14.3f}"
              f"{np.median(nis[sel]):12.3f}")

print(f"\nsign split")
for lbl, sel in (("accepted", acc), ("rejected", ~acc)):
    if sel.sum():
        p = (par[sel] > 0).mean()
        print(f"  {lbl:<10} wants s larger {p:5.1%}   smaller {1-p:5.1%}")

bias_all, bias_acc = par.mean(), par[acc].mean() if acc.sum() else np.nan
print(f"\nverdict")
print(f"  full population bias {bias_all:+.4f} m/s")
print(f"  accepted-only  bias {bias_acc:+.4f} m/s")
if acc.sum() and (~acc).sum():
    if abs(bias_all) > 2 * abs(bias_acc) or \
            (par[~acc].mean() * bias_all > 0 and abs(par[~acc].mean()) > 0.05):
        print("  -> ASYMMETRIC: the gate is removing one sign. `s` stalls at the "
              "value\n     that makes the SURVIVORS look unbiased, not the true one.")
    else:
        print("  -> symmetric: the gate is not what is holding `s` back.")
        