#!/usr/bin/env python3
"""
residual_structure.py -- is the velocity residual a SCALE error or a CONSTANT
offset? and is the filter's confidence in `s` justified?

Projects each residual onto h:  par = (y . h)/|h|   [m/s]

    z = k*h        (scale error)      ->  par = (k-1)*|h|   slope, no intercept
    z = h + c*hhat (constant offset)  ->  par = c           intercept, no slope

`s` is the right knob only for the first. A constant offset cannot be removed
by any value of `s` -- it comes from the pairing (VEL_DELAY), from K_VEL, or
from the lever arm.

Run:  python3 residual_structure.py <innovation_log.csv> [t_max_rel]
"""
import sys

import numpy as np
import pandas as pd

CHI2_3_MEDIAN = 2.366                    # median of chi-square, 3 dof
CSV = sys.argv[1]
T_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0

d = pd.read_csv(CSV)
d["t"] -= d.t.min()
v = d[(d.kind == "velocity") & (d.t < T_MAX)].copy()
y = v[["y0", "y1", "y2"]].to_numpy(float)
h = v[["h0", "h1", "h2"]].to_numpy(float)
hn = np.linalg.norm(h, axis=1)
keep = (hn > 1e-6) & v.accepted.to_numpy().astype(bool)
y, h, hn = y[keep], h[keep], hn[keep]
tt = v.t.to_numpy(float)[keep]
nis = v.nis.to_numpy(float)[keep]
s_ser = v.s.to_numpy(float)[keep]
sig_ser = v.sigma_s.to_numpy(float)[keep]
par = (y * h).sum(1) / hn
n = len(par)

print(f"{CSV}\n  {n} ACCEPTED velocity updates, t < {T_MAX:.0f} s")
if n < 10:
    sys.exit("  too few to regress")

# ---- 1. slope vs intercept ------------------------------------------------
A = np.vstack([hn, np.ones(n)]).T
(slope, icept), res, *_ = np.linalg.lstsq(A, par, rcond=None)
pred = A @ [slope, icept]
ss_tot = ((par - par.mean()) ** 2).sum()
r2 = 1 - ((par - pred) ** 2).sum() / max(ss_tot, 1e-12)
# each model alone
slope_only = float(np.linalg.lstsq(hn[:, None], par, rcond=None)[0][0])
rms_slope = float(np.sqrt(np.mean((par - slope_only * hn) ** 2)))
icept_only = float(par.mean())
rms_icept = float(np.sqrt(np.mean((par - icept_only) ** 2)))

print("\n1. SCALE vs CONSTANT   par = slope*|h| + intercept")
print(f"   joint fit     slope {slope:+.4f}   intercept {icept:+.4f} m/s"
      f"   R2 {r2:.3f}")
print(f"   slope only    slope {slope_only:+.4f}          rms {rms_slope:.4f}")
print(f"   intercept only            {icept_only:+.4f} m/s  rms {rms_icept:.4f}")
print(f"   |h| range {hn.min():.3f} .. {hn.max():.3f} m/s  (lever for the fit)")
if rms_slope < 0.8 * rms_icept:
    print("   -> SCALE-like: `s` is the right knob")
elif rms_icept < 0.8 * rms_slope:
    print("   -> CONSTANT-like: no value of `s` removes this "
          "(pairing / K_VEL / lever arm)")
else:
    print("   -> MIXED or under-determined: check the |h| range above")

s_now = float(np.median(s_ser))
print(f"\n   s during window {s_now:.3f}   implied by slope "
      f"{s_now * (1 + slope_only):.3f}")

# ---- 2. is the confidence justified? --------------------------------------
print("\n2. NIS CONSISTENCY  (accepted only)")
print(f"   median NIS {np.median(nis):7.3f}   vs chi2(3) median {CHI2_3_MEDIAN}")
infl = np.median(nis) / CHI2_3_MEDIAN
print(f"   -> S is understated by ~{infl:.2f}x; R_speed needs ~{infl:.2f}x "
      f"inflation\n      (or the residual carries an unmodelled bias, "
      f"which the fit above tells apart)")

# ---- 3. did sigma_s collapse before the bias was removed? -----------------
print("\n3. CONFIDENCE vs BIAS OVER TIME (10 s buckets)")
print(f"   {'t':>7} {'n':>4} {'s':>9} {'sigma_s':>9} {'mean par':>10} "
      f"{'rel err implied':>16}")
for lo in np.arange(0, T_MAX, 10):
    sel = (tt >= lo) & (tt < lo + 10)
    if sel.sum() > 2:
        mp = par[sel].mean()
        rel = mp / max(np.median(hn[sel]), 1e-9)
        print(f"   {lo:7.0f} {sel.sum():4d} {s_ser[sel][-1]:9.4f} "
              f"{sig_ser[sel][-1]:9.4f} {mp:+10.4f} {rel:+15.1%}")
print("\n   sigma_s/s at the end: "
      f"{sig_ser[-1] / max(s_ser[-1], 1e-9):.3%}  "
      f"vs implied relative error {par.mean() / max(np.median(hn), 1e-9):+.1%}")