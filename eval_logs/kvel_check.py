#!/usr/bin/env python3
"""
kvel_check.py -- STEP 32: measure K_VEL against mocap, with no filter involved.

    z = K_VEL * v_true       (DJI UNDER-reports)

so K_VEL is the ratio of DJI-integrated path to mocap path over the same
window. Integration, not regression: differentiating mocap at 100 Hz aliases
noise and attenuated fitted gains by ~4.2x in the earlier analysis
(filter_design.md 10), while integration does not.

Why it matters: init computes s = (|v_dji| / K_VEL) / vo_speed, so
s is proportional to 1/K_VEL. F9_02 init gives 3.627 while mocap says truth is
~3.79-3.96; K_VEL = 0.91 * 3.627 / 3.85 = 0.857 would close that gap exactly.

Also bins the gain by speed, which measures the quantisation dead-zone droop
directly -- and therefore says whether V_LOW = 0.4 is the right threshold for
the scale-hold gate in update_velocity.

The mocap clock offset is FITTED (velocity cross-correlation) rather than
assumed; a lag biases the ratio when speed is changing.

Run:
    python3 kvel_check.py F9_02 F9_02_mocap [--win 3.0] [--vmin 0.15]
"""
import argparse
import sys

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    sys.exit("source your ROS 2 workspace first")

T_VEL = "/drone_1/speed_vector"
T_MOCAP = "/optitrack/rigid_bodies/dji_mini4"
DT = 0.01


def read_bag(path, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    have = {t.name: t.type for t in r.get_all_topics_and_types()}
    out = {t: [] for t in topics if t in have}
    while r.has_next():
        tp, data, _ = r.read_next()
        if tp in out:
            out[tp].append(deserialize_message(data, get_message(have[tp])))
    return out


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def mocap_speed(t, p, smooth_s=0.10, dt=DT):
    """Smoothed mocap speed on a uniform grid.

    NOT raw arc length. Summing |dp| at 100 Hz adds the NOISE's own path
    length: at 1 mm per-sample noise a true 0.9 m of travel measures 1.08 m,
    so the ratio reads K_VEL = 0.75 when the truth is 0.91. The inflation
    scales as noise/(speed*dt), which is why the raw method produced a
    monotone rise with speed that LOOKS like dead-zone droop and is not.
    Resample-then-smooth-then-differentiate first (filter_design.md 10).
    """
    g = np.arange(t[0], t[-1], dt)
    k = max(int(round(smooth_s / dt)) | 1, 3)
    w = np.ones(k) / k
    v = []
    for i in range(3):
        y = np.convolve(np.interp(g, t, p[:, i]), w, mode="same")
        v.append(np.gradient(y, dt))
    return g, np.linalg.norm(np.stack(v, 1), axis=1)


def path_len(t, sp, t0, t1):
    """Mocap path over [t0, t1] as the integral of smoothed speed."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:
        return np.nan
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(sp[m], t[m]))


def raw_arc(t, p, t0, t1):
    """Raw arc length -- kept only to REPORT the noise inflation."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:
        return np.nan
    return float(np.sum(np.linalg.norm(np.diff(p[m], axis=0), axis=1)))


def dji_path(t, v, t0, t1):
    """Integral of |v_dji| dt over the window, trapezoid on the native stamps."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:
        return np.nan, np.nan
    tt, vv = t[m], np.linalg.norm(v[m], axis=1)
    # np.trapz was removed in numpy 2; trapezoid is the same function
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(vv, tt)), float(vv.mean())


def fit_offset(t_v, v_d, t_m, p_m, lo, hi, span=0.5):
    g = np.arange(lo, hi, DT)
    vm = np.stack([np.gradient(np.interp(g, t_m, p_m[:, i]), DT)
                   for i in range(3)], 1)
    k = 11
    vm = np.stack([np.convolve(vm[:, i], np.ones(k)/k, "same")
                   for i in range(3)], 1)
    a = np.linalg.norm(vm, axis=1)
    a = (a - a.mean()) / (a.std() + 1e-12)
    best, bc = 0.0, -np.inf
    for d in np.arange(-span, span, DT):
        b = np.interp(g + d, t_v, np.linalg.norm(v_d, axis=1))
        sd = b.std()
        if sd < 1e-9:
            continue
        c = float((a * ((b - b.mean()) / sd)).mean())
        if c > bc:
            bc, best = c, float(d)
    return best, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--win", type=float, default=3.0, help="window length s")
    ap.add_argument("--vmin", type=float, default=0.15)
    a = ap.parse_args()

    f = read_bag(a.flight_bag, [T_VEL])
    mo = read_bag(a.mocap_bag, [T_MOCAP])[T_MOCAP]
    t_v = np.array([stamp(x) for x in f[T_VEL]])
    v_d = np.array([[x.vector.x, x.vector.y, x.vector.z] for x in f[T_VEL]])
    t_m = np.array([stamp(x) for x in mo])
    p_m = np.array([[x.pose.position.x, x.pose.position.y,
                     x.pose.position.z] for x in mo])
    keep = np.concatenate([[True], (np.abs(np.diff(p_m, axis=0)).sum(1) > 0)])
    t_m, p_m = t_m[keep], p_m[keep]

    floor = np.percentile(p_m[:, 2], 5)
    air = p_m[:, 2] > floor + 0.20
    lo = t_m[np.argmax(air)] + 2.0
    hi = t_m[len(air) - 1 - np.argmax(air[::-1])] - 2.0
    print(f"{a.flight_bag}: airborne {hi-lo:.1f} s, "
          f"{int((~keep).sum())} mocap dups dropped")

    d, c = fit_offset(t_v, v_d, t_m, p_m, lo + 1, hi - 1)
    print(f"fitted mocap offset {d:+.3f} s (corr {c:.3f}) -- applied\n")
    t_m = t_m - d

    g_m, sp_m = mocap_speed(t_m, p_m)
    rows = []
    t0 = lo
    while t0 + a.win < hi:
        t1 = t0 + a.win
        Lm = path_len(g_m, sp_m, t0, t1)
        Lraw = raw_arc(t_m, p_m, t0, t1)
        Ld, vbar = dji_path(t_v, v_d, t0, t1)
        if np.isfinite(Lm) and np.isfinite(Ld) and Lm > a.vmin * a.win:
            rows.append((vbar, Ld / Lm, Lm, Lraw / max(Lm, 1e-9)))
        t0 = t1
    if len(rows) < 5:
        sys.exit("too few windows")
    R = np.array(rows)
    print(f"{len(R)} windows of {a.win:.1f} s\n")

    print(f"{'speed bin m/s':>16}{'n':>5}{'K_VEL':>9}{'sd':>8}{'path m':>9}"
          f"{'rawinfl':>9}")
    edges = [0.0, 0.2, 0.3, 0.4, 0.6, 0.9, 1.3, 9.9]
    for i in range(len(edges) - 1):
        m = (R[:, 0] >= edges[i]) & (R[:, 0] < edges[i+1])
        if m.sum() >= 3:
            print(f"{edges[i]:7.1f}-{edges[i+1]:<8.1f}{m.sum():5d}"
                  f"{np.median(R[m,1]):9.4f}{R[m,1].std():8.4f}"
                  f"{R[m,2].sum():9.1f}{np.median(R[m,3]):9.3f}")

    fast = R[R[:, 0] >= 0.4]
    print(f"\nALL windows      K_VEL median {np.median(R[:,1]):.4f}")
    if len(fast) >= 3:
        print(f"windows >= 0.4   K_VEL median {np.median(fast[:,1]):.4f}  "
              f"(n={len(fast)})")
    tot = R[:, 2].sum()
    print(f"whole-flight     K_VEL = {np.sum(R[:,1]*R[:,2])/tot:.4f} "
          f"(path-weighted, {tot:.0f} m)")
    kv = np.median(fast[:, 1]) if len(fast) >= 3 else np.median(R[:, 1])
    print(f"\n'rawinfl' is raw arc length / smoothed path. Values well above 1.00\n"
          f"   mean the RAW method was measuring mocap noise, not motion.")
    print(f"\ncurrent K_VEL = 0.91 -> init `s` would change by "
          f"x{0.91/kv:.4f}")
    print(f"   F9_02 init s = 3.627 -> {3.627*0.91/kv:.3f}   "
          f"(mocap says truth ~3.79-3.96)")


if __name__ == "__main__":
    main()