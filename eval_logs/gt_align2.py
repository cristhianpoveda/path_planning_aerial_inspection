#!/usr/bin/env python3
"""
gt_align2.py -- STEP 15b: align on VELOCITY, with a data-health check first.

Why v1 failed: mocap z is a plateau. Correlating two plateaus gives a broad
peak (0.9204 against a 0.9144 sidelobe -- 0.006 of separation). Differentiate
first: takeoff, landing and every accel/decel corner become sharp edges, and
edges are what a cross-correlation can localise.

Two INDEPENDENT cues are computed. If they agree the alignment is real; if
they disagree by more than a sample period, it is not, and no amount of peak
polish fixes that.

  cue A: |v| from mocap        vs |v| from /drone_1/speed_vector
  cue B: vertical speed, mocap vs speed_vector z (NED -> ENU sign flip)

Also reports mocap HEALTH. A static-window orientation sd of exactly 0.000 deg
is not a quiet sensor -- it means the pose is being repeated, either because
the rigid body is untracked and Motive is holding the last value, or because a
second publisher mirrors the topic (filter_design 3). Numbers derived from
repeated samples are fiction, so this runs before anything else.

Run:
    python3 gt_align2.py F9_02 F9_02_mocap [--out f9_02_aligned.npz]
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

T_ALT = "/drone_1/relative_altitude"
T_VEL = "/drone_1/speed_vector"
T_MOCAP = "/optitrack/rigid_bodies/dji_mini4"


def read_bag(path, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    out = {t: [] for t in topics}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic in out:
            out[topic].append(deserialize_message(data,
                                                  get_message(types[topic])))
    return out


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def resample_diff(t, x, grid, smooth_s=0.10):
    """Resample to a uniform grid FIRST, then differentiate (filter_design 10:
    differentiating mocap at native rate aliases noise)."""
    dt = grid[1] - grid[0]
    y = np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], 1)
    k = max(int(round(smooth_s / dt)) | 1, 3)
    w = np.ones(k) / k
    y = np.stack([np.convolve(y[:, i], w, mode="same") for i in range(y.shape[1])], 1)
    return np.gradient(y, dt, axis=0)


def xcorr_offset(t_a, v_a, t_b, v_b, span=5.0, dt=0.01, t_lo=None, t_hi=None):
    lo = max(t_a[0], t_b[0]) + span
    hi = min(t_a[-1], t_b[-1]) - span
    if t_lo is not None:
        lo = max(lo, t_lo)
    if t_hi is not None:
        hi = min(hi, t_hi)
    grid = np.arange(lo, hi, dt)
    a = np.interp(grid, t_a, v_a)
    # A cue with no variance over the window carries no timing information;
    # normalising it produces a meaningless correlation that peaks at the edge
    # of the lag range. Refuse rather than return a number.
    if a.std() < 1e-6 or np.interp(grid, t_b, v_b).std() < 1e-6:
        return float("nan"), 0.0, 0.0
    a = (a - a.mean()) / (a.std() + 1e-12)
    lags = np.arange(-span, span, dt)
    c = np.empty(len(lags))
    for i, L in enumerate(lags):
        b = np.interp(grid + L, t_b, v_b)
        b = (b - b.mean()) / (b.std() + 1e-12)
        c[i] = float((a * b).mean())
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k-1], c[k], c[k+1]
        k += 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-15)
    d = float(-span + k * dt)
    side = c[np.abs(np.arange(-span, span, dt) - d) > 0.5]
    return d, float(c.max()), float(side.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    f = read_bag(a.flight_bag, [T_ALT, T_VEL])
    m = read_bag(a.mocap_bag, [T_MOCAP])
    mo = m[T_MOCAP]

    t_mo = np.array([stamp(x) for x in mo])
    p_mo = np.array([[x.pose.position.x, x.pose.position.y,
                      x.pose.position.z] for x in mo])
    q_mo = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                      x.pose.orientation.z, x.pose.orientation.w] for x in mo])

    # ---- 0. MOCAP HEALTH ------------------------------------------------
    print("0. MOCAP HEALTH")
    dp = np.linalg.norm(np.diff(p_mo, axis=0), axis=1)
    dq = np.linalg.norm(np.diff(q_mo, axis=0), axis=1)
    rep_p = float((dp == 0).mean())
    rep_q = float((dq == 0).mean())
    dts = np.diff(t_mo)
    print(f"   n={len(t_mo)}  span {t_mo[-1]-t_mo[0]:.1f} s  "
          f"nominal {len(t_mo)/(t_mo[-1]-t_mo[0]):.1f} Hz")
    print(f"   dt: median {np.median(dts)*1e3:.2f} ms  "
          f"p95 {np.percentile(dts,95)*1e3:.2f}  max {dts.max()*1e3:.1f}  "
          f"zero-dt {int((dts==0).sum())}")
    print(f"   bit-identical consecutive POSITION {rep_p:6.1%}   "
          f"ORIENTATION {rep_q:6.1%}")
    if rep_p > 0.10 or rep_q > 0.10:
        print("   *** repeated samples: the effective rate is lower than the "
              "nominal one.\n       Deduplicate before differentiating, or "
              "velocity will be a comb. ***")

    z0 = np.median(p_mo[t_mo < t_mo[0] + 10.0, 2])
    air = p_mo[:, 2] > z0 + 0.15
    t_to, t_land = t_mo[np.argmax(air)], t_mo[len(air) - 1 - np.argmax(air[::-1])]
    stat = t_mo < t_to - 2.0
    print(f"   takeoff +{t_to-t_mo[0]:.1f} s   landing +{t_land-t_mo[0]:.1f} s "
          f"  airborne {t_land-t_to:.1f} s")
    print(f"   STATIC window: position sd {np.std(p_mo[stat],0).round(6)} m")
    print(f"                  unique positions "
          f"{len(np.unique(p_mo[stat], axis=0))} of {int(stat.sum())}")
    if len(np.unique(p_mo[stat], axis=0)) < 0.5 * stat.sum():
        print("   *** the rigid body is NOT being tracked on the ground -- "
              "Motive is holding\n       a stale pose. Static-window frame "
              "constants are meaningless. ***")

    # ---- 1. two-cue alignment ------------------------------------------
    t_v = np.array([stamp(x) for x in f[T_VEL]])
    v_dji = np.array([[x.vector.x, x.vector.y, x.vector.z] for x in f[T_VEL]])
    grid = np.arange(t_mo[0], t_mo[-1], 0.01)
    v_mo = resample_diff(t_mo, p_mo, grid)

    print("\n1. ALIGNMENT (mocap clock relative to flight-bag clock)")
    res = {}
    dA, pA, sA = xcorr_offset(t_v, np.linalg.norm(v_dji, axis=1),
                              grid, np.linalg.norm(v_mo, axis=1),
                              t_lo=t_to, t_hi=t_land)
    res["A |v|"] = (dA, pA, sA)
    dB, pB, sB = xcorr_offset(t_v, -v_dji[:, 2], grid, v_mo[:, 2],
                              t_lo=t_to, t_hi=t_land)
    res["B v_z"] = (dB, pB, sB)
    t_a = np.array([stamp(x) for x in f[T_ALT]])
    z_a = np.array([float(x.altitude) for x in f[T_ALT]])
    gz = np.arange(t_a[0], t_a[-1], 0.01)
    dz = resample_diff(t_a, z_a[:, None], gz)[:, 0]
    dC, pC, sC = xcorr_offset(gz, dz, grid, v_mo[:, 2], t_lo=t_to, t_hi=t_land)
    res["C dz/dt"] = (dC, pC, sC)

    print(f"   {'cue':<10}{'offset s':>11}{'peak':>8}{'sidelobe':>10}{'margin':>9}")
    for k, (d, p, s) in res.items():
        f = f"{d:+11.4f}" if np.isfinite(d) else f"{'flat':>11}"
        print(f"   {k:<10}{f}{p:8.4f}{s:10.4f}{p-s:9.4f}")
    ds = np.array([v[0] for v in res.values() if np.isfinite(v[0])])
    if len(ds) == 0:
        print("   *** no cue had usable variance ***")
        return
    spread = ds.max() - ds.min()
    print(f"   spread across cues {spread*1e3:.1f} ms")
    if spread < 0.05 and min(v[1]-v[2] for v in res.values()) > 0.05:
        print(f"   -> AGREED. use offset {np.median(ds):+.4f} s")
    else:
        print("   *** cues disagree or peaks are flat -- alignment NOT "
              "established ***")

    if a.out:
        np.savez(a.out, offset=float(np.median(ds)), t_mocap=t_mo,
                 p_mocap=p_mo, q_mocap=q_mo, t_takeoff=t_to, t_land=t_land)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()