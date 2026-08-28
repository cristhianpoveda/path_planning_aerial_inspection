#!/usr/bin/env python3
"""
channel_lags.py -- STEP 16: per-channel latency of DJI telemetry vs mocap.

If the two clocks are compatible (they are -- three cues agreed to 18 ms), then
a lag measured against mocap is NOT a clock offset. It is either a real
transport/measurement latency on that channel, or a residual common clock
offset shared by all of them. The two are separable the moment you measure
more than one channel:

    lag_channel = clock_offset + latency_channel

so the DIFFERENCES between channels are the real per-channel latencies, free of
whatever the common term is. That difference is exactly what the filter needs:
re-stamping is only correct for the part that is NOT common-mode
(filter_design 5.1/5.3).

Channels, each with a cue chosen to be sharp and unbiased:
  attitude   yaw RATE          (yaw is immune to the tilt-under-accel bias
                                that corrupts roll/pitch -- same choice as
                                check_g2.py's sync_mocap force="yaw_rate")
  altitude   d(altitude)/dt    vs mocap v_z
  velocity   speed_vector z    vs mocap v_z   (NED -> ENU sign flip)
  velocity   |speed_vector|    vs mocap |v|

Mocap is DEDUPLICATED first: 7.6% of consecutive poses in F9_02 are
bit-identical, and interpolating through them makes velocity a comb.
Orientation is CONJUGATED (check_g2.py rev 2, section E).

Run:
    python3 channel_lags.py F9_02 F9_02_mocap
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
T_ATT = "/drone_1/attitude"
T_MOCAP = "/optitrack/rigid_bodies/dji_mini4"
DT = 0.01


def read_bag(path, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    out = {t: [] for t in topics}
    while r.has_next():
        tp, data, _ = r.read_next()
        if tp in out:
            out[tp].append(deserialize_message(data, get_message(types[tp])))
    return out


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def quat_yaw_conj(q):
    """Yaw of the CONJUGATED mocap quaternion, vectorised. q is (n,4) xyzw."""
    x, y, z, w = (-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3])
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def smooth_diff(t, x, grid, smooth_s=0.10):
    dt = grid[1] - grid[0]
    y = np.interp(grid, t, x)
    k = max(int(round(smooth_s / dt)) | 1, 3)
    y = np.convolve(y, np.ones(k) / k, mode="same")
    return np.gradient(y, dt)


def lag(t_a, v_a, t_b, v_b, lo, hi, span=1.0):
    """Lag of channel `a` behind reference `b`, in seconds, POSITIVE = a is late.

    Sign is verified by construction: delaying `a` by +0.075 s and measuring
    returns +0.0751.
    """
    grid = np.arange(max(lo, t_a[0], t_b[0]) + span,
                     min(hi, t_a[-1], t_b[-1]) - span, DT)
    a = np.interp(grid, t_a, v_a)
    if a.std() < 1e-9:
        return float("nan"), 0.0, 0.0
    a = (a - a.mean()) / a.std()
    lags = np.arange(-span, span, DT)
    c = np.empty(len(lags))
    for i, L in enumerate(lags):
        b = np.interp(grid + L, t_b, v_b)
        sd = b.std()
        c[i] = float((a * ((b - b.mean()) / sd)).mean()) if sd > 1e-9 else 0.0
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k-1], c[k], c[k+1]
        k += 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-15)
    d = float(-span + k * DT)
    side = c[np.abs(lags - d) > 0.3]
    # the correlation finds the shift applied to b; the lag of a is its negative
    return -d, float(c.max()), float(side.max() if len(side) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--span", type=float, default=1.0)
    a = ap.parse_args()

    f = read_bag(a.flight_bag, [T_ALT, T_VEL, T_ATT])
    mo = read_bag(a.mocap_bag, [T_MOCAP])[T_MOCAP]

    t_m = np.array([stamp(x) for x in mo])
    p_m = np.array([[x.pose.position.x, x.pose.position.y,
                     x.pose.position.z] for x in mo])
    q_m = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                     x.pose.orientation.z, x.pose.orientation.w] for x in mo])

    # dedupe: a repeated pose carries no new information and flattens v
    keep = np.concatenate([[True], (np.abs(np.diff(p_m, axis=0)).sum(1) > 0)])
    print(f"mocap {len(t_m)} poses, {int((~keep).sum())} duplicates dropped "
          f"-> {int(keep.sum())} unique "
          f"({keep.sum()/(t_m[-1]-t_m[0]):.1f} Hz effective)")
    t_m, p_m, q_m = t_m[keep], p_m[keep], q_m[keep]

    z0 = np.median(p_m[t_m < t_m[0] + 10, 2])
    air = p_m[:, 2] > z0 + 0.15
    t_to = t_m[np.argmax(air)] + 2.0
    t_ld = t_m[len(air) - 1 - np.argmax(air[::-1])] - 2.0
    print(f"airborne window {t_ld - t_to:.1f} s\n")

    grid = np.arange(t_m[0], t_m[-1], DT)
    vz_m = smooth_diff(t_m, p_m[:, 2], grid)
    vh_m = np.hypot(smooth_diff(t_m, p_m[:, 0], grid),
                    smooth_diff(t_m, p_m[:, 1], grid))
    v_m = np.sqrt(vh_m**2 + vz_m**2)
    yaw_m = np.unwrap(quat_yaw_conj(q_m))
    wz_m = smooth_diff(t_m, yaw_m, grid)

    t_a = np.array([stamp(x) for x in f[T_ATT]])
    rpy = np.array([[x.roll, x.pitch, x.yaw] for x in f[T_ATT]])
    if np.abs(rpy).max() > 2 * np.pi:
        rpy = np.radians(rpy)
    ga = np.arange(t_a[0], t_a[-1], DT)
    wz_d = smooth_diff(t_a, np.unwrap(rpy[:, 2]), ga)

    t_z = np.array([stamp(x) for x in f[T_ALT]])
    z_d = np.array([float(x.altitude) for x in f[T_ALT]])
    gz = np.arange(t_z[0], t_z[-1], DT)
    dz_d = smooth_diff(t_z, z_d, gz)

    t_v = np.array([stamp(x) for x in f[T_VEL]])
    v_d = np.array([[x.vector.x, x.vector.y, x.vector.z] for x in f[T_VEL]])

    rows = [
        ("attitude  yaw rate", ga, wz_d, grid, wz_m),
        ("altitude  dz/dt", gz, dz_d, grid, vz_m),
        ("velocity  v_z", t_v, -v_d[:, 2], grid, vz_m),
        ("velocity  |v|", t_v, np.linalg.norm(v_d, axis=1), grid, v_m),
    ]
    print(f"{'channel':<20}{'lag s':>10}{'peak':>8}{'sidelobe':>10}{'margin':>9}")
    got = {}
    for lbl, ta, va, tb, vb in rows:
        d, pk, sd = lag(ta, va, tb, vb, t_to, t_ld, a.span)
        got[lbl] = d
        ds = f"{d:+10.4f}" if np.isfinite(d) else f"{'flat':>10}"
        print(f"{lbl:<20}{ds}{pk:8.4f}{sd:10.4f}{pk-sd:9.4f}")

    att = got["attitude  yaw rate"]
    print("\nDIFFERENTIAL vs attitude  (this is what re-stamping must correct)")
    for k, v in got.items():
        if k.startswith("attitude"):
            continue
        print(f"   {k:<20}{v - att:+.4f} s")
    print("\n   common term (clock offset + shared latency) is the attitude "
          f"lag itself: {att:+.4f} s\n   it cancels in every difference above.")


if __name__ == "__main__":
    main()
    