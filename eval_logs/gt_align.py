#!/usr/bin/env python3
"""
gt_align.py -- STEP 15a: align a flight bag to its mocap bag, then read the
two things the static ground window gives you for free.

The drone bag and the mocap bag are recorded by different processes, possibly
on different machines, so their clocks are NOT known to agree. Every ATE, RPE
or residual-vs-truth number downstream is wrong by whatever that offset is, and
a 100 ms offset at 0.8 m/s is 8 cm of pure artefact. So this runs first and
nothing else runs until its correlation peak is sharp.

Alignment cue: relative_altitude vs mocap z. Takeoff is a step in both, and
both bags contain >=15 s of ground before it, so the feature is unambiguous.

Also reports, from the STATIC pre-takeoff window:
  * mocap roll/pitch vs DJI roll/pitch  -> the optitrack_map->map tilt,
    re-fitted for THIS bag rather than inherited from G2_yaw_reference
  * mocap yaw vs DJI yaw                -> the yaw datum offset

Run:
    python3 gt_align.py F9_02 F9_02_mocap [--conj] [--out f9_02_aligned.npz]
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

TOPIC_ALT = "/drone_1/relative_altitude"
TOPIC_ATT = "/drone_1/attitude"
TOPIC_MOCAP = "/optitrack/rigid_bodies/dji_mini4"


def read_bag(path, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in out:
            out[topic].append(deserialize_message(
                data, get_message(types[topic])))
    return out


def stamp(msg):
    h = msg.header.stamp
    return h.sec + h.nanosec * 1e-9


def quat_to_rpy(x, y, z, w):
    """(x,y,z,w) -> roll, pitch, yaw (ZYX), vectorised."""
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def best_offset(t_a, v_a, t_b, v_b, span=5.0, dt=0.01):
    """Offset d (seconds) such that b(t) best matches a(t + d)."""
    t0 = max(t_a[0], t_b[0]) + span
    t1 = min(t_a[-1], t_b[-1]) - span
    grid = np.arange(t0, t1, dt)
    a = np.interp(grid, t_a, v_a)
    a = (a - a.mean()) / (a.std() + 1e-12)
    lags = np.arange(-span, span, dt)
    corr = np.empty(len(lags))
    for i, L in enumerate(lags):
        b = np.interp(grid + L, t_b, v_b)
        b = (b - b.mean()) / (b.std() + 1e-12)
        corr[i] = float((a * b).mean())
    k = int(np.argmax(corr))
    # parabolic refinement on the peak
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k-1], corr[k], corr[k+1]
        d = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-15)
    else:
        d = 0.0
    return float(lags[k] + d * dt), float(corr[k]), lags, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--conj", action="store_true",
                    help="conjugate the mocap quaternion (filter_design 3)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    f = read_bag(a.flight_bag, [TOPIC_ALT, TOPIC_ATT])
    m = read_bag(a.mocap_bag, [TOPIC_MOCAP])

    t_alt = np.array([stamp(x) for x in f[TOPIC_ALT]])
    z_alt = np.array([float(x.altitude) for x in f[TOPIC_ALT]])
    t_att = np.array([stamp(x) for x in f[TOPIC_ATT]])
    rpy_dji = np.array([[x.roll, x.pitch, x.yaw] for x in f[TOPIC_ATT]])
    if np.abs(rpy_dji).max() > 2 * np.pi:
        rpy_dji = np.radians(rpy_dji)
        deg_note = " (published in DEGREES, converted)"
    else:
        deg_note = " (published in radians)"

    mo = m[TOPIC_MOCAP]
    t_mo = np.array([stamp(x) for x in mo])
    p_mo = np.array([[x.pose.position.x, x.pose.position.y,
                      x.pose.position.z] for x in mo])
    q = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                   x.pose.orientation.z, x.pose.orientation.w] for x in mo])
    if a.conj:
        q[:, :3] *= -1.0

    print(f"flight bag  {a.flight_bag}")
    print(f"  altitude  n={len(t_alt):6d}  {len(t_alt)/(t_alt[-1]-t_alt[0]):6.2f} Hz")
    print(f"  attitude  n={len(t_att):6d}{deg_note}")
    print(f"mocap bag   {a.mocap_bag}")
    print(f"  pose      n={len(t_mo):6d}  {len(t_mo)/(t_mo[-1]-t_mo[0]):6.2f} Hz")
    print(f"  raw start difference {t_mo[0] - t_alt[0]:+.3f} s")

    # ---- 1. clock offset ---------------------------------------------
    d, peak, lags, corr = best_offset(t_alt, z_alt, t_mo, p_mo[:, 2])
    side = corr[np.abs(lags - d) > 0.5]
    print("\n1. CLOCK OFFSET (mocap relative to flight bag)")
    print(f"   offset {d:+.4f} s   peak corr {peak:.4f}   "
          f"next-best outside 0.5 s {side.max():.4f}")
    if peak < 0.9:
        print("   *** WEAK PEAK -- do not trust anything downstream ***")
    elif peak - side.max() < 0.05:
        print("   *** AMBIGUOUS -- peak barely beats the sidelobes ***")
    else:
        print("   -> sharp; use this offset for every comparison")
    t_mo_al = t_mo - d

    # ---- 2. takeoff, and the static window ---------------------------
    z0 = np.median(p_mo[t_mo_al < t_mo_al[0] + 10.0, 2])
    moving = p_mo[:, 2] > z0 + 0.15
    t_takeoff = t_mo_al[np.argmax(moving)] if moving.any() else None
    print(f"\n2. TAKEOFF at t = {t_takeoff:.2f} "
          f"({t_takeoff - t_mo_al[0]:.1f} s into the mocap bag)")
    stat = (t_mo_al < t_takeoff - 2.0)
    print(f"   static window {stat.sum()} mocap samples "
          f"({t_mo_al[stat][-1] - t_mo_al[stat][0]:.1f} s)")

    r_mo, p_mo_a, y_mo = quat_to_rpy(*q.T)
    sd = np.array([r_mo[stat].std(), p_mo_a[stat].std(), y_mo[stat].std()])
    print(f"   mocap rpy sd over the static window (deg): "
          f"{np.degrees(sd).round(3)}")

    ds = (t_att > t_mo_al[stat][0]) & (t_att < t_mo_al[stat][-1])
    if ds.sum() < 5:
        print("   too few DJI attitude samples in the static window")
        return
    dji = rpy_dji[ds].mean(0)
    moc = np.array([np.median(r_mo[stat]), np.median(p_mo_a[stat]),
                    np.median(y_mo[stat])])

    print("\n3. STATIC-WINDOW FRAME CONSTANTS  (mocap - DJI, degrees)")
    for lbl, i in (("roll ", 0), ("pitch", 1), ("yaw  ", 2)):
        diff = np.degrees((moc[i] - dji[i] + np.pi) % (2*np.pi) - np.pi)
        print(f"   {lbl}  mocap {np.degrees(moc[i]):+8.3f}   "
              f"DJI {np.degrees(dji[i]):+8.3f}   diff {diff:+8.3f}")
    tilt = np.hypot(moc[0] - dji[0], moc[1] - dji[1])
    bearing = np.degrees(np.arctan2(moc[1] - dji[1], moc[0] - dji[0]))
    print(f"   tilt magnitude {np.degrees(tilt):.3f} deg at bearing "
          f"{bearing:+.1f} deg")
    print(f"   currently in the static transform: roll -1.112, pitch -0.300 "
          f"deg (mag 1.152)")

    if a.out:
        np.savez(a.out, offset=d, t_mocap=t_mo_al, p_mocap=p_mo, q_mocap=q,
                 t_alt=t_alt, z_alt=z_alt, t_att=t_att, rpy_dji=rpy_dji,
                 t_takeoff=t_takeoff)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
    