#!/usr/bin/env python3
"""
channel_lags2.py -- STEP 17: per-channel latency, now including VO.

Adds the row that actually matters. `update_velocity` never compares velocity
against attitude or against mocap; it compares it against the VO increment. So
the quantity VEL_DELAY must correct is

        velocity stamp  -  VO stamp

and both are measured here against the same mocap reference, so the common
clock term cancels in the difference exactly as it does for attitude.

Two VO cues, both chosen to be independent of things VO does not know:

  vo |v|    |dp_vo|/dt  vs mocap |v|.  VO is unscaled, but normalised
            cross-correlation is scale-invariant, so an unknown constant
            factor does not move the peak.
  vo |w|    angular rate from consecutive VO quaternions vs mocap angular
            rate. Rotation MAGNITUDE is invariant to the VO frame as well as
            to its scale, so this cue is immune to R_n_v being unknown.

VO increments are masked where they are not trustworthy: dt outside
[0.5, 2.0] x nominal, samples with pose_valid False, and any increment that
spans a vo_epoch change (a map rebuild moves the frame, so |dp| is meaningless
across it).

Run:
    python3 channel_lags2.py F9_02 F9_02_mocap
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
T_VO = "/drone_1/vo/pose"
T_VOS = "/drone_1/vo/status"
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


def quat_yaw_conj(q):
    x, y, z, w = (-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3])
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def quat_rate(t, q):
    """|angular rate| between consecutive quaternions. Frame-invariant."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    d = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
    ang = 2.0 * np.arccos(d)
    dt = np.diff(t)
    ok = dt > 1e-6
    tm = 0.5 * (t[1:] + t[:-1])            # MIDPOINT, not the end of the
    return tm[ok], ang[ok] / dt[ok]        # interval: end-stamping biases
                                           # the lag by +dt/2 (26 ms at 19.4 Hz)


def smooth_diff(t, x, grid, smooth_s=0.10):
    dt = grid[1] - grid[0]
    y = np.interp(grid, t, x)
    k = max(int(round(smooth_s / dt)) | 1, 3)
    return np.gradient(np.convolve(y, np.ones(k) / k, mode="same"), dt)


def smooth_resample(t, x, grid, smooth_s=0.10):
    dt = grid[1] - grid[0]
    y = np.interp(grid, t, x)
    k = max(int(round(smooth_s / dt)) | 1, 3)
    return np.convolve(y, np.ones(k) / k, mode="same")


def lag(t_a, v_a, t_b, v_b, lo, hi, span=1.0):
    """Lag of channel `a` behind reference `b`, seconds. POSITIVE = a is late.
    Sign verified by construction (see the self-test at the bottom)."""
    grid = np.arange(max(lo, t_a[0], t_b[0]) + span,
                     min(hi, t_a[-1], t_b[-1]) - span, DT)
    if len(grid) < 100:
        return float("nan"), 0.0, 0.0
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
    return -d, float(c.max()), float(side.max() if len(side) else 0.0)


def vo_speed(t_vo, p_vo, epoch, valid):
    """|dp|/dt from consecutive VO poses, with the untrustworthy ones masked."""
    dt = np.diff(t_vo)
    nom = float(np.median(dt))
    sp = np.linalg.norm(np.diff(p_vo, axis=0), axis=1) / np.maximum(dt, 1e-9)
    ok = (dt > 0.5 * nom) & (dt < 2.0 * nom)
    if epoch is not None:
        ok &= (np.diff(epoch) == 0)
    if valid is not None:
        ok &= valid[1:] & valid[:-1]
    # a map jump survives the epoch test if the epoch field never moved
    ok &= sp < 20.0 * np.median(sp[ok]) if ok.any() else ok
    tm = 0.5 * (t_vo[1:] + t_vo[:-1])      # midpoint, see quat_rate()
    return tm[ok], sp[ok], nom, float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--span", type=float, default=1.0)
    a = ap.parse_args()

    f = read_bag(a.flight_bag, [T_ALT, T_VEL, T_ATT, T_VO, T_VOS])
    mo = read_bag(a.mocap_bag, [T_MOCAP])[T_MOCAP]

    t_m = np.array([stamp(x) for x in mo])
    p_m = np.array([[x.pose.position.x, x.pose.position.y,
                     x.pose.position.z] for x in mo])
    q_m = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                     x.pose.orientation.z, x.pose.orientation.w] for x in mo])
    keep = np.concatenate([[True], (np.abs(np.diff(p_m, axis=0)).sum(1) > 0)])
    print(f"mocap {len(t_m)} poses, {int((~keep).sum())} duplicates dropped "
          f"-> {int(keep.sum())} ({keep.sum()/(t_m[-1]-t_m[0]):.1f} Hz)")
    t_m, p_m, q_m = t_m[keep], p_m[keep], q_m[keep]

    z0 = np.median(p_m[t_m < t_m[0] + 10, 2])
    air = p_m[:, 2] > z0 + 0.15
    t_to = t_m[np.argmax(air)] + 2.0
    t_ld = t_m[len(air) - 1 - np.argmax(air[::-1])] - 2.0
    print(f"airborne window {t_ld - t_to:.1f} s")

    grid = np.arange(t_m[0], t_m[-1], DT)
    vz_m = smooth_diff(t_m, p_m[:, 2], grid)
    v_m = np.sqrt(smooth_diff(t_m, p_m[:, 0], grid) ** 2
                  + smooth_diff(t_m, p_m[:, 1], grid) ** 2 + vz_m ** 2)
    tw_m, w_m_raw = quat_rate(t_m, q_m)
    w_m = smooth_resample(tw_m, w_m_raw, grid)

    rows = []

    t_a = np.array([stamp(x) for x in f[T_ATT]])
    rpy = np.array([[x.roll, x.pitch, x.yaw] for x in f[T_ATT]])
    if np.abs(rpy).max() > 2 * np.pi:
        rpy = np.radians(rpy)
    ga = np.arange(t_a[0], t_a[-1], DT)
    rows.append(("attitude  yaw rate", ga,
                 smooth_diff(t_a, np.unwrap(rpy[:, 2]), ga), grid, w_m * 0 +
                 smooth_resample(*quat_rate(t_m, q_m), grid) * 0 +
                 smooth_diff(t_m, np.unwrap(quat_yaw_conj(q_m)), grid)))

    t_z = np.array([stamp(x) for x in f[T_ALT]])
    z_d = np.array([float(x.altitude) for x in f[T_ALT]])
    gz = np.arange(t_z[0], t_z[-1], DT)
    rows.append(("altitude  dz/dt", gz, smooth_diff(t_z, z_d, gz), grid, vz_m))

    t_v = np.array([stamp(x) for x in f[T_VEL]])
    v_d = np.array([[x.vector.x, x.vector.y, x.vector.z] for x in f[T_VEL]])
    rows.append(("velocity  v_z", t_v, -v_d[:, 2], grid, vz_m))
    rows.append(("velocity  |v|", t_v, np.linalg.norm(v_d, axis=1), grid, v_m))

    if T_VO in f and len(f[T_VO]) > 50:
        t_vo = np.array([stamp(x) for x in f[T_VO]])
        p_vo = np.array([[x.pose.position.x, x.pose.position.y,
                          x.pose.position.z] for x in f[T_VO]])
        q_vo = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                          x.pose.orientation.z, x.pose.orientation.w]
                         for x in f[T_VO]])
        epoch = valid = None
        if T_VOS in f and len(f[T_VOS]):
            t_s = np.array([stamp(x) for x in f[T_VOS]])
            e = np.array([float(getattr(x, "vo_epoch", 0)) for x in f[T_VOS]])
            pv = np.array([bool(getattr(x, "pose_valid", True))
                           for x in f[T_VOS]], bool)
            epoch = np.interp(t_vo, t_s, e).round()
            valid = np.interp(t_vo, t_s, pv.astype(float)) > 0.5
            print(f"vo/status: {int(e.max()-e.min())} epoch changes, "
                  f"pose_valid {pv.mean():.1%}")
        tv, sv, nom, frac = vo_speed(t_vo, p_vo, epoch, valid)
        print(f"vo/pose: {len(t_vo)} poses at {1/nom:.1f} Hz, "
              f"{frac:.1%} of increments usable")
        rows.append(("vo        |v|", tv, sv, grid, v_m))
        tw, wv = quat_rate(t_vo, q_vo)
        rows.append(("vo        |w|", tw, wv, grid, w_m))
    else:
        print("vo/pose absent or too short -- VO rows skipped")

    print(f"\n{'channel':<20}{'lag s':>10}{'peak':>8}{'sidelobe':>10}{'margin':>9}")
    got = {}
    for lbl, ta, va, tb, vb in rows:
        d, pk, sd = lag(ta, va, tb, vb, t_to, t_ld, a.span)
        got[lbl] = d
        ds = f"{d:+10.4f}" if np.isfinite(d) else f"{'flat':>10}"
        print(f"{lbl:<20}{ds}{pk:8.4f}{sd:10.4f}{pk-sd:9.4f}")

    att = got.get("attitude  yaw rate", np.nan)
    print("\nDIFFERENTIAL vs attitude (diagnostic)")
    for k, v in got.items():
        if not k.startswith("attitude"):
            print(f"   {k:<20}{v - att:+.4f} s")

    vo = got.get("vo        |v|", np.nan)
    if np.isfinite(vo):
        print("\nDIFFERENTIAL vs VO  <-- this is VEL_DELAY")
        for k in ("velocity  |v|", "velocity  v_z", "altitude  dz/dt"):
            if k in got:
                print(f"   {k:<20}{got[k] - vo:+.4f} s")
        print(f"\n   VEL_DELAY = {got.get('velocity  |v|', np.nan) - vo:+.4f} s"
              f"   (currently 0.010 in params.py)")


if __name__ == "__main__":
    main()