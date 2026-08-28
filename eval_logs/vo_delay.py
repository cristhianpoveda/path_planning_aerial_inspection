#!/usr/bin/env python3
"""
vo_delay.py -- STEP 19: measure VO_DELAY precisely, and check it is constant.

camera_decoder_node stamps every frame with get_clock().now() at the moment it
leaves the decoder. No latency is subtracted -- data_flow.md describes
"decode time minus VIDEO_LATENCY" but the code does not implement it. So the
VO stamp is late by the whole OcuSync -> phone -> TCP -> PyAV path.

Two improvements over channel_lags2.py:

  1. MASKING. That script computed VO angular rate from ALL poses, including
     ones spanning an epoch change or with pose_valid False. A map rebuild
     injects an enormous spurious rotation, which is why the |w| peaks came
     back at 0.34/0.71/0.16. Masked properly, |w| is the sharpest cue there is.

  2. A MOCAP-FREE cue. Rotation MAGNITUDE is invariant to frame and scale, so
     VO |w| can be compared directly against DJI attitude |w| -- both in the
     flight bag, no mocap, no clock question at all. This measures
     VO - attitude in one step, which is the differential the filter needs.

Also splits the flight into segments and reports the delay in each: a constant
can be corrected with one parameter, a drifting one cannot.

Run:
    python3 vo_delay.py F9_02 [--mocap F9_02_mocap] [--segments 4]
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

T_ATT = "/drone_1/attitude"
T_VO = "/drone_1/vo/pose"
T_VOS = "/drone_1/vo/status"
T_ALT = "/drone_1/relative_altitude"
T_MOCAP = "/optitrack/rigid_bodies/dji_mini4"
DT = 0.005
PEAK_MIN = 0.40      # below this the correlation found noise
MARGIN_MIN = 0.10    # peak must beat the sidelobes by this much


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


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]])


def quat_to_R(q):
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def rate_from_R(t, Rs, baseline_s=0.20, mask=None):
    """|angular rate| over a FIXED TIME BASELINE, stamped at the MIDPOINT.

    Fixed baseline, not consecutive samples: differentiating orientation at
    100+ Hz turns 0.05 deg of noise into 7 deg/s (check_g2.py, windowed_rate).
    Midpoint, not the interval end: end-stamping biases the lag by +dt/2.
    """
    t = np.asarray(t, float)
    out_t, out_w = [], []
    j = 0
    for i in range(len(t)):
        while j < i and t[i] - t[j] > baseline_s:
            j += 1
        if j == i:
            continue
        if mask is not None and not mask[i - 1:i + 1].all():
            continue
        if mask is not None and not mask[j:i + 1].all():
            continue
        dR = Rs[j].T @ Rs[i]
        v = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                            dR[1, 0] - dR[0, 1]])
        ang = np.arctan2(np.linalg.norm(v), (np.trace(dR) - 1.0) / 2.0)
        out_t.append(0.5 * (t[i] + t[j]))
        out_w.append(ang / (t[i] - t[j]))
    return np.array(out_t), np.array(out_w)


def lag(t_a, v_a, t_b, v_b, lo, hi, span=1.0):
    """Lag of `a` behind `b`, seconds. POSITIVE = a is late."""
    grid = np.arange(max(lo, t_a[0], t_b[0]) + span,
                     min(hi, t_a[-1], t_b[-1]) - span, DT)
    if len(grid) < 200:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("--mocap", default=None)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--span", type=float, default=1.0)
    ap.add_argument("--baseline", type=float, default=0.20)
    a = ap.parse_args()

    f = read_bag(a.flight_bag, [T_ATT, T_VO, T_VOS, T_ALT])

    t_a = np.array([stamp(x) for x in f[T_ATT]])
    rpy = np.array([[x.roll, x.pitch, x.yaw] for x in f[T_ATT]])
    if np.abs(rpy).max() > 2 * np.pi:
        rpy = np.radians(rpy)
    R_a = [rpy_to_R(*r) for r in rpy]
    tw_a, w_a = rate_from_R(t_a, R_a, a.baseline)

    t_vo = np.array([stamp(x) for x in f[T_VO]])
    q_vo = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                      x.pose.orientation.z, x.pose.orientation.w]
                     for x in f[T_VO]])
    p_vo = np.array([[x.pose.position.x, x.pose.position.y,
                      x.pose.position.z] for x in f[T_VO]])
    R_vo = [quat_to_R(q) for q in q_vo]

    ok = np.ones(len(t_vo), bool)
    if T_VOS in f and len(f[T_VOS]):
        t_s = np.array([stamp(x) for x in f[T_VOS]])
        e = np.array([float(getattr(x, "vo_epoch", 0)) for x in f[T_VOS]])
        pv = np.array([bool(getattr(x, "pose_valid", True)) for x in f[T_VOS]])
        ep = np.interp(t_vo, t_s, e).round()
        ok &= np.interp(t_vo, t_s, pv.astype(float)) > 0.5
        ok &= np.concatenate([[True], np.diff(ep) == 0])
        print(f"vo/status: {int(e.max()-e.min())} epochs, "
              f"pose_valid {pv.mean():.1%}, mask keeps {ok.mean():.1%}")
    tw_v, w_v = rate_from_R(t_vo, R_vo, a.baseline, mask=ok)
    print(f"vo {len(t_vo)} poses at {len(t_vo)/(t_vo[-1]-t_vo[0]):.1f} Hz, "
          f"{len(tw_v)} usable rate samples")

    # Airborne window. Prefer mocap z: it has a true ground datum. The
    # altitude fallback uses the 5th percentile as the floor, not the head of
    # the record -- F6c does not start on the ground, and a median of the
    # first 5% of samples then sits ABOVE the flight, inverting the window.
    moc = None
    if a.mocap:
        moc = read_bag(a.mocap, [T_MOCAP])[T_MOCAP]
        t_m0 = np.array([stamp(x) for x in moc])
        z_m0 = np.array([x.pose.position.z for x in moc])
        floor = np.percentile(z_m0, 5)
        airb, tref = z_m0 > floor + 0.20, t_m0
    else:
        tref = np.array([stamp(x) for x in f[T_ALT]])
        z = np.array([float(x.altitude) for x in f[T_ALT]])
        airb = z > np.percentile(z, 5) + 0.20
    if not airb.any():
        sys.exit("could not find an airborne window")
    lo = tref[np.argmax(airb)] + 2.0
    hi = tref[len(airb) - 1 - np.argmax(airb[::-1])] - 2.0
    if hi <= lo:
        sys.exit(f"airborne window is empty ({hi-lo:.1f} s)")
    print(f"airborne window {hi - lo:.1f} s\n")

    print("MOCAP-FREE:  VO |w|  vs  DJI attitude |w|   (= VO - attitude)")
    d, pk, sd = lag(tw_v, w_v, tw_a, w_a, lo, hi, a.span)
    print(f"   whole flight   {d:+.4f} s   peak {pk:.4f}  margin {pk-sd:.4f}")

    print(f"\n   per segment ({a.segments}):")
    edges = np.linspace(lo, hi, a.segments + 1)
    ds = []
    for i in range(a.segments):
        di, pi, si = lag(tw_v, w_v, tw_a, w_a, edges[i], edges[i+1], a.span)
        # A segment with little rotation has nothing for the correlation to
        # lock onto. Its "delay" is the argmax of noise, and including it in
        # the spread manufactures drift that is not there.
        good = np.isfinite(di) and pi > PEAK_MIN and (pi - si) > MARGIN_MIN
        ds.append(di if good else np.nan)
        txt = f"{di:+.4f}" if np.isfinite(di) else "  flat"
        print(f"     {edges[i]-lo:6.1f}-{edges[i+1]-lo:6.1f} s   {txt}   "
              f"peak {pi:.3f}  margin {pi-si:.3f}"
              f"{'' if good else '   <- rejected, no usable peak'}")
    ds = np.array([x for x in ds if np.isfinite(x)])
    print(f"   {len(ds)} of {a.segments} segments usable")
    if len(ds) > 1:
        print(f"   spread {(ds.max()-ds.min())*1e3:.0f} ms   "
              f"sd {ds.std()*1e3:.0f} ms   mean {ds.mean():+.4f} s")
        print("   -> " + ("CONSTANT within this flight"
                          if ds.std() < 0.030 else
                          "DRIFTING within this flight"))

    if moc is not None:
        mo = moc
        t_m = np.array([stamp(x) for x in mo])
        p_m = np.array([[x.pose.position.x, x.pose.position.y,
                         x.pose.position.z] for x in mo])
        q_m = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                         x.pose.orientation.z, x.pose.orientation.w]
                        for x in mo])
        keep = np.concatenate([[True],
                               (np.abs(np.diff(p_m, axis=0)).sum(1) > 0)])
        t_m, q_m = t_m[keep], q_m[keep]
        q_m = q_m * np.array([-1., -1., -1., 1.])       # conjugate
        R_m = [quat_to_R(q) for q in q_m]
        tw_m, w_m = rate_from_R(t_m, R_m, a.baseline)
        print("\nCROSS-CHECK vs mocap |w|")
        for lbl, tt, vv in (("VO", tw_v, w_v), ("attitude", tw_a, w_a)):
            di, pi, si = lag(tt, vv, tw_m, w_m, lo, hi, a.span)
            print(f"   {lbl:<10}{di:+.4f} s   peak {pi:.4f}  "
                  f"margin {pi-si:.4f}")


if __name__ == "__main__":
    main()