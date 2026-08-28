#!/usr/bin/env python3
"""
evaluate_pose.py -- STEP 23: the estimated pose against OptiTrack.

Answers, in order:
  1. is `s` right?  Umeyama with scale FREE, per segment. The fitted scale is
     the residual factor: 1.00 means `s` is correct and any external
     disagreement is in the reference, not the filter.
  2. what is the actual accuracy?  RPE over 1 s and 2 s windows, per segment,
     which is what filter_design.md 7 asks for (p_x, p_y are dead-reckoned, so
     ATE measures elapsed time, not quality -- it is NOT reported).
  3. where does the error live?  RPE is reported three ways: frame-free
     magnitude ratio (no alignment at all), rotation-only, and rotation+scale.
     The differences separate a scale error from a frame error from noise.

Three timeline facts this depends on, all verified in the code:
  * localisation/pose is stamped with the VO stamp AFTER VO_DELAY is
    subtracted (ekf_node._publish, frontend.Scheduler.on_vo), so it sits on
    the ATTITUDE timeline.
  * vo/status is stamped with the RAW camera stamp (slam_node.cpp:
    st.header.stamp = msg->header.stamp), ~VO_DELAY ahead of the pose
    timeline. Epoch boundaries are shifted before use.
  * mocap is on its own clock; the offset is FITTED here rather than assumed,
    by minimising post-alignment RMSE, and reported so it can be checked
    against the independently measured attitude lag.

Segmentation is on vo_epoch: a map rebuild gives the new map an arbitrary
scale, so a trajectory-wide number would average incomparable things.

Run:
    python3 evaluate_pose.py POSE_BAG MOCAP_BAG --flight FLIGHT_BAG \
        [--vo-delay 0.40] [--min-seg 15]
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

T_POSE = "/drone_1/localisation/pose"
T_MOCAP = "/optitrack/rigid_bodies/dji_mini4"
T_VOS = "/drone_1/vo/status"
# optitrack_map -> map, as published by the evaluation launch file
TILT_RPY = (-0.019412, -0.005237, 0.0)


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
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                              np.cos(y), np.sin(y))
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def umeyama(P, Q, with_scale=True):
    """Fit Q ~ c*R@P + t. P, Q are (n,3). Returns c, R, t, rmse."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    S = Y.T @ X / len(P)
    U, D, Vt = np.linalg.svd(S)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0                    # keep it a rotation, not a reflection
    R = U @ W @ Vt
    c = float((D * np.diag(W)).sum() / (X ** 2).sum() * len(P)) if with_scale else 1.0
    t = mq - c * R @ mp
    rmse = float(np.sqrt(np.mean(np.sum((Q - (c * (R @ P.T).T + t)) ** 2, 1))))
    return c, R, t, rmse


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

def yaw_of(q, conjugate=False):
    """Yaw from (x,y,z,w), vectorised. The mocap driver publishes the
    CONJUGATE (check_g2.py rev 2 section E), so conjugate=True for it."""
    x, y, z, w = q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy(), q[:, 3]
    if conjugate:
        x, y, z = -x, -y, -z
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def rpe(P, Q, R, c, dt_grid, win_s):
    """Relative pose error over win_s. P, Q are on a common uniform grid."""
    k = int(round(win_s / dt_grid))
    if k < 1 or k >= len(P):
        return None
    dP = c * (R @ (P[k:] - P[:-k]).T).T
    dQ = Q[k:] - Q[:-k]
    e = np.linalg.norm(dP - dQ, axis=1)
    L = np.linalg.norm(dQ, axis=1)
    m = L > 0.05
    if m.sum() < 5:
        return None
    return dict(rms=float(np.sqrt(np.mean(e[m] ** 2))),
                rel=float(100 * np.median(e[m] / L[m])),
                ratio=float(np.median(np.linalg.norm(dP[m], axis=1) / L[m])),
                n=int(m.sum()))


def fit_offset(t_e, p_e, t_g, p_g, lo, hi, grid, scan):
    """Fit the mocap time offset on ONE segment.

    Fitting globally on the longest segment is wrong: Umeyama with scale free
    trades time offset against scale, so a mis-scaled segment drags the offset
    with it and then corrupts every other segment's numbers.
    """
    best, best_r = 0.0, np.inf
    for d in np.arange(-scan, scan, 0.005):
        g = np.arange(lo + 1, hi - 1, grid)
        if len(g) < 50:
            return float("nan"), np.inf
        P = np.stack([np.interp(g, t_e, p_e[:, i]) for i in range(3)], 1)
        Q = np.stack([np.interp(g + d, t_g, p_g[:, i]) for i in range(3)], 1)
        _, _, _, r = umeyama(P, Q, True)
        if r < best_r:
            best_r, best = r, float(d)
    return best, best_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--flight", default=None, help="bag with vo/status")
    ap.add_argument("--vo-delay", type=float, default=0.40)
    ap.add_argument("--min-seg", type=float, default=15.0)
    ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--scan", type=float, default=0.30)
    a = ap.parse_args()

    pb = read_bag(a.pose_bag, [T_POSE])
    if T_POSE not in pb or len(pb[T_POSE]) < 20:
        sys.exit(f"{T_POSE} missing or too short in {a.pose_bag}")
    t_e = np.array([stamp(x) for x in pb[T_POSE]])
    p_e = np.array([[x.pose.pose.position.x, x.pose.pose.position.y,
                     x.pose.pose.position.z] for x in pb[T_POSE]])

    q_e = np.array([[x.pose.pose.orientation.x, x.pose.pose.orientation.y,
                     x.pose.pose.orientation.z, x.pose.pose.orientation.w]
                    for x in pb[T_POSE]])

    mb = read_bag(a.mocap_bag, [T_MOCAP])[T_MOCAP]
    t_g = np.array([stamp(x) for x in mb])
    p_g = np.array([[x.pose.position.x, x.pose.position.y,
                     x.pose.position.z] for x in mb])
    q_g = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                     x.pose.orientation.z, x.pose.orientation.w] for x in mb])
    keep = np.concatenate([[True], (np.abs(np.diff(p_g, axis=0)).sum(1) > 0)])
    t_g, p_g, q_g = t_g[keep], p_g[keep], q_g[keep]


    print(f"estimate  {len(t_e)} poses, {len(t_e)/(t_e[-1]-t_e[0]):5.1f} Hz, "
          f"{t_e[-1]-t_e[0]:.1f} s")
    print(f"mocap     {len(t_g)} poses ({int((~keep).sum())} dups dropped), "
          f"{len(t_g)/(t_g[-1]-t_g[0]):5.1f} Hz")
    ov = min(t_e[-1], t_g[-1]) - max(t_e[0], t_g[0])
    print(f"overlap   {ov:.1f} s")
    if ov < 20:
        sys.exit("too little overlap")

    # ---- segments -----------------------------------------------------
    bounds = [t_e[0], t_e[-1]]
    if a.flight:
        fs = read_bag(a.flight, [T_VOS])
        if T_VOS in fs and fs[T_VOS]:
            t_s = np.array([stamp(x) for x in fs[T_VOS]])
            ep = np.array([int(getattr(x, "vo_epoch", 0)) for x in fs[T_VOS]])
            # vo/status carries the RAW camera stamp; the pose timeline is
            # VO_DELAY earlier. Shift before using as a boundary.
            ch = t_s[1:][np.diff(ep) != 0] - a.vo_delay
            bounds = sorted(set([t_e[0]] + [c for c in ch
                                            if t_e[0] < c < t_e[-1]] + [t_e[-1]]))
            print(f"vo_epoch: {int(ep.max()-ep.min())} changes -> "
                  f"{len(bounds)-1} segments")
    segs = [(bounds[i], bounds[i+1]) for i in range(len(bounds) - 1)
            if bounds[i+1] - bounds[i] >= a.min_seg]
    print(f"{len(segs)} segments >= {a.min_seg:.0f} s\n")
    if not segs:
        sys.exit("no segment long enough")

    # ---- fit the mocap time offset on the longest segment --------------
    print("Per-segment mocap time offset (fitted independently):")
    offs = {}
    for i, (s0, s1) in enumerate(segs):
        d, r = fit_offset(t_e, p_e, t_g, p_g, s0, s1, a.grid, a.scan)
        offs[i] = d
        rail = np.isfinite(d) and abs(abs(d) - a.scan) < 0.01
        print(f"   seg {i}  {d:+.4f} s   rmse {r:.4f} m"
              + ("   <- AT THE SCAN LIMIT: not a fit. `s` is probably not "
                 "constant within this segment." if rail else ""))
    good = np.array([v for v in offs.values() if np.isfinite(v)])
    print(f"   median {np.median(good):+.4f} s, spread "
          f"{(good.max()-good.min())*1e3:.0f} ms")
    print(f"   expected ~ +0.0345 s (the attitude lag). Segments that "
          f"disagree with\n   the median are mis-scaled, not mis-timed -- "
          f"check their `scale` column.\n")

    # ---- per segment ---------------------------------------------------
    R_tilt = rpy_to_R(*TILT_RPY)
    hdr = (f"{'seg':>4}{'dur':>7}{'n':>6}{'scale':>8}{'yaw':>8}{'rmse':>8}"
           f"{'RPE1s%':>9}{'RPE2s%':>9}{'ratio1s':>9}{'pub1s%':>9}"
           f"{'toff':>8}{'ro1s%':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for i, (s0, s1) in enumerate(segs):
        g = np.arange(s0 + 1, s1 - 1, a.grid)
        P = np.stack([np.interp(g, t_e, p_e[:, i]) for i in range(3)], 1)
        d_seg = offs[i] if np.isfinite(offs[i]) else float(np.median(good))
        Q = np.stack([np.interp(g + d_seg, t_g, p_g[:, i]) for i in range(3)], 1)
        c, R, _, rm = umeyama(P, Q, True)
        _, R1, _, _ = umeyama(P, Q, False)
        r1 = rpe(P, Q, R, c, a.grid, 1.0)
        r2 = rpe(P, Q, R, c, a.grid, 2.0)
        rf = rpe(P, Q, np.eye(3), 1.0, a.grid, 1.0)      # frame-free ratio
        rp = rpe(P, Q, R_tilt, 1.0, a.grid, 1.0)         # published transform
        # Frame fitted, scale NOT corrected: what the LIVE system produces.
        # RPE1s% has the scale error removed by Umeyama, which the running
        # filter cannot do. The gap between the two columns IS the cost of
        # the scale error.
        ro = rpe(P, Q, R1, 1.0, a.grid, 1.0)
        ye = np.unwrap(np.interp(g, t_e, np.unwrap(yaw_of(q_e)))
                       - np.interp(g + d_seg, t_g,
                                   np.unwrap(yaw_of(q_g, conjugate=True))))
        ye = np.degrees(ye - np.median(ye))     # constant offset is a frame
        drift = float(np.polyfit(g - g[0], ye, 1)[0] * (g[-1] - g[0]))
        k1 = int(round(1.0 / a.grid))
        dye = ye[k1:] - ye[:-k1]                 # yaw change WITHIN 1 s
        print(f"     yaw error: sd {ye.std():5.2f} deg, "
              f"p2p {ye.max()-ye.min():6.2f}, drift {drift:+6.2f} deg "
              f"over {g[-1]-g[0]:.0f} s | within 1 s: sd {dye.std():5.2f} deg "
              f"-> predicts RPE {200*np.sin(np.radians(dye.std())/2):.1f} %")
        if r1 is None:
            continue
        rows.append((c, r1["rel"], r2["rel"] if r2 else np.nan,
                     ro["rel"] if ro else np.nan))
        print(f"{i:>4}{s1-s0:7.1f}{len(g):6d}{c:8.4f}{rot_angle(R1):8.2f}"
              f"{rm:8.3f}{r1['rel']:9.2f}"
              f"{(r2['rel'] if r2 else np.nan):9.2f}"
              f"{(rf['ratio'] if rf else np.nan):9.3f}"
              f"{(rp['rel'] if rp else np.nan):9.2f}"
              f"{d_seg:8.3f}"
              f"{(ro['rel'] if ro else np.nan):9.2f}")

    if not rows:
        return
    cs = np.array([r[0] for r in rows])
    print(f"\nSCALE  fitted residual factor {cs.mean():.4f} +- {cs.std():.4f}")
    print(f"   the filter's `s` should be multiplied by this to match truth")
    print(f"   1.00 -> `s` is correct;  1.05 -> `s` is 5% LOW")
    rel = np.array([r[1] for r in rows])
    ope = np.array([r[3] for r in rows])
    print(f"\nRPE 1 s   scale-corrected  {rel.mean():6.2f} % mean "
          f"(range {rel.min():.2f}-{rel.max():.2f})")
    print(f"          AS THE SYSTEM RUNS {np.nanmean(ope):6.2f} % mean "
          f"(range {np.nanmin(ope):.2f}-{np.nanmax(ope):.2f})")
    print("   the gap between those two lines is the cost of the scale error")

if __name__ == "__main__":
    main()