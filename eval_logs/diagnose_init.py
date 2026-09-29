#!/usr/bin/env python3
"""diagnose_init.py -- why did the EKF not initialise?

    Airborne      altitude > INIT_ALT_MIN
    VO tracking   pose_valid
    Transit       >= INIT_PATH_M of path at >= V_LOW
    Gimbal        brackets the VO stamp

Also reports VO epoch history, tracking-state timeline, and the DJI velocity
profile, so a refusal can be attributed to speed, to VO, or to neither.

    python3 diagnose_init.py --bag step7
    python3 diagnose_init.py --bag step7 --mocap-bag step7        # same bag ok
"""
import argparse
import math
from collections import Counter

import numpy as np

INIT_ALT_MIN = 0.30      # m,   ekf params
INIT_PATH_M = 3.0        # m,   ekf params
V_LOW = 0.40             # m/s, ekf params
K_VEL = 0.87             # DJI velocity reads low by this factor


def read_bag(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    print(f"  topics in {path}:")
    for name in sorted(types):
        print(f"    {name:<52} {types[name]}")
    print()

    out = {t: [] for t in topics}
    while reader.has_next():
        topic, raw, t_recv = reader.read_next()
        if topic not in out:
            continue
        out[topic].append((t_recv * 1e-9,
                           deserialize_message(raw, get_message(types[topic]))))
    return out, types


def sec(rows):
    return np.array([r[0] for r in rows]) if rows else np.array([])


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--ns', default='/drone_1')
    ap.add_argument('--v-low', type=float, default=V_LOW)
    ap.add_argument('--init-path', type=float, default=INIT_PATH_M)
    ap.add_argument('--k-vel', type=float, default=K_VEL)
    ap.add_argument('--image-topic', default='camera/image/compressed',
                    help='node_graph.mmd is right, node_interfaces.md has drifted')
    a = ap.parse_args()

    n = a.ns.rstrip('/')
    T = {
        'vo_pose': f'{n}/vo/pose',
        'vo_status': f'{n}/vo/status',
        'speed': f'{n}/speed_vector',
        'alt': f'{n}/relative_altitude',
        'att': f'{n}/attitude',
        'gimbal': f'{n}/gimbal_joint_attitude',
        'image': a.image_topic if a.image_topic.startswith('/')
                 else f'{n}/{a.image_topic}',
        'loc_pose': f'{n}/localisation/pose',
        'loc_status': f'{n}/localisation/status',
        'cmd': f'{n}/command/vel',
        'mocap_pose': f'{n}/mocap/pose',
    }
    print(f"reading {a.bag}")
    d, types = read_bag(a.bag, list(T.values()))

    def rows(k):
        return d.get(T[k], [])

    # ---------------------------------------------------------- inventory
    section("TOPIC INVENTORY")
    t0 = min([sec(rows(k))[0] for k in T if len(rows(k))], default=0.0)
    t1 = max([sec(rows(k))[-1] for k in T if len(rows(k))], default=0.0)
    print(f"  bag span {t1 - t0:.1f} s\n")
    print(f"  {'topic':<28}{'msgs':>8}{'Hz':>8}   span")
    for k in T:
        r = rows(k)
        if not r:
            print(f"  {k:<28}{0:>8}{'--':>8}   MISSING")
            continue
        t = sec(r)
        span = t[-1] - t[0]
        hz = (len(t) - 1) / span if span > 0 else 0.0
        print(f"  {k:<28}{len(t):>8}{hz:>8.1f}   "
              f"{t[0] - t0:6.1f} -> {t[-1] - t0:6.1f} s")

    # ------------------------------------------------------ 3.1 airborne
    section("REQUIREMENT 1/4 -- AIRBORNE  (altitude > INIT_ALT_MIN)")
    if not rows('alt'):
        print("  relative_altitude MISSING -- cannot check")
    else:
        t = sec(rows('alt'))
        z = np.array([m.altitude for _, m in rows('alt')])
        ok = z > INIT_ALT_MIN
        print(f"  altitude: min {z.min():.3f}  max {z.max():.3f} m")
        print(f"  above {INIT_ALT_MIN} m for {ok.sum()} of {len(z)} samples "
              f"({100.0 * ok.mean():.0f} %)")
        if ok.any():
            print(f"  first airborne at t = {t[np.argmax(ok)] - t0:.1f} s")
            print("  PASS")
        else:
            print("  FAIL -- never airborne by the altitude key")

    # ---------------------------------------------------- 3.1 VO tracking
    section("REQUIREMENT 2/4 -- VO TRACKING  (pose_valid)")
    vp = rows('vo_pose')
    vs = rows('vo_status')
    if not vp and not vs:
        print("  vo/pose AND vo/status both MISSING from the bag.")
        print("  Either slam_node published on a different domain or with an")
        print("  incompatible QoS, or the topic names differ. The console log")
        print("  showed tracking OK for ~148 s, so VO itself was running.")
    if vp:
        t = sec(vp)
        print(f"  vo/pose: {len(t)} msgs, "
              f"{t[0] - t0:.1f} -> {t[-1] - t0:.1f} s, "
              f"{(len(t) - 1) / max(t[-1] - t[0], 1e-9):.1f} Hz")
        gaps = np.diff(t)
        big = gaps[gaps > 0.5]
        print(f"  gaps > 0.5 s: {len(big)}"
              + (f"  largest {big.max():.1f} s" if len(big) else ""))
        p = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z]
                      for _, m in vp])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        print(f"  VO path length (unscaled): {seg.sum():.2f}")
    if vs:
        t = sec(vs)
        states, epochs, valid = [], [], []
        for _, m in vs:
            states.append(getattr(m, 'state', getattr(m, 'data', '?')))
            epochs.append(getattr(m, 'vo_epoch', -1))
            valid.append(bool(getattr(m, 'pose_valid', False)))
        valid = np.array(valid)
        print(f"\n  vo/status: {len(t)} msgs")
        print(f"  state histogram: {dict(Counter(states))}")
        if hasattr(vs[0][1], 'pose_valid'):
            print(f"  pose_valid: {valid.sum()} of {len(valid)} "
                  f"({100.0 * valid.mean():.0f} %)")
        ep = np.array(epochs)
        if (ep >= 0).any():
            ch = np.where(np.diff(ep) != 0)[0]
            print(f"  vo_epoch: {ep.min()} -> {ep.max()}, "
                  f"{len(ch)} changes")
            for i in ch:
                print(f"    t={t[i + 1] - t0:7.1f} s  epoch {ep[i]} -> {ep[i + 1]}")
            span = t[-1] - t[0]
            if len(ch):
                print(f"  rebuild rate: one per {span / len(ch):.0f} s "
                      f"({'ABORT-level, see 4.1' if span / len(ch) < 30 else 'acceptable'})")

    # ------------------------------------------------------- 3.1 transit
    section("REQUIREMENT 3/4 -- TRANSIT  "
            f"(>= {a.init_path} m of path at >= {a.v_low} m/s)")
    sp = rows('speed')
    if not sp:
        print("  speed_vector MISSING -- cannot check")
    else:
        t = sec(sp)
        v = np.array([[m.vector.x, m.vector.y, m.vector.z] for _, m in sp])
        # The node measures path from the DJI velocity, corrected by K_VEL.
        spd = np.linalg.norm(v, axis=1) / a.k_vel
        dt = np.diff(t, prepend=t[0])
        fast = spd >= a.v_low

        print(f"  speed (K_VEL-corrected): max {spd.max():.3f}  "
              f"mean {spd.mean():.3f} m/s")
        print(f"  samples >= {a.v_low} m/s: {fast.sum()} of {len(spd)} "
              f"({100.0 * fast.mean():.1f} %)")
        print(f"  time above {a.v_low} m/s: {dt[fast].sum():.1f} s")

        path_fast = float((spd * dt)[fast].sum())
        path_all = float((spd * dt).sum())
        print(f"\n  PATH AT OR ABOVE {a.v_low} m/s: {path_fast:.2f} m")
        print(f"  path at any speed:          {path_all:.2f} m")
        print(f"  requirement:                {a.init_path:.2f} m")
        print("  " + ("PASS" if path_fast >= a.init_path
                      else f"FAIL -- short by {a.init_path - path_fast:.2f} m"))

        # Longest continuous run above V_LOW, which is what a single transit
        # actually delivers.
        runs, cur = [], 0.0
        for f, s_, dt_ in zip(fast, spd, dt):
            if f:
                cur += s_ * dt_
            elif cur > 0:
                runs.append(cur)
                cur = 0.0
        if cur > 0:
            runs.append(cur)
        if runs:
            runs.sort(reverse=True)
            print(f"\n  continuous runs above {a.v_low} m/s: "
                  f"{len(runs)}, longest {runs[0]:.2f} m")
            print(f"  top 5: {[round(x, 2) for x in runs[:5]]}")

        # Speed histogram, to show where the flight actually sat.
        print("\n  speed distribution:")
        edges = [0, 0.05, 0.15, 0.30, 0.40, 0.60, 1.00, 99]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (spd >= lo) & (spd < hi)
            bar = "#" * int(60 * m.mean())
            print(f"    {lo:4.2f}-{hi:5.2f} m/s  {100 * m.mean():5.1f} %  {bar}")

    # -------------------------------------------------------- 3.1 gimbal
    section("REQUIREMENT 4/4 -- GIMBAL BRACKETS THE VO STAMP")
    g = rows('gimbal')
    if not g:
        print("  gimbal_joint_attitude MISSING -- cannot check")
    elif not vp:
        print("  vo/pose missing, so bracketing cannot be evaluated")
    else:
        tg, tv = sec(g), sec(vp)
        print(f"  gimbal {len(tg)} msgs at "
              f"{(len(tg) - 1) / max(tg[-1] - tg[0], 1e-9):.1f} Hz")
        inside = ((tv >= tg[0]) & (tv <= tg[-1])).mean()
        print(f"  VO stamps inside the gimbal span: {100 * inside:.1f} %")
        print("  " + ("PASS" if inside > 0.95 else "CHECK"))

    # ------------------------------------------------------------ outcome
    section("EKF OUTPUT")
    lp, ls = rows('loc_pose'), rows('loc_status')
    if not lp and not ls:
        print("  localisation/pose and localisation/status both MISSING.")
        print("  The EKF published nothing at all, consistent with a refusal")
        print("  to initialise.")
    if ls:
        t = sec(ls)
        st = [m.state for _, m in ls]
        deg = np.array([m.degraded for _, m in ls])
        fl = Counter(f for _, m in ls for f in m.flags)
        print(f"  localisation/status: {len(t)} msgs")
        print(f"  state histogram: {dict(Counter(st))}")
        print(f"  degraded: {100 * deg.mean():.0f} % of samples")
        print(f"  flags: {dict(fl)}")
        sc = np.array([m.scale for _, m in ls])
        ss = np.array([m.sigma_scale for _, m in ls])
        good = sc > 0
        if good.any():
            print(f"  scale: {sc[good].min():.3f} -> {sc[good].max():.3f}, "
                  f"sigma/scale {np.median(ss[good] / sc[good]):.3f}")
    if lp:
        t = sec(lp)
        print(f"  localisation/pose: {len(t)} msgs, "
              f"{t[0] - t0:.1f} -> {t[-1] - t0:.1f} s")

    # -------------------------------------------------------- commanded
    section("WHAT THE CONTROLLER ASKED FOR")
    c = rows('cmd')
    if not c:
        print("  command/vel MISSING")
    else:
        t = sec(c)
        v = np.array([[m.twist.linear.x, m.twist.linear.y, m.twist.linear.z]
                      for _, m in c])
        h = np.linalg.norm(v[:, :2], axis=1)
        dt = np.diff(t, prepend=t[0])
        print(f"  command/vel: {len(t)} msgs at "
              f"{(len(t) - 1) / max(t[-1] - t[0], 1e-9):.1f} Hz")
        print(f"  commanded horizontal speed: max {h.max():.3f} m/s")
        print(f"  time commanded >= {a.v_low} m/s: "
              f"{dt[h >= a.v_low].sum():.1f} s")
        print(f"  commanded path: {float((h * dt).sum()):.2f} m")
        print("\n  If the commanded path is well above the achieved path at")
        print(f"  >= {a.v_low} m/s, the aircraft spent the transit accelerating")
        print("  and decelerating rather than cruising: the steps were too")
        print("  short for the speed cap to be reached.")

    # ------------------------------------------------------------ verdict
    section("VERDICT")
    print("  Compare, in order:")
    print("   1. Was vo/pose in the bag at all? If not, the failure is")
    print("      plumbing (domain or QoS), not VO and not speed.")
    print("   2. Did PATH AT OR ABOVE V_LOW reach INIT_PATH_M? If not, the")
    print("      refusal is correct and the fix is longer or faster transits.")
    print("   3. Was the longest continuous run shorter than INIT_PATH_M?")
    print("      Scale collection needs the path within one tracking epoch.")
    print("   4. How many vo_epoch changes? Each one invalidates s.")


if __name__ == '__main__':
    main()