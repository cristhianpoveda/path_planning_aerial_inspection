#!/usr/bin/env python3
"""analyse_shadow.py -- would flying on the EKF have worked?

Compares two controllers that saw the SAME setpoint but different feedback:

    /drone_1/command/vel          real, fed by mocap  -> flew the aircraft
    /drone_1/shadow/command/vel   shadow, fed by the EKF -> published nowhere

    python3 analyse_shadow.py --bag step13 --kp-real 0.6 --kp-shadow 0.3
"""
import argparse
import math

import numpy as np


def read_bag(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, raw, t_recv = reader.read_next()
        if topic in out:
            out[topic].append((t_recv * 1e-9,
                               deserialize_message(raw, get_message(types[topic]))))
    return out


def twist_xy(rows):
    t = np.array([r[0] for r in rows])
    v = np.array([[r[1].twist.linear.x, r[1].twist.linear.y,
                   r[1].twist.linear.z] for r in rows])
    return t, v


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def circ_mean_sd(deg):
    r = np.radians(np.asarray(deg))
    c, s = np.cos(r).mean(), np.sin(r).mean()
    mean = math.degrees(math.atan2(s, c))
    R = math.hypot(c, s)
    sd = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12)))))
    return mean, sd


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--ns', default='/drone_1')
    ap.add_argument('--kp-real', type=float, default=0.6)
    ap.add_argument('--kp-shadow', type=float, default=0.3)
    ap.add_argument('--v-max', type=float, default=0.5,
                    help='saturated samples are excluded; the implied error '
                         'is not recoverable there')
    ap.add_argument('--min-cmd', type=float, default=0.03,
                    help='ignore samples where the real controller was '
                         'essentially idle')
    ap.add_argument('--grid', type=float, default=0.05)
    a = ap.parse_args()

    n = a.ns.rstrip('/')
    T = {'real': f'{n}/command/vel', 'shadow': f'{n}/shadow/command/vel',
         'en': f'{n}/controller/enable', 'st': f'{n}/localisation/status',
         'vo': f'{n}/vo/status', 'mocap': f'{n}/mocap/pose'}
    print(f"reading {a.bag}")
    d = read_bag(a.bag, list(T.values()))
    for k, v in T.items():
        if not d[v]:
            print(f"  MISSING {v}")

    tr, vr = twist_xy(d[T['real']])
    ts, vs = twist_xy(d[T['shadow']])
    print(f"  real   {len(tr)} msgs over {tr[-1] - tr[0]:.0f} s")
    print(f"  shadow {len(ts)} msgs over {ts[-1] - ts[0]:.0f} s")

    # -------------------------------------------------- armed window only
    en = [(t, m.data) for t, m in d[T['en']]]
    print(f"\n  enable events: {[(round(t - tr[0], 1), v) for t, v in en]}")
    t_on = next((t for t, v in en if v), None)
    t_off = next((t for t, v in en if not v), tr[-1])
    if t_on is None:
        print("  never armed; nothing to compare")
        return
    print(f"  armed window: {t_on - tr[0]:.0f} to {t_off - tr[0]:.0f} s "
          f"({t_off - t_on:.0f} s)")

    grid = np.arange(max(t_on, ts[0]), min(t_off, ts[-1]), a.grid)
    R = np.column_stack([np.interp(grid, tr, vr[:, k]) for k in range(3)])
    S = np.column_stack([np.interp(grid, ts, vs[:, k]) for k in range(3)])

    mr = np.linalg.norm(R[:, :2], axis=1)
    ms = np.linalg.norm(S[:, :2], axis=1)

    sat = (mr > 0.98 * a.v_max) | (ms > 0.98 * a.v_max)
    idle = mr < a.min_cmd
    use = ~sat & ~idle
    print(f"  {len(grid)} samples, {sat.sum()} saturated, {idle.sum()} idle, "
          f"{use.sum()} usable")
    if use.sum() < 50:
        print("  too few usable samples")
        return

    # ============================================== 1. COMMAND DIFFERENCE
    section("1. COMMANDED VELOCITY DIFFERENCE  (what the aircraft would have done)")
    dv = np.linalg.norm((S - R)[:, :2], axis=1)
    print(f"  |shadow - real| horizontal, m/s:")
    print(f"    median {np.median(dv[use]):.3f}, "
          f"p90 {np.percentile(dv[use], 90):.3f}, max {dv[use].max():.3f}")
    print(f"  real command magnitude: median {np.median(mr[use]):.3f} m/s")
    print(f"  relative difference: "
          f"{np.median(dv[use]) / max(np.median(mr[use]), 1e-9):.2f}x")

    dz = np.abs(S[:, 2] - R[:, 2])
    print(f"\n  vertical difference, m/s: median {np.median(dz[use]):.3f}, "
          f"p90 {np.percentile(dz[use], 90):.3f}")
    print("  Vertical is observed by the altitude update, so it should be the")
    print("  best-behaved axis. If it is not, the problem is not just scale.")

    # ================================================ 2. DIRECTION AGREEMENT
    section("2. DIRECTION AGREEMENT  (the stability question)")
    ang = np.degrees(np.arctan2(S[:, 1], S[:, 0])
                     - np.arctan2(R[:, 1], R[:, 0]))
    ang = wrap180(ang)
    au = ang[use]
    mean, sd = circ_mean_sd(au)
    over90 = float((np.abs(au) > 90.0).mean())
    print(f"  angle(shadow, real): mean {mean:+.1f} deg, sd {sd:.1f} deg")
    print(f"  |angle| > 90 deg: {100 * over90:.1f} % of usable samples")
    print("\n  distribution:")
    hist, edges = np.histogram(au, bins=12, range=(-180, 180))
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(50 * h / max(hist.max(), 1))
        print(f"    {lo:+5.0f}..{hi:+5.0f}  {h:>6}  {bar}")

    print()
    if over90 < 0.05:
        print("  -> The shadow command almost always reduces the error.")
        print("     Closing the loop on the EKF would converge, more slowly")
        print("     and less directly than on mocap, but it would converge.")
    elif over90 < 0.20:
        print("  -> Mostly convergent, with excursions that drive the wrong")
        print("     way. Flyable at low gain with a safety pilot; not")
        print("     something to leave unattended.")
    else:
        print("  -> The shadow command drives AWAY from the setpoint a large")
        print("     fraction of the time. Closing the loop on this estimate")
        print("     would not converge.")

    # ============================================ 3. IMPLIED POSITION ERROR
    section("3. IMPLIED POSITION ERROR  (command / kp, in metres)")
    er = R[:, :2] / a.kp_real
    es = S[:, :2] / a.kp_shadow
    de = np.linalg.norm(es - er, axis=1)
    print(f"  real believed error:   median "
          f"{np.median(np.linalg.norm(er[use], axis=1)):.3f} m")
    print(f"  shadow believed error: median "
          f"{np.median(np.linalg.norm(es[use], axis=1)):.3f} m")
    print(f"  disagreement: median {np.median(de[use]):.3f} m, "
          f"p90 {np.percentile(de[use], 90):.3f}, max {de[use].max():.3f}")
    print("\n  This is the number seen in flight as a discrepancy between the")
    print("  two controllers. It is the EKF's position error projected onto")
    print("  what the controller would have acted on.")

    # ================================================= 4. BY EKF HEALTH
    if d[T['st']]:
        section("4. SPLIT BY EKF HEALTH")
        tst = np.array([r[0] for r in d[T['st']]])
        deg = np.array([1.0 if r[1].degraded else 0.0 for r in d[T['st']]])
        dg = np.interp(grid, tst, deg) > 0.5
        for label, m in (("state OK", use & ~dg), ("DEGRADED", use & dg)):
            if m.sum() < 30:
                print(f"  {label:<12} n={m.sum():>6}  too few")
                continue
            mn, s_ = circ_mean_sd(ang[m])
            o90 = float((np.abs(ang[m]) > 90.0).mean())
            print(f"  {label:<12} n={m.sum():>6}  "
                  f"angle {mn:+7.1f} +- {s_:5.1f}  "
                  f">90deg {100 * o90:5.1f} %  "
                  f"err diff {np.median(de[m]):.3f} m")
        print("\n  If OK is much better than DEGRADED, the health flag is")
        print("  doing its job and gating on it makes the loop safe.")

    # ================================================= 5. BY EPOCH
    if d[T['vo']]:
        section("5. SPLIT BY vo_epoch")
        tv = np.array([r[0] for r in d[T['vo']]])
        ev = np.array([r[1].vo_epoch for r in d[T['vo']]])
        eg = np.interp(grid, tv, ev).round().astype(int)
        print(f"  {'epoch':>6}{'n':>7}{'angle':>9}{'sd':>7}"
              f"{'>90deg %':>10}{'err diff m':>12}")
        for e_ in sorted(set(eg[use])):
            m = use & (eg == e_)
            if m.sum() < 30:
                continue
            mn, s_ = circ_mean_sd(ang[m])
            o90 = float((np.abs(ang[m]) > 90.0).mean())
            print(f"  {e_:>6}{m.sum():>7}{mn:>9.1f}{s_:>7.1f}"
                  f"{100 * o90:>10.1f}{np.median(de[m]):>12.3f}")

    # ================================================= 6. TIME COURSE
    section("6. TIME COURSE  (30 s blocks)")
    print(f"  {'t rel':>8}{'n':>7}{'angle':>9}{'>90deg %':>10}"
          f"{'err diff m':>12}{'degraded %':>12}")
    edges = np.arange(grid[0], grid[-1], 30.0)
    for lo in edges:
        m = use & (grid >= lo) & (grid < lo + 30.0)
        if m.sum() < 20:
            continue
        mn, _ = circ_mean_sd(ang[m])
        o90 = float((np.abs(ang[m]) > 90.0).mean())
        dgf = 100 * dg[m].mean() if d[T['st']] else float('nan')
        print(f"  {lo - grid[0]:>8.0f}{m.sum():>7}{mn:>9.1f}"
              f"{100 * o90:>10.1f}{np.median(de[m]):>12.3f}{dgf:>12.1f}")


if __name__ == '__main__':
    main()
