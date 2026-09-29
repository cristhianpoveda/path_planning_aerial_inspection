#!/usr/bin/env python3
"""analyse_frame.py -- is the EKF position error a FRAME error or an
INCREMENT error?

    python3 analyse_frame.py --bag step10c
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


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def circ_mean_sd(deg):
    r = np.radians(np.asarray(deg))
    c, s = np.cos(r).mean(), np.sin(r).mean()
    mean = math.degrees(math.atan2(s, c))
    R = math.hypot(c, s)
    sd = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12)))))
    return mean, sd


def rot2(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def best_yaw(A, B):
    """Yaw theta minimising || R(theta) A - B ||, rows are 2D vectors."""
    num = np.sum(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0])
    den = np.sum(A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])
    return math.atan2(num, den)


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--ns', default='/drone_1')
    ap.add_argument('--window', type=float, default=1.0)
    ap.add_argument('--grid', type=float, default=0.05)
    ap.add_argument('--min-move', type=float, default=0.05,
                    help='ignore windows with less than this much true motion')
    a = ap.parse_args()

    n = a.ns.rstrip('/')
    T = {'ekf': f'{n}/localisation/pose', 'st': f'{n}/localisation/status',
         'mocap': f'{n}/mocap/pose', 'vo': f'{n}/vo/status',
         'att': f'{n}/attitude'}
    print(f"reading {a.bag}")
    d = read_bag(a.bag, list(T.values()))

    te = np.array([r[0] for r in d[T['ekf']]])
    pe = np.array([[r[1].pose.pose.position.x, r[1].pose.pose.position.y,
                    r[1].pose.pose.position.z] for r in d[T['ekf']]])
    ye = np.degrees([quat_yaw(r[1].pose.pose.orientation) for r in d[T['ekf']]])

    tm = np.array([r[0] for r in d[T['mocap']]])
    pm = np.array([[r[1].pose.pose.position.x, r[1].pose.pose.position.y,
                    r[1].pose.pose.position.z] for r in d[T['mocap']]])
    ym = np.degrees([quat_yaw(r[1].pose.pose.orientation) for r in d[T['mocap']]])

    # =========================================================== 0. HEADING
    section("0. HEADING AT TAKEOFF AND AT EKF INIT")
    print(f"  mocap yaw at bag start : {ym[0]:+7.1f} deg")
    i_init = int(np.argmin(np.abs(tm - te[0])))
    print(f"  mocap yaw at EKF init  : {ym[i_init]:+7.1f} deg")
    print(f"  mocap yaw range        : {ym.min():+7.1f} to {ym.max():+7.1f} deg")
    print("\n  The takeoff heading sets the odom<->mocap rotation for this")
    print("  session. It does NOT change the rigid-body-to-base_link offset,")
    print("  which is a Motive definition. The two are not separable from")
    print("  position and orientation alone, which is why the test below uses")
    print("  only the CONSTANCY of the direction offset.")

    # ======================================================= resample
    t0, t1 = te[0], te[-1]
    grid = np.arange(t0, t1, a.grid)
    E = np.column_stack([np.interp(grid, te, pe[:, k]) for k in range(3)])
    M = np.column_stack([np.interp(grid, tm, pm[:, k]) for k in range(3)])

    k = int(round(a.window / a.grid))
    dE = E[k:, :2] - E[:-k, :2]
    dM = M[k:, :2] - M[:-k, :2]
    tw = grid[:-k]

    mag_m = np.linalg.norm(dM, axis=1)
    sel = mag_m > a.min_move
    dE, dM, tw, mag_m = dE[sel], dM[sel], tw[sel], mag_m[sel]
    mag_e = np.linalg.norm(dE, axis=1)
    print(f"\n  {sel.sum()} windows of {a.window:.0f} s with > "
          f"{100 * a.min_move:.0f} cm of true motion")

    if sel.sum() < 20:
        print("  too few moving windows to analyse")
        return

    # ================================================== 1. DIRECTION OFFSET
    section("1. DIRECTION OFFSET  (EKF displacement vs mocap displacement)")
    ang = np.degrees(np.arctan2(dE[:, 1], dE[:, 0])
                     - np.arctan2(dM[:, 1], dM[:, 0]))
    ang = wrap180(ang)
    mean, sd = circ_mean_sd(ang)
    half = len(ang) // 2
    m1, _ = circ_mean_sd(ang[:half])
    m2, _ = circ_mean_sd(ang[half:])

    print(f"  offset: mean {mean:+.1f} deg, circular sd {sd:.1f} deg")
    print(f"  first half {m1:+.1f}, second half {m2:+.1f}, "
          f"change {wrap180(m2 - m1):+.1f} deg")
    print("\n  distribution:")
    hist, edges = np.histogram(ang, bins=12, range=(-180, 180))
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(50 * h / max(hist.max(), 1))
        print(f"    {lo:+5.0f}..{hi:+5.0f}  {h:>5}  {bar}")

    if sd < 20.0:
        print("\n  -> CONSTANT offset. This is a FRAME error: a single yaw")
        print("     correction would fix it.")
    elif sd < 50.0:
        print("\n  -> partly constant. A frame correction helps but does not")
        print("     account for all of it.")
    else:
        print("\n  -> NOT constant. The increment directions are wrong in a way")
        print("     no fixed frame correction can fix.")

    # ================================================= 2. MAGNITUDE RATIO
    section("2. MAGNITUDE RATIO  (|EKF displacement| / |mocap displacement|)")
    ratio = mag_e / mag_m
    print(f"  median {np.median(ratio):.3f}, "
          f"p10 {np.percentile(ratio, 10):.3f}, "
          f"p90 {np.percentile(ratio, 90):.3f}")
    print(f"  spread p90/p10 = {np.percentile(ratio, 90) / max(np.percentile(ratio, 10), 1e-9):.1f}x")
    r1, r2 = np.median(ratio[:half]), np.median(ratio[half:])
    print(f"  first half {r1:.3f}, second half {r2:.3f}")
    print("\n  A tight ratio near a single value means one scale error.")
    print("  A wide spread means the increments vary in length independently")
    print("  of any scale, which no scalar can fix.")

    # ============================================ 3. ERROR DECOMPOSITION
    section("3. ERROR DECOMPOSITION  (median relative error per window)")

    def rel(pred):
        return np.median(np.linalg.norm(pred - dM, axis=1) / mag_m)

    raw = rel(dE)

    s_g = float(np.sum(dE * dM) / max(np.sum(dE * dE), 1e-12))
    e_s = rel(s_g * dE)

    th_g = best_yaw(dE, dM)
    R = rot2(th_g)
    e_r = rel(dE @ R.T)

    dEs = s_g * dE
    th_sr = best_yaw(dEs, dM)
    e_sr = rel(dEs @ rot2(th_sr).T)

    # per-window rotation: rotate each increment onto the true direction
    unit_m = dM / mag_m[:, None]
    per_rot = unit_m * mag_e[:, None]
    e_pr = rel(per_rot)

    # per-window scale: keep the EKF direction, use the true magnitude
    unit_e = dE / np.maximum(mag_e, 1e-9)[:, None]
    per_sc = unit_e * mag_m[:, None]
    e_ps = rel(per_sc)

    rows = [("raw, as the system runs", raw, ""),
            (f"+ global scale ({s_g:.3f})", e_s, ""),
            (f"+ global rotation ({math.degrees(th_g):+.1f} deg)", e_r,
             "<- frame fix ceiling"),
            (f"+ global scale and rotation", e_sr, ""),
            ("+ PER-WINDOW rotation (direction perfect)", e_pr,
             "residual = magnitude error"),
            ("+ PER-WINDOW scale (magnitude perfect)", e_ps,
             "residual = direction error")]
    print(f"  {'correction':<44}{'err %':>8}   note")
    print("  " + "-" * 74)
    for label, v, note in rows:
        print(f"  {label:<44}{100 * v:>8.1f}   {note}")

    print("\n  Read the last two rows. If 'direction perfect' leaves a small")
    print("  residual, the error is almost entirely direction, i.e. frame.")
    print("  If 'magnitude perfect' leaves a small residual, it is almost")
    print("  entirely scale. If both leave large residuals, the increments")
    print("  disagree with truth in both, and the propagation input is bad.")

    # ================================================= 4. PER EPOCH
    if d[T['vo']]:
        section("4. PER vo_epoch SEGMENT")
        tv = np.array([x[0] for x in d[T['vo']]])
        ev = np.array([x[1].vo_epoch for x in d[T['vo']]])
        eg = np.interp(tw, tv, ev).round().astype(int)
        print(f"  {'epoch':>6}{'n':>6}{'dir mean':>10}{'dir sd':>9}"
              f"{'ratio':>8}{'raw %':>8}{'+rot %':>8}")
        for e_ in sorted(set(eg)):
            i = eg == e_
            if i.sum() < 15:
                continue
            m_, s_ = circ_mean_sd(ang[i])
            th = best_yaw(dE[i], dM[i])
            er = np.median(np.linalg.norm(dE[i] @ rot2(th).T - dM[i], axis=1)
                           / mag_m[i])
            rw = np.median(np.linalg.norm(dE[i] - dM[i], axis=1) / mag_m[i])
            print(f"  {e_:>6}{i.sum():>6}{m_:>10.1f}{s_:>9.1f}"
                  f"{np.median(ratio[i]):>8.3f}{100 * rw:>8.1f}{100 * er:>8.1f}")

    # ================================================= 5. SCALE STATE
    if d[T['st']]:
        section("5. FILTER SCALE STATE vs EMPIRICAL RATIO")
        ts = np.array([r[0] for r in d[T['st']]])
        ss = np.array([r[1].scale for r in d[T['st']]])
        sg = np.array([r[1].sigma_scale for r in d[T['st']]])
        print(f"  {'t rel':>8}{'s':>10}{'sigma_s':>10}{'sigma/s':>10}"
              f"{'emp ratio':>11}")
        for frac in np.linspace(0, 1, 12):
            tq = ts[0] + frac * (ts[-1] - ts[0])
            i = int(np.argmin(np.abs(ts - tq)))
            w = np.abs(tw - tq) < 3.0
            emp = np.median(ratio[w]) if w.sum() > 3 else float('nan')
            print(f"  {ts[i] - ts[0]:>8.0f}{ss[i]:>10.3f}{sg[i]:>10.3f}"
                  f"{sg[i] / max(ss[i], 1e-9):>10.3f}{emp:>11.3f}")
        print("\n  The empirical ratio is what `s` would have to be for the")
        print("  increments to come out the right length. If `s` is chasing it")
        print("  and never settling, the true scale is not constant, which")
        print("  means the VO increments are not a consistent scaling of truth.")


if __name__ == '__main__':
    main()