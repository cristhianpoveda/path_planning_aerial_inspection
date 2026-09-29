#!/usr/bin/env python3
"""analyse_ekf_vs_mocap.py -- what has to change before closing the loop on the EKF.

"""
import argparse
import math

import numpy as np

TELEMETRY_LAG_S = 0.0732
CMD_LATENCY_S = 0.28          # controller_plant_model.md 8
VO_DELAY_S = 0.40             # filter_design.md 4


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


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def circ_stats(deg):
    """Mean and sd of an angle series, wrap-safe."""
    r = np.radians(deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    mean = math.degrees(math.atan2(s, c))
    R = math.hypot(c, s)
    sd = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12)))))
    return mean, sd


def interp_series(t_src, y_src, t_q):
    if np.ndim(y_src) == 1:
        return np.interp(t_q, t_src, y_src)
    return np.column_stack([np.interp(t_q, t_src, y_src[:, k])
                            for k in range(y_src.shape[1])])


def fit_yaw_scale_2d(P, Q):
    """Least-squares s, theta, t minimising || s R(theta) P + t - Q ||, in 2D."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    num = np.sum(Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])
    den = np.sum(Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])
    th = math.atan2(num, den)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    PR = Pc @ R.T
    s = float(np.sum(PR * Qc) / max(np.sum(PR * PR), 1e-12))
    t = Q.mean(axis=0) - s * (R @ P.mean(axis=0))
    return s, th, R, t


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--ns', default='/drone_1')
    ap.add_argument('--rpe-window', type=float, default=1.0)
    ap.add_argument('--grid', type=float, default=0.05)
    a = ap.parse_args()

    n = a.ns.rstrip('/')
    T = {'ekf': f'{n}/localisation/pose',
         'st': f'{n}/localisation/status',
         'mocap': f'{n}/mocap/pose',
         'att': f'{n}/attitude',
         'vo': f'{n}/vo/status',
         'cmd': f'{n}/command/vel'}

    print(f"reading {a.bag}")
    d = read_bag(a.bag, list(T.values()))
    for k, v in T.items():
        if not d[v]:
            print(f"  MISSING {v}")

    # ---------------------------------------------------------------- series
    te = np.array([r[0] for r in d[T['ekf']]])
    pe = np.array([[r[1].pose.pose.position.x, r[1].pose.pose.position.y,
                    r[1].pose.pose.position.z] for r in d[T['ekf']]])
    ye = np.array([math.degrees(quat_yaw(r[1].pose.pose.orientation))
                   for r in d[T['ekf']]])

    tm = np.array([r[0] for r in d[T['mocap']]])
    pm = np.array([[r[1].pose.pose.position.x, r[1].pose.pose.position.y,
                    r[1].pose.pose.position.z] for r in d[T['mocap']]])
    ym = np.array([math.degrees(quat_yaw(r[1].pose.pose.orientation))
                   for r in d[T['mocap']]])

    ta = np.array([r[0] for r in d[T['att']]])
    ya = np.array([r[1].yaw for r in d[T['att']]])      # degrees

    t0, t1 = te[0], te[-1]
    print(f"\n  EKF publishing from {t0 - tm[0]:.1f} s into the bag, "
          f"for {t1 - t0:.1f} s")

    grid = np.arange(t0, t1, a.grid)
    Pe = interp_series(te, pe, grid)
    Pm = interp_series(tm, pm, grid)
    Ye = np.interp(grid, te, unwrap_deg(ye))
    Ym = np.interp(grid, tm, unwrap_deg(ym))
    Ya = np.interp(grid, ta, unwrap_deg(ya))

    # ============================================================= 1. HEADING
    section("1. HEADING DATUMS")

    pairs = [("EKF yaw  - DJI attitude", Ye - Ya),
             ("mocap    - DJI attitude", Ym - Ya),
             ("EKF yaw  - mocap       ", Ye - Ym),
             ("mocap    + DJI attitude", Ym + Ya)]

    print(f"  {'pair':<26}{'mean':>9}{'sd':>8}{'drift/min':>11}   verdict")
    print("  " + "-" * 74)
    results = {}
    for label, series in pairs:
        w = wrap180(series)
        mean, sd = circ_stats(w)
        # slope of the unwrapped difference, deg per minute
        slope = np.polyfit(grid - grid[0], np.unwrap(np.radians(series)), 1)[0]
        slope = math.degrees(slope) * 60.0
        verdict = ("CONSTANT" if sd < 5.0 and abs(slope) < 2.0
                   else "wandering" if abs(slope) >= 2.0
                   else "noisy")
        results[label.strip()] = (mean, sd, slope)
        print(f"  {label:<26}{mean:>9.1f}{sd:>8.1f}{slope:>11.2f}   {verdict}")

    print("\n  Reading:")
    print("   * 'mocap + DJI attitude' CONSTANT means DJI yaw is the NEGATIVE of")
    print("     mocap yaw plus an offset, i.e. DJI is clockwise-positive while")
    print("     mocap is REP-103 counter-clockwise-positive.")
    print("   * 'EKF yaw - DJI attitude' CONSTANT means the odom datum is fixed")
    print("     relative to DJI attitude: yaw_source=dji_attitude is safe and")
    print("     yaw_offset_deg is that mean.")
    print("   * 'EKF yaw - mocap' small sd means the published orientation is")
    print("     usable directly: yaw_source=ekf_pose is safe.")

    # ============================================================ 2. POSITION
    section("2. POSITION ACCURACY  (odom -> mocap fit)")

    s_fit, th, R2, t2 = fit_yaw_scale_2d(Pe[:, :2], Pm[:, :2])
    print(f"  whole-segment fit: yaw {math.degrees(th):+.1f} deg, "
          f"scale {s_fit:.4f}")
    print(f"  fitted yaw vs 'EKF yaw - mocap' mean "
          f"{results['EKF yaw  - mocap'][0]:+.1f} deg  "
          f"(agreement means the frame fit and the orientation agree)")

    Pa = (s_fit * (Pe[:, :2] @ R2.T)) + t2
    err = np.linalg.norm(Pa - Pm[:, :2], axis=1)
    print(f"\n  ATE (2D, after alignment): mean {err.mean():.3f} m, "
          f"median {np.median(err):.3f}, p95 {np.percentile(err, 95):.3f}, "
          f"max {err.max():.3f}")
    print("  ATE grows without bound by construction (filter_design.md 9);")
    print("  it is reported for context, never as a pass criterion.")

    dz = Pe[:, 2] - Pm[:, 2]
    print(f"  z error: mean {dz.mean():+.3f} m, sd {dz.std():.3f}  "
          f"(altitude is observed, so this one is meaningful)")

    # ---- RPE over short windows, per vo_epoch segment
    k = int(round(a.rpe_window / a.grid))

    def rpe(idx):
        if len(idx) < k + 5:
            return None
        P, Q = Pe[idx], Pm[idx]
        de = P[k:, :2] - P[:-k, :2]
        dg = Q[k:, :2] - Q[:-k, :2]
        mag = np.linalg.norm(dg, axis=1)
        sel = mag > 0.05                    # ignore near-stationary windows
        if sel.sum() < 5:
            return None
        s_, th_, R_, _ = fit_yaw_scale_2d(P[:, :2], Q[:, :2])
        de_a = s_ * (de @ R_.T)
        e = np.linalg.norm(de_a[sel] - dg[sel], axis=1) / mag[sel]
        # unscaled: frame fixed, scale left as the filter produced it
        de_u = de @ R_.T
        eu = np.linalg.norm(de_u[sel] - dg[sel], axis=1) / mag[sel]
        return (100 * np.median(e), 100 * np.median(eu), s_, sel.sum())

    print(f"\n  RPE over {a.rpe_window:.0f} s windows, moving segments only:")
    r = rpe(np.arange(len(grid)))
    if r:
        print(f"    whole flight: RPE {r[0]:.1f} %  (scale-corrected), "
              f"{r[1]:.1f} % as it runs, fitted scale {r[2]:.3f}, n={r[3]}")

    # per epoch
    if d[T['vo']]:
        tv = np.array([x[0] for x in d[T['vo']]])
        ev = np.array([x[1].vo_epoch for x in d[T['vo']]])
        eg = np.interp(grid, tv, ev).round().astype(int)
        print("\n    per vo_epoch segment:")
        print(f"      {'epoch':>6}{'dur s':>8}{'RPE%':>8}{'raw%':>8}"
              f"{'scale':>8}{'n':>6}")
        for e_ in sorted(set(eg)):
            idx = np.where(eg == e_)[0]
            dur = len(idx) * a.grid
            rr = rpe(idx)
            if rr and dur > 10.0:
                print(f"      {e_:>6}{dur:>8.0f}{rr[0]:>8.1f}{rr[1]:>8.1f}"
                      f"{rr[2]:>8.3f}{rr[3]:>6}")
            elif dur > 10.0:
                print(f"      {e_:>6}{dur:>8.0f}{'--':>8}{'--':>8}"
                      f"{'--':>8}{'--':>6}")

    # ============================================================== 3. HEALTH
    section("3. HEALTH FLAGS")
    st = d[T['st']]
    if st:
        sc = np.array([m.scale for _, m in st])
        ss = np.array([m.sigma_scale for _, m in st])
        deg = np.array([m.degraded for _, m in st])
        rel = ss / np.maximum(sc, 1e-9)
        print(f"  scale:        {sc.min():.3f} -> {sc.max():.3f}, "
              f"median {np.median(sc):.3f}")
        print(f"  sigma_scale:  {ss.min():.3f} -> {ss.max():.3f}, "
              f"median {np.median(ss):.3f}")
        print(f"  RELATIVE sigma_scale/scale: median {np.median(rel):.4f}, "
              f"p95 {np.percentile(rel, 95):.4f}")
        print(f"  degraded: {100 * deg.mean():.0f} % of samples")
        print()
        for thr in (0.05, 0.10, 0.20):
            print(f"  if the gate were RELATIVE at {thr:.2f}: "
                  f"{100 * (rel > thr).mean():5.1f} % would be degraded")
        for thr in (0.20, 0.50, 1.00):
            print(f"  if the gate were ABSOLUTE at {thr:.2f}: "
                  f"{100 * (ss > thr).mean():5.1f} % would be degraded")
        print("\n  An absolute threshold on sigma_scale cannot work when scale")
        print("  itself is order 8: clearing 0.20 absolute needs 2.5 % relative.")

    # ====================================================== 4. IMPLICATIONS
    section("4. CONTROLLER IMPLICATIONS")
    L = CMD_LATENCY_S + VO_DELAY_S
    print(f"  loop delay with EKF feedback: {CMD_LATENCY_S:.2f} + "
          f"{VO_DELAY_S:.2f} = {L:.2f} s")
    for pm_deg in (60, 45):
        wc = math.radians(90 - pm_deg) / L
        print(f"    {pm_deg} deg phase margin -> omega_c {wc:.2f} rad/s, "
              f"Kp <= {wc / 1.5:.2f} (plant gain 1.5), "
              f"{wc / 1.0:.2f} (plant gain 1.0)")
    print(f"\n  With mocap feedback the delay was only {CMD_LATENCY_S:.2f} s,")
    print("  which is why kp_xy=0.6 felt comfortable. It will not transfer.")
    if r:
        print(f"\n  Measured RPE {r[1]:.1f} % as the system runs. Over a 1 s")
        print("  window at 0.5 m/s that is roughly "
              f"{0.005 * r[1]:.3f} m of position error per second of travel,")
        print("  which is the floor on closed-loop tracking with this estimator.")


if __name__ == '__main__':
    main()