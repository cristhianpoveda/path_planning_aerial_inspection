#!/usr/bin/env python3
"""vo_delay.py -- measure VO_DELAY, and judge VO quality independently of the EKF.

    python3 vo_delay.py --bag step10c
    python3 vo_delay.py --bag step10c --selftest
"""
import argparse
import math

import numpy as np

TELEMETRY_LAG_S = 0.0732     # filter_design.md 4, velocity vs mocap


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


def resampled_speed(t, p, grid, smooth_s=0.10):
    """Resample position onto a uniform grid, then differentiate and smooth.

    filter_design.md 12.4: resample -> smooth -> differentiate. Point-to-point
    differentiation of a jittery series measures the noise's own path.
    """
    dt = grid[1] - grid[0]
    pg = np.column_stack([np.interp(grid, t, p[:, k]) for k in range(3)])
    v = np.gradient(pg, dt, axis=0)
    k = max(1, int(round(smooth_s / dt)))
    if k > 1:
        w = np.ones(k) / k
        v = np.column_stack([np.convolve(v[:, i], w, mode='same')
                             for i in range(3)])
    return np.linalg.norm(v, axis=1), pg


def best_lag(grid, a_sig, b_sig, max_lag=1.0):
    """Lag L maximising correlation of a(t) with b(t - L).

    Positive L means a LAGS b: the a series describes motion that happened L
    seconds earlier.
    """
    dt = grid[1] - grid[0]
    a = a_sig - a_sig.mean()
    b = b_sig - b_sig.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None, None
    a /= a.std()
    b /= b.std()
    n = int(max_lag / dt)
    lags = np.arange(-n, n + 1)
    cc = np.array([np.dot(a, np.roll(b, k)) / len(a) for k in lags])
    i = int(np.argmax(cc))
    return float(lags[i] * dt), float(cc[i])


def rot2(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def fit_yaw_scale(A, B):
    """s, theta minimising || s R(theta) A - B ||, rows are 2D vectors."""
    num = np.sum(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0])
    den = np.sum(A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])
    th = math.atan2(num, den)
    AR = A @ rot2(th).T
    s = float(np.sum(AR * B) / max(np.sum(AR * AR), 1e-12))
    return s, th


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def selftest():
    print("SELF-TEST: injecting a known VO lag\n")
    dt = 0.01
    grid = np.arange(0, 120, dt)
    true_lag = 0.27
    # true motion: a few transits
    v_true = 0.5 * (np.sin(2 * np.pi * grid / 20.0) > 0.3).astype(float)
    p_true = np.column_stack([np.cumsum(v_true) * dt,
                              np.zeros_like(grid), np.zeros_like(grid)])
    # VO sees the same motion but its stamps are `true_lag` late
    p_vo = np.column_stack([np.interp(grid - true_lag, grid, p_true[:, 0]),
                            np.zeros_like(grid), np.zeros_like(grid)])
    s_m, _ = resampled_speed(grid, p_true, grid)
    s_v, _ = resampled_speed(grid, p_vo, grid)
    lag, cc = best_lag(grid, s_v, s_m)
    print(f"  injected {true_lag:.3f} s -> recovered {lag:.3f} s "
          f"(corr {cc:.3f})")
    print(f"  error {1000 * abs(lag - true_lag):.0f} ms")

    A = np.random.default_rng(0).normal(size=(400, 2))
    s_t, th_t = 0.31, math.radians(-73.0)
    B = (A @ rot2(th_t).T) * s_t
    s_f, th_f = fit_yaw_scale(A, B)
    print(f"\n  injected scale {s_t:.3f} yaw {math.degrees(th_t):+.1f} "
          f"-> recovered {s_f:.3f} {math.degrees(th_f):+.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag')
    ap.add_argument('--ns', default='/drone_1')
    ap.add_argument('--grid', type=float, default=0.01)
    ap.add_argument('--window', type=float, default=1.0)
    ap.add_argument('--min-move', type=float, default=0.05)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.bag:
        print("need --bag or --selftest")
        return

    n = a.ns.rstrip('/')
    T = {'vo': f'{n}/vo/pose', 'st': f'{n}/vo/status',
         'mocap': f'{n}/mocap/pose', 'spd': f'{n}/speed_vector'}
    print(f"reading {a.bag}")
    d = read_bag(a.bag, list(T.values()))

    tv = np.array([r[0] for r in d[T['vo']]])
    pv = np.array([[r[1].pose.position.x, r[1].pose.position.y,
                    r[1].pose.position.z] for r in d[T['vo']]])
    tm = np.array([r[0] for r in d[T['mocap']]])
    pm = np.array([[r[1].pose.pose.position.x, r[1].pose.pose.position.y,
                    r[1].pose.pose.position.z] for r in d[T['mocap']]])

    print(f"  vo/pose {len(tv)} msgs, mocap {len(tm)} msgs")

    t0 = max(tv[0], tm[0])
    t1 = min(tv[-1], tm[-1])
    grid = np.arange(t0, t1, a.grid)
    s_vo, pv_g = resampled_speed(tv, pv, grid)
    s_mo, pm_g = resampled_speed(tm, pm, grid)

    # ==================================================== 1. VO_DELAY
    section("1. VO_DELAY  (vo/pose stamp vs physical motion)")
    lag, cc = best_lag(grid, s_vo, s_mo)
    if lag is None:
        print("  insufficient variation to correlate")
        return
    print(f"  WHOLE FLIGHT (unreliable if the scale changes per epoch):")
    print(f"  against mocap:        {lag:+.3f} s   (correlation {cc:.3f})")
    print("  Positive means the vo/pose stamp is LATE: the motion it describes")
    print(f"  happened {1000 * lag:.0f} ms before the stamp says.")

    if d[T['spd']]:
        ts = np.array([r[0] for r in d[T['spd']]])
        vs = np.array([[r[1].vector.x, r[1].vector.y, r[1].vector.z]
                       for r in d[T['spd']]])
        s_dji = np.interp(grid, ts, np.linalg.norm(vs, axis=1))
        lag2, cc2 = best_lag(grid, s_vo, s_dji)
        if lag2 is not None:
            print(f"\n  against DJI velocity: {lag2:+.3f} s   "
                  f"(correlation {cc2:.3f})")
            print(f"  DJI velocity itself lags truth by "
                  f"{1000 * TELEMETRY_LAG_S:.0f} ms, so this implies "
                  f"{lag2 + TELEMETRY_LAG_S:+.3f} s")
            print("  Agreement between the two routes is the consistency check.")

    # ---- per-epoch, where the VO scale is constant -------------------
    MIN_CORR = 0.40
    print("\n  PER EPOCH  (scale is constant within an epoch)")
    good = []
    if d[T['st']]:
        tst = np.array([r[0] for r in d[T['st']]])
        est = np.array([r[1].vo_epoch for r in d[T['st']]])
        eg = np.interp(grid, tst, est).round().astype(int)
        print(f"    {'epoch':>6}{'dur s':>8}{'lag s':>9}{'corr':>8}   use")
        for e_ in sorted(set(eg)):
            i = np.where(eg == e_)[0]
            if len(i) * a.grid < 15.0:
                continue
            g2 = grid[i]
            l2, c2 = best_lag(g2, s_vo[i], s_mo[i])
            if l2 is None:
                continue
            ok = c2 >= MIN_CORR and l2 > 0.0
            if ok:
                good.append((l2, len(i)))
            print(f"    {e_:>6}{len(i) * a.grid:>8.0f}{l2:>+9.3f}{c2:>8.3f}"
                  f"   {'yes' if ok else 'REJECT'}")

    print()
    if good:
        w = np.array([g[1] for g in good], float)
        v = np.array([g[0] for g in good], float)
        est_lag = float(np.sum(v * w) / w.sum())
        print(f"  >>> VO_DELAY = {est_lag:.3f} s   "
              f"from {len(good)} usable epoch(s), "
              f"spread {v.min():.3f}-{v.max():.3f}")
        print("  >>> Weighted by segment length.")
    else:
        print("  >>> NO USABLE SEGMENT.")
        print("  >>> Every epoch was either too short, too frozen, or gave a")
        print(f"  >>> correlation below {MIN_CORR}. A lag from a flat")
        print("  >>> correlation surface is noise, not a measurement.")
        print("  >>> Do NOT change VO_DELAY on this bag. A negative result")
        print("  >>> here is physically impossible and is the tell.")

    # ============================================ 2. VO INCREMENT QUALITY
    section("2. VO INCREMENT QUALITY  (at the measured lag, filter not involved)")
    k = int(round(a.window / a.grid))
    # shift VO backwards by the measured lag so it aligns with mocap
    shift = int(round(lag / a.grid))
    pv_s = np.roll(pv_g, -shift, axis=0)

    dV = pv_s[k:, :2] - pv_s[:-k, :2]
    dM = pm_g[k:, :2] - pm_g[:-k, :2]
    tw = grid[:-k]
    mag = np.linalg.norm(dM, axis=1)
    sel = mag > a.min_move
    if shift > 0:
        sel[-shift:] = False
    dV, dM, tw, mag = dV[sel], dM[sel], tw[sel], mag[sel]
    print(f"  {sel.sum()} windows of {a.window:.0f} s with > "
          f"{100 * a.min_move:.0f} cm of true motion")

    if sel.sum() < 20:
        print("  too few windows")
        return

    ev = None
    if d[T['st']]:
        tst = np.array([r[0] for r in d[T['st']]])
        est = np.array([r[1].vo_epoch for r in d[T['st']]])
        ev = np.interp(tw, tst, est).round().astype(int)

    def report(label, iv):
        if iv.sum() < 15:
            return
        A, B, m = dV[iv], dM[iv], mag[iv]
        s_f, th_f = fit_yaw_scale(A, B)
        pred = s_f * (A @ rot2(th_f).T)
        err = np.median(np.linalg.norm(pred - B, axis=1) / m)
        ratio = np.linalg.norm(A, axis=1) / m
        frozen = float((ratio < 0.05 * max(s_f, 1e-9)).mean())
        print(f"  {label:<12}{iv.sum():>6}{s_f:>10.3f}"
              f"{math.degrees(th_f):>10.1f}{100 * err:>9.1f}"
              f"{100 * frozen:>9.1f}")

    print(f"\n  {'segment':<12}{'n':>6}{'scale':>10}{'yaw':>10}"
          f"{'resid %':>9}{'frozen %':>9}")
    print("  " + "-" * 56)
    report("all", np.ones(len(dV), bool))
    if ev is not None:
        for e_ in sorted(set(ev)):
            report(f"epoch {e_}", ev == e_)

    print("\n  scale    : VO units per metre for that segment")
    print("  yaw      : VO world frame vs mocap, arbitrary per epoch")
    print("  resid %  : what is left after the best single scale and rotation")
    print("  frozen % : windows where VO moved under 5 % of what it should")
    print("\n  A low residual with low frozen means VO is a clean rigid scaling")
    print("  of truth, and any remaining EKF error is in the filter or the")
    print("  frontend. A high frozen fraction means VO itself stalls.")


if __name__ == '__main__':
    main()
