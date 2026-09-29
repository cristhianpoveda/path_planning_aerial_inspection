#!/usr/bin/env python3
"""yaw_windows.py -- find the yaw-impulse windows in a flight npz.

    python3 yaw_windows.py c03.npz                 # table
    python3 yaw_windows.py c03.npz --pairs         # "t0 t1" lines only
"""
import argparse
import math

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    D = {}
    for k in z.files:
        g, f = k.split("/", 1)
        D.setdefault(g, {})[f] = z[k]
    return D


def quat_yaw(q):
    q = np.atleast_2d(np.asarray(q, float))
    x, y, zc, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * zc + x * y), 1.0 - 2.0 * (y * y + zc * zc))


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--min-dyaw", type=float, default=5.0, help="deg")
    ap.add_argument("--max-dpos", type=float, default=0.15, help="m")
    ap.add_argument("--any", action="store_true",
                    help="include yaw steps that also translate")
    ap.add_argument("--pad", type=float, default=4.0,
                    help="seconds added before the step and after the segment")
    ap.add_argument("--min-dur", type=float, default=30.0,
                    help="seconds; windows shorter than this are extended")
    ap.add_argument("--max-len", type=float, default=90.0)
    ap.add_argument("--pairs", action="store_true",
                    help="print only 't0 t1' lines, for a shell loop")
    a = ap.parse_args()

    D = load(a.npz)
    t_ref = min(np.asarray(g["t_recv"], float).min() for g in D.values()
                if len(g.get("t_recv", [])))
    t_end_file = max(np.asarray(g["t_recv"], float).max() for g in D.values()
                     if len(g.get("t_recv", [])))

    if "setpoint" not in D:
        raise SystemExit("no setpoint topic in this npz")
    ts = np.asarray(D["setpoint"]["t_recv"], float)
    sp = np.asarray(D["setpoint"]["p"], float)
    yaw = quat_yaw(D["setpoint"]["q"])
    order = np.argsort(ts)
    ts, sp, yaw = ts[order], sp[order], yaw[order]

    dyaw = np.degrees(np.abs(wrap_pi(np.diff(yaw))))
    dpos = np.linalg.norm(np.diff(sp, axis=0), axis=1)
    changed = (dyaw > 0.5) | (dpos > 0.02)
    change_i = np.where(changed)[0] + 1

    cmd_t = cmd_w = None
    if "command.vel" in D and "ang" in D["command.vel"]:
        cmd_t = np.asarray(D["command.vel"]["t_recv"], float)
        cmd_w = np.degrees(np.asarray(D["command.vel"]["ang"], float)[:, 2])
    m_t = m_y = None
    if "mocap" in D:
        m_t = np.asarray(D["mocap"]["t_hdr"], float)
        if not np.isfinite(m_t).all():
            m_t = np.asarray(D["mocap"]["t_recv"], float)
        m_y = np.unwrap(quat_yaw(D["mocap"]["q"]))
        o = np.argsort(m_t)
        m_t, m_y = m_t[o], m_y[o]

    rows = []
    for i in change_i:
        dy = math.degrees(abs(wrap_pi(yaw[i] - yaw[i - 1])))
        dp = float(np.linalg.norm(sp[i] - sp[i - 1]))
        if dy < a.min_dyaw:
            continue
        if not a.any and dp > a.max_dpos:
            continue
        nxt = change_i[change_i > i]
        seg_end = ts[nxt[0]] if nxt.size else ts[-1]
        t0 = ts[i] - a.pad
        t1 = min(seg_end + a.pad, ts[i] + a.max_len)
        if t1 - t0 < a.min_dur:
            t1 = min(t0 + a.min_dur, t_end_file)
        peak = float("nan")
        if cmd_t is not None:
            m = (cmd_t >= t0) & (cmd_t <= t1)
            if m.any():
                peak = float(np.max(np.abs(cmd_w[m])))
        gt_dyaw = gt_p90 = float("nan")
        if m_t is not None:
            m = (m_t >= t0) & (m_t <= t1)
            if m.sum() > 10:
                yy = m_y[m]
                gt_dyaw = float(np.degrees(yy.max() - yy.min()))
                w = np.abs(np.diff(yy) / np.maximum(np.diff(m_t[m]), 1e-6))
                gt_p90 = float(np.degrees(np.percentile(w, 90)))
        rows.append(dict(t0=t0 - t_ref, t1=t1 - t_ref, dyaw=dy, dpos=dp,
                         peak=peak, gt_dyaw=gt_dyaw, gt_p90=gt_p90))

    if a.pairs:
        for r in rows:
            print(f"{r['t0']:.1f} {r['t1']:.1f}")
        return

    if not rows:
        print("no yaw steps found. Try --any, or lower --min-dyaw.")
        return
    print(f"{len(rows)} windows "
          f"({'all yaw steps' if a.any else 'rotation-dominant only'})")
    print(f"{'t0':>8s} {'t1':>8s} {'dur':>6s} {'sp dyaw':>8s} {'sp dpos':>8s} "
          f"{'cmd w':>7s} {'gt dyaw':>8s} {'gt |w|p90':>10s}")
    for r in rows:
        print(f"{r['t0']:8.1f} {r['t1']:8.1f} {r['t1']-r['t0']:6.1f} "
              f"{r['dyaw']:8.1f} {r['dpos']:8.3f} {r['peak']:7.1f} "
              f"{r['gt_dyaw']:8.1f} {r['gt_p90']:10.1f}")
    print("\ndeg and m. Run the analysis only on windows where gt dyaw is "
          "large:\na setpoint change the aircraft did not act on carries no "
          "timing information.")


if __name__ == "__main__":
    main()
