#!/usr/bin/env python3
"""Which parts of a panel bag are usable? Before any optics fitting.

Standalone. Drop next to the bags with panel.py and run:

    python3 panel_triage.py --bag dof_sweep_20260907_1349
    python3 panel_triage.py --bag speed_1420 --mode sweep
"""

import argparse
import sys

import numpy as np

TOPIC_IMG = "/drone_1/camera/image/compressed"
TOPIC_DRONE = "/optitrack/rigid_bodies/dji_mini4"
TOPIC_PANEL = "/optitrack/rigid_bodies/panel"
TOPIC_STATUS = "/drone_1/localisation/status"

STALL_FACTOR = 4.0
HOVER_SPEED = 0.05      # m/s, mocap
HOVER_MIN_S = 2.0
STANDOFF_TOL = 0.03     # m, spread within a hover
SWEEP_MIN_S = 1.5
SWEEP_SPEED_TOL = 0.08  # m/s, spread within a constant-speed leg


def read_bag(path, topics):
    import rclpy.serialization
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    out = {t: [] for t in topics}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic not in out:
            continue
        try:
            m = rclpy.serialization.deserialize_message(
                data, get_message(types[topic]))
        except Exception:
            continue
        out[topic].append(m)
    return out


def stamp(msg):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def pose_array(msgs):
    return np.array([[stamp(m), m.pose.position.x, m.pose.position.y,
                      m.pose.position.z, m.pose.orientation.x,
                      m.pose.orientation.y, m.pose.orientation.z,
                      m.pose.orientation.w] for m in msgs])


def quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def runs_where(mask, t, min_dur):
    out, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if t[j - 1] - t[i] >= min_dur:
            out.append((i, j))
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--mode", choices=["hover", "sweep", "both"],
                    default="both")
    ap.add_argument("--panel-normal", type=int, default=None,
                    help="which axis of the panel rigid body is its outward "
                         "normal: 0=x, 1=y, 2=z. Default: chosen "
                         "automatically as the axis and sign giving the "
                         "smallest median incidence, since the aircraft spent "
                         "the flight facing the panel.")
    ap.add_argument("--min-standoff", type=float, default=0.6)
    ap.add_argument("--max-standoff", type=float, default=3.0)
    ap.add_argument("--max-incidence-deg", type=float, default=35.0)
    a = ap.parse_args()

    d = read_bag(a.bag, [TOPIC_IMG, TOPIC_DRONE, TOPIC_PANEL, TOPIC_STATUS])
    if not d[TOPIC_IMG]:
        sys.exit("no camera frames in the bag")

    # ---------------------------------------------------------- camera health
    ti = np.array([stamp(m) for m in d[TOPIC_IMG]])
    ti.sort()
    dt = np.diff(ti)
    med = float(np.median(dt))
    stall = dt > STALL_FACTOR * med
    print(f"camera: {len(ti)} frames over {ti[-1] - ti[0]:.1f} s, "
          f"median interval {med * 1e3:.1f} ms ({1 / med:.1f} Hz)")
    if stall.any():
        lost = float(dt[stall].sum() - stall.sum() * med)
        print(f"  {int(stall.sum())} stalls, longest {dt[stall].max():.2f} s, "
              f"{lost:.1f} s lost ({100 * lost / (ti[-1] - ti[0]):.1f} % of "
              "the bag)")
        for k in np.argsort(-dt)[:5]:
            if stall[k]:
                print(f"    t+{ti[k] - ti[0]:7.1f} s  gap {dt[k]:.2f} s")
    else:
        print("  no stalls")

    # ------------------------------------------------------- panel is tracked
    if not d[TOPIC_PANEL]:
        sys.exit("no panel poses; standoff cannot be established")
    P = pose_array(d[TOPIC_PANEL])
    uniq = len(np.unique(P[:, 1:8], axis=0))
    sd = P[:, 1:4].std(axis=0) * 1e3
    print(f"\npanel rigid body: {len(P)} samples, {uniq} unique poses, "
          f"position sd {np.round(sd, 3).tolist()} mm")
    if uniq <= 1:
        sys.exit("the panel rigid body is NOT tracked -- one stored pose "
                 "republished. Standoff from it would be meaningless. This is "
                 "the same failure /optitrack/rigid_bodies/calib_board had.")
    if sd.max() > 5.0:
        print("  WARNING: the panel moved during the bag; standoff is not a "
              "constant per hover")

    # ------------------------------------------------------------- geometry
    D = pose_array(d[TOPIC_DRONE])
    t = D[:, 0]
    p_panel = np.array([np.interp(t, P[:, 0], P[:, i]) for i in (1, 2, 3)]).T
    R0 = quat_to_R(P[len(P) // 2, 4:8])
    v_rel = D[:, 1:4] - p_panel
    standoff = np.linalg.norm(v_rel, axis=1)
    u = v_rel / standoff[:, None]

    def inc_for(axis, sign):
        c = np.clip(u @ (sign * R0[:, axis]), -1, 1)
        return np.degrees(np.arccos(c))

    if a.panel_normal is None:
        
        cands = [(float(np.median(inc_for(ax, sg))), ax, sg)
                 for ax in (0, 1, 2) for sg in (1, -1)]
        cands.sort()
        med, axis, sign = cands[0]
        print(f"\npanel normal: auto-selected {'+' if sign > 0 else '-'}"
              f"{'xyz'[axis]} (median incidence {med:.1f} deg); "
              "alternatives "
              + ", ".join(f"{'+' if s > 0 else '-'}{'xyz'[x]}={m:.0f}"
                          for m, x, s in cands[1:]))
    else:
        axis, sign = a.panel_normal, 1
        for sg in (1, -1):
            if np.median(inc_for(axis, sg)) < 90:
                sign = sg
    incidence = inc_for(axis, sign)

    w = 21
    from scipy.signal import savgol_filter
    sm = np.column_stack([savgol_filter(D[:, i], w, 2) for i in (1, 2, 3)])
    speed = np.linalg.norm(np.gradient(sm, t, axis=0), axis=1)

    ok_geom = ((standoff > a.min_standoff) & (standoff < a.max_standoff)
               & (incidence < a.max_incidence_deg))
    print(f"\ngeometry: standoff {standoff.min():.2f}-{standoff.max():.2f} m, "
          f"incidence median {np.median(incidence):.1f} deg, "
          f"{100 * ok_geom.mean():.0f} % of the bag inside the window")
    if np.median(incidence) > 60:
        print("  WARNING: incidence near 90 deg suggests --panel-normal is "
              "the wrong axis")

    def frames_in(t0, t1):
        k = (ti >= t0) & (ti <= t1)
        n = int(k.sum())
        g = np.diff(ti[k])
        return n, float(g[g > STALL_FACTOR * med].sum()) if len(g) else 0.0

    if a.mode in ("hover", "both"):
        print("\nHOVER segments (for MTF50, line pairs, grey steps)")
        print("   t0      t1    dur   standoff   incid   frames  stalled")
        segs = runs_where(ok_geom & (speed < HOVER_SPEED), t, HOVER_MIN_S)
        for i, j in segs:
            s = standoff[i:j]
            if s.ptp() > STANDOFF_TOL:
                continue
            n, lost = frames_in(t[i], t[j - 1])
            flag = "  STALLED" if lost > 0.2 else ""
            print(f"  {t[i] - t[0]:6.1f} {t[j - 1] - t[0]:6.1f} "
                  f"{t[j - 1] - t[i]:5.1f}  {s.mean():6.3f} m  "
                  f"{np.median(incidence[i:j]):5.1f}  {n:6d}  "
                  f"{lost:5.2f} s{flag}")

    if a.mode in ("sweep", "both"):
        print("\nSWEEP segments (for the exposure-time fit)")
        print("   t0      t1    dur   speed    standoff   frames  stalled")
        moving = ok_geom & (speed > 0.15)
        for i, j in runs_where(moving, t, SWEEP_MIN_S):
            v = speed[i:j]
            if v.ptp() > SWEEP_SPEED_TOL:
                continue
            n, lost = frames_in(t[i], t[j - 1])
            print(f"  {t[i] - t[0]:6.1f} {t[j - 1] - t[0]:6.1f} "
                  f"{t[j - 1] - t[i]:5.1f}  {v.mean():5.2f} m/s "
                  f"{standoff[i:j].mean():6.3f} m {n:6d}  {lost:5.2f} s")

    print("\n# Take the hover segments to fit b_0, k_dof and the band, and the")
    print("# sweep segments at a common standoff to fit t_exp from the growth")
    print("# of edge width with speed.")


if __name__ == "__main__":
    main()
