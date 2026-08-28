#!/usr/bin/env python3
"""
yaw_datum.py -- STEP 35: which DJI datum carries the 52 degree offset?

The nav frame is seeded from DJI ATTITUDE yaw, and DJI VELOCITY (NED) is
referenced to magnetic north. A constant ~52 deg rotation was measured between
DJI velocity heading and the VO-derived heading in the nav frame. One of the
two DJI datums is displaced; mocap breaks the tie because each can be compared
against it independently:

    A = heading(mocap velocity)  -  heading(ned_to_enu(v_dji))
    B = yaw(mocap orientation)   -  yaw(DJI attitude)

Both carry the same unknown mocap-frame yaw, so it CANCELS in A - B, which is
the quantity the filter sees as the 52 deg.

    A - B ~= 0      -> neither DJI datum is displaced relative to the other;
                       the 52 deg lives in the VO chain (R_LINK_OPTICAL, the
                       gimbal yaw negation, camera extrinsic).
    A ~= 0, B ~= -52 -> DJI ATTITUDE yaw is displaced. The nav-frame seed is
                       wrong; seed R_n_v from velocity heading instead.
    B ~= 0, A ~= +52 -> DJI VELOCITY uses a different north than attitude.
                       dtheta_z is doing real work; keep the yaw row enabled
                       and give velocity its own yaw offset.

Method notes, both learned the hard way earlier in this investigation:
  * mocap velocity comes from resample-smooth-differentiate, NOT raw
    differences: raw arc length at 100 Hz measures the noise's own path and
    inflated a K_VEL estimate by 7-13%.
  * mocap orientation is CONJUGATED (check_g2.py rev 2 section E).
  * only windows above V_LOW are used: below the 0.1 m/s quantum the DJI
    velocity DIRECTION is as unreliable as its magnitude.
  * the mocap clock offset is fitted, not assumed.

Run:
    python3 yaw_datum.py F9_02 F9_02_mocap [--vlow 0.4]
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

T_VEL = "/drone_1/speed_vector"
T_ATT = "/drone_1/attitude"
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


def ned_to_enu(v):
    """(x, y, z) -> (y, x, -z). Same mapping as filter.ned_to_enu."""
    return np.stack([v[:, 1], v[:, 0], -v[:, 2]], 1)


def yaw_of(q, conjugate=False):
    x, y, z, w = q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy(), q[:, 3]
    if conjugate:
        x, y, z = -x, -y, -z
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def smooth_vel(t, p, grid, smooth_s=0.15):
    k = max(int(round(smooth_s / DT)) | 1, 3)
    w = np.ones(k) / k
    out = []
    for i in range(3):
        y = np.convolve(np.interp(grid, t, p[:, i]), w, mode="same")
        out.append(np.gradient(y, DT))
    return np.stack(out, 1)


def circ(a):
    """Circular mean and sd of angles (radians)."""
    s, c = np.mean(np.sin(a)), np.mean(np.cos(a))
    R = min(float(np.hypot(s, c)), 1.0)   # clip: float error can push R
    return (float(np.arctan2(s, c)),      # above 1 and make log positive
            float(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def fit_offset(t_v, v_d, grid, sp_m, span=0.5):
    a = (sp_m - sp_m.mean()) / (sp_m.std() + 1e-12)
    vd = np.linalg.norm(v_d, axis=1)
    best, bc = 0.0, -np.inf
    for d in np.arange(-span, span, DT):
        b = np.interp(grid + d, t_v, vd)
        sd = b.std()
        if sd < 1e-9:
            continue
        c = float((a * ((b - b.mean()) / sd)).mean())
        if c > bc:
            bc, best = c, float(d)
    return best, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--vlow", type=float, default=0.4)
    a = ap.parse_args()

    f = read_bag(a.flight_bag, [T_VEL, T_ATT])
    mo = read_bag(a.mocap_bag, [T_MOCAP])[T_MOCAP]

    t_m = np.array([stamp(x) for x in mo])
    p_m = np.array([[x.pose.position.x, x.pose.position.y,
                     x.pose.position.z] for x in mo])
    q_m = np.array([[x.pose.orientation.x, x.pose.orientation.y,
                     x.pose.orientation.z, x.pose.orientation.w] for x in mo])
    keep = np.concatenate([[True], (np.abs(np.diff(p_m, axis=0)).sum(1) > 0)])
    t_m, p_m, q_m = t_m[keep], p_m[keep], q_m[keep]

    t_v = np.array([stamp(x) for x in f[T_VEL]])
    v_ned = np.array([[x.vector.x, x.vector.y, x.vector.z] for x in f[T_VEL]])
    v_enu = ned_to_enu(v_ned)

    t_a = np.array([stamp(x) for x in f[T_ATT]])
    rpy = np.array([[x.roll, x.pitch, x.yaw] for x in f[T_ATT]])
    if np.abs(rpy).max() > 2 * np.pi:
        rpy = np.radians(rpy)

    floor = np.percentile(p_m[:, 2], 5)
    air = p_m[:, 2] > floor + 0.20
    lo = t_m[np.argmax(air)] + 2.0
    hi = t_m[len(air) - 1 - np.argmax(air[::-1])] - 2.0
    grid = np.arange(max(lo, t_v[0], t_a[0]) + 1, min(hi, t_v[-1], t_a[-1]) - 1, DT)
    vm = smooth_vel(t_m, p_m, grid)
    spm = np.linalg.norm(vm, axis=1)

    d, c = fit_offset(t_v, v_enu, grid, spm)
    print(f"{a.flight_bag}: airborne {hi-lo:.1f} s, "
          f"fitted mocap offset {d:+.3f} s (corr {c:.3f})")
    vm = smooth_vel(t_m + d, p_m, grid)
    ym = np.interp(grid, t_m + d, np.unwrap(yaw_of(q_m, conjugate=True)))

    ve = np.stack([np.interp(grid, t_v, v_enu[:, i]) for i in range(3)], 1)
    ya = np.interp(grid, t_a, np.unwrap(rpy[:, 2]))

    fast = (np.linalg.norm(vm[:, :2], axis=1) > a.vlow) & \
           (np.linalg.norm(ve[:, :2], axis=1) > a.vlow)
    print(f"windows above V_LOW={a.vlow}: {fast.sum()} of {len(grid)} "
          f"({fast.mean():.1%})\n")
    if fast.sum() < 200:
        sys.exit("not enough fast samples -- use a transit-heavy bag")

    hd_m = np.arctan2(vm[fast, 1], vm[fast, 0])
    hd_d = np.arctan2(ve[fast, 1], ve[fast, 0])
    A, A_sd = circ(wrap(hd_m - hd_d))
    B, B_sd = circ(wrap(ym[fast] - ya[fast]))
    AB, AB_sd = circ(wrap((hd_m - hd_d) - (ym[fast] - ya[fast])))

    print(f"{'quantity':<38}{'deg':>9}{'sd':>8}")
    print(f"{'A  mocap heading - DJI velocity heading':<38}"
          f"{np.degrees(A):9.2f}{np.degrees(A_sd):8.2f}")
    print(f"{'B  mocap yaw - DJI attitude yaw':<38}"
          f"{np.degrees(B):9.2f}{np.degrees(B_sd):8.2f}")
    print(f"{'A - B  (what the filter sees)':<38}"
          f"{np.degrees(AB):9.2f}{np.degrees(AB_sd):8.2f}")

    ab = abs(np.degrees(AB))
    print()
    if ab < 10:
        print("-> A - B ~= 0: the two DJI datums AGREE. The 52 deg is NOT "
              "between them;\n   it lives in the VO chain (R_LINK_OPTICAL, "
              "gimbal yaw negation, extrinsic).")
    elif abs(np.degrees(A)) < 15:
        print("-> DJI ATTITUDE yaw is displaced: velocity agrees with mocap, "
              "attitude does not.\n   Seed R_n_v from velocity heading, not "
              "attitude yaw.")
    elif abs(np.degrees(B)) < 15:
        print("-> DJI VELOCITY uses a different north than attitude. "
              "dtheta_z is doing real\n   work: keep the yaw row enabled and "
              "give velocity its own yaw offset.")
    else:
        print("-> BOTH differ from mocap. A - B is still the filter-relevant "
              "number;\n   the common part is the mocap frame's own yaw.")

    print("\nper-third (is it constant?)")
    idx = np.where(fast)[0]
    for j in range(3):
        sl = idx[j * len(idx) // 3:(j + 1) * len(idx) // 3]
        if len(sl) < 50:
            continue
        a3, _ = circ(wrap(np.arctan2(vm[sl, 1], vm[sl, 0])
                          - np.arctan2(ve[sl, 1], ve[sl, 0])))
        b3, _ = circ(wrap(ym[sl] - ya[sl]))
        print(f"   third {j}: A {np.degrees(a3):+7.2f}  B {np.degrees(b3):+7.2f}"
              f"  A-B {np.degrees(wrap(a3-b3)):+7.2f}  n={len(sl)}")


if __name__ == "__main__":
    main()