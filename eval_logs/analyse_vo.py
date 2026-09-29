#!/usr/bin/env python3
"""analyse_vo.py -- why does VO lose tracking?

Tests, against mocap ground truth, whether tracking failure correlates with:
  * image rate            (fewer frames -> more motion between them)
  * translation speed     (baseline per frame)
  * yaw rate              (rotation without parallax)
  * inter-frame baseline  (the quantity that actually matters)
  * map point count       (scene texture)

Also reports recovery time after each epoch change, and the aircraft state
during the healthy periods versus the unhealthy ones.

    python3 analyse_vo.py --bag step8
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


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def rate_series(t, window=5.0, step=1.0):
    """Windowed message rate."""
    if len(t) < 2:
        return np.array([]), np.array([])
    grid = np.arange(t[0] + window, t[-1], step)
    r = np.array([((t >= g - window) & (t < g)).sum() / window for g in grid])
    return grid, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--ns', default='/drone_1')
    a = ap.parse_args()

    n = a.ns.rstrip('/')
    T = {'img': f'{n}/camera/image/compressed',
         'vo': f'{n}/vo/status',
         'vop': f'{n}/vo/pose',
         'mocap': f'{n}/mocap/pose',
         'speed': f'{n}/speed_vector'}

    print(f"reading {a.bag}")
    d = read_bag(a.bag, list(T.values()))
    for k, v in T.items():
        if not d[v]:
            print(f"  MISSING {v}")

    # ------------------------------------------------------------- mocap
    tm = np.array([r[0] for r in d[T['mocap']]])
    pm = np.array([[r[1].pose.pose.position.x, r[1].pose.pose.position.y,
                    r[1].pose.pose.position.z] for r in d[T['mocap']]])
    ym = np.unwrap([quat_yaw(r[1].pose.pose.orientation)
                    for r in d[T['mocap']]])

    # ------------------------------------------------------------- vo/status
    tv = np.array([r[0] for r in d[T['vo']]])
    valid = np.array([bool(r[1].pose_valid) for r in d[T['vo']]])
    epoch = np.array([int(r[1].vo_epoch) for r in d[T['vo']]])
    nmp = np.array([int(r[1].n_map_points) for r in d[T['vo']]])
    nkp = np.array([int(r[1].n_keypoints) for r in d[T['vo']]])
    state = [r[1].tracking_state for r in d[T['vo']]]

    ti = np.array([r[0] for r in d[T['img']]]) if d[T['img']] else np.array([])

    t0 = min(tm[0], tv[0])

    # =========================================================== 1. RATES
    section("1. RATES OVER TIME  (5 s windows)")
    if len(ti):
        g, r = rate_series(ti)
        print(f"  image rate: min {r.min():.1f}  median {np.median(r):.1f}  "
              f"max {r.max():.1f} Hz")
        print(f"    first 60 s: {r[g < g[0] + 60].mean():.1f} Hz")
        print(f"    last 60 s:  {r[g > g[-1] - 60].mean():.1f} Hz")
        lo = g[r < 0.6 * np.median(r)]
        if len(lo):
            print(f"    below 60 % of median for {len(lo)} s, "
                  f"first at t={lo[0] - t0:.0f} s")
    gv, rv = rate_series(tv)
    print(f"  vo/status rate: median {np.median(rv):.1f} Hz")
    if len(ti):
        print(f"  status/image ratio: {len(tv) / max(len(ti), 1):.3f}  "
              f"(1.0 means SLAM sees every frame it is sent)")

    # ================================================= 2. PER-FRAME MOTION
    section("2. MOTION BETWEEN CONSECUTIVE VO FRAMES")
    sel = (tv >= tm[0]) & (tv <= tm[-1])
    tvv = tv[sel]
    P = np.column_stack([np.interp(tvv, tm, pm[:, k]) for k in range(3)])
    Y = np.interp(tvv, tm, ym)

    dt = np.diff(tvv)
    base = np.linalg.norm(np.diff(P, axis=0), axis=1)      # m between frames
    dyaw = np.abs(np.degrees(np.diff(Y)))                  # deg between frames
    spd = base / np.maximum(dt, 1e-6)
    yrate = dyaw / np.maximum(dt, 1e-6)
    v_ok = valid[sel][1:]

    print(f"  frame interval: median {1000 * np.median(dt):.0f} ms")
    print(f"  baseline between frames: median {100 * np.median(base):.1f} cm, "
          f"p95 {100 * np.percentile(base, 95):.1f} cm")
    print(f"  yaw between frames:      median {np.median(dyaw):.2f} deg, "
          f"p95 {np.percentile(dyaw, 95):.2f} deg")

    print(f"\n  {'condition':<34}{'n':>7}{'valid %':>10}")
    print("  " + "-" * 51)

    def bucket(label, mask):
        if mask.sum() < 20:
            return
        print(f"  {label:<34}{mask.sum():>7}{100 * v_ok[mask].mean():>10.1f}")

    for lo, hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5),
                   (0.5, 0.8), (0.8, 99)]:
        bucket(f"speed {lo:.2f}-{hi:.2f} m/s", (spd >= lo) & (spd < hi))
    print()
    for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 12), (12, 999)]:
        bucket(f"yaw rate {lo}-{hi} deg/s", (yrate >= lo) & (yrate < hi))
    print()
    for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 99)]:
        bucket(f"baseline {100*lo:.0f}-{100*hi:.0f} cm", (base >= lo) & (base < hi))
    print()
    for lo, hi in [(0, 0.08), (0.08, 0.12), (0.12, 99)]:
        bucket(f"frame gap {1000*lo:.0f}-{1000*hi:.0f} ms", (dt >= lo) & (dt < hi))

    print("\n  A flat 'valid %' column means that variable does not drive")
    print("  tracking failure. A monotonic fall means it does.")

    # ================================================== 3. MAP POINTS
    section("3. MAP POINTS AND KEYPOINTS")
    print(f"  keypoints:  median {np.median(nkp):.0f}")
    print(f"  map points: median {np.median(nmp):.0f}, "
          f"p10 {np.percentile(nmp, 10):.0f}, p90 {np.percentile(nmp, 90):.0f}")
    print(f"  map points when   valid: median {np.median(nmp[valid]):.0f}")
    if (~valid).sum():
        print(f"  map points when invalid: median {np.median(nmp[~valid]):.0f}")
    print(f"  ratio map_points/keypoints when valid: "
          f"{np.median(nmp[valid] / np.maximum(nkp[valid], 1)):.3f}")
    print("\n  ORB-SLAM3 indoors typically holds 150-300 map points. Well under")
    print("  100 means features are found but not triangulating: too little")
    print("  texture, too little parallax, or both.")

    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 9999)]:
        m = (nmp >= lo) & (nmp < hi)
        if m.sum() > 20:
            print(f"    map points {lo:>4}-{hi:<5} n={m.sum():>6}  "
                  f"valid {100 * valid[m].mean():5.1f} %")

    # ================================================== 4. EPOCHS
    section("4. EPOCH CHANGES AND RECOVERY")
    ch = np.where(np.diff(epoch) != 0)[0]
    print(f"  {len(ch)} epoch changes in {tv[-1] - tv[0]:.0f} s "
          f"(one per {(tv[-1] - tv[0]) / max(len(ch), 1):.0f} s)")
    print(f"\n  {'t (s)':>8}{'epoch':>8}{'valid before':>14}{'recover s':>12}")
    for i in ch:
        before = valid[max(0, i - 50):i + 1].mean()
        after = valid[i + 1:i + 400]
        ta = tv[i + 1:i + 400]
        rec = float('nan')
        if after.any():
            j = np.argmax(after)
            rec = ta[j] - tv[i + 1]
        print(f"  {tv[i + 1] - t0:>8.1f}{epoch[i + 1]:>8}"
              f"{100 * before:>13.0f}%{rec:>12.1f}")

    # ================================================== 5. HEALTHY PERIODS
    section("5. WHAT THE AIRCRAFT WAS DOING")
    runs, cur = [], None
    for k in range(len(tvv) - 1):
        if v_ok[k] and cur is None:
            cur = k
        elif not v_ok[k] and cur is not None:
            if tvv[k] - tvv[cur] > 3.0:
                runs.append((cur, k))
            cur = None
    if cur is not None and tvv[-1] - tvv[cur] > 3.0:
        runs.append((cur, len(tvv) - 1))

    print(f"  {len(runs)} continuous valid runs longer than 3 s")
    if runs:
        durs = [tvv[b] - tvv[a_] for a_, b in runs]
        print(f"  longest {max(durs):.0f} s, median {np.median(durs):.0f} s, "
              f"total {sum(durs):.0f} s of {tvv[-1] - tvv[0]:.0f} s "
              f"({100 * sum(durs) / (tvv[-1] - tvv[0]):.0f} %)")
        print(f"\n  {'start s':>9}{'dur s':>8}{'mean speed':>12}"
              f"{'mean |yawrate|':>16}{'map pts':>9}")
        for a_, b in sorted(runs, key=lambda r: tvv[r[0]])[:15]:
            m = slice(a_, b)
            nm = nmp[sel][a_:b]
            print(f"  {tvv[a_] - t0:>9.1f}{tvv[b] - tvv[a_]:>8.1f}"
                  f"{spd[m].mean():>12.3f}{yrate[m].mean():>16.2f}"
                  f"{np.median(nm):>9.0f}")

    section("VERDICT")
    print("  Read section 2 first. If 'valid %' falls with speed or with")
    print("  baseline, the hypothesis is confirmed and the fix is to fly")
    print("  slower or raise the frame rate. If it is flat across all four")
    print("  buckets, motion is not the cause and section 3 is where to look:")
    print("  a median map point count under 100 points at scene texture.")


if __name__ == '__main__':
    main()