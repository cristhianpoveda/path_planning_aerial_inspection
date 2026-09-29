#!/usr/bin/env python3
"""Position error decomposition: range component and accumulated component.

    python3 position_error.py --bag checkerboard_02 --mocap checkerboard_02_mocap
    python3 position_error.py --summary pe_*.json
"""

import argparse
import glob
import json
import sys

import numpy as np

TOPIC_POSE = "/drone_1/localisation/pose"
TOPIC_VO = "/drone_1/vo/status"
TOPIC_STATUS = "/drone_1/localisation/status"
TOPIC_SPEED = "/drone_1/speed_vector"
TOPIC_DRONE = "/optitrack/rigid_bodies/dji_mini4"

MIN_EPOCH_S = 20.0
MIN_PATH_M = 1.0


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
            out[topic].append(rclpy.serialization.deserialize_message(
                data, get_message(types[topic])))
        except Exception:
            pass
    return out


def st(m):
    s = m.header.stamp
    return s.sec + s.nanosec * 1e-9


def umeyama_rigid(src, dst):
    """Best rigid transform src -> dst, no scale.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(A)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def run(args):
    from scipy.signal import savgol_filter
    from forecast_validation import clock_offset

    d = read_bag(args.bag, [TOPIC_POSE, TOPIC_VO, TOPIC_STATUS, TOPIC_SPEED])
    m = read_bag(args.mocap, [TOPIC_DRONE])
    if not d[TOPIC_POSE] or not m[TOPIC_DRONE]:
        sys.exit("missing pose or mocap")

    pt = np.array([st(x) for x in d[TOPIC_POSE]])
    pp = np.array([[x.pose.pose.position.x, x.pose.pose.position.y,
                    x.pose.pose.position.z] for x in d[TOPIC_POSE]])
    vt = np.array([st(x) for x in d[TOPIC_VO]])
    vep = np.array([int(x.vo_epoch) for x in d[TOPIC_VO]])
    stt = np.array([st(x) for x in d[TOPIC_STATUS]])
    flags = [";".join(x.flags) for x in d[TOPIC_STATUS]]
    spt = np.array([st(x) for x in d[TOPIC_SPEED]])
    spv = np.array([np.linalg.norm([x.vector.x, x.vector.y, x.vector.z])
                    for x in d[TOPIC_SPEED]])

    mt = np.array([st(x) for x in m[TOPIC_DRONE]])
    mp = np.array([[x.pose.position.x, x.pose.position.y, x.pose.position.z]
                   for x in m[TOPIC_DRONE]])
    g = np.arange(mt[0], mt[-1], 0.01)
    ms = np.column_stack([savgol_filter(np.interp(g, mt, mp[:, i]), 21, 2)
                          for i in range(3)])
    mv = np.linalg.norm(np.gradient(ms, g, axis=0), axis=1)
    lag, peak = clock_offset(spt, spv, g, mv)
    print(f"{args.bag}: clock offset {lag:+.3f} s (peak {peak:.3f})")

    epochs = (np.interp(pt, vt, vep).round().astype(int) if len(vt)
              else np.zeros(len(pt), int))
    out = dict(bag=args.bag, clock_lag=lag, epochs={})
    print(f"  {'ep':>3} {'dur':>6} {'path':>7} {'e_s':>7} {'total':>8} "
          f"{'range':>8} {'accum':>8} {'per m':>8}  flags")
    for ep in sorted(set(epochs)):
        k = epochs == ep
        if k.sum() < 20:
            continue
        t = pt[k]
        if t[-1] - t[0] < MIN_EPOCH_S:
            continue
        est = pp[k]
        truth = np.column_stack([np.interp(t + lag, g, ms[:, i])
                                 for i in range(3)])
        path = float(np.sum(np.linalg.norm(np.diff(truth, axis=0), axis=1)))
        if path < MIN_PATH_M:
            continue

        n0 = max(12, int(0.03 * len(t)))
        R, tr = umeyama_rigid(est[:n0], truth[:n0])
        aligned = (R @ est.T).T + tr
        err = aligned - truth

        total = np.linalg.norm(err, axis=1)
        horiz = np.linalg.norm(err[:, :2], axis=1)
        L = np.linalg.norm(aligned - aligned[0], axis=1)

        # scale error over 1 s windows carrying >= 5 cm of true motion
        ratios = []
        w0 = t[0]
        while w0 < t[-1] - 1.0:
            kk = (t >= w0) & (t <= w0 + 1.0)
            if kk.sum() >= 5:
                de = float(np.linalg.norm(aligned[kk][-1] - aligned[kk][0]))
                dm = float(np.linalg.norm(truth[kk][-1] - truth[kk][0]))
                if dm >= 0.05 and de > 1e-6:
                    ratios.append(dm / de)
            w0 += 1.0
        e_scale = abs(1.0 - float(np.median(ratios))) if ratios else float("nan")
        rng_c = e_scale * L
        accum = np.sqrt(np.maximum(total ** 2 - rng_c ** 2, 0.0))
        radial = rng_c

        ks = (stt >= t[0]) & (stt <= t[-1])
        fl = set()
        for f_ in np.array(flags)[ks]:
            fl.update(x.split("=")[0] for x in f_.split(";") if x)

        row = dict(dur_s=float(t[-1] - t[0]), path_m=path,
                   scale_error=e_scale, L_max_m=float(L.max()),
                   total_final_m=float(total[-1]),
                   total_max_m=float(total.max()),
                   range_m=float(np.median(radial)),
                   range_final_m=float(radial[-1]),
                   accum_m=float(np.median(accum)),
                   accum_final_m=float(accum[-1]),
                   accum_per_m=float(accum[-1] / path),
                   accum_per_s=float(accum[-1] / (t[-1] - t[0])),
                   horiz_final_m=float(horiz[-1]),
                   flags=sorted(fl))
        out["epochs"][int(ep)] = row
        print(f"  {ep:3d} {row['dur_s']:6.1f} {path:7.2f} {e_scale:7.4f} "
              f"{row['total_final_m']:8.3f} {row['range_final_m']:8.3f} "
              f"{row['accum_final_m']:8.3f} {row['accum_per_m']:8.4f}  "
              f"{','.join(sorted(fl))[:26]}")

    with open(args.out or f"pe_{args.bag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote pe_{args.bag}.json")


def summary(paths):
    BAD = ("S_CLAMPED", "SIGMA_S_MAX")
    files = []
    for p in paths:
        files += glob.glob(p)
    keep, drop = [], []
    for p in sorted(files):
        with open(p) as f:
            d = json.load(f)
        for ep, r in d["epochs"].items():
            (drop if any(x in BAD for x in r["flags"]) else keep).append(
                (d["bag"], int(ep), r))
    print(f"{len(keep)} usable epochs, {len(drop)} excluded\n")
    print(f"{'bag':<18}{'ep':>3}{'dur':>7}{'path':>7}{'e_s':>8}{'total':>8}"
          f"{'range':>8}{'accum':>8}{'per m':>9}")
    for b, e, r in keep:
        print(f"{b:<18}{e:>3}{r['dur_s']:>7.1f}{r['path_m']:>7.2f}"
              f"{r['scale_error']:>8.4f}{r['total_final_m']:>8.3f}"
              f"{r['range_final_m']:>8.3f}{r['accum_final_m']:>8.3f}"
              f"{r['accum_per_m']:>9.4f}")
    if drop:
        print("\nexcluded:")
        for b, e, r in drop:
            print(f"  {b} ep {e}: {','.join(r['flags'])}")
    if not keep:
        return
    rg = np.array([r["range_final_m"] for _, _, r in keep])
    ac = np.array([r["accum_final_m"] for _, _, r in keep])
    pm = np.array([r["accum_per_m"] for _, _, r in keep])
    ps = np.array([r["accum_per_s"] for _, _, r in keep])
    print(f"\npooled over {len(keep)} epochs")
    print(f"  range component      median {np.median(rg):.3f} m  "
          f"range {rg.min():.3f}-{rg.max():.3f}")
    print(f"  accumulated          median {np.median(ac):.3f} m  "
          f"range {ac.min():.3f}-{ac.max():.3f}")
    print(f"  accumulated per m    median {np.median(pm):.4f} m/m")
    print(f"  accumulated per s    median {np.median(ps):.4f} m/s")
    print(f"  accumulated / range  median {np.median(ac) / max(np.median(rg), 1e-9):.2f}")
    print("\n  The larger of the two is the dominant term in the total and")
    print("  identifies which limitation binds: range is the one the")
    print("  objective acts on, accumulated is the one no planner reaches.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag")
    ap.add_argument("--mocap")
    ap.add_argument("--out")
    ap.add_argument("--summary", nargs="+")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    elif a.bag and a.mocap:
        run(a)
    else:
        ap.error("give --bag with --mocap, or --summary")


if __name__ == "__main__":
    main()
