#!/usr/bin/env python3
"""Predicted / reported / realised scale uncertainty, per flight.

    python3 forecast_validation.py --bag checkerboard_01 --mocap checkerboard_01_mocap
    python3 forecast_validation.py --summary fv_*.json
"""

import argparse
import glob
import json
import sys

import numpy as np

TOPIC_STATUS = "/drone_1/localisation/status"
TOPIC_POSE = "/drone_1/localisation/pose"
TOPIC_VO = "/drone_1/vo/status"
TOPIC_SPEED = "/drone_1/speed_vector"
TOPIC_DRONE = "/optitrack/rigid_bodies/dji_mini4"

WIN_S = 1.0            # window for the scale fit
MIN_MOVE_M = 0.05      # [M] envelope table: windows carrying >= 5 cm


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


def clock_offset(t_a, v_a, t_b, v_b, max_lag=3.0, rate=20.0):
    """Speed-profile cross-correlation. Returns lag such that t_b + lag ~ t_a."""
    from scipy.signal import correlate
    t0, t1 = max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])
    if t1 - t0 < 5:
        return 0.0, 0.0
    g = np.arange(t0, t1, 1.0 / rate)
    a = np.interp(g, t_a, v_a)
    b = np.interp(g, t_b, v_b)
    a = a - a.mean()
    b = b - b.mean()
    c = correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a)) / rate
    k = np.abs(lags) <= max_lag
    lag = float(lags[k][np.argmax(c[k])])
    peak = float(c[k].max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return lag, peak


def run(args):
    from scipy.signal import savgol_filter
    d = read_bag(args.bag, [TOPIC_STATUS, TOPIC_POSE, TOPIC_VO, TOPIC_SPEED])
    m = read_bag(args.mocap, [TOPIC_DRONE])
    if not d[TOPIC_STATUS] or not d[TOPIC_POSE]:
        sys.exit(f"{args.bag} has no localisation/status or pose")
    if not m[TOPIC_DRONE]:
        sys.exit(f"{args.mocap} has no drone rigid body")

    stt = np.array([st(x) for x in d[TOPIC_STATUS]])
    s_s = np.array([x.scale for x in d[TOPIC_STATUS]])
    s_sig = np.array([x.sigma_scale for x in d[TOPIC_STATUS]])
    flags = [";".join(x.flags) for x in d[TOPIC_STATUS]]

    pt = np.array([st(x) for x in d[TOPIC_POSE]])
    pp = np.array([[x.pose.pose.position.x, x.pose.pose.position.y,
                    x.pose.pose.position.z] for x in d[TOPIC_POSE]])

    vt = np.array([st(x) for x in d[TOPIC_VO]])
    vep = np.array([int(x.vo_epoch) for x in d[TOPIC_VO]])

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

    # ------------------------------------------------- realised scale
    epochs = np.interp(pt, vt, vep).round().astype(int) if len(vt) else \
        np.zeros(len(pt), int)
    rows = []
    t0 = pt[0]
    while t0 < pt[-1] - WIN_S:
        t1 = t0 + WIN_S
        k = (pt >= t0) & (pt <= t1)
        if k.sum() >= 5:
            de = float(np.linalg.norm(pp[k][-1] - pp[k][0]))
            a = np.interp([t0 + lag, t1 + lag], g, ms[:, 0])
            b = np.interp([t0 + lag, t1 + lag], g, ms[:, 1])
            c = np.interp([t0 + lag, t1 + lag], g, ms[:, 2])
            dm = float(np.linalg.norm([a[1] - a[0], b[1] - b[0], c[1] - c[0]]))
            if dm >= MIN_MOVE_M and de > 1e-6:
                rows.append((t0, int(np.median(epochs[k])), dm / de,
                             abs(de - dm) / max(dm, 1e-9)))
        t0 += WIN_S
    if not rows:
        sys.exit("no windows carried enough motion")
    R = np.array([[r[0], r[1], r[2], r[3]] for r in rows])

    # ------------------------------------------------- predicted
    pred = {}
    try:
        sys.path.insert(0, args.pkg)
        from drone_navigation.planning.forecast import Forecast, ForecastConfig
        from drone_localisation.ekf.params import EkfParams
        P = EkfParams()
        for ep in sorted(set(R[:, 1].astype(int))):
            k = R[:, 1] == ep
            t_lo, t_hi = R[k, 0].min(), R[k, 0].max() + WIN_S
            sub = (g >= t_lo + lag) & (g <= t_hi + lag)
            if sub.sum() < 50:
                continue
            s0 = float(np.median(s_s[(stt >= t_lo) & (stt <= t_hi)]))
            f = Forecast(P, ForecastConfig(s_true=max(s0, 1e-3), dwell_s=0.0))
            f.reset(sigma_s0=0.10 * max(s0, 1e-3))
            gg = g[sub]
            vv = np.gradient(ms[sub], gg, axis=0)
            for i in range(1, len(gg)):
                f.step(vv[i], float(gg[i] - gg[i - 1]))
            pred[int(ep)] = f.sigma_s_ratio
    except Exception as e:
        print(f"  predicted column unavailable: {e}")

    # ------------------------------------------------- report
    out = dict(bag=args.bag, clock_lag=lag, clock_peak=peak, epochs={})
    print(f"  {'epoch':>5} {'n':>4} {'predicted':>10} {'reported':>9} "
          f"{'realised':>9} {'|1-scale|':>10} {'flags':>14}")
    for ep in sorted(set(R[:, 1].astype(int))):
        k = R[:, 1] == ep
        sc = float(np.median(R[k, 2]))
        t_lo, t_hi = R[k, 0].min(), R[k, 0].max() + WIN_S
        ks = (stt >= t_lo) & (stt <= t_hi)
        rep = (float(np.median(s_sig[ks] / np.maximum(s_s[ks], 1e-9)))
               if ks.any() else float("nan"))
        fl = set()
        for f_ in np.array(flags)[ks]:
            fl.update(x.split("=")[0] for x in f_.split(";") if x)
        row = dict(n_windows=int(k.sum()), predicted=pred.get(int(ep)),
                   reported=rep, realised_scale=sc,
                   realised_error=abs(1.0 - sc),
                   rpe_median=float(np.median(R[k, 3])),
                   flags=sorted(fl))
        out["epochs"][int(ep)] = row
        pv = f"{row['predicted']:.4f}" if row["predicted"] else "   --"
        print(f"  {ep:5d} {int(k.sum()):4d} {pv:>10} {rep:9.4f} "
              f"{sc:9.4f} {abs(1 - sc):10.4f} {','.join(sorted(fl))[:14]:>14}")

    with open(args.out or f"fv_{args.bag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote fv_{args.bag}.json")
    
BAD_FLAGS = ("S_CLAMPED", "SIGMA_S_MAX")
MIN_WINDOWS = 10
MAX_RPE = 0.30


def usable(r):
    return (r["n_windows"] >= MIN_WINDOWS
            and not any(f in BAD_FLAGS for f in r["flags"])
            and r["rpe_median"] < MAX_RPE)


def summary(paths):
    files = []
    for p in paths:
        files += glob.glob(p)
    rows = []
    for p in sorted(files):
        with open(p) as f:
            d = json.load(f)
        for ep, r in d["epochs"].items():
            rows.append((d["bag"], int(ep), r))
    keep = [(b, e, r) for b, e, r in rows if usable(r)]
    drop = [(b, e, r) for b, e, r in rows if not usable(r)]
    print(f"{len(keep)} of {len(rows)} epochs usable\n")
    print(f"{'bag':<18}{'ep':>3}{'n':>5}{'pred':>8}{'rep':>8}{'scale':>8}"
          f"{'|1-s|':>8}{'RPE':>7}{'r/real':>8}{'p/real':>8}")
    for bag, ep, r in keep:
        pv = f"{r['predicted']:.4f}" if r["predicted"] else "  --"
        e = r["realised_error"]
        print(f"{bag:<18}{ep:>3}{r['n_windows']:>5}{pv:>8}"
              f"{r['reported']:>8.4f}{r['realised_scale']:>8.4f}"
              f"{e:>8.4f}{r['rpe_median']:>7.3f}"
              f"{e / max(r['reported'], 1e-9):>8.2f}"
              f"{e / max(r['predicted'] or 1e-9, 1e-9):>8.2f}")
    if drop:
        print(f"\nexcluded ({len(drop)}):")
        for bag, ep, r in drop:
            why = []
            if r["n_windows"] < MIN_WINDOWS:
                why.append(f"n={r['n_windows']}")
            bad = [f for f in r["flags"] if f in BAD_FLAGS]
            if bad:
                why.append(",".join(bad))
            if r["rpe_median"] >= MAX_RPE:
                why.append(f"RPE={r['rpe_median']:.2f}")
            print(f"  {bag} ep {ep}: {'; '.join(why)}")

    if not keep:
        print("\nno usable epochs")
        return
    pr = np.array([r["predicted"] for _, _, r in keep
                   if r["predicted"] is not None])
    rp = np.array([r["reported"] for _, _, r in keep])
    re = np.array([r["realised_error"] for _, _, r in keep])
    rpe = np.array([r["rpe_median"] for _, _, r in keep])
    print(f"\npooled over {len(keep)} epochs")
    if len(pr):
        print(f"  predicted sigma_s/s    median {np.median(pr):.4f}  "
              f"range {pr.min():.4f}-{pr.max():.4f}")
    print(f"  reported  sigma_s/s    median {np.median(rp):.4f}  "
          f"range {rp.min():.4f}-{rp.max():.4f}")
    print(f"  realised  |1 - scale|  median {np.median(re):.4f}  "
          f"range {re.min():.4f}-{re.max():.4f}")
    if len(pr):
        f = re / pr
        print(f"  optimism, realised/predicted  median {np.median(f):.2f}  "
              f"range {f.min():.2f}-{f.max():.2f}")
    f2 = re / rp
    print(f"  optimism, realised/reported   median {np.median(f2):.2f}  "
          f"range {f2.min():.2f}-{f2.max():.2f}")
    print(f"  RPE 1 s                median {np.median(rpe):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag")
    ap.add_argument("--mocap")
    ap.add_argument("--out")
    ap.add_argument("--pkg", default="/ros2_ws/src/drone_navigation")
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
