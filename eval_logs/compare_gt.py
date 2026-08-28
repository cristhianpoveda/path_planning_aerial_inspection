#!/usr/bin/env python3
"""
compare_gt.py -- ATE / RPE of localisation/pose against OptiTrack.

    python3 compare_gt.py F9_02_est F9_02_mocap --out ./gt_out

Why Umeyama rather than a hardcoded transform: the `odom` origin is wherever
the drone was when the filter INITIALISED (filter_design.md 9 sets p = 0
there), not at takeoff, and its yaw datum is DJI attitude at that instant.
Neither is knowable in advance, so the transform is estimated from the
trajectories.

Two alignments, and the difference between them is the point:

  * WITH scale    -> the recovered factor IS the scale error. This is the
                     number NIS cannot give you: filter_design.md 10 warns
                     that a wrong scale converges self-consistently.
  * WITHOUT scale -> ATE. Letting the alignment absorb scale would hide the
                     quantity of interest.

Positions only. The mocap conjugate-orientation bug never touched position, so
none of that applies here.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_positions(bag, topic_hint):
    """Return (t, p) using HEADER stamps -- both bags share the original
    recording clock, so header stamps are directly comparable."""
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    topic = next((n for n in types if topic_hint in n), None)
    if topic is None:
        raise SystemExit(f"no topic matching '{topic_hint}' in {bag}\n"
                         f"  available: {list(types)}")
    print(f"  {bag}: {topic}  ({types[topic]})")
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    T, P = [], []
    while r.has_next():
        _, data, t_rec = r.read_next()
        m = deserialize_message(data, get_message(types[topic]))
        st = m.header.stamp
        t = st.sec + st.nanosec * 1e-9
        if t <= 0.0:
            t = t_rec * 1e-9
        pos = m.pose.pose.position if hasattr(m.pose, "pose") else m.pose.position
        T.append(t)
        P.append([pos.x, pos.y, pos.z])
    T, P = np.array(T), np.array(P, float)
    o = np.argsort(T)
    T, P = T[o], P[o]
    keep = np.concatenate([[True], np.diff(T) > 1e-6])   # domain-0 duplicates
    return T[keep], P[keep]


def umeyama(src, dst, with_scale):
    """Least-squares similarity dst ~ c*R*src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    c = (np.trace(np.diag(sig) @ W) / (S ** 2).sum() * len(src)) if with_scale else 1.0
    t = mu_d - c * R @ mu_s
    return c, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("est_bag")
    ap.add_argument("mocap_bag")
    ap.add_argument("--out", default="./gt_out")
    ap.add_argument("--max-offset", type=float, default=0.5)
    ap.add_argument("--topic", default="localisation/pose", help="topic hint for the estimate bag")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("reading:")
    t_e, p_e = read_positions(args.est_bag, args.topic)
    t_m, p_m = read_positions(args.mocap_bag, "rigid_bodies")
    print(f"\nest  : {len(t_e)} poses, {t_e[-1] - t_e[0]:.1f} s")
    print(f"mocap: {len(t_m)} poses, {t_m[-1] - t_m[0]:.1f} s, "
          f"{1.0 / np.median(np.diff(t_m)):.0f} Hz")

    # ---- time offset: scan for the one minimising rigid ATE ----------------
    # The two recorders share a clock but not a latency; a residual offset of
    # tens of ms was measured across all bags (mocap_clock_lag_s).
    best = None
    for dt in np.arange(-args.max_offset, args.max_offset + 1e-9, 0.005):
        lo = max(t_e[0], t_m[0] + dt)
        hi = min(t_e[-1], t_m[-1] + dt)
        m = (t_e >= lo) & (t_e <= hi)
        if m.sum() < 100:
            continue
        g = np.column_stack([np.interp(t_e[m] - dt, t_m, p_m[:, i])
                             for i in range(3)])
        _, R, tr = umeyama(p_e[m], g, with_scale=False)
        e = np.linalg.norm((R @ p_e[m].T).T + tr - g, axis=1)
        rms = float(np.sqrt((e ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, dt, m, g)
    if best is None:
        raise SystemExit("no overlapping window -- check the bags cover the "
                         "same flight")
    _, dt_best, mask, gt = best
    est = p_e[mask]
    t = t_e[mask]
    print(f"\ntime offset: {dt_best * 1e3:+.0f} ms   "
          f"overlap {t[-1] - t[0]:.1f} s, {len(t)} samples")

    # ---- alignment WITH scale: this is the scale error --------------------
    c, Rs, ts = umeyama(est, gt, with_scale=True)
    print(f"\n--- scale ---")
    print(f"  Umeyama scale factor : {c:.4f}   ({100 * (c - 1):+.1f}%)")
    if abs(c - 1) < 0.05:
        print("  => scale is correct to within 5%. sigma_scale is honest.")
    else:
        print("  => the filter's `s` is wrong by this factor. NIS could not "
              "have revealed it (10).")

    # ---- alignment WITHOUT scale: ATE -------------------------------------
    _, R, tr = umeyama(est, gt, with_scale=False)
    al_rigid = (R @ est.T).T + tr
    al_sim = c * (Rs @ est.T).T + ts
    err = al_rigid - gt          # ATE: scale NOT absorbed
    err_sim = al_sim - gt        # shape only
    ate = np.linalg.norm(err, axis=1)
    ate_sim = np.linalg.norm(err_sim, axis=1)
    print(f"\n--- ATE (rigid alignment, scale NOT absorbed) ---")
    print(f"  RMSE {np.sqrt((ate ** 2).mean()):.3f} m   "
          f"median {np.median(ate):.3f}   p95 {np.percentile(ate, 95):.3f}   "
          f"max {ate.max():.3f}")
    for i, ax_ in enumerate("xyz"):
        print(f"    {ax_}: rmse {np.sqrt((err[:, i] ** 2).mean()):.3f} m   "
              f"bias {err[:, i].mean():+.3f}")
    print(f"\n--- shape only (scale absorbed) ---")
    print(f"  RMSE {np.sqrt((ate_sim ** 2).mean()):.3f} m   "
          f"median {np.median(ate_sim):.3f}   "
          f"p95 {np.percentile(ate_sim, 95):.3f}")
    for i, ax_ in enumerate("xyz"):
        print(f"    {ax_}: rmse {np.sqrt((err_sim[:, i] ** 2).mean()):.3f} m")
    print("  (a large z bias with small x/y is the p_z vs b split -- 5.2)")

    # ---- RPE over short windows (7 prefers this to ATE) -------------------
    print(f"\n--- RPE (short windows; 7 prefers this to ATE) ---")
    for win in (1.0, 2.0, 5.0):
        d_e, d_g = [], []
        j = 0
        for i in range(len(t)):
            while j < len(t) and t[j] - t[i] < win:
                j += 1
            if j >= len(t):
                break
            d_e.append(c * np.linalg.norm(est[j] - est[i]))
            d_g.append(np.linalg.norm(gt[j] - gt[i]))
        if len(d_e) < 20:
            continue
        d_e, d_g = np.array(d_e), np.array(d_g)
        rel = np.abs(d_e - d_g)
        moving = d_g > 0.05
        print(f"  {win:.0f} s: RMSE {np.sqrt((rel ** 2).mean()):.3f} m"
              + (f"   relative {100 * np.median(rel[moving] / d_g[moving]):.1f}%"
                 if moving.sum() > 10 else ""))

    # ---- plots -------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    ax[0].plot(gt[:, 0], gt[:, 1], lw=1, label="mocap")
    ax[0].plot(al_sim[:, 0], al_sim[:, 1], lw=1, label="EKF (aligned)")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    for i, ax_ in enumerate("xyz"):
        ax[1].plot(t - t[0], err[:, i], lw=1, label=ax_)
    ax[1].set_ylabel("error (m)"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(t - t[0], gt[:, 2], lw=1, label="mocap z")
    ax[2].plot(t - t[0], al_sim[:, 2], lw=1, label="EKF z")
    ax[2].set_ylabel("altitude (m)"); ax[2].set_xlabel("s")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle(f"{Path(args.est_bag).name} vs OptiTrack")
    fig.savefig(out / "gt_compare.png", dpi=110, bbox_inches="tight")
    print(f"\nplot: {out / 'gt_compare.png'}")


if __name__ == "__main__":
    main()
    