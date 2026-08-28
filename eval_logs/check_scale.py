#!/usr/bin/env python3
"""
check_scale.py -- what SHOULD `s` be?

    python3 check_scale.py F9_02 F9_02_mocap

`s` converts VO units to metres. It is directly measurable from the bags with
no filter involved: for each consecutive vo/pose pair, compare how far mocap
says the drone actually moved against how far VO says it moved.

    s = true_distance / vo_distance

This separates two very different failures:

  * median lands near a stable value, but the filter picked something else
    -> the init estimator is choosing bad samples (a node bug)
  * median is unstable or scattered over decades
    -> VO scale is not consistent on this flight (upstream, not the filter)

The filter reported s = 4.46 on F9_02, while Umeyama against ground truth says
it should have been ~0.73.
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

DT_MIN, DT_MAX = 0.010, 0.150      # must match params.py
V_LOW = 0.40                       # must match params.py


def read(bag, hint):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    topic = next((n for n in types if hint in n), None)
    if topic is None:
        raise SystemExit(f"no topic matching '{hint}' in {bag}: {list(types)}")
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
    keep = np.concatenate([[True], np.diff(T) > 1e-6])
    print(f"  {Path(bag).name}: {topic}  {keep.sum()} samples")
    return T[keep], P[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag", help="bag containing /drone_1/vo/pose")
    ap.add_argument("mocap_bag")
    ap.add_argument("--out", default="./scale_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("reading:")
    t_v, p_v = read(args.flight_bag, "vo/pose")
    t_m, p_m = read(args.mocap_bag, "rigid_bodies")

    lo, hi = max(t_v[0], t_m[0]), min(t_v[-1], t_m[-1])
    m = (t_v >= lo) & (t_v <= hi)
    t_v, p_v = t_v[m], p_v[m]
    print(f"\noverlap {hi - lo:.1f} s, {len(t_v)} VO poses")

    # mocap interpolated onto VO stamps -- no differentiation of mocap, so no
    # native-rate noise amplification (filter_design.md 10)
    p_g = np.column_stack([np.interp(t_v, t_m, p_m[:, i]) for i in range(3)])

    rows = []
    for k in range(1, len(t_v)):
        dt = t_v[k] - t_v[k - 1]
        if not (DT_MIN <= dt <= DT_MAX):
            continue
        d_vo = float(np.linalg.norm(p_v[k] - p_v[k - 1]))
        d_gt = float(np.linalg.norm(p_g[k] - p_g[k - 1]))
        if d_vo < 1e-6:
            continue
        rows.append((t_v[k], d_gt / d_vo, d_gt / dt, d_vo / dt))

    if len(rows) < 50:
        raise SystemExit("too few usable increments")

    t = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    v_true = np.array([r[2] for r in rows])
    v_vo = np.array([r[3] for r in rows])
    t0 = t[0]

    def report(label, sel):
        if sel.sum() < 20:
            print(f"  {label:22s} too few samples")
            return
        x = s[sel]
        print(f"  {label:22s} n={sel.sum():5d}  median {np.median(x):7.4f}  "
              f"IQR [{np.percentile(x, 25):.4f}, {np.percentile(x, 75):.4f}]  "
              f"p5-p95 [{np.percentile(x, 5):.4f}, {np.percentile(x, 95):.4f}]")

    print(f"\n--- s = true_distance / vo_distance ---")
    report("all increments", np.ones(len(s), bool))
    report(f"moving (>{V_LOW} m/s)", v_true > V_LOW)
    report("slow (0.1-0.4 m/s)", (v_true > 0.1) & (v_true <= V_LOW))
    report("near-stationary", v_true <= 0.1)

    print(f"\n  filter chose s = 4.4615")
    print(f"  Umeyama implies  ~0.73")

    # ---- does it drift over the flight? -----------------------------------
    print(f"\n--- s over the flight (moving increments only) ---")
    mv = v_true > V_LOW
    if mv.sum() > 100:
        edges = np.linspace(t[0], t[-1], 11)
        print(f"  {'window (s)':>14} {'n':>5} {'median s':>10}")
        for a, b in zip(edges[:-1], edges[1:]):
            w = mv & (t >= a) & (t < b)
            if w.sum() >= 10:
                print(f"  {a - t0:6.0f}-{b - t0:6.0f} {w.sum():5d} "
                      f"{np.median(s[w]):10.4f}")

    # ---- what the init estimator would have picked -------------------------
    first = np.where(v_true > V_LOW)[0][:20]
    if len(first) == 20:
        print(f"\n  first 20 moving increments -> median s = "
              f"{np.median(s[first]):.4f}  (t = {t[first[0]] - t0:.1f}"
              f"-{t[first[-1]] - t0:.1f} s)")

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].semilogy(t - t0, s, ".", ms=2, alpha=.4)
    ax[0].axhline(4.4615, color="r", ls="--", lw=1, label="filter s = 4.46")
    ax[0].axhline(0.73, color="g", ls="--", lw=1, label="Umeyama ~0.73")
    ax[0].set_ylabel("s per increment"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(t - t0, v_true, lw=1, label="mocap speed")
    ax[1].axhline(V_LOW, color="k", ls=":", lw=.8)
    ax[1].set_ylabel("true speed (m/s)"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(t - t0, v_vo, lw=1, label="VO speed (unscaled)")
    ax[2].set_ylabel("VO speed (vo units/s)"); ax[2].set_xlabel("s")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle(f"{Path(args.flight_bag).name} — VO scale from ground truth")
    fig.savefig(out / "scale.png", dpi=110, bbox_inches="tight")
    print(f"\nplot: {out / 'scale.png'}")


if __name__ == "__main__":
    main()
    