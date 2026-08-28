#!/usr/bin/env python3
"""
check_g2.py — analyse the G2 yaw-reference bag.  (rev 3)

CONFIRMED (rev 2, section E): the mocap driver publishes the CONJUGATE
orientation, i.e. R_body_world where ROS expects R_world_body. Applying R.T
gives slope -0.00 and sd 0.63 deg, against slope +2.00 and sd 142 deg
as-published. Position is unaffected. This script now applies that transform by
default (--convention inverted) and still prints section E as a check.

Sections:
  A. mocap stream health (dedupe, gaps)
  B. static heading holds -- detected from DJI yaw RANGE, not from a derivative
  C. discontinuities
  D. DJI heading vs DJI attitude yaw
  E. orientation-convention verification
  F. mocap<->nav tilt fit, and the sigma_rp that remains after removing it

Usage:
    python3 check_g2.py G2_yaw_reference --out ./g2_out
    python3 check_g2.py G2_yaw_reference --convention as-published
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from source_characterisation import (
    read_bag, quat_to_R, R_to_rpy, rpy_to_R, wrap, resample,
    effective_rate, sync_mocap,
)

HEADING_TOPIC = "/drone_1/heading"
HOLD_WIN_S = 2.0           # sliding window for the "is it holding?" test
HOLD_RANGE_DEG = 3.0       # max DJI yaw excursion within the window
HOLD_MIN_S = 3.0           # minimum hold length to report
JUMP_DEG = 45.0
DUP_DT = 1e-3
GAP_ABS_S = 0.100          # absolute gap threshold; the 3x-median test is far
                           # too sensitive at 145 Hz with a mixed dt distribution
RATE_BASELINE_S = 0.20     # finite-difference baseline for the DISPLAYED rate

C_YUP = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])

CONVENTIONS = {
    "as-published": lambda R: R,
    "inverted":     lambda R: R.T,
    "yup":          lambda R: C_YUP @ R @ C_YUP.T,
    "yup-inverted": lambda R: (C_YUP @ R @ C_YUP.T).T,
}


def circ_mean_std(a):
    s, c = np.mean(np.sin(a)), np.mean(np.cos(a))
    m = np.arctan2(s, c)
    R = np.hypot(s, c)
    return m, np.sqrt(-2.0 * np.log(max(R, 1e-12)))


def sign_align(Q):
    Q = np.asarray(Q, float).copy()
    if len(Q) < 2:
        return Q
    flip = np.cumprod(np.where(np.sum(Q[1:] * Q[:-1], axis=1) < 0, -1.0, 1.0))
    Q[1:] *= flip[:, None]
    return Q


def read_heading(bagpath):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=str(bagpath), storage_id=""),
               rosbag2_py.ConverterOptions("", ""))
        types = {t.name: t.type for t in r.get_all_topics_and_types()}
        if HEADING_TOPIC not in types:
            return None
        msgtype = get_message(types[HEADING_TOPIC])
        r.set_filter(rosbag2_py.StorageFilter(topics=[HEADING_TOPIC]))
        ts, vs = [], []
        while r.has_next():
            _, data, trec = r.read_next()
            m = deserialize_message(data, msgtype)
            v = getattr(m, "data", getattr(m, "heading", None))
            if v is None:
                continue
            hdr = getattr(m, "header", None)
            ts.append(hdr.stamp.sec + hdr.stamp.nanosec * 1e-9
                      if hdr is not None else trec * 1e-9)
            vs.append(float(v))
        return (np.array(ts), np.array(vs)) if ts else None
    except Exception as e:                                   # noqa: BLE001
        print(f"  heading read failed: {e}")
        return None


def windowed_rate(t, Rs, baseline_s):
    """Angular rate (deg/s) over a FIXED TIME BASELINE, not consecutive samples.

    Differentiating orientation at the native 145 Hz turns ~0.05 deg of mocap
    noise into ~7 deg/s, and ~0.2 deg into ~30 deg/s -- which is why the naive
    rate never fell below the static threshold. Over a 0.2 s baseline the same
    noise contributes well under 1 deg/s.
    """
    t = np.asarray(t, float)
    n = len(t)
    w = np.zeros(n)
    j = 0
    for i in range(n):
        while j < i and t[i] - t[j] > baseline_s:
            j += 1
        if j == i:
            w[i] = w[i - 1] if i else 0.0
            continue
        dt = t[i] - t[j]
        dR = Rs[j].T @ Rs[i]
        v = 0.5 * np.array([dR[2, 1] - dR[1, 2],
                            dR[0, 2] - dR[2, 0],
                            dR[1, 0] - dR[0, 1]])
        ang = np.arctan2(np.linalg.norm(v), (np.trace(dR) - 1.0) / 2.0)
        w[i] = np.degrees(ang / dt)
    return w


def find_holds(t, yaw_rad, win_s, range_deg, min_s):
    """Holds = windows where DJI yaw RANGE stays small. No differentiation."""
    y = np.degrees(np.unwrap(yaw_rad))
    n = len(t)
    is_hold = np.zeros(n, bool)
    lo = 0
    for i in range(n):
        while t[i] - t[lo] > win_s:
            lo += 1
        if i - lo >= 2 and (y[lo:i + 1].max() - y[lo:i + 1].min()) < range_deg:
            is_hold[lo:i + 1] = True
    idx = np.where(is_hold)[0]
    if not len(idx):
        return []
    blocks = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
    return [b for b in blocks if t[b[-1]] - t[b[0]] >= min_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--out", default="./g2_out")
    ap.add_argument("--convention", default="inverted", choices=list(CONVENTIONS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conv = CONVENTIONS[args.convention]

    d = read_bag(args.bag)
    for k in ("att", "mocap"):
        if k not in d:
            print(f"FAIL: '{k}' missing from bag")
            return

    lag = sync_mocap(d, force="yaw_rate")

    # ---------------------------------------------------------------- A
    print("\n--- A. data health ---")
    print(f"mocap orientation convention applied: {args.convention}")
    print(f"mocap clock lag removed: {(lag or 0)*1e3:+.0f} ms "
          f"(corr {d.get('_mocap_corr', float('nan')):.3f})")

    tm = np.asarray(d["mocap"]["t"], float)
    Qm = np.asarray(d["mocap"]["q"], float)
    order = np.argsort(tm, kind="stable")
    tm, Qm = tm[order], Qm[order]
    keep = np.concatenate([[True], np.diff(tm) > DUP_DT])
    n_dup = int((~keep).sum())
    tm, Qm = tm[keep], Qm[keep]
    dtm = np.diff(tm)
    nom = float(np.median(dtm)) if len(dtm) else 0.0
    print(f"mocap: {len(tm)} unique poses ({n_dup} duplicates dropped), "
          f"{1.0/nom if nom else float('nan'):.1f} Hz median")
    gaps = np.where(dtm > GAP_ABS_S)[0]
    print(f"mocap gaps (>{GAP_ABS_S*1e3:.0f} ms): {len(gaps)}"
          + (f", longest {dtm[gaps].max()*1e3:.0f} ms" if len(gaps) else "")
          + f"   [max dt overall {dtm.max()*1e3:.0f} ms]")

    ta = np.asarray(d["att"]["t"], float)
    rpy_a = np.radians(np.asarray(d["att"]["rpy"], float))
    print(f"attitude: {len(ta)} msgs, {effective_rate(ta):.1f} Hz")

    m = (ta >= tm[0]) & (ta <= tm[-1])
    ta, rpy_a = ta[m], rpy_a[m]
    Q = sign_align(Qm)
    q = np.column_stack([np.interp(ta, tm, Q[:, i]) for i in range(4)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R_m_raw = [quat_to_R(qq) for qq in q]
    R_m = [conv(R) for R in R_m_raw]
    R_d = [rpy_to_R(*r) for r in rpy_a]
    errs = np.array([R_to_rpy(R_m[k].T @ R_d[k]) for k in range(len(ta))])

    # ---------------------------------------------------------------- B
    blocks = find_holds(ta, rpy_a[:, 2], HOLD_WIN_S, HOLD_RANGE_DEG, HOLD_MIN_S)
    print(f"\n--- B. static heading holds ({len(blocks)} found, expect 4) ---")
    print(f"{'#':>2} {'t0(s)':>7} {'dur':>5} {'yaw_off':>9} {'sd':>6} "
          f"{'roll_off':>9} {'pitch_off':>10} {'n':>5}")
    yaw_offs = []
    for i, b in enumerate(blocks):
        ym, ysd = circ_mean_std(errs[b, 2])
        rm, _ = circ_mean_std(errs[b, 0])
        pm, _ = circ_mean_std(errs[b, 1])
        yaw_offs.append(np.degrees(ym))
        print(f"{i:>2} {ta[b[0]]-ta[0]:7.1f} {ta[b[-1]]-ta[b[0]]:5.1f} "
              f"{np.degrees(ym):9.2f} {np.degrees(ysd):6.2f} "
              f"{np.degrees(rm):9.2f} {np.degrees(pm):10.2f} {len(b):5d}")

    if len(yaw_offs) >= 2:
        yo = np.radians(np.array(yaw_offs))
        dev = np.degrees(np.abs(wrap(yo - yo[0])))
        print(f"\nmax deviation from hold 0: {dev.max():.2f} deg")
        if dev.max() < 5.0:
            print("=> PASS: offset CONSTANT across headings. Rigid body is "
                  "sound and sigma_yaw is characterisable.")
        elif np.abs(dev - 180.0).min() < 15.0:
            print("=> one hold differs by ~180 deg -- genuine mocap yaw "
                  "ambiguity remaining.")
        else:
            print("=> offsets disagree but not by 180 deg. Check occlusion.")

    # ---------------------------------------------------------------- C
    print("\n--- C. discontinuities ---")
    base = np.radians(yaw_offs[0]) if yaw_offs else 0.0
    resid_yaw = np.degrees(wrap(errs[:, 2] - base))
    step = np.degrees(np.abs(wrap(np.diff(np.radians(resid_yaw)))))
    jumps = np.where(step > JUMP_DEG)[0]
    print(f"residual jumps > {JUMP_DEG:.0f} deg: {len(jumps)}")
    if len(jumps):
        near = sum(1 for j in jumps
                   if len(gaps) and np.min(np.abs(tm[gaps] - ta[j])) < 0.2)
        print(f"  within 200 ms of a mocap gap: {near}/{len(jumps)}")

    w_disp = windowed_rate(ta, R_m, RATE_BASELINE_S)
    print(f"max |rate| over a {RATE_BASELINE_S:.2f} s baseline = "
          f"{w_disp.max():.0f} deg/s")

    # ---------------------------------------------------------------- D
    print("\n--- D. DJI heading vs DJI attitude yaw ---")
    h = read_heading(args.bag)
    if h is None:
        print("  /drone_1/heading not in bag -- skipped")
    else:
        th, hd = h
        dh = wrap(resample(th, np.radians(hd), ta) - rpy_a[:, 2])
        hm, hsd = circ_mean_std(dh)
        print(f"  all samples : offset {np.degrees(hm):+.2f} deg, "
              f"sd {np.degrees(hsd):.2f} deg")
        if blocks:
            bidx = np.concatenate(blocks)
            hm2, hsd2 = circ_mean_std(dh[bidx])
            print(f"  holds only  : offset {np.degrees(hm2):+.2f} deg, "
                  f"sd {np.degrees(hsd2):.2f} deg")
            print("  (if the holds-only sd is much smaller, the spread is "
                  "sampling skew during rotation, not a compass fault)")

    # ---------------------------------------------------------------- E
    print("\n--- E. orientation-convention check ---")
    dji_yaw = np.unwrap(rpy_a[:, 2])
    print(f"  {'hypothesis':16s} {'sd(deg)':>9} {'slope':>7} {'offset(deg)':>12}")
    for name, f in CONVENTIONS.items():
        e = np.array([R_to_rpy(f(R_m_raw[k]).T @ R_d[k])[2]
                      for k in range(len(ta))])
        mu, sd = circ_mean_std(e)
        slope = float(np.polyfit(dji_yaw, np.unwrap(e), 1)[0])
        flag = "  <-- applied" if name == args.convention else ""
        print(f"  {name:16s} {np.degrees(sd):9.2f} {slope:+7.2f} "
              f"{np.degrees(mu):12.2f}{flag}")

    # ---------------------------------------------------------------- F
    # A constant tilt between the mocap and nav frames projects into roll/pitch
    # as the drone yaws:  r = a cos(psi) + b sin(psi),  p = -a sin(psi) + b cos(psi)
    print("\n--- F. mocap<->nav tilt, and residual sigma_rp ---")
    psi = rpy_a[:, 2]
    A = np.vstack([
        np.column_stack([np.cos(psi), np.sin(psi)]),
        np.column_stack([-np.sin(psi), np.cos(psi)]),
    ])
    y = np.concatenate([errs[:, 0], errs[:, 1]])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ np.array([a, b])
    sd_before = float(np.std(y))
    sd_after = float(np.std(y - fit))
    print(f"  tilt magnitude {np.degrees(np.hypot(a, b)):.3f} deg, "
          f"direction {np.degrees(np.arctan2(b, a)):+.1f} deg")
    print(f"  sigma_rp before removing tilt: {np.degrees(sd_before):.3f} deg "
          f"({sd_before:.5f} rad)")
    print(f"  sigma_rp after  removing tilt: {np.degrees(sd_after):.3f} deg "
          f"({sd_after:.5f} rad)")
    if sd_after < 0.7 * sd_before:
        print("  => a real frame tilt. The 0.045 rad sigma_rp in "
              "filter_design.md 5.1 is inflated by it; use the corrected value.")

    # ---------------------------------------------------------------- plot
    mocap_yaw = np.unwrap(np.array([R_to_rpy(R)[2] for R in R_m]))
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(ta - ta[0], np.degrees(dji_yaw), lw=1, label="DJI yaw")
    ax[0].plot(ta - ta[0], np.degrees(mocap_yaw), lw=1,
               label=f"mocap yaw ({args.convention})")
    ax[0].set_ylabel("yaw (deg)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ta - ta[0], resid_yaw, ".", ms=2)
    ax[1].set_ylabel("yaw resid (deg)"); ax[1].grid(alpha=.3)
    ax[2].plot(ta - ta[0], np.degrees(errs[:, 0]), ".", ms=2, label="roll")
    ax[2].plot(ta - ta[0], np.degrees(errs[:, 1]), ".", ms=2, label="pitch")
    ax[2].plot(ta - ta[0], np.degrees(fit[:len(ta)]), "-", lw=1, label="tilt fit")
    ax[2].set_ylabel("r/p resid (deg)"); ax[2].legend(); ax[2].grid(alpha=.3)
    ax[3].plot(ta - ta[0], w_disp, lw=1)
    for b in blocks:
        ax[3].axvspan(ta[b[0]] - ta[0], ta[b[-1]] - ta[0], color="g", alpha=.15)
    ax[3].set_ylabel(f"rate over {RATE_BASELINE_S:.2f}s (deg/s)")
    ax[3].set_xlabel("s"); ax[3].grid(alpha=.3)
    fig.suptitle(f"{Path(args.bag).name} — G2 ({args.convention})")
    fig.savefig(out / "g2_yaw_reference.png", dpi=110, bbox_inches="tight")
    print(f"\nplot: {out/'g2_yaw_reference.png'}")


if __name__ == "__main__":
    main()
    