#!/usr/bin/env python3
"""
source_characterisation.py
--------------------------
Characterise DJI telemetry (altitude, velocity, attitude, gimbal) against
OptiTrack ground truth from rosbag2 recordings, and emit the constants needed
by filter_design.md sections 5 and 10.

Clock policy
------------
Telemetry topics carry header.stamp in the ground-station ROS clock (mapped
from the phone monotonic clock by ClockOffsetTracker), so header stamps are
used for them -- these are measurement times.

OptiTrack stamps come from the mocap host, whose clock may not be synced, so
the bag RECORD time is used for mocap instead. The residual offset between the
two is estimated by cross-correlation and removed before any fit.

Usage
-----
    source install/setup.bash
    python3 source_characterisation.py BAG_DIR [BAG_DIR ...] --out ./out

Each bag is analysed independently; whichever analyses the data supports are
run. Pass --only alt,vel,att,gimbal to restrict.

Deps: numpy, matplotlib, rosbag2_py (from ROS 2), PyYAML.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- config
ALT_TOPIC = "/drone_1/relative_altitude"
SPEED_TOPIC = "/drone_1/speed_vector"
ATT_TOPIC = "/drone_1/attitude"
GIMBAL_TOPIC = "/drone_1/gimbal_joint_attitude"
MOCAP_TOPIC = "/optitrack/rigid_bodies/dji_mini4"

Q_ALT = 0.1        # altitude quantisation (m)
Q_SPEED = 0.1      # speed quantisation (m/s)

RESAMPLE_HZ = 20.0
VZ_THRESH = 0.15   # m/s, climb/descend classification from mocap
HOVER_V = 0.05     # m/s, hover detection
HOVER_MIN_S = 5.0


# --------------------------------------------------------------------------- reading
def read_bag(bagpath):
    """Return dict of topic -> dict(arrays). Telemetry uses header stamps,
    mocap uses record time."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bagpath), storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    raw = {k: [] for k in
           (ALT_TOPIC, SPEED_TOPIC, ATT_TOPIC, GIMBAL_TOPIC, MOCAP_TOPIC)}

    while reader.has_next():
        topic, data, t_rec = reader.read_next()
        if topic not in raw:
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        raw[topic].append((t_rec * 1e-9, msg))

    out = {}

    def hdr(m):
        s = m.header.stamp
        return s.sec + s.nanosec * 1e-9

    if raw[ALT_TOPIC]:
        out["alt"] = {
            "t": np.array([hdr(m) for _, m in raw[ALT_TOPIC]]),
            "t_rec": np.array([tr for tr, _ in raw[ALT_TOPIC]]),
            "z": np.array([float(m.altitude) for _, m in raw[ALT_TOPIC]]),
        }
    if raw[SPEED_TOPIC]:
        out["vel"] = {
            "t": np.array([hdr(m) for _, m in raw[SPEED_TOPIC]]),
            "t_rec": np.array([tr for tr, _ in raw[SPEED_TOPIC]]),
            "v": np.array([[m.vector.x, m.vector.y, m.vector.z]
                           for _, m in raw[SPEED_TOPIC]], float),
        }
    for key, topic in (("att", ATT_TOPIC), ("gimbal", GIMBAL_TOPIC)):
        if raw[topic]:
            out[key] = {
                "t": np.array([hdr(m) for _, m in raw[topic]]),
                "rpy": np.array([[m.roll, m.pitch, m.yaw]
                                 for _, m in raw[topic]], float),
            }
    if raw[MOCAP_TOPIC]:
        P, Q, T = [], [], []
        for tr, m in raw[MOCAP_TOPIC]:
            p = m.pose.position if hasattr(m, "pose") else m.position
            q = m.pose.orientation if hasattr(m, "pose") else m.orientation
            P.append([p.x, p.y, p.z])
            Q.append([q.x, q.y, q.z, q.w])
            T.append(tr)                      # RECORD time, see module docstring
        out["mocap"] = {"t": np.array(T), "p": np.array(P, float),
                        "q": np.array(Q, float)}
    return out


# --------------------------------------------------------------------------- maths
def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_rpy(R):
    """ZYX intrinsic -> roll, pitch, yaw (rad)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def rpy_to_R(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch),
                              np.sin(pitch), np.cos(yaw), np.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def resample(t_src, y_src, t_dst):
    y_src = np.asarray(y_src)
    if y_src.ndim == 1:
        return np.interp(t_dst, t_src, y_src)
    return np.column_stack([np.interp(t_dst, t_src, y_src[:, i])
                            for i in range(y_src.shape[1])])


def mocap_velocity(t, P, win=9, v_max=6.0):
    """Robust differentiation: dedupe stamps, smooth, reject glitches."""
    keep = np.concatenate([[True], np.diff(t) > 1e-4])
    t, P = t[keep], P[keep]
    k = np.ones(win) / win
    Ps = np.column_stack([np.convolve(P[:, i], k, mode="same") for i in range(3)])
    V = np.gradient(Ps, t, axis=0)
    bad = np.linalg.norm(V, axis=1) > v_max
    for i in range(3):
        V[bad, i] = np.interp(t[bad], t[~bad], V[~bad, i])
    return t, V


def xcorr_lag(t, a, b, max_lag=1.0, fs=RESAMPLE_HZ, min_std=1e-6):
    """Lag (s) to shift b so it best matches a. Positive = b is late.
    Returns (nan, nan) if either signal is too flat to correlate."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan"), float("nan")
    if a.std() < min_std or b.std() < min_std:
        return float("nan"), float("nan")
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    n = int(max_lag * fs)
    lags = np.arange(-n, n + 1)
    c = []
    for l in lags:
        aa = a[max(0, l):len(a) + min(0, l)]
        bb = b[max(0, -l):len(b) + min(0, -l)]
        if len(aa) < 20 or aa.std() < 1e-9 or bb.std() < 1e-9:
            c.append(np.nan)
        else:
            c.append(np.corrcoef(aa, bb)[0, 1])
    c = np.array(c)
    if np.all(np.isnan(c)):
        return float("nan"), float("nan")
    i = int(np.nanargmax(c))
    if 0 < i < len(c) - 1 and np.isfinite(c[i - 1]) and np.isfinite(c[i + 1]):
        d = 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2 * c[i] + c[i + 1] + 1e-12)
    else:
        d = 0.0
    return (lags[i] + d) / fs, float(c[i])


def onepole(x, dt, tau):
    if tau <= 1e-6:
        return x.copy()
    a = np.exp(-dt / tau)
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = a * y[k - 1] + (1 - a) * x[k]
    return y


def effective_rate(t):
    d = np.diff(t)
    d = d[d > 0]
    return (1.0 / np.median(d)) if len(d) else float("nan")


def quant_check(v, q):
    r = np.abs(v / q - np.round(v / q))
    return float(np.mean(r < 1e-6))


# --------------------------------------------------------------------------- sync
def sync_mocap(d, min_corr=0.5):
    """Estimate and remove the mocap-clock offset. Returns lag (s), 0 if
    no signal in the bag supports a reliable estimate."""
    if "mocap" not in d:
        return 0.0
    tel = [v["t"] for k, v in d.items() if k != "mocap" and isinstance(v, dict)]
    if not tel:
        return 0.0
    t0 = max(d["mocap"]["t"][0], min(t[0] for t in tel))
    t1 = min(d["mocap"]["t"][-1], max(t[-1] for t in tel))
    if t1 - t0 < 5:
        return 0.0
    grid = np.arange(t0, t1, 1.0 / RESAMPLE_HZ)
    fs = RESAMPLE_HZ

    cands = []
    if "alt" in d:
        cands.append(("altitude",
                      resample(d["alt"]["t"], d["alt"]["z"], grid),
                      resample(d["mocap"]["t"], d["mocap"]["p"][:, 2], grid)))
    if "vel" in d:
        mt, V = mocap_velocity(d["mocap"]["t"], d["mocap"]["p"])
        cands.append(("speed",
                      np.linalg.norm(resample(d["vel"]["t"], d["vel"]["v"], grid), axis=1),
                      np.linalg.norm(resample(mt, V, grid), axis=1)))
    if "att" in d:
        ya = np.unwrap(resample(d["att"]["t"],
                                np.unwrap(d["att"]["rpy"][:, 2]), grid))
        ym = np.unwrap(np.array([R_to_rpy(quat_to_R(q))[2] for q in d["mocap"]["q"]]))
        yb = np.unwrap(resample(d["mocap"]["t"], ym, grid))
        cands.append(("yaw_rate",
                      np.gradient(ya, 1.0 / fs),
                      np.gradient(yb, 1.0 / fs)))

    best = (None, float("nan"), -np.inf)
    for label, a, b in cands:
        lag, corr = xcorr_lag(grid, a, b, fs=fs)
        if np.isfinite(corr) and corr > best[2]:
            best = (label, lag, corr)
    label, lag, corr = best

    if label is None or corr < min_corr:
        print(f"  mocap sync: no reliable signal (best {label}, corr {corr:.2f})"
              f" -- offset NOT applied")
        d["_mocap_lag"] = 0.0
        d["_mocap_corr"] = corr
        return 0.0

    d["mocap"]["t"] = d["mocap"]["t"] + lag
    d["_mocap_lag"] = lag
    d["_mocap_corr"] = corr
    d["_mocap_sync_signal"] = label
    return lag


# --------------------------------------------------------------------------- altitude
def characterise_altitude(d, out, name):
    if "alt" not in d or "mocap" not in d:
        return {}
    t = d["alt"]["t"]
    z = d["alt"]["z"]
    m = (t >= d["mocap"]["t"][0]) & (t <= d["mocap"]["t"][-1])
    t, z = t[m], z[m]
    z_true = resample(d["mocap"]["t"], d["mocap"]["p"][:, 2], t)
    z_true = z_true - np.median(z_true[:min(20, len(z_true))])   # takeoff datum
    mt, V = mocap_velocity(d["mocap"]["t"], d["mocap"]["p"])
    vz = resample(mt, V[:, 2], t)

    res = {"rate_hz": effective_rate(t),
           "quantised_frac": quant_check(z, Q_ALT)}

    up = vz > VZ_THRESH
    dn = vz < -VZ_THRESH
    hv = ~up & ~dn                      # hover: neither climbing nor descending
    # Joint fit, one shared intercept:
    #   z_dji = (1+k_up)*z*I_up + (1+k_dn)*z*I_dn + (1+k_hov)*z*I_hov + b
    if up.sum() > 30 and dn.sum() > 30:
        A = np.column_stack([z_true * up, z_true * dn, z_true * hv,
                             np.ones_like(z_true)])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        res["k_up"] = float(coef[0] - 1.0)
        res["k_down"] = float(coef[1] - 1.0)
        res["k_hover"] = float(coef[2] - 1.0)      # empirical K_DEFAULT
        res["b"] = float(coef[3])
        res["n_up"], res["n_down"], res["n_hover"] = (
            int(up.sum()), int(dn.sum()), int(hv.sum()))
    else:
        A = np.column_stack([z_true, np.ones_like(z_true)])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        res["k_combined"] = float(coef[0] - 1.0)
        res["b"] = float(coef[1])
        res["note"] = "insufficient climb/descend data to split k"
    pred = A @ coef
    res["fit_rms"] = float(np.std(z - pred))

    # Noise from hover segments only, quantisation floor removed.
    hov = np.abs(vz) < HOVER_V
    if hov.sum() > 50:
        e = (z - pred)[hov]
        var = float(np.var(e))
        floor = Q_ALT ** 2 / 12.0
        res["R_alt_measured"] = var
        res["R_alt_sensor"] = max(var - floor, 0.0)
        res["quant_floor"] = floor
        res["sigma_alt"] = float(np.sqrt(var))

    # Ground effect: bias vs height across hover blocks.
    if hov.sum() > 50:
        hb = []
        idx = np.where(hov)[0]
        splits = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1)
        for s in splits:
            if len(s) < int(HOVER_MIN_S * res["rate_hz"]):
                continue
            hb.append((float(np.mean(z_true[s])), float(np.mean((z - pred)[s]))))
        if len(hb) >= 2:
            res["hover_bias_vs_height"] = hb
            hh = np.array(hb)
            slope = np.polyfit(hh[:, 0], hh[:, 1], 1)[0]
            res["ground_effect_slope"] = float(slope)

    # VZ_DEAD: |vz| where the two k branches differ by less than sigma_alt.
    if "k_up" in res and "sigma_alt" in res:
        dk = abs(res["k_up"] - res["k_down"])
        zt = float(np.percentile(np.abs(z_true), 90)) or 1.0
        res["k_branch_diff_at_typical_height_m"] = float(dk * zt)
        res["k_hover_between_branches"] = bool(
            min(res["k_up"], res["k_down"]) <= res["k_hover"]
            <= max(res["k_up"], res["k_down"]))
        res["height_where_branches_matter_m"] = float(
            res["sigma_alt"] / max(dk, 1e-9))

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t - t[0], z, ".", ms=2, label="z_dji")
    ax[0].plot(t - t[0], z_true, "-", lw=1, label="z_mocap")
    ax[0].legend(); ax[0].set_ylabel("m"); ax[0].grid(alpha=.3)
    ax[1].plot(t - t[0], z - pred, ".", ms=2)
    ax[1].set_ylabel("residual (m)"); ax[1].set_xlabel("s"); ax[1].grid(alpha=.3)
    fig.suptitle(f"{name} — altitude")
    fig.savefig(out / f"{name}_altitude.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return res


# --------------------------------------------------------------------------- velocity
def characterise_velocity(d, out, name):
    if "vel" not in d or "mocap" not in d or "att" not in d:
        return {}
    t0 = max(d["vel"]["t"][0], d["mocap"]["t"][0], d["att"]["t"][0])
    t1 = min(d["vel"]["t"][-1], d["mocap"]["t"][-1], d["att"]["t"][-1])
    grid = np.arange(t0, t1, 1.0 / RESAMPLE_HZ)
    dt = 1.0 / RESAMPLE_HZ

    v_dji = resample(d["vel"]["t"], d["vel"]["v"], grid)
    p_grid = resample(d["mocap"]["t"], d["mocap"]["p"], grid)
    k = np.ones(5) / 5
    p_grid = np.column_stack([np.convolve(p_grid[:, i], k, mode="same")
                              for i in range(3)])
    v_world = np.gradient(p_grid, 1.0 / RESAMPLE_HZ, axis=0)
    yaw = np.unwrap(resample(d["att"]["t"], np.unwrap(d["att"]["rpy"][:, 2]), grid))

    res = {"rate_hz": effective_rate(d["vel"]["t"]),
           "quantised_frac": quant_check(d["vel"]["v"].ravel(), Q_SPEED)}

    # Body-frame reference: rotate world velocity by -yaw about vertical.
    c, s = np.cos(yaw), np.sin(yaw)
    v_body = np.column_stack([
        c * v_world[:, 0] + s * v_world[:, 1],
        -s * v_world[:, 0] + c * v_world[:, 1],
        v_world[:, 2]])

    moving = np.linalg.norm(v_dji, axis=1) > 0.15
    if moving.sum() < 50:
        res["note"] = "insufficient motion"
        return res

    def fit(ref):
        M, *_ = np.linalg.lstsq(ref[moving], v_dji[moving], rcond=None)
        pred = ref @ M
        rms = float(np.sqrt(np.mean((v_dji[moving] - pred[moving]) ** 2)))
        return M.T, rms

    M_w, rms_w = fit(v_world)
    M_b, rms_b = fit(v_body)
    res["fit_rms_world"] = rms_w
    res["fit_rms_body"] = rms_b
    res["frame"] = "world" if rms_w < rms_b else "body"
    res["frame_confidence"] = float(max(rms_w, rms_b) / (min(rms_w, rms_b) + 1e-12))
    M = M_w if rms_w < rms_b else M_b
    res["M"] = M.tolist()

    # Snap to signed permutation for axis_perm / signs.
    perm, signs = [], []
    for r in range(3):
        j = int(np.argmax(np.abs(M[r])))
        perm.append(j)
        signs.append(int(np.sign(M[r, j])))
    res["axis_perm"] = perm
    res["signs"] = signs
    res["diag_gains"] = [float(M[r, perm[r]]) for r in range(3)]

    # Lag model: grid search delay + one-pole tau on the resolved reference.
    ref = (v_world if res["frame"] == "world" else v_body) @ M.T
    best = None
    for dly in np.arange(0.0, 0.41, dt):
        n = int(round(dly / dt))
        for tau in np.arange(0.0, 0.51, 0.01):
            r = np.column_stack([onepole(ref[:, i], dt, tau) for i in range(3)])
            if n:
                r = np.vstack([np.repeat(r[:1], n, axis=0), r[:-n]])
            e = v_dji[moving] - r[moving]
            c = float(np.mean(e ** 2))
            if best is None or c < best[0]:
                best = (c, dly, tau, r)
    _, dly, tau, ref_lagged = best
    res["velocity_delay_s"] = float(dly)
    res["velocity_tau_s"] = float(tau)
    e = v_dji[moving] - ref_lagged[moving]
    res["R_speed_infl"] = [float(np.var(e[:, i])) for i in range(3)]
    res["residual_rms"] = [float(np.std(e[:, i])) for i in range(3)]

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, lab in enumerate("xyz"):
        ax[i].plot(grid - grid[0], v_dji[:, i], ".", ms=2, label="dji")
        ax[i].plot(grid - grid[0], ref_lagged[:, i], "-", lw=1, label="mocap->fit")
        ax[i].set_ylabel(f"v{lab} (m/s)"); ax[i].grid(alpha=.3)
    ax[0].legend(); ax[2].set_xlabel("s")
    fig.suptitle(f"{name} — velocity  frame={res['frame']}  "
                 f"delay={dly*1e3:.0f} ms  tau={tau*1e3:.0f} ms")
    fig.savefig(out / f"{name}_velocity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return res


# --------------------------------------------------------------------------- attitude
def characterise_attitude(d, out, name):
    if "att" not in d or "mocap" not in d:
        return {}
    t = d["att"]["t"]
    m = (t >= d["mocap"]["t"][0]) & (t <= d["mocap"]["t"][-1])
    t = t[m]
    rpy = np.radians(d["att"]["rpy"][m])
    if len(t) < 50:
        return {}

    q = np.column_stack([np.interp(t, d["mocap"]["t"], d["mocap"]["q"][:, i])
                         for i in range(4)])
    res = {"rate_hz": effective_rate(t)}

    # Constant offset between the mocap rigid-body frame and base_link.
    errs = []
    for k in range(len(t)):
        R_m = quat_to_R(q[k])
        R_d = rpy_to_R(*rpy[k])
        errs.append(R_to_rpy(R_m.T @ R_d))
    errs = np.array(errs)
    offset = np.array([np.median(errs[:, 0]), np.median(errs[:, 1]),
                       np.arctan2(np.median(np.sin(errs[:, 2])),
                                  np.median(np.cos(errs[:, 2])))])
    res["mocap_to_base_link_offset_deg"] = list(np.degrees(offset))

    resid = np.column_stack([wrap(errs[:, i] - offset[i]) for i in range(3)])
    res["sigma_roll_deg"] = float(np.degrees(np.std(resid[:, 0])))
    res["sigma_pitch_deg"] = float(np.degrees(np.std(resid[:, 1])))
    res["sigma_yaw_deg"] = float(np.degrees(np.std(resid[:, 2])))
    res["sigma_rp_rad"] = float(np.radians(
        0.5 * (res["sigma_roll_deg"] + res["sigma_pitch_deg"])))
    res["sigma_yaw_rad"] = float(np.radians(res["sigma_yaw_deg"]))
    res["yaw_drift_deg"] = float(np.degrees(resid[-20:, 2].mean()
                                            - resid[:20, 2].mean()))

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, lab in enumerate(("roll", "pitch", "yaw")):
        ax[i].plot(t - t[0], np.degrees(resid[:, i]), ".", ms=2)
        ax[i].set_ylabel(f"{lab} err (deg)"); ax[i].grid(alpha=.3)
    ax[2].set_xlabel("s")
    fig.suptitle(f"{name} — attitude residual vs mocap")
    fig.savefig(out / f"{name}_attitude.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return res


# --------------------------------------------------------------------------- gimbal
def characterise_gimbal(d, out, name):
    if "gimbal" not in d:
        return {}
    t, rpy = d["gimbal"]["t"], d["gimbal"]["rpy"].copy()
    res = {"rate_hz": effective_rate(t)}

    # Unsigned 16-bit tenths-of-degree encoding: values above half-scale are
    # negative angles. Flag per axis on the RAW data, then unwrap.
    for i, lab in enumerate(("roll", "pitch", "yaw")):
        if np.any(rpy[:, i] > 3276.75):
            res[f"{lab}_needs_unwrap"] = True
            rpy[rpy[:, i] > 3276.75, i] -= 6553.5

    for i, lab in enumerate(("roll", "pitch", "yaw")):
        res[f"{lab}_min_deg"] = float(np.min(rpy[:, i]))
        res[f"{lab}_max_deg"] = float(np.max(rpy[:, i]))
        res[f"{lab}_range_deg"] = float(np.ptp(rpy[:, i]))
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--out", default="./characterisation_out")
    ap.add_argument("--only", default="alt,vel,att,gimbal")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only.split(","))
    allres = {}

    for bag in args.bags:
        name = Path(bag).name
        print(f"\n=== {name} ===")
        d = read_bag(bag)
        present = [k for k in ("alt", "vel", "att", "gimbal", "mocap") if k in d]
        print("topics:", ", ".join(present))
        if "mocap" not in d:
            print("  WARNING: no mocap -- only rate/quantisation available")

        lag = sync_mocap(d)
        r = {"mocap_clock_lag_s": lag,
             "mocap_sync_corr": d.get("_mocap_corr")}
        if lag:
            print(f"mocap clock offset removed: {lag*1e3:+.0f} ms "
                  f"(corr {d.get('_mocap_corr', 0):.3f})")

        if "alt" in want:
            r["altitude"] = characterise_altitude(d, out, name)
        if "vel" in want:
            r["velocity"] = characterise_velocity(d, out, name)
        if "att" in want:
            r["attitude"] = characterise_attitude(d, out, name)
        if "gimbal" in want:
            r["gimbal"] = characterise_gimbal(d, out, name)

        allres[name] = r
        print(json.dumps(r, indent=2, default=float))

    (out / "results.json").write_text(json.dumps(allres, indent=2, default=float))
    print(f"\nwritten: {out/'results.json'} and plots")


if __name__ == "__main__":
    main()
    