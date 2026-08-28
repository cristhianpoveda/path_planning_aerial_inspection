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
def read_bag(bagpath, mocap_bagpath=None):
    """Return dict of topic -> dict(arrays). Telemetry uses header stamps,
    mocap uses record time.

    mocap_bagpath: optional second bag holding MOCAP_TOPIC, recorded in
    ROS_DOMAIN_ID 0 at full rate. domain_bridge delivers ~15 Hz with
    multi-second stalls against ~103 Hz at source, so mocap is recorded
    separately. Both recorders run on the same host, so record times share one
    clock and no extra alignment is needed. When given, it REPLACES any mocap
    found in the main bag.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    def _drain(uri, wanted, sink):
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(uri), storage_id=""),
            rosbag2_py.ConverterOptions("", ""),
        )
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        n = 0
        while reader.has_next():
            topic, data, t_rec = reader.read_next()
            if topic not in wanted:
                continue
            msg = deserialize_message(data, get_message(types[topic]))
            sink[topic].append((t_rec * 1e-9, msg))
            n += 1
        return n

    raw = {k: [] for k in
           (ALT_TOPIC, SPEED_TOPIC, ATT_TOPIC, GIMBAL_TOPIC, MOCAP_TOPIC)}

    _drain(bagpath, set(raw), raw)

    if mocap_bagpath is not None:
        n_bridged = len(raw[MOCAP_TOPIC])
        raw[MOCAP_TOPIC] = []
        n = _drain(mocap_bagpath, {MOCAP_TOPIC}, raw)
        print(f"  mocap from {Path(mocap_bagpath).name}: {n} msgs "
              f"(replacing {n_bridged} bridged)")

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
            # The mocap4r2 driver publishes the CONJUGATE orientation:
            # R_body_world where ROS convention is R_world_body. Verified on
            # G2_yaw_reference -- as-published gives a yaw residual whose slope
            # against DJI yaw is +2.00 with sd 142 deg (the signature of an
            # inverted yaw sense); conjugating gives slope -0.00, sd 0.63 deg.
            # Position is unaffected, which is consistent with a quaternion
            # conjugation bug touching orientation only.
            Q.append([-q.x, -q.y, -q.z, q.w])
            T.append(tr)                      # RECORD time, see module docstring
        # Domain 0 carries two publishers of this topic (the driver and a
        # zenoh bridge), so some poses arrive twice, microseconds apart.
        T, P, Q = np.array(T), np.array(P, float), np.array(Q, float)
        order = np.argsort(T, kind="stable")
        T, P, Q = T[order], P[order], Q[order]
        keep = np.concatenate([[True], np.diff(T) > 1e-3])
        out["mocap"] = {"t": T[keep], "p": P[keep], "q": Q[keep],
                        "n_duplicates": int((~keep).sum())}
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
def sync_mocap(d, min_corr=0.5, force="altitude"):
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

    if force is not None:
        forced = [c for c in cands if c[0] == force]
        if forced:
            cands = forced

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

def vertical_integral_gain(d, win_s=3.0):
    """K_VEL from integrated DJI vz vs mocap dz. No differentiation of mocap,
    and a 75 ms delay only affects the window endpoints. Needs no yaw."""
    if "vel" not in d or "mocap" not in d:
        return {}
    t0 = max(d["vel"]["t"][0], d["mocap"]["t"][0])
    t1 = min(d["vel"]["t"][-1], d["mocap"]["t"][-1])
    if t1 - t0 < 3 * win_s:
        return {}
    grid = np.arange(t0, t1, 1.0 / RESAMPLE_HZ)
    vz = resample(d["vel"]["t"], d["vel"]["v"][:, 2], grid)      # down-positive
    z_m = resample(d["mocap"]["t"], d["mocap"]["p"][:, 2], grid)

    n = int(win_s * RESAMPLE_HZ)
    num, den = [], []
    for i in range(0, len(grid) - n, n // 2):
        dz_dji = -np.trapz(vz[i:i + n], grid[i:i + n])          # sign: NED -> up
        dz_mocap = z_m[i + n - 1] - z_m[i]
        if abs(dz_mocap) < 0.20:                                 # need real motion
            continue
        num.append(dz_dji)
        den.append(dz_mocap)
    if len(num) < 4:
        return {"note": "insufficient vertical displacement"}
    num, den = np.array(num), np.array(den)
    g = float(np.sum(num * den) / np.sum(den * den))             # TLS-ish slope
    return {"K_VEL_integral": g,
            "K_VEL_integral_n_windows": len(num),
            "K_VEL_integral_scatter": float(np.std(num / den))}

# --------------------------------------------------------------------------- altitude
def characterise_altitude(d, out, name):
    if "alt" not in d or "mocap" not in d:
        return {}
    t = d["alt"]["t"]
    z = d["alt"]["z"]
    m = (t >= d["mocap"]["t"][0]) & (t <= d["mocap"]["t"][-1])
    t, z = t[m], z[m]
    z_true = resample(d["mocap"]["t"], d["mocap"]["p"][:, 2], t)
    grounded = (z == 0.0)
    if grounded.sum() >= 10:
        z_true = z_true - np.median(z_true[grounded])
    else:
        z_true = z_true - np.median(z_true[:min(20, len(z_true))])
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
    res.update(vertical_integral_gain(d))
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

    # Mask on the REFERENCE, not on v_dji: selecting on the dependent variable
    # truncates the regression and biases the slope.
    moving = np.linalg.norm(v_dji, axis=1) > 0.15
    if moving.sum() < 50:
        res["note"] = "insufficient motion"
        return res

    def lsq_trim(ref, mask, passes=2):
        M, *_ = np.linalg.lstsq(ref[mask], v_dji[mask], rcond=None)
        for _ in range(passes):
            e = np.linalg.norm(v_dji - ref @ M, axis=1)
            thr = 3.0 * np.median(e[mask]) + 1e-9
            keep = mask & (e < thr)
            if keep.sum() < 50:
                break
            M, *_ = np.linalg.lstsq(ref[keep], v_dji[keep], rcond=None)
        res["fit_trim_frac"] = float(1.0 - keep.sum() / max(mask.sum(), 1))
        return M

    def fit(ref):
        M = lsq_trim(ref, moving)
        pred = ref @ M
        return M.T, float(np.sqrt(np.mean((v_dji[moving] - pred[moving]) ** 2)))

    def apply_lag(x, dly, tau):
        """Delay by a FRACTIONAL number of samples, then one-pole."""
        y = np.column_stack([onepole(x[:, i], dt, tau) for i in range(3)])
        if dly > 0:
            src = np.arange(len(y)) - dly / dt
            src = np.clip(src, 0, len(y) - 1)
            y = np.column_stack([np.interp(src, np.arange(len(y)), y[:, i])
                                 for i in range(3)])
        return y

    M_w, rms_w = fit(v_world)
    M_b, rms_b = fit(v_body)
    res["fit_rms_world"] = rms_w
    res["fit_rms_body"] = rms_b
    res["frame"] = "world" if rms_w < rms_b else "body"
    res["frame_confidence"] = float(max(rms_w, rms_b) / (min(rms_w, rms_b) + 1e-12))
    M = M_w if rms_w < rms_b else M_b
    ref_raw = v_world if res["frame"] == "world" else v_body
    res["M_naive"] = M.tolist()          # gain WITH the delay still in it

    # --- alternate: delay/tau given M, then M given delay/tau -------------
    for _ in range(4):
        ref = ref_raw @ M.T
        best = None
        for dly in np.arange(0.0, 0.41, 0.005):        # 5 ms grid
            for tau in np.arange(0.0, 0.21, 0.005):
                r = apply_lag(ref, dly, tau)
                c = float(np.mean((v_dji[moving] - r[moving]) ** 2))
                if best is None or c < best[0]:
                    best = (c, dly, tau)
        _, dly, tau = best
        ref_lagged_basis = apply_lag(ref_raw, dly, tau)
        M = lsq_trim(ref_lagged_basis, moving).T
        M = M.T

    ref_lagged = apply_lag(ref_raw, dly, tau) @ M.T
    res["M"] = M.tolist()
    res["velocity_delay_s"] = float(dly)
    res["velocity_tau_s"] = float(tau)

    # Convention-free isotropic gain: invariant to any orthogonal factor.
    Hh = np.array(M)[:2, :2]
    res["gain_horizontal"] = float(np.sqrt(abs(np.linalg.det(Hh))))
    res["gain_z"] = float(abs(M[2][2]))
    res["det_horizontal"] = float(np.linalg.det(Hh))   # <0 => NED/ENU reflection

    perm, signs = [], []
    for r_ in range(3):
        j = int(np.argmax(np.abs(M[r_])))
        perm.append(j)
        signs.append(int(np.sign(M[r_][j])))
    res["axis_perm"] = perm
    res["signs"] = signs
    res["diag_gains"] = [float(M[r_][perm[r_]]) for r_ in range(3)]

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

    Q = np.asarray(d["mocap"]["q"], dtype=float).copy()
    flip = np.cumprod(np.where(np.sum(Q[1:] * Q[:-1], axis=1) < 0, -1.0, 1.0))
    Q[1:] *= flip[:, None]
    q = np.column_stack([np.interp(t, d["mocap"]["t"], Q[:, i]) for i in range(4)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)

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

    def rate_mag(ts, Rs):
        ts = np.asarray(ts, float)
        keep = np.concatenate([[True], np.diff(ts) > 1e-3])   # drop duplicate stamps
        ts = ts[keep]
        Rs = [Rs[i] for i in np.where(keep)[0]]
        w = np.zeros(len(ts))
        for i in range(1, len(ts)):
            dR = Rs[i - 1].T @ Rs[i]
            v = 0.5 * np.array([dR[2, 1] - dR[1, 2],
                                dR[0, 2] - dR[2, 0],
                                dR[1, 0] - dR[0, 1]])
            # atan2 form: numerically stable at small angles, unlike arccos
            ang = np.arctan2(np.linalg.norm(v), (np.trace(dR) - 1.0) / 2.0)
            w[i] = ang / (ts[i] - ts[i - 1])
        if len(w) > 1:
            w[0] = w[1]
        cap = np.radians(400.0)          # physically impossible above this
        bad = w > cap
        if bad.any() and (~bad).sum() > 2:
            w[bad] = np.interp(ts[bad], ts[~bad], w[~bad])
        return ts, w

    g2 = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    kk = np.ones(5) / 5
    td, wd = rate_mag(t, [rpy_to_R(*r) for r in rpy])
    tm, wm = rate_mag(d["mocap"]["t"], [quat_to_R(qq) for qq in d["mocap"]["q"]])
    w_d = np.convolve(resample(td, wd, g2), kk, mode="same")
    w_m = np.convolve(resample(tm, wm, g2), kk, mode="same")
    lag, corr = xcorr_lag(g2, w_m, w_d, max_lag=0.5)   # positive => DJI is late
    res["att_delay_s"] = float(lag)
    res["att_delay_corr"] = float(corr)
    res["att_rate_max_dps"] = float(np.degrees(np.nanmax(w_m)))

    # ---- sigma_rp vs horizontal acceleration -----------------------------
    # sigma_rp tracks ACCELERATION, not angular rate: 0.08 deg static,
    # 0.43 deg hover+yaw at 59 deg/s, 0.59 deg vertical-only, but 2.4-3.8 deg
    # when translating. That is accelerometer-referenced tilt estimation:
    # linear acceleration corrupts the gravity vector, giving atan(a/g).
    # Resample to the grid FIRST, then smooth, then differentiate -- twice.
    ga = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    p_g = resample(d["mocap"]["t"], d["mocap"]["p"], ga)
    ks = np.ones(11) / 11.0                       # longer window: 2nd derivative
    p_g = np.column_stack([np.convolve(p_g[:, i], ks, mode="same")
                           for i in range(3)])
    v_g = np.gradient(p_g, 1.0 / RESAMPLE_HZ, axis=0)
    a_g = np.gradient(v_g, 1.0 / RESAMPLE_HZ, axis=0)
    a_h = np.linalg.norm(a_g[:, :2], axis=1)
    edge = 15                                     # drop convolution edges
    a_h[:edge] = np.nan
    a_h[-edge:] = np.nan

    a_at = resample(ga, np.nan_to_num(a_h, nan=0.0), t)
    good = np.isfinite(a_at) & (t > t[0] + 1.0) & (t < t[-1] - 1.0)
    rp_mag = np.hypot(resid[:, 0], resid[:, 1])   # rad

    if good.sum() > 50:
        A = np.column_stack([a_at[good], np.ones(good.sum())])
        slope, icept = np.linalg.lstsq(A, rp_mag[good], rcond=None)[0]
        res["rp_vs_accel_slope_s2_per_m"] = float(slope)
        res["rp_vs_accel_intercept_rad"] = float(icept)
        res["rp_vs_accel_inv_g"] = float(1.0 / 9.81)   # expected slope
        res["a_horiz_max"] = float(np.nanmax(a_at[good]))
        res["a_horiz_p95"] = float(np.nanpercentile(a_at[good], 95))

        # Two-tier sigma_rp for the acceleration-scheduled R_att.
        quiet = good & (a_at < 0.20)
        dyn = good & (a_at > 0.50)
        if quiet.sum() > 20:
            res["sigma_rp_quiet_rad"] = float(np.std(rp_mag[quiet]))
            res["n_quiet"] = int(quiet.sum())
        if dyn.sum() > 20:
            res["sigma_rp_dynamic_rad"] = float(np.std(rp_mag[dyn]))
            res["n_dynamic"] = int(dyn.sum())
    else:
        res["note_accel"] = "insufficient acceleration coverage"

    fig, ax = plt.subplots(4, 1, figsize=(11, 11))
    for i, lab in enumerate(("roll", "pitch", "yaw")):
        ax[i].plot(t - t[0], np.degrees(resid[:, i]), ".", ms=2)
        ax[i].set_ylabel(f"{lab} err (deg)")
        ax[i].grid(alpha=.3)
        ax[i].sharex(ax[0])
    ax[2].set_xlabel("s")

    if "rp_vs_accel_slope_s2_per_m" in res:
        ax[3].plot(a_at[good], np.degrees(rp_mag[good]), ".", ms=2, alpha=.4)
        xs = np.linspace(0, np.nanmax(a_at[good]), 50)
        ax[3].plot(xs, np.degrees(res["rp_vs_accel_slope_s2_per_m"] * xs
                                  + res["rp_vs_accel_intercept_rad"]),
                   "-", lw=1.5, label=f"fit, slope="
                                      f"{res['rp_vs_accel_slope_s2_per_m']:.4f}")
        ax[3].plot(xs, np.degrees(xs / 9.81), "--", lw=1.2, label="atan(a/g)")
        ax[3].set_xlabel("|a_horiz| (m/s^2)")
        ax[3].set_ylabel("|roll,pitch| resid (deg)")
        ax[3].legend(); ax[3].grid(alpha=.3)
    else:
        ax[3].axis("off")

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
    ap.add_argument("--mocap-bag", default=None,
                    help="explicit mocap bag for ALL bags; default is to look "
                         "for <bag>_mocap alongside each one")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only.split(","))
    allres = {}

    for bag in args.bags:
        name = Path(bag).name
        if name.endswith("_mocap"):
            continue                       # companion bag, paired automatically
        print(f"\n=== {name} ===")

        mocap_bag = args.mocap_bag
        if mocap_bag is None:
            cand = Path(str(bag).rstrip("/") + "_mocap")
            mocap_bag = str(cand) if cand.exists() else None
        d = read_bag(bag, mocap_bag)
        if mocap_bag is None:
            print("  no paired _mocap bag -- using mocap from the main bag")
        present = [k for k in ("alt", "vel", "att", "gimbal", "mocap") if k in d]
        print("topics:", ", ".join(present))
        if "mocap" not in d:
            print("  WARNING: no mocap -- only rate/quantisation available")

        lag = sync_mocap(d)
        r = {"mocap_clock_lag_s": lag,
             "mocap_sync_corr": d.get("_mocap_corr"),
             "mocap_sync_signal": d.get("_mocap_sync_signal")}
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
    