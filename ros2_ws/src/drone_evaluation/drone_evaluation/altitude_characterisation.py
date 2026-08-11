#!/usr/bin/env python3
"""
analyze_altitude_bias.py
------------------------
Characterise DJI telemetry altitude (and, lightly, speed) against OptiTrack
ground truth from a rosbag2 recording.

What it computes (maps to the agreed measurement list):
  1. Effective rate + quantisation confirmation per telemetry topic.
  2. Aligned error signal  e(t) = z_dji - z_mocap_rel   (mocap resampled to telemetry stamps).
  3. Transport-lag estimate via cross-correlation on the step transitions (report-only by default).
  4. Per-hover-segment bias (mean e) and noise (std e).
  5. Scale-factor vs offset: linear fit of per-hover bias against true altitude.
  6. Noise vs quantisation floor (q^2/12) -> recovered true sensor noise + the R to give the EKF.
  7. Allan deviation on the longest hover (bias-instability / correlation time) if long enough.
  8. Hysteresis (asc vs desc) and ground-effect (lowest hover) as differences of segment means.

Clock handling: uses the rosbag RECORD timestamp for every topic, so all streams share
one clock (your recording machine). This avoids the un-synced OptiTrack PC clock problem.

Deps:  numpy, matplotlib (both pip). Reading uses rosbag2_py from your ROS 2 install.
Run with the workspace sourced so custom types resolve:
  source install/setup.bash
  python3 analyze_altitude_bias.py /path/to/bag_dir --out ./out
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOTE: reading is done via rosbag2_py (below) so custom types resolve from the
# sourced workspace. Run with:  source install/setup.bash

Q_ALT = 0.1        # confirmed altitude quantisation step (m)
Q_SPEED = 0.1      # confirmed speed quantisation step (m/s)

# Fixed topics / config (not parameters — confirmed for this platform)
ALT_TOPIC = "/drone_1/altitude_agl"
SPEED_TOPIC = "/drone_1/speed_vector"
MOCAP_TOPIC = "/optitrack/rigid_bodies/dji_mini4"
ATT_TOPIC = "/drone_1/attitude"          # std_msgs/String, e.g. "pitch <v> roll <v> yaw <v>"
VERT_AXIS = 2                     # OptiTrack vertical axis = z (confirmed)


# ----------------------------------------------------------------------------- reading
def _get_scalar(msg):
    """drone_interfaces/AltitudeAglStamped -> float (metres)."""
    return float(msg.altitude)


def _get_vec3(msg):
    """geometry_msgs/Vector3Stamped -> (x, y, z)."""
    v = msg.vector
    return float(v.x), float(v.y), float(v.z)


def _hdr_sec(msg):
    """Header stamp in seconds (for latency diagnostics)."""
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def _get_position(msg):
    """PoseStamped (or similar) -> (x, y, z), defensively."""
    if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
        p = msg.pose.position
    elif hasattr(msg, "position"):
        p = msg.position
    elif hasattr(msg, "transform"):          # TransformStamped fallback
        p = msg.transform.translation
    else:
        raise TypeError("Cannot find a position field on mocap message.")
    return float(p.x), float(p.y), float(p.z)


def _get_quat(msg):
    """PoseStamped (or similar) -> (x, y, z, w) orientation, or None."""
    o = None
    if hasattr(msg, "pose") and hasattr(msg.pose, "orientation"):
        o = msg.pose.orientation
    elif hasattr(msg, "orientation"):
        o = msg.orientation
    elif hasattr(msg, "transform"):
        o = msg.transform.rotation
    if o is None:
        return None
    return float(o.x), float(o.y), float(o.z), float(o.w)


def quat_to_R(q):
    """(x,y,z,w) -> 3x3 rotation matrix (body->world for a body-frame vector)."""
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


import json


def parse_attitude_string(s):
    """Parse the attitude/gimbal JSON string -> (roll,pitch,yaw) in radians, or None.
    Matches the node's convention: degrees in topic; pitch and yaw negated; roll as-is."""
    try:
        j = json.loads(s.replace("'", '"'))
        roll = np.radians(float(j["roll"]))
        pitch = np.radians(-float(j["pitch"]))
        yaw = np.radians(-float(j["yaw"]))
    except (ValueError, KeyError):
        return None
    return roll, pitch, yaw


def quat_norm(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else q


def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])


def quat_inv(q):
    x, y, z, w = quat_norm(q)
    return np.array([-x, -y, -z, w])


def quat_geodesic_deg(qa, qb):
    """Smallest rotation angle (deg) between two quaternions (xyzw)."""
    d = abs(float(np.dot(quat_norm(qa), quat_norm(qb))))
    d = min(1.0, d)
    return np.degrees(2 * np.arccos(d))


def quat_to_euler(q):
    """(x,y,z,w) -> (roll,pitch,yaw) rad, ZYX."""
    x, y, z, w = quat_norm(q)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sp, -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def mocap_health(t, P, v_max=6.0):
    """Report mocap stream quality. Returns (mask_good, stats).
    Flags: large frame gaps, and position jumps implying speed > v_max (glitches)."""
    dt = np.diff(t)
    med_dt = np.median(dt)
    gaps = int(np.sum(dt > 3 * med_dt))
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    implied_v = step / np.maximum(dt, 1e-6)
    jumps = implied_v > v_max                      # glitch between sample i and i+1
    # a glitch contaminates both endpoints -> mark both
    bad = np.zeros(len(t), dtype=bool)
    bad[:-1] |= jumps
    bad[1:] |= jumps
    stats = dict(med_dt=med_dt, gaps=gaps, n_jump=int(jumps.sum()),
                 frac_bad=float(bad.mean()))
    return ~bad, stats


def mocap_world_velocity(t, P, mask_good, win=5):
    """Robust world velocity from mocap position:
    drop glitch samples, then differentiate over a short window (least-squares slope)."""
    tg, Pg = t[mask_good], P[mask_good]
    v = np.zeros_like(Pg)
    h = win // 2
    for i in range(len(tg)):
        a, b = max(0, i - h), min(len(tg), i + h + 1)
        if b - a >= 2:
            tt = tg[a:b] - tg[a:b].mean()
            for k in range(3):
                # slope of local linear fit = velocity
                v[i, k] = np.polyfit(tt, Pg[a:b, k], 1)[0]
    return tg, v


def read_bag(bagpath):
    """Return dict of numpy arrays, timestamps in seconds (bag record time, zeroed).
    Uses rosbag2_py so custom types resolve from the sourced workspace.
    Topics are fixed module constants (ALT_TOPIC, SPEED_TOPIC, MOCAP_TOPIC, ATT_TOPIC)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    alt_t, alt_v, alt_hdr = [], [], []
    spd_t, spd_v = [], []
    moc_t, moc_p = [], []

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bagpath), storage_id=""),   # auto-detect db3/mcap
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if ALT_TOPIC not in type_map:
        raise SystemExit(f"Altitude topic {ALT_TOPIC} not in bag. Present: {list(type_map)}")
    if MOCAP_TOPIC not in type_map:
        raise SystemExit(f"Mocap topic {MOCAP_TOPIC} not in bag. Present: {list(type_map)}")
    has_speed = SPEED_TOPIC in type_map
    if not has_speed:
        print(f"  [info] speed topic not in this bag -> skipping velocity.")

    msgcls = {name: get_message(t) for name, t in type_map.items()}
    moc_q = []
    att_t, att_rpy = [], []                         # raw (roll,pitch,yaw) as parsed
    has_att = ATT_TOPIC in type_map

    while reader.has_next():
        topic, raw, ts = reader.read_next()
        t = ts * 1e-9  # bag RECORD time (single common clock)
        if topic == alt_topic:
            m = deserialize_message(raw, msgcls[topic])
            alt_t.append(t); alt_v.append(_get_scalar(m)); alt_hdr.append(_hdr_sec(m))
        elif has_speed and topic == SPEED_TOPIC:
            m = deserialize_message(raw, msgcls[topic])
            spd_t.append(t); spd_v.append(_get_vec3(m))
        elif has_att and topic == ATT_TOPIC:
            m = deserialize_message(raw, msgcls[topic])
            rpy = parse_attitude_string(m.data)     # std_msgs/String
            if rpy is not None:
                att_t.append(t); att_rpy.append(rpy)
        elif topic == MOCAP_TOPIC:
            m = deserialize_message(raw, msgcls[topic])
            moc_t.append(t); moc_p.append(_get_position(m))
            moc_q.append(_get_quat(m))

    if not alt_t:
        raise SystemExit("No altitude messages read.")
    if not moc_t:
        raise SystemExit("No mocap messages read.")

    t0 = min(alt_t[0], moc_t[0])
    data = {
        "alt_t": np.array(alt_t) - t0,
        "alt_v": np.array(alt_v),
        "alt_hdr": np.array(alt_hdr),
        "alt_rec_abs": np.array(alt_t),
        "moc_t": np.array(moc_t) - t0,
        "moc_p": np.array(moc_p),
    }
    if moc_q and all(q is not None for q in moc_q):
        data["moc_q"] = np.array(moc_q)             # (N,4) xyzw
    if att_t:
        rpy_rad = np.array(att_rpy)                 # (N,3) roll,pitch,yaw in radians
        aq = np.array([_euler_to_quat(*r) for r in rpy_rad])
        data["att_t"] = np.array(att_t) - t0
        data["att_q"] = aq                          # (N,4) xyzw
        data["att_rpy"] = rpy_rad
    if spd_t:
        data["spd_t"] = np.array(spd_t) - t0
        data["spd_v"] = np.array(spd_v)
    return data


def _euler_to_quat(roll, pitch, yaw):
    """ZYX euler (rad) -> (x,y,z,w)."""
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


# ----------------------------------------------------------------------------- helpers
def effective_rate(t):
    dt = np.diff(t)
    return 1.0 / np.median(dt), dt


def confirm_quant(v, q, name):
    d = np.diff(v)
    d = d[np.abs(d) > 1e-6]                      # drop repeats
    if len(d) == 0:
        print(f"  [{name}] signal is constant.")
        return
    ratios = np.round(d / q)
    resid = np.abs(d - ratios * q)
    print(f"  [{name}] step multiples-of-{q}: max residual = {resid.max():.4f} "
          f"(should be ~0); unique |steps| ~ {sorted(set(np.round(np.abs(d), 3)))[:6]} ...")


def pick_vertical_axis(moc_p):
    """OptiTrack may be Y-up or Z-up. Pick the axis with the largest range (the climb)."""
    ranges = moc_p.max(axis=0) - moc_p.min(axis=0)
    axis = int(np.argmax(ranges))
    return axis, ranges


def find_hover_segments(moc_t, moc_vert, v_thresh=0.05, min_dur=8.0, margin=2.0):
    """Detect near-stationary vertical segments from dense mocap. Returns list of (t0, t1)."""
    vel = np.gradient(moc_vert, moc_t)
    # light smoothing
    k = max(1, int(0.3 / np.median(np.diff(moc_t))))   # ~0.3 s window
    if k > 1:
        vel = np.convolve(vel, np.ones(k) / k, mode="same")
    still = np.abs(vel) < v_thresh
    segs = []
    i = 0
    n = len(still)
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            t0, t1 = moc_t[i], moc_t[j - 1]
            if (t1 - t0) >= min_dur:
                segs.append((t0 + margin, t1 - margin))   # trim transitions
            i = j
        else:
            i += 1
    return [s for s in segs if s[1] > s[0]]


def xcorr_lag(t_a, a, t_b, b, fs=20.0):
    """Estimate lag (s) that best aligns a onto b, via cross-correlation on a uniform grid."""
    t0 = max(t_a[0], t_b[0]); t1 = min(t_a[-1], t_b[-1])
    if t1 <= t0:
        return 0.0
    grid = np.arange(t0, t1, 1.0 / fs)
    ai = np.interp(grid, t_a, a); bi = np.interp(grid, t_b, b)
    ai -= ai.mean(); bi -= bi.mean()
    if ai.std() < 1e-9 or bi.std() < 1e-9:
        return 0.0
    c = np.correlate(ai, bi, mode="full")
    lag = (np.argmax(c) - (len(ai) - 1)) / fs
    return lag


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="Path to the rosbag2 directory (.db3 or .mcap inside).")
    ap.add_argument("--marker-offset", type=float, default=0.061,
                    help="mocap marker height above ground when landed (m); datum = z_mocap - this")
    ap.add_argument("--v-thresh", type=float, default=0.05, help="hover vertical-vel threshold (m/s)")
    ap.add_argument("--move-thresh", type=float, default=0.10,
                    help="min |vertical vel| (m/s) for a sample to count as a transition")
    ap.add_argument("--mocap-vmax", type=float, default=6.0,
                    help="max plausible drone speed (m/s); position jumps above this are glitches")
    ap.add_argument("--vel-win", type=int, default=5,
                    help="window (samples) for robust mocap velocity differentiation")
    ap.add_argument("--speed-resid-max", type=float, default=0.30,
                    help="max residual-std sum (m/s) to consider the speed frame resolved")
    ap.add_argument("--min-hover", type=float, default=6.0, help="min hover duration (s)")
    ap.add_argument("--trim", type=float, default=2.0,
                    help="seconds trimmed off each hover end to drop transitions (lower for short holds)")
    ap.add_argument("--apply-lag", action="store_true", help="apply estimated transport lag")
    ap.add_argument("--out", default="./altitude_analysis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = read_bag(args.bag)

    # --- latency (measurement) -------------------------------------------------
    if "alt_hdr" in d and np.all(d["alt_hdr"] > 0):
        lat = (d["alt_rec_abs"] - d["alt_hdr"]) * 1e3
        print(f"latency_ms: median={np.median(lat):.1f} iqr=[{np.percentile(lat,25):.1f},"
              f"{np.percentile(lat,75):.1f}]")

    # --- rate + quantisation ---------------------------------------------------
    r_alt, _ = effective_rate(d["alt_t"])
    sigma_q = Q_ALT / np.sqrt(12.0)
    print(f"rate_hz: {r_alt:.3f}   n: {len(d['alt_t'])}   q: {Q_ALT}   sigma_q: {sigma_q:.4f}")

    # --- vertical axis + datum -------------------------------------------------
    axis = VERT_AXIS                                # z, confirmed
    moc_vert = d["moc_p"][:, axis]
    moc_rel = moc_vert - args.marker_offset
    z_start = np.median(moc_vert[d["moc_t"] < d["moc_t"][0] + 1.0])
    grounded = abs(z_start - args.marker_offset) < 0.15
    print(f"axis: {'xyz'[axis]}   marker_offset: {args.marker_offset:.3f}   "
          f"z_start: {z_start:.3f}   grounded_start: {grounded}")

    # --- lag + aligned error ---------------------------------------------------
    lag = xcorr_lag(d["alt_t"], d["alt_v"], d["moc_t"], moc_rel)
    print(f"xcorr_lag_ms: {lag*1e3:.1f}")
    moc_t_use = d["moc_t"] - lag if args.apply_lag else d["moc_t"]
    moc_at_alt = np.interp(d["alt_t"], moc_t_use, moc_rel)
    err = d["alt_v"] - moc_at_alt

    # --- TRANSITION-BASED FIT (primary): e = k*alt + b over moving samples ------
    # velocity of the true altitude at each telemetry sample (up-positive)
    moc_v = np.gradient(moc_rel, d["moc_t"])
    moc_v_at_alt = np.interp(d["alt_t"], d["moc_t"], moc_v)
    moving = np.abs(moc_v_at_alt) > args.move_thresh
    print(f"\n[transition_fit]  moving_samples: {int(moving.sum())} / {len(err)}")
    if moving.sum() >= 20:
        A = np.vstack([moc_at_alt[moving], np.ones(moving.sum())]).T
        (k, b), res, *_ = np.linalg.lstsq(A, err[moving], rcond=None)
        resid = err[moving] - (k * moc_at_alt[moving] + b)
        sigma_res = float(np.std(resid))
        n = moving.sum()
        # slope/intercept std errors
        xm = moc_at_alt[moving]
        Sxx = np.sum((xm - xm.mean()) ** 2)
        se_k = sigma_res / np.sqrt(Sxx)
        se_b = sigma_res * np.sqrt(1.0 / n + xm.mean() ** 2 / Sxx)
        print(f"scale_factor_k: {k*100:+.2f} %   (se {se_k*100:.2f} %)")
        print(f"offset_b: {b:+.4f} m   (se {se_b:.4f} m)")
        print(f"residual_sigma: {sigma_res:.4f} m")
        # up vs down hysteresis
        up = moving & (moc_v_at_alt > 0); dn = moving & (moc_v_at_alt < 0)
        if up.sum() > 20 and dn.sum() > 20:
            ku = np.polyfit(moc_at_alt[up], err[up], 1)[0]
            kd = np.polyfit(moc_at_alt[dn], err[dn], 1)[0]
            print(f"hysteresis_k_up: {ku*100:+.2f} %   k_down: {kd*100:+.2f} %")
    else:
        k = b = sigma_res = np.nan
        print("scale_factor_k: n/a (insufficient motion)")

    # --- hover segments (secondary) -------------------------------------------
    segs = find_hover_segments(d["moc_t"], moc_rel, args.v_thresh, args.min_hover, args.trim)
    rows = []
    print(f"\n[hovers]  found: {len(segs)}")
    for (t0, t1) in segs:
        m = (d["alt_t"] >= t0) & (d["alt_t"] <= t1)
        if m.sum() < 5:
            continue
        true_alt = float(np.mean(moc_at_alt[m]))
        if true_alt < -0.2:
            continue
        bias = float(np.mean(err[m])); std = float(np.std(err[m]))
        bins = np.round(d["alt_v"][m] / Q_ALT).astype(int)
        n_cross = int(np.sum(np.abs(np.diff(bins)) > 0))
        dithered = n_cross >= 1
        rows.append((0.5*(t0+t1), true_alt, bias, std, m.sum(),
                     1.0 if dithered else 0.0, len(np.unique(bins)), n_cross))
        print(f"  alt={true_alt:5.2f}  bias={bias:+.3f}  std={std:.3f}  "
              f"{'dither' if dithered else 'midbin'}  cross={n_cross}  n={m.sum()}")
    rows = np.array(rows) if rows else np.empty((0, 8))

    # --- R for the filter ------------------------------------------------------
    R_floor = sigma_q ** 2
    if np.isfinite(sigma_res):
        R_use = max(sigma_res ** 2, R_floor)
        print(f"\nR_alt: {R_use:.5f}   (residual_sigma^2={sigma_res**2:.5f}, floor q^2/12={R_floor:.5f})")
    elif len(rows):
        dith = rows[:, 5] > 0.5
        s = np.median(rows[dith, 3]) if dith.any() else np.median(rows[:, 3])
        print(f"\nR_alt: {max(s**2, R_floor):.5f}   (floor q^2/12={R_floor:.5f})")

    # --- plots (only the useful two) -------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(d["alt_t"], d["alt_v"], ".", ms=2, label="DJI altitude")
    plt.plot(d["moc_t"], moc_rel, "-", lw=0.8, label="OptiTrack (rel. to ground)")
    plt.xlabel("t (s)"); plt.ylabel("altitude (m)"); plt.legend(); plt.grid(True)
    plt.title("Altitude: telemetry vs ground truth")
    plt.savefig(os.path.join(args.out, "altitude_timeseries.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # error vs true altitude, with the transition fit line
    plt.figure(figsize=(6, 4.5))
    if 'moving' in dir() and moving.sum() >= 20:
        plt.plot(moc_at_alt[moving], err[moving], ".", ms=2, alpha=0.4, label="moving samples")
        xs = np.linspace(moc_at_alt[moving].min(), moc_at_alt[moving].max(), 50)
        if np.isfinite(k):
            plt.plot(xs, k * xs + b, "-", color="tab:red",
                     label=f"fit: {k*100:+.2f}% + {b:+.3f} m")
    if len(rows):
        dm = rows[:, 5] > 0.5
        if dm.any():
            plt.plot(rows[dm, 1], rows[dm, 2], "o", color="tab:green", label="hover (dither)")
        if (~dm).any():
            plt.errorbar(rows[~dm, 1], rows[~dm, 2], yerr=Q_ALT/2, fmt="s",
                         color="tab:gray", alpha=0.7, capsize=3, label="hover (midbin bound)")
    plt.xlabel("true altitude (m)"); plt.ylabel("error e = dji - mocap (m)")
    plt.legend(); plt.grid(True); plt.title("Altitude error vs height (scale-factor fit)")
    plt.savefig(os.path.join(args.out, "error_vs_altitude.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # --- CSV -------------------------------------------------------------------
    if len(rows):
        np.savetxt(os.path.join(args.out, "hover_segments.csv"), rows, delimiter=",",
                   header="t_center,true_alt,bias,std,n,dithered,n_bins,n_cross", comments="")

    # --- SPEED: frame diagnosis + covariance -----------------------------------
    if "spd_t" in d:
        characterise_speed(d, moc_rel, moc_v, axis, args)

    # --- ATTITUDE: DJI vs mocap ------------------------------------------------
    if "att_t" in d and "moc_q" in d:
        characterise_attitude(d, args)


def characterise_speed(d, moc_rel, moc_v, axis, args):
    """Diagnose which frame the telemetry velocity is in, then estimate R per axis.
    Uses glitch-cleaned mocap velocity and aborts if the reference is non-physical."""
    print("\n[speed]")
    tel = d["spd_v"]
    ts = d["spd_t"]
    r_spd, _ = effective_rate(ts)
    print(f"rate_hz: {r_spd:.3f}   n: {len(ts)}   q: {Q_SPEED}   sigma_q: {Q_SPEED/np.sqrt(12):.4f}")

    # --- mocap health + cleaning ---
    mask_good, hs = mocap_health(d["moc_t"], d["moc_p"], v_max=args.mocap_vmax)
    print(f"mocap_health: med_dt={hs['med_dt']*1e3:.1f}ms  gaps={hs['gaps']}  "
          f"pos_jumps(>{args.mocap_vmax}m/s)={hs['n_jump']}  bad_frac={hs['frac_bad']*100:.1f}%")
    tg, vw = mocap_world_velocity(d["moc_t"], d["moc_p"], mask_good, win=args.vel_win)
    vw_at = np.vstack([np.interp(ts, tg, vw[:, i]) for i in range(3)]).T
    speed_true = np.linalg.norm(vw_at, axis=1)

    # --- HARD GUARD: reference must be physically plausible ---
    if speed_true.max() > args.mocap_vmax:
        print(f"ABORT: cleaned mocap velocity still reaches {speed_true.max():.1f} m/s "
              f"(> {args.mocap_vmax}). Ground truth is unreliable; speed R not computed. "
              f"Increase --vel-win, lower --mocap-vmax, or inspect the mocap stream.")
        return
    print(f"|v|_true range [{speed_true.min():.2f},{speed_true.max():.2f}] m/s  (after cleaning)")

    moving = speed_true > args.move_thresh
    if moving.sum() < 20:
        print("insufficient motion for speed characterisation.")
        return
    print(f"moving_samples: {int(moving.sum())} / {len(ts)}")

    # --- yaw-excitation check: can this flight even separate body vs world? -----
    if "moc_q" in d:
        yaws = np.array([np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),
                                    1-2*(q[1]**2+q[2]**2)) for q in d["moc_q"]])
        yaw_range = float(np.ptp(np.unwrap(yaws)))
        print(f"yaw_excitation: {np.degrees(yaw_range):.0f} deg "
              f"({'sufficient' if yaw_range > np.radians(60) else 'INSUFFICIENT -> body~world, '
               'frame cannot be resolved'})")

    # --- velocity lag: telemetry velocity is pre-filtered, so it trails truth ---
    # cross-correlate telemetry speed magnitude vs true speed on a uniform grid.
    tel_mag = np.linalg.norm(tel, axis=1)
    lag_v = xcorr_lag(ts, tel_mag, ts, speed_true, fs=10.0)   # both on ts grid
    print(f"velocity_lag_ms: {lag_v*1e3:.0f}  (telemetry trails truth if positive)")
    # apply the lag to the reference so residuals reflect noise, not phase delay
    ts_shift = ts - lag_v

    # reference velocities sampled at the LAG-SHIFTED telemetry stamps
    vw_at = np.vstack([np.interp(ts_shift, tg, vw[:, i]) for i in range(3)]).T

    def sums(a, b, mask):
        return np.std((a - b)[mask], axis=0)

    print("frame_test (residual std per axis, lower = better fit):")
    stdW = sums(tel, vw_at, moving)
    print(f"  H1 world : std=[{stdW[0]:.3f},{stdW[1]:.3f},{stdW[2]:.3f}]  sum={stdW.sum():.3f}")
    best = ("world", stdW, vw_at)
    if "moc_q" in d:
        Rmats = np.array([quat_to_R(q) for q in d["moc_q"]])
        idx = np.searchsorted(d["moc_t"], ts).clip(0, len(d["moc_t"]) - 1)
        v_body = np.einsum('nij,nj->ni', np.transpose(Rmats[idx], (0, 2, 1)), vw_at)
        stdB = sums(tel, v_body, moving)
        print(f"  H2 body  : std=[{stdB[0]:.3f},{stdB[1]:.3f},{stdB[2]:.3f}]  sum={stdB.sum():.3f}")
        if stdB.sum() < best[1].sum():
            best = ("body", stdB, v_body)
    else:
        print("  H2 body  : no mocap orientation in bag -> cannot test body frame.")

    # frame is only 'resolved' if one hypothesis clearly wins AND residuals are small
    ref = best[2]
    perms = [(0,1,2),(1,0,2),(0,2,1),(2,1,0),(1,2,0),(2,0,1)]
    signs = [(1,1,1),(-1,1,1),(1,-1,1),(1,1,-1),(-1,-1,1),(-1,1,-1),(1,-1,-1),(-1,-1,-1)]
    bestmap = None
    for p in perms:
        for s in signs:
            cand = np.column_stack([s[i]*tel[:, p[i]] for i in range(3)])
            st = np.std((cand - ref)[moving], axis=0)
            if bestmap is None or st.sum() < bestmap[0]:
                bestmap = (st.sum(), p, s, st)
    _, p, s, st = bestmap
    print(f"best_frame: {best[0]}   axis_perm: {p}   signs: {s}")
    print(f"residual_std_per_axis: [{st[0]:.3f},{st[1]:.3f},{st[2]:.3f}] m/s")

    resolved = st.sum() < args.speed_resid_max
    floor = (Q_SPEED**2) / 12.0
    R_raw = np.maximum(st**2, floor)

    # --- pre-filtering handling: inflate R by the decorrelation factor ----------
    # For AR(1) noise with lag-1 autocorr rho, effective independent samples shrink;
    # a practical EKF fix is to inflate R by (1+rho)/(1-rho) so the filter is not
    # overconfident about correlated measurements.
    cand = np.column_stack([s[i]*tel[:, p[i]] for i in range(3)])
    res = cand - ref                        # ref = winning frame (world or body)
    rho = np.zeros(3)
    for i in range(3):
        r = res[moving, i] - res[moving, i].mean()
        if len(r) > 5 and r.std() > 1e-6:
            rho[i] = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    infl = np.where(np.abs(rho) < 0.99, (1 + rho) / (1 - rho), 1.0)
    infl = np.clip(infl, 1.0, 50.0)
    R_infl = R_raw * infl

    print(f"resid_std_lagfixed: [{st[0]:.3f},{st[1]:.3f},{st[2]:.3f}] m/s")
    print(f"autocorr_lag1: [{rho[0]:+.2f},{rho[1]:+.2f},{rho[2]:+.2f}]  "
          f"inflation: [{infl[0]:.1f},{infl[1]:.1f},{infl[2]:.1f}]x")
    if resolved:
        print(f"R_speed_raw:   [{R_raw[0]:.4f},{R_raw[1]:.4f},{R_raw[2]:.4f}] (m/s)^2")
        print(f"R_speed_infl:  [{R_infl[0]:.4f},{R_infl[1]:.4f},{R_infl[2]:.4f}] (m/s)^2  <- use this")
    else:
        print(f"R_speed: NOT TRUSTED (residual sum {st.sum():.2f} > {args.speed_resid_max} m/s). "
              f"Need sufficient yaw excitation + sustained horizontal motion to resolve the frame.")


def characterise_attitude(d, args):
    """Compare DJI attitude to OptiTrack: constant frame offset, residual after removing it,
    per-axis roll/pitch/yaw error, lag, and R estimate."""
    print("\n[attitude]")
    at, aq = d["att_t"], d["att_q"]
    mt, mq = d["moc_t"], d["moc_q"]
    r_att, _ = effective_rate(at)
    print(f"rate_hz: {r_att:.3f}   n: {len(at)}")

    # nearest mocap quaternion at each attitude stamp
    idx = np.searchsorted(mt, at).clip(0, len(mt) - 1)
    mq_at = mq[idx]

    # DJI and mocap may be in different fixed frames: q_dji = q_off * q_mocap.
    # Estimate the constant offset q_off from the median relative rotation, remove it,
    # then the residual is the real attitude error.
    q_off = np.median(np.array([quat_mul(aq[i], quat_inv(mq_at[i]))
                                for i in range(len(aq))]), axis=0)
    q_off = quat_norm(q_off)
    ang = np.array([quat_geodesic_deg(quat_mul(q_off, mq_at[i]), aq[i])
                    for i in range(len(aq))])
    print(f"frame_offset_deg: {quat_geodesic_deg(q_off,(0,0,0,1)):.1f}  "
          f"(constant DJI<->mocap frame rotation; expected if different conventions)")
    print(f"attitude_error_deg: median={np.median(ang):.2f}  p95={np.percentile(ang,95):.2f}")

    # per-axis (roll/pitch/yaw) error after offset removal
    err_rpy = []
    for i in range(len(aq)):
        qcorr = quat_mul(q_off, mq_at[i])                 # mocap mapped into DJI frame
        qe = quat_mul(aq[i], quat_inv(qcorr))             # residual rotation
        err_rpy.append(quat_to_euler(qe))
    err_rpy = np.degrees(np.array(err_rpy))
    for i, ax in enumerate(("roll", "pitch", "yaw")):
        print(f"  {ax}_err_deg: mean={err_rpy[:,i].mean():+.2f} std={err_rpy[:,i].std():.2f}")

    # lag: correlate DJI yaw-rate vs mocap yaw-rate
    ay = np.unwrap([quat_to_euler(q)[2] for q in aq])
    my = np.unwrap([quat_to_euler(q)[2] for q in mq_at])
    lag = xcorr_lag(at, np.gradient(ay, at), at, np.gradient(my, at), fs=10.0)
    print(f"attitude_lag_ms: {lag*1e3:.0f}")

    # R (rad^2) from residual std, useful if fusing attitude as an orientation measurement
    R_att = np.radians(err_rpy.std(axis=0)) ** 2
    print(f"R_att_rpy_rad2: [{R_att[0]:.5f},{R_att[1]:.5f},{R_att[2]:.5f}]  "
          f"(use if fusing attitude; else attitude is an INPUT for velocity rotation)")


if __name__ == "__main__":
    main()
