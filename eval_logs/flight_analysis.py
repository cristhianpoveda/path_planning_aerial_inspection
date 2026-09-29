#!/usr/bin/env python3
"""flight_analysis.py -- closed-loop flight analysis against OptiTrack.

Input is the .npz produced by bag_to_npz.py (flight bag + mocap bag).
Output is a text report, a JSON dump of every number, and an optional
per-step CSV.

    python3 flight_analysis.py controller_03.npz --json c03.json --steps c03_steps.csv
    python3 flight_analysis.py --selftest

What it computes, in the order the report prints it:

  1  inventory          rates, gaps, header-vs-receive offsets per topic
  2  stamp chain        the VO_DELAY actually applied, telemetry packet
                        sharing, command rate/jitter, pose age at the command
  3  mocap sanity       duplicate fraction, tracked window, and which yaw
                        convention relates mocap to DJI attitude
  4  time alignment     per-channel lag against mocap, all referenced to the
                        attitude channel, plus VO_DELAY by two routes
  5  velocity           K_VEL and the DJI-velocity yaw datum, by complex fit
  6  plant              per-axis closed-loop gain, command latency, duty
  7  estimator          per-VO-epoch scale, RPE, drift, timing sensitivity
  8  tracking           per-setpoint-step response in both frames, hold drift,
                        and the single commanded-vs-achieved similarity fit
  9  control law        one-step re-simulation of the controller from the bag
 10  health             degraded duty, epoch rate, video gaps vs commands
"""
import argparse
import csv
import json
import math
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================ small maths ==


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def quat_yaw(q):
    """Yaw about z from (N,4) xyzw quaternions."""
    q = np.atleast_2d(np.asarray(q, float))
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_conj(q):
    q = np.atleast_2d(np.asarray(q, float)).copy()
    q[:, :3] *= -1.0
    return q


def quat_rel_angle(q):
    """Angle of the relative rotation between consecutive quaternions, rad.

    Frame independent and scale independent, which is what makes it usable
    for timing work across VO map rebuilds.
    """
    q = np.asarray(q, float)
    d = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def savgol_coeffs(win, poly, deriv, dt):
    """Savitzky-Golay coefficients, computed rather than imported (no scipy)."""
    half = win // 2
    x = np.arange(-half, half + 1) * dt
    A = np.vander(x, poly + 1, increasing=True)
    C = np.linalg.pinv(A)
    return C[deriv] * math.factorial(deriv)


def smooth_deriv(y, dt, win=11, poly=2, deriv=1):
    """Local-polynomial derivative of a uniformly sampled signal.

    Returns (value, valid) with the half-window at each end marked invalid.
    """
    y = np.asarray(y, float)
    win = int(win) | 1
    if y.size < win + 2:
        return np.zeros_like(y), np.zeros(y.shape[0], bool)
    c = savgol_coeffs(win, poly, deriv, dt)
    out = np.convolve(y, c[::-1], mode="same")
    valid = np.ones(y.shape[0], bool)
    half = win // 2
    valid[:half] = False
    valid[-half:] = False
    return out, valid


def mono(t, *arrays):
    """Sort by time and keep strictly increasing stamps."""
    t = np.asarray(t, float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    arrays = [np.asarray(a)[order] for a in arrays]
    keep = np.ones(t.size, bool)
    keep[1:] = np.diff(t) > 0
    return (t[keep],) + tuple(a[keep] for a in arrays)


def resample(t, y, grid, max_gap):
    """Linear interpolation onto grid with a validity mask.

    A grid point is valid only if it lies inside [t0, t1] and the source
    interval bracketing it is no longer than max_gap.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    single = y.ndim == 1
    Y = y[:, None] if single else y
    if t.size < 2:
        out = np.zeros((grid.size, Y.shape[1]))
        return (out[:, 0] if single else out), np.zeros(grid.size, bool)
    j = np.clip(np.searchsorted(t, grid), 1, t.size - 1)
    gap = t[j] - t[j - 1]
    valid = (grid >= t[0]) & (grid <= t[-1]) & (gap <= max_gap)
    out = np.empty((grid.size, Y.shape[1]))
    for k in range(Y.shape[1]):
        out[:, k] = np.interp(grid, t, Y[:, k])
    valid = valid & np.isfinite(out).all(axis=1)
    out[~valid] = np.nan
    return (out[:, 0] if single else out), valid


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 0 else 0.0


def lag_xcorr(t_a, a, t_b, b, fs=50.0, max_lag=1.5, max_gap=0.3,
              highpass_s=None, t_lo=None, t_hi=None):
    """Time shift of signal b relative to signal a, by masked correlation.

    Returns dict(lag, corr, n). lag > 0 means b is DELAYED by that much
    relative to a, i.e. a(t) matches b(t + lag).
    """
    t_a = np.asarray(t_a, float)
    t_b = np.asarray(t_b, float)
    lo = max(t_a[0], t_b[0]) if t_lo is None else t_lo
    hi = min(t_a[-1], t_b[-1]) if t_hi is None else t_hi
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 5.0:
        return dict(lag=float("nan"), corr=0.0, n=0)
    grid = np.arange(lo, hi, 1.0 / fs)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ka = np.isfinite(a)
    kb = np.isfinite(b)
    if ka.sum() < 10 or kb.sum() < 10:
        return dict(lag=float("nan"), corr=0.0, n=0)
    A, vA = resample(t_a[ka], a[ka], grid, max_gap)
    B, vB = resample(t_b[kb], b[kb], grid, max_gap)
    A = np.nan_to_num(A)
    B = np.nan_to_num(B)
    if highpass_s:
        w = max(3, int(highpass_s * fs) | 1)
        k = np.ones(w) / w
        for arr, v in ((A, vA), (B, vB)):
            base = np.convolve(np.where(v, arr, 0.0), k, mode="same")
            arr -= base
    K = int(round(max_lag * fs))
    best = (-2.0, 0, 0)
    curve = np.full(2 * K + 1, np.nan)
    for i, k in enumerate(range(-K, K + 1)):
        # correlate A[n] with B[n + k]
        if k >= 0:
            ia, ib = slice(0, A.size - k), slice(k, A.size)
        else:
            ia, ib = slice(-k, A.size), slice(0, A.size + k)
        m = vA[ia] & vB[ib]
        n = int(m.sum())
        if n < 50:
            continue
        c = _pearson(A[ia][m], B[ib][m])
        curve[i] = c
        if c > best[0]:
            best = (c, k, n)
    corr, k, n = best
    if n == 0:
        return dict(lag=float("nan"), corr=0.0, n=0)
    lag = k / fs
    i = k + K
    if 0 < i < curve.size - 1 and np.all(np.isfinite(curve[i - 1:i + 2])):
        y0, y1, y2 = curve[i - 1:i + 2]
        den = (y0 - 2 * y1 + y2)
        if den != 0:
            lag += 0.5 * (y0 - y2) / den / fs
    return dict(lag=float(lag), corr=float(corr), n=int(n))


def umeyama(X, Y, with_scale=True):
    """Least-squares similarity mapping X onto Y: Y ~= s R X + t."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    C = Yc.T @ Xc / X.shape[0]
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var = float((Xc ** 2).sum() / X.shape[0])
        s = float(np.trace(np.diag(D) @ S) / var) if var > 0 else 1.0
    else:
        s = 1.0
    t = my - s * R @ mx
    res = Y - (s * (R @ X.T).T + t)
    return dict(s=s, R=R, t=t,
                rms=float(np.sqrt((res ** 2).sum(1).mean())),
                yaw_deg=float(np.degrees(math.atan2(R[1, 0], R[0, 0]))))


def complex_gain(a_xy, b_xy):
    """Fit a_xy ~= K * Rot(psi) * b_xy over horizontal vector pairs.

    Solved in the complex plane: c = sum(a conj(b)) / sum(|b|^2).
    Returns (K, psi_rad, residual_fraction, n).
    """
    a = a_xy[:, 0] + 1j * a_xy[:, 1]
    b = b_xy[:, 0] + 1j * b_xy[:, 1]
    den = float(np.sum(np.abs(b) ** 2))
    if den <= 0 or a.size < 10:
        return float("nan"), float("nan"), float("nan"), int(a.size)
    c = np.sum(a * np.conj(b)) / den
    res = a - c * b
    frac = float(np.sqrt(np.mean(np.abs(res) ** 2) /
                         max(np.mean(np.abs(a) ** 2), 1e-12)))
    return float(np.abs(c)), float(np.angle(c)), frac, int(a.size)


def ned_to_enu(v):
    v = np.atleast_2d(np.asarray(v, float))
    return np.column_stack([v[:, 1], v[:, 0], -v[:, 2]])


def zoh(t_src, v_src, t_q):
    """Zero-order hold: value of the last sample at or before each query."""
    t_src = np.asarray(t_src, float)
    idx = np.searchsorted(t_src, t_q, side="right") - 1
    ok = idx >= 0
    idx = np.clip(idx, 0, t_src.size - 1)
    v = np.asarray(v_src)[idx]
    age = t_q - t_src[idx]
    return v, age, ok


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def rate_stats(t):
    d = np.diff(np.asarray(t, float))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return dict(hz=float("nan"), med_dt=float("nan"), p99_dt=float("nan"),
                    max_dt=float("nan"), n_gap_500ms=0)
    return dict(hz=float(1.0 / np.median(d)), med_dt=float(np.median(d)),
                p99_dt=pct(d, 99), max_dt=float(d.max()),
                n_gap_500ms=int((d > 0.5).sum()))


# ================================================================== report ==


class Report:
    def __init__(self):
        self.lines = []
        self.data = {}

    def h(self, title):
        self.lines.append("")
        self.lines.append("=" * 78)
        self.lines.append(title)
        self.lines.append("=" * 78)

    def p(self, s=""):
        self.lines.append(s)

    def kv(self, section, **kw):
        self.data.setdefault(section, {}).update(kw)

    def text(self):
        return "\n".join(self.lines)


# ==================================================================== data ==


def load(path):
    z = np.load(path, allow_pickle=False)
    D = {}
    for key in z.files:
        g, f = key.split("/", 1)
        D.setdefault(g, {})[f] = z[key]
    return D


def trim(D, t0, t1):
    """Keep only messages whose receive time falls in [t0, t1], seconds from
    the earliest message in the file.."""
    start = min(np.asarray(g["t_recv"], float).min() for g in D.values()
                if len(g.get("t_recv", [])))
    lo = start + (t0 if t0 is not None else -1e9)
    hi = start + (t1 if t1 is not None else 1e9)
    out = {}
    for g, rec in D.items():
        t = np.asarray(rec["t_recv"], float)
        m = (t >= lo) & (t <= hi)
        if m.sum() == 0:
            continue
        out[g] = {k: (np.asarray(v)[m] if len(np.asarray(v)) == t.size else v)
                  for k, v in rec.items()}
    return out


def have(D, group, *fields):
    if group not in D:
        return False
    return all(f in D[group] for f in fields)


def tsel(D, group, which):
    """Time vector for a group: 'hdr' falls back to 'recv' when unusable."""
    g = D[group]
    if which == "hdr" and "t_hdr" in g:
        t = np.asarray(g["t_hdr"], float)
        if np.isfinite(t).all() and t.size and (t.max() - t.min()) > 1.0:
            return t
    return np.asarray(g["t_recv"], float)


# =============================================================== sections ==


def sec_inventory(D, rep, ctx):
    rep.h("1  INVENTORY")
    rep.p(f"{'group':26s} {'n':>7s} {'dur_s':>8s} {'Hz':>7s} "
          f"{'p99 dt':>8s} {'max dt':>8s} {'gaps>0.5s':>9s} {'hdr-recv med':>13s}")
    inv = {}
    for g in sorted(D):
        tr = np.asarray(D[g]["t_recv"], float)
        if tr.size == 0:
            continue
        st = rate_stats(tr)
        off = float("nan")
        if "t_hdr" in D[g]:
            th = np.asarray(D[g]["t_hdr"], float)
            d = th - tr
            d = d[np.isfinite(d)]
            if d.size:
                off = float(np.median(d))
        rep.p(f"{g:26s} {tr.size:7d} {tr[-1]-tr[0]:8.1f} {st['hz']:7.2f} "
              f"{st['p99_dt']:8.3f} {st['max_dt']:8.3f} {st['n_gap_500ms']:9d} "
              f"{off:13.3f}")
        inv[g] = dict(n=int(tr.size), dur=float(tr[-1] - tr[0]),
                      hdr_minus_recv=off, **st)
    rep.p()
    rep.p("hdr-recv is the publisher stamp minus the recorder stamp. It is "
          "negative by\nthe transport delay for a live topic; a large "
          "magnitude means the stamp is\nnot a receive time (telemetry, "
          "localisation/pose) and says how far back it\nreaches.")
    rep.kv("inventory", **inv)
    ctx["inventory"] = inv


def sec_stamp_chain(D, rep, ctx, args):
    rep.h("2  STAMP CHAIN")
    out = {}

    # --- VO_DELAY as actually applied by the filter -----------------------
    if (have(D, "vo.pose", "t_hdr", "t_recv") and
            have(D, "localisation.pose", "t_hdr", "t_recv")):
        tvr, tvh = mono(np.asarray(D["vo.pose"]["t_recv"], float),
                        np.asarray(D["vo.pose"]["t_hdr"], float))
        tlr = np.asarray(D["localisation.pose"]["t_recv"], float)
        tlh = np.asarray(D["localisation.pose"]["t_hdr"], float)
        j = np.clip(np.searchsorted(tvr, tlr), 1, tvr.size - 1)
        pick = np.where(np.abs(tvr[j] - tlr) < np.abs(tvr[j - 1] - tlr),
                        j, j - 1)
        close = np.abs(tvr[pick] - tlr) < 0.25
        d = tvh[pick][close] - tlh[close]
        d = d[np.isfinite(d)]
        if d.size:
            med = float(np.median(d))
            sd = float(np.std(d))
            near = float(np.mean(np.abs(d - med) < 5e-3))
            rep.p(f"vo/pose stamp - localisation/pose stamp: "
                  f"median {med:+.4f} s, sd {sd:.4f}, "
                  f"{100*near:.1f} % within 5 ms of the median "
                  f"({d.size} matched frames)")
            rep.p("  -> this IS the VO_DELAY the node ran with. If it is not "
                  "the value you\n     think you launched, the launch "
                  "argument did not take.")
            out["vo_delay_applied"] = med
            ctx["vo_delay_applied"] = med
    # --- telemetry packet sharing ----------------------------------------
    if have(D, "attitude", "t_hdr") and have(D, "speed_vector", "t_hdr"):
        ta = np.unique(np.asarray(D["attitude"]["t_hdr"], float))
        ts = np.unique(np.asarray(D["speed_vector"]["t_hdr"], float))
        shared = float(np.mean(np.isin(ts, ta))) if ts.size else 0.0
        rep.p(f"speed_vector stamps also present in attitude: "
              f"{100*shared:.1f} %  (one FC packet, one stamp)")
        out["telemetry_shared_stamp_frac"] = shared
    if have(D, "relative_altitude", "t_hdr") and have(D, "attitude", "t_hdr"):
        ta = np.unique(np.asarray(D["attitude"]["t_hdr"], float))
        tz = np.unique(np.asarray(D["relative_altitude"]["t_hdr"], float))
        shared = float(np.mean(np.isin(tz, ta))) if tz.size else 0.0
        rep.p(f"relative_altitude stamps also present in attitude: "
              f"{100*shared:.1f} %")
        out["altitude_shared_stamp_frac"] = shared

    # --- command rate and jitter -----------------------------------------
    if have(D, "command.vel", "t_recv"):
        tc = np.asarray(D["command.vel"]["t_recv"], float)
        st = rate_stats(tc)
        rep.p(f"command/vel: {st['hz']:.2f} Hz, p99 period {st['p99_dt']*1e3:.0f} ms, "
              f"max {st['max_dt']*1e3:.0f} ms, {st['n_gap_500ms']} gaps > 500 ms")
        rep.p("  a gap above the phone's 200 ms watchdog means the aircraft "
              "hovered on\n  its own for that interval, whatever the "
              "controller thought it was doing.")
        out["command_rate"] = st
        n_over_200 = int((np.diff(tc) > 0.2).sum())
        rep.p(f"  command periods over 200 ms: {n_over_200}")
        out["command_gaps_over_watchdog"] = n_over_200

    # --- pose age at the command ------------------------------------------
    if have(D, "command.vel", "t_recv") and have(D, "localisation.pose",
                                                 "t_recv", "t_hdr"):
        tc = np.asarray(D["command.vel"]["t_recv"], float)
        tpr, tph = mono(np.asarray(D["localisation.pose"]["t_recv"], float),
                        np.asarray(D["localisation.pose"]["t_hdr"], float))
        stamp, _, ok = zoh(tpr, tph, tc)
        arrival, _, _ = zoh(tpr, tpr, tc)
        age_stamp = tc[ok] - stamp[ok]        # command time minus pose stamp
        age_arr = tc[ok] - arrival[ok]        # staleness of the last pose
        rep.p(f"pose age at command, against the pose STAMP: "
              f"median {np.median(age_stamp)*1e3:.0f} ms, "
              f"p90 {pct(age_stamp,90)*1e3:.0f} ms")
        rep.p(f"pose age at command, against pose ARRIVAL: "
              f"median {np.median(age_arr)*1e3:.0f} ms, "
              f"p90 {pct(age_arr,90)*1e3:.0f} ms")
        rep.p("  the first is the age the loop BELIEVES it has: it is the "
              "arrival delay\n  plus whatever VO_DELAY the node subtracted. "
              "The true age is that plus\n  (measured VO delay - applied "
              "VO_DELAY), assembled in section 11.")
        out["pose_age_stamp_med"] = float(np.median(age_stamp))
        out["pose_age_stamp_p90"] = pct(age_stamp, 90)
        out["pose_age_arrival_med"] = float(np.median(age_arr))
        ctx["pose_age_stamp_med"] = float(np.median(age_stamp))
    rep.kv("stamp_chain", **out)


def sec_mocap(D, rep, ctx, args):
    rep.h("3  MOCAP SANITY AND CONVENTION")
    out = {}
    if not have(D, "mocap", "p", "q"):
        rep.p("no mocap in the npz -- sections 3 to 8 will be skipped")
        return
    t = np.asarray(D["mocap"]["t_hdr"], float)
    if not np.isfinite(t).all() or (t.max() - t.min()) < 1.0:
        t = np.asarray(D["mocap"]["t_recv"], float)
        rep.p("mocap header stamps unusable; using receive stamps")
    p = np.asarray(D["mocap"]["p"], float)
    q = np.asarray(D["mocap"]["q"], float)

    dup = np.zeros(p.shape[0], bool)
    dup[1:] = np.all(p[1:] == p[:-1], axis=1) & np.all(q[1:] == q[:-1], axis=1)
    rep.p(f"samples {p.shape[0]}, duration {t[-1]-t[0]:.1f} s, "
          f"bit-identical repeats {100*dup.mean():.1f} %")
    rep.p("  repeats are dropped here: interpolating through them turns "
          "velocity\n  into a comb.")
    t, p, q, = mono(t[~dup], p[~dup], q[~dup])
    st = rate_stats(t)
    rep.p(f"after dedupe: {st['hz']:.1f} Hz, max gap {st['max_dt']*1e3:.0f} ms, "
          f"{st['n_gap_500ms']} gaps > 500 ms")

    # untracked windows show up as long gaps; report where they are
    d = np.diff(t)
    big = np.where(d > 0.5)[0]
    if big.size:
        rep.p(f"  untracked windows > 0.5 s: " +
              ", ".join(f"{t[i]-t[0]:.0f}s+{d[i]:.1f}s" for i in big[:10]) +
              (" ..." if big.size > 10 else ""))
    out["rate"] = st
    out["dup_frac"] = float(dup.mean())

    yaw_plain = np.unwrap(quat_yaw(q))
    yaw_conj = np.unwrap(quat_yaw(quat_conj(q)))
    ctx["mocap"] = dict(t=t, p=p, q=q, yaw_plain=yaw_plain, yaw_conj=yaw_conj)
    rep.kv("mocap", **out)


def _mocap_yaw_convention(D, rep, ctx):
    """Pick the mocap quaternion handling that makes mocap yaw and DJI yaw
    differ by a constant, and report how constant that is."""
    if "mocap" not in ctx or not have(D, "attitude", "yaw"):
        return
    m = ctx["mocap"]
    ta = tsel(D, "attitude", "hdr")
    ya = np.radians(np.asarray(D["attitude"]["yaw"], float))
    ta, ya = mono(ta, ya)
    ya_u = np.unwrap(ya)
    lo, hi = max(m["t"][0], ta[0]), min(m["t"][-1], ta[-1])
    grid = np.arange(lo, hi, 0.05)
    A, vA = resample(ta, ya_u, grid, 0.5)
    best = None
    for name, ym in (("as published", m["yaw_plain"]),
                     ("conjugated", m["yaw_conj"])):
        M, vM = resample(m["t"], ym, grid, 0.2)
        for sign in (+1.0, -1.0):
            v = vA & vM
            if v.sum() < 100:
                continue
            d = wrap_pi(M[v] - sign * A[v])
            c, s = np.cos(d).mean(), np.sin(d).mean()
            sd = math.degrees(math.sqrt(max(-2.0 * math.log(
                max(math.hypot(c, s), 1e-12)), 0.0)))
            off = math.degrees(math.atan2(s, c))
            cand = dict(handling=name, sign=sign, offset_deg=off, sd_deg=sd,
                        n=int(v.sum()))
            if best is None or sd < best["sd_deg"]:
                best = cand
    if best is None:
        return
    rep.p(f"mocap yaw = {best['sign']:+.0f} x DJI attitude yaw + "
          f"{best['offset_deg']:+.1f} deg, quaternion {best['handling']}, "
          f"sd {best['sd_deg']:.2f} deg over {best['n']} samples")
    rep.p("  a small sd is the check that both conventions and the rigid-body "
          "mounting\n  are what you think; the constant is per session and "
          "must not be inherited.")
    ctx["mocap_yaw"] = (m["yaw_plain"] if best["handling"] == "as published"
                        else m["yaw_conj"])
    ctx["yaw_convention"] = best
    rep.kv("mocap", yaw_convention=best)


def rpy_to_quat(roll, pitch, yaw):
    """ZYX roll-pitch-yaw to (N,4) xyzw quaternions."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.column_stack([sr * cp * cy - cr * sp * sy,
                            cr * sp * cy + sr * cp * sy,
                            cr * cp * sy - sr * sp * cy,
                            cr * cp * cy + sr * sp * sy])


def omega_signal(t, q, max_dt=0.5):
    """Rotation-rate magnitude between consecutive attitudes.
    """
    t = np.asarray(t, float)
    w = quat_rel_angle(q) / np.maximum(np.diff(t), 1e-6)
    tm = 0.5 * (t[1:] + t[:-1])
    ok = (np.diff(t) > 0) & (np.diff(t) < max_dt) & np.isfinite(w)
    return tm[ok], w[ok]


def deriv_signal(t, y, fs, win_s=0.15, max_gap=0.5):
    """Resample, smooth, differentiate. Never point to point."""
    t, y = mono(np.asarray(t, float), np.asarray(y, float))
    if t.size < 5:
        return np.array([]), np.array([])
    g = np.arange(t[0], t[-1], 1.0 / fs)
    Y, v = resample(t, y, g, max_gap)
    d, vd = smooth_deriv(np.nan_to_num(Y), 1.0 / fs, max(5, int(win_s * fs) | 1))
    k = v & vd
    return g[k], d[k]


def sec_alignment(D, rep, ctx, args):
    rep.h("4  TIME ALIGNMENT")
    if "mocap" not in ctx:
        return
    m = ctx["mocap"]
    fs = args.fs
    _mocap_yaw_convention(D, rep, ctx)
    yaw_m = ctx.get("mocap_yaw", m["yaw_plain"])
    yaw_sign = ctx.get("yaw_convention", {}).get("sign", -1.0)

    # ---- ground truth on a uniform grid ---------------------------------
    grid = np.arange(m["t"][0], m["t"][-1], 1.0 / fs)
    P, vP = resample(m["t"], m["p"], grid, 0.2)
    Y, vY = resample(m["t"], np.unwrap(yaw_m), grid, 0.2)
    win = max(5, int(0.15 * fs) | 1)
    vel = np.zeros_like(P)
    vd = np.ones(grid.size, bool)
    for k in range(3):
        vel[:, k], vk = smooth_deriv(np.nan_to_num(P[:, k]), 1.0 / fs, win)
        vd &= vk
    wz, wv = smooth_deriv(np.nan_to_num(Y), 1.0 / fs, win)
    valid = vP & vY & vd & wv
    speed = np.linalg.norm(vel, axis=1)
    ctx["gt"] = dict(t=grid, p=P, v=vel, yaw=Y, wz=wz, valid=valid,
                     speed=speed)
    rep.p(f"mocap resampled to {fs:.0f} Hz, smoothing window "
          f"{win/fs*1000:.0f} ms, {100*valid.mean():.1f} % valid")
    rep.p(f"mocap speed: median {np.nanmedian(speed[valid]):.3f} m/s, "
          f"p90 {pct(speed[valid],90):.3f}, max {np.nanmax(speed[valid]):.3f}")
    rep.p(f"mocap yaw rate: p90 |w| "
          f"{pct(np.abs(wz[valid]),90)*180/math.pi:.1f} deg/s")

    gt_t = grid[valid]
    gt_yawrate = wz[valid]
    gt_vz = vel[valid, 2]
    gt_speed = speed[valid]
    gt_om_t, gt_om = omega_signal(m["t"], m["q"])

    lags = {}
    L = lambda ta, a, tb, b, **kw: lag_xcorr(ta, a, tb, b, fs=fs,
                                             max_lag=args.max_lag, **kw)

    # ---- DJI attitude ----------------------------------------------------
    att_om = None
    if have(D, "attitude", "yaw"):
        ta = tsel(D, "attitude", "hdr")
        roll = np.radians(np.asarray(D["attitude"]["roll"], float))
        pitchd = np.asarray(D["attitude"]["pitch"], float)
        pitch = np.radians(pitchd)
        yawd = np.asarray(D["attitude"]["yaw"], float)
        ta, roll, pitch, yawd = mono(ta, roll, pitch, yawd)
        tay, way = deriv_signal(ta, np.unwrap(np.radians(yawd)), fs)
        lags["attitude"] = L(gt_t, gt_yawrate, tay, yaw_sign * way,
                             max_gap=0.5)
        q_att = rpy_to_quat(roll, pitch, -np.radians(yawd))
        att_om = omega_signal(ta, q_att)

    # ---- altitude --------------------------------------------------------
    if have(D, "relative_altitude", "altitude"):
        tz, dz = deriv_signal(tsel(D, "relative_altitude", "hdr"),
                              np.asarray(D["relative_altitude"]["altitude"],
                                         float), fs)
        lags["altitude"] = L(gt_t, gt_vz, tz, dz, max_gap=0.5)

    # ---- DJI velocity ----------------------------------------------------
    if have(D, "speed_vector", "v"):
        tv, vv = mono(tsel(D, "speed_vector", "hdr"),
                      np.asarray(D["speed_vector"]["v"], float))
        lags["velocity"] = L(gt_t, gt_speed, tv,
                             np.linalg.norm(vv, axis=1), max_gap=0.5)

    # ---- VO --------------------------------------------------------------
    vo_om = None
    if have(D, "vo.pose", "q"):
        tvo, qvo = mono(tsel(D, "vo.pose", "hdr"),
                        np.asarray(D["vo.pose"]["q"], float))
        tm, w = omega_signal(tvo, qvo)
        if have(D, "vo.status", "vo_epoch"):
            ts, ep = mono(tsel(D, "vo.status", "hdr"),
                          np.asarray(D["vo.status"]["vo_epoch"], float))
            e_at, _, _ = zoh(ts, ep, tm)
            seam = np.zeros(tm.size, bool)
            seam[1:] = e_at[1:] != e_at[:-1]
            tm, w = tm[~seam], w[~seam]
        vo_om = (tm, w)
        lags["vo_vs_mocap"] = L(gt_om_t, gt_om, tm, w, max_gap=0.3,
                                highpass_s=args.highpass)
        if att_om is not None:
            lags["vo_vs_attitude"] = L(att_om[0], att_om[1], tm, w,
                                       max_gap=0.3, highpass_s=args.highpass)

    # ---- published EKF pose ---------------------------------------------
    if have(D, "localisation.pose", "p"):
        tp, pp = mono(tsel(D, "localisation.pose", "hdr"),
                      np.asarray(D["localisation.pose"]["p"], float))
        tz2, dz2 = deriv_signal(tp, pp[:, 2], fs)
        lags["ekf_z_rate"] = L(gt_t, gt_vz, tz2, dz2, max_gap=0.5)

    # ---- commands --------------------------------------------------------
    if have(D, "command.vel", "lin"):
        tc, lin = mono(tsel(D, "command.vel", "hdr"),
                       np.asarray(D["command.vel"]["lin"], float))
        lags["command_vz"] = L(gt_t, gt_vz, tc, lin[:, 2], max_gap=0.3)
        lags["command_speed"] = L(
            gt_t, np.linalg.norm(vel[valid][:, :2], axis=1), tc,
            np.linalg.norm(lin[:, :2], axis=1), max_gap=0.3)

    # ---- anchor ----------------------------------------------------------
    anchor, ref = None, float("nan")
    for cand in ("attitude", "altitude", "velocity"):
        r = lags.get(cand)
        if r and r["corr"] > 0.5 and abs(r["lag"]) < 0.9 * args.max_lag:
            anchor, ref = cand, r["lag"]
            break
    rep.p()
    rep.p(f"{'channel':16s} {'lag vs mocap':>13s} {'vs anchor':>10s} "
          f"{'corr':>6s} {'n':>7s}")
    for k, v in lags.items():
        rel = v["lag"] - ref if np.isfinite(ref) else float("nan")
        flag = "  <- pinned at max_lag" if abs(
            v["lag"]) > 0.98 * args.max_lag else ""
        rep.p(f"{k:16s} {v['lag']:+13.3f} {rel:+10.3f} "
              f"{v['corr']:6.2f} {v['n']:7d}{flag}")
    rep.p()
    if anchor is None:
        rep.p("NO USABLE ANCHOR: no telemetry channel correlated with mocap "
              "above 0.5.\nEverything downstream of here runs unshifted and "
              "should be treated as\nunaligned. Check that the two bags "
              "overlap in time and that the mocap\nrigid body was tracked.")
    else:
        rep.p(f"anchored on '{anchor}': the mocap clock is shifted by "
              f"{-ref:+.3f} s onto the\nflight clock, which sets that "
              f"channel's residual lag to zero. Attitude,\naltitude and "
              "velocity share one telemetry packet and one stamp, so the "
              "choice\nof anchor moves everything by at most their physical "
              "differential (tens of ms).")
    rep.p("lag > 0 means the channel is DELAYED against mocap. Only the "
          "'vs anchor'\ncolumn is free of the unknown mocap-to-laptop clock "
          "offset. Trust a row only\nif corr is high and it is not pinned at "
          "the search limit.")

    if "vo_vs_attitude" in lags:
        v = lags["vo_vs_attitude"]
        rep.p()
        rep.p(f"VO_DELAY, mocap-free (VO |w| against DJI attitude |w|): "
              f"{v['lag']:.3f} s, corr {v['corr']:.2f}, n {v['n']}")
    if "vo_vs_mocap" in lags and np.isfinite(ref):
        v = lags["vo_vs_mocap"]
        rep.p(f"VO_DELAY, mocap route (VO |w| against mocap |w|, minus the "
              f"anchor): {v['lag']-ref:.3f} s, corr {v['corr']:.2f}")
    if "vo_delay_applied" in ctx:
        rep.p(f"VO_DELAY applied by the node this flight: "
              f"{ctx['vo_delay_applied']:.3f} s")
        rep.p("  measured minus applied is a constant bias on every published "
              "pose stamp.\n  Section 7 re-measures it independently as the "
              "shift that minimises RPE.")

    ctx["lags"] = lags
    ctx["lag_ref"] = ref
    ctx["anchor"] = anchor
    rep.kv("alignment", lags=lags, attitude_ref_lag=ref, anchor=anchor)

    if have(D, "mocap", "t_recv", "t_hdr"):
        d = np.asarray(D["mocap"]["t_recv"], float) - \
            np.asarray(D["mocap"]["t_hdr"], float)
        d = d[np.isfinite(d)]
        if d.size:
            rep.p()
            rep.p(f"mocap receive minus mocap header: median "
                  f"{np.median(d)*1e3:.0f} ms. If both bags were recorded on "
                  f"the same\nhost this is transport plus publisher latency, "
                  f"and it should be within a few\ntens of ms of the "
                  f"{-ref if np.isfinite(ref) else float('nan'):+.3f} s "
                  f"anchor offset above. A large disagreement means the two\n"
                  f"clocks are not the same and only the anchored numbers are "
                  f"usable.")
            rep.kv("alignment", mocap_recv_minus_hdr=float(np.median(d)))


def _gt_frame(ctx):
    """Ground truth resampled onto the flight timebase (attitude-referenced)."""
    if "gt" not in ctx:
        return None
    g = dict(ctx["gt"])
    shift = -ctx.get("lag_ref", 0.0)
    if not np.isfinite(shift):
        shift = 0.0
    g["t"] = g["t"] + shift
    return g


def sec_velocity(D, rep, ctx, args):
    rep.h("5  DJI VELOCITY: K_VEL AND YAW DATUM")
    gt = _gt_frame(ctx)
    if gt is None or not have(D, "speed_vector", "v"):
        rep.p("skipped")
        return
    fs = args.fs
    tv = tsel(D, "speed_vector", "hdr")
    v_ned = np.asarray(D["speed_vector"]["v"], float)
    tv, v_ned = mono(tv, v_ned)
    v_enu = ned_to_enu(v_ned)
    lag = ctx.get("lags", {}).get("velocity", {}).get("lag", float("nan"))
    ref = ctx.get("lag_ref", 0.0)
    shift = (lag - ref) if np.isfinite(lag) and np.isfinite(ref) else 0.0

    grid = gt["t"]
    V, vV = resample(tv - shift, v_enu, grid, 0.5)
    ok = vV & gt["valid"] & (gt["speed"] > args.v_low)
    n = int(ok.sum())
    if n < 100:
        rep.p(f"only {n} samples above {args.v_low} m/s -- not enough")
        return
    K, psi, frac, _ = complex_gain(V[ok][:, :2], gt["v"][ok][:, :2])
    # magnitude route, integrated path ratio
    dt = 1.0 / fs
    path_dji = float(np.sum(np.linalg.norm(V[ok], axis=1)) * dt)
    path_gt = float(np.sum(gt["speed"][ok]) * dt)
    K_path = path_dji / path_gt if path_gt > 0 else float("nan")
    # vertical
    a = V[ok][:, 2]
    b = gt["v"][ok][:, 2]
    K_z = float((a @ b) / (b @ b)) if (b @ b) > 0 else float("nan")

    rep.p(f"samples above {args.v_low} m/s: {n}  "
          f"(velocity shifted by {shift*1e3:+.0f} ms to the attitude timebase)")
    rep.p(f"horizontal complex fit   K_VEL = {K:.4f}, "
          f"yaw datum = {math.degrees(psi):+.2f} deg, "
          f"residual {100*frac:.1f} % of signal")
    rep.p(f"integrated path ratio    K_VEL = {K_path:.4f}  "
          f"({path_dji:.1f} m of DJI path against {path_gt:.1f} m of mocap)")
    rep.p(f"vertical regression      K_vz  = {K_z:.4f}")
    rep.p()
    rep.p("K_VEL is DJI speed divided by true speed, which is how the filter "
          "uses it\n(z = v_enu / K_VEL). The yaw datum is the rotation from "
          "the room frame to\nDJI's magnetic-north ENU; the filter absorbs it "
          "in delta-theta_z, so it is a\ndiagnostic, not a correction.")

    # speed dependence: is the low-speed droop still there?
    rep.p()
    rep.p(f"{'speed bin m/s':>14s} {'n':>7s} {'gain':>7s}")
    edges = [0.05, 0.15, 0.3, 0.5, 0.8, 1.2, 5.0]
    bins = {}
    okb = vV & gt["valid"] & (gt["speed"] > 0.02)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = okb & (gt["speed"] >= lo) & (gt["speed"] < hi)
        if m.sum() < 50:
            continue
        g = float(np.sum(np.linalg.norm(V[m], axis=1)) /
                  np.sum(gt["speed"][m]))
        bins[f"{lo}-{hi}"] = dict(n=int(m.sum()), gain=g)
        rep.p(f"{lo:6.2f}-{hi:<6.2f} {int(m.sum()):7d} {g:7.3f}")
    rep.p("  a gain that droops below ~0.4 m/s is the quantisation dead zone; "
          "it is a\n  bias, so inflating R cannot remove it.")
    ctx["K_VEL"] = K
    rep.kv("velocity", K_VEL_complex=K, K_VEL_path=K_path,
           yaw_datum_deg=math.degrees(psi), residual_frac=frac,
           K_vz=K_z, n=n, shift_applied=shift, bins=bins)


def sec_plant(D, rep, ctx, args):
    rep.h("6  PLANT AND COMMAND FIDELITY (closed loop)")
    gt = _gt_frame(ctx)
    if gt is None or not have(D, "command.vel", "lin"):
        rep.p("skipped")
        return
    fs = args.fs
    tc = tsel(D, "command.vel", "hdr")
    lin = np.asarray(D["command.vel"]["lin"], float)
    ang = np.asarray(D["command.vel"]["ang"], float)
    tc, lin, ang = mono(tc, lin, ang)

    grid = gt["t"]
    C, vC = resample(tc, lin, grid, 0.3)
    A, vA = resample(tc, ang[:, 2], grid, 0.3)
    active = vC & gt["valid"] & (np.linalg.norm(C[:, :2], axis=1) > 0.02)

    # command latency from the horizontal channel
    r = lag_xcorr(grid, np.where(gt["valid"],
                                 np.linalg.norm(gt["v"][:, :2], axis=1), np.nan),
                  tc, np.linalg.norm(lin[:, :2], axis=1), fs=fs,
                  max_lag=args.max_lag, max_gap=0.3)
    # positive lag means the command LEADS the motion by that much
    tau = -r["lag"]
    rep.p(f"command to motion latency: {tau*1e3:.0f} ms "
          f"(corr {r['corr']:.2f}, n {r['n']})")
    rep.p("  measured in closed loop, so it is the same quantity the pulse "
          "tests give\n  only if the correlation is high; treat a low corr as "
          "no measurement.")
    if not np.isfinite(tau):
        tau = 0.0

    Cs, vCs = resample(tc + tau, lin, grid, 0.3)
    As, vAs = resample(tc + tau, ang[:, 2], grid, 0.3)
    yaw = gt["yaw"]
    c, s = np.cos(-yaw), np.sin(-yaw)
    v_body = np.column_stack([c * gt["v"][:, 0] - s * gt["v"][:, 1],
                              s * gt["v"][:, 0] + c * gt["v"][:, 1],
                              gt["v"][:, 2]])
    m = vCs & gt["valid"] & (np.linalg.norm(Cs[:, :2], axis=1) > 0.05)
    out = {}
    if m.sum() > 100:
        K, psi, frac, n = complex_gain(v_body[m][:, :2], Cs[m][:, :2])
        rep.p(f"horizontal: achieved = {K:.3f} x commanded, rotated "
              f"{math.degrees(psi):+.1f} deg, residual {100*frac:.0f} %, n {n}")
        rep.p("  the rotation is the body-frame mismatch between the command "
              "axes and the\n  rigid body, i.e. the mocap nose offset plus "
              "any sign error. A residual\n  much above ~30 % means the "
              "linear model is not what is happening.")
        out.update(K_horizontal=K, body_offset_deg=math.degrees(psi),
                   residual_frac=frac, n_horizontal=int(n))
        # per-axis, using the fitted body offset so the axes are separated
        cc, ss = math.cos(-psi), math.sin(-psi)
        vb = np.column_stack([cc * v_body[:, 0] - ss * v_body[:, 1],
                              ss * v_body[:, 0] + cc * v_body[:, 1]])
        for i, name in ((0, "vx"), (1, "vy")):
            sel = m & (np.abs(Cs[:, i]) > 0.05)
            if sel.sum() > 50:
                g = float((vb[sel, i] @ Cs[sel, i]) / (Cs[sel, i] @ Cs[sel, i]))
                fwd = sel & (Cs[:, i] > 0)
                bwd = sel & (Cs[:, i] < 0)
                gf = (float((vb[fwd, i] @ Cs[fwd, i]) /
                            (Cs[fwd, i] @ Cs[fwd, i])) if fwd.sum() > 30
                      else float("nan"))
                gb = (float((vb[bwd, i] @ Cs[bwd, i]) /
                            (Cs[bwd, i] @ Cs[bwd, i])) if bwd.sum() > 30
                      else float("nan"))
                rep.p(f"  {name}: gain {g:.3f}  (+{gf:.3f} / -{gb:.3f}), "
                      f"n {int(sel.sum())}")
                out[f"K_{name}"] = g
                out[f"K_{name}_pos"] = gf
                out[f"K_{name}_neg"] = gb
    mz = vCs & gt["valid"] & (np.abs(Cs[:, 2]) > 0.03)
    if mz.sum() > 50:
        g = float((gt["v"][mz, 2] @ Cs[mz, 2]) / (Cs[mz, 2] @ Cs[mz, 2]))
        rep.p(f"  vz: gain {g:.3f}, n {int(mz.sum())}")
        out["K_vz"] = g
    ma = vAs & gt["valid"] & (np.abs(As) > math.radians(2))
    if ma.sum() > 50:
        g = float((gt["wz"][ma] @ As[ma]) / (As[ma] @ As[ma]))
        rep.p(f"  yaw rate: achieved / command_vel.angular.z = {g:.3f}, "
              f"n {int(ma.sum())}")
        rep.p(f"    command/vel already carries the x{args.yaw_scale} "
              f"compensation, so this is the\n    plant's own yaw gain and "
              f"should read near 1/{args.yaw_scale:.2f} = "
              f"{1/args.yaw_scale:.3f}. Delivered rate against\n    what the "
              f"controller asked for before compensation: "
              f"{g*args.yaw_scale:.3f} (want 1.0).\n    A negative value "
              f"means the sign chain is inverted somewhere.")
        out["K_yaw_vs_published"] = g
        out["K_yaw_effective"] = g * args.yaw_scale

    # duty: how much of the flight was saturated, in deadband, or gated off
    tot = int(vC.sum())
    if tot:
        sat = float(np.mean(np.linalg.norm(C[vC][:, :2], axis=1) >
                            0.99 * args.v_max_xy))
        zero = float(np.mean(np.linalg.norm(C[vC], axis=1) < 1e-6))
        rep.p()
        rep.p(f"command duty: {100*sat:.1f} % horizontally saturated at "
              f"{args.v_max_xy} m/s, {100*zero:.1f} % exactly zero "
              f"(gated, disabled or in deadband)")
        out["duty_saturated"] = sat
        out["duty_zero"] = zero
    ctx["cmd_latency"] = tau
    rep.kv("plant", latency=tau, latency_corr=r["corr"], **out)


def epochs(D, ctx, args):
    """VO epoch segments, in the localisation/pose stamp timebase."""
    if not have(D, "vo.status", "vo_epoch"):
        return []
    ts = tsel(D, "vo.status", "hdr")
    ep = np.asarray(D["vo.status"]["vo_epoch"], float)
    ts, ep = mono(ts, ep)
    delay = ctx.get("vo_delay_applied", 0.0)
    if not np.isfinite(delay):
        delay = 0.0
    ts = ts - delay
    idx = np.concatenate([[0], np.where(np.diff(ep) != 0)[0] + 1, [ts.size]])
    segs = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < 2:
            continue
        segs.append(dict(epoch=int(ep[a]), t0=float(ts[a]),
                         t1=float(ts[b - 1])))
    return segs


def _rpe_stats(de, dg):
    err = np.linalg.norm(de - dg, axis=1)
    ng = np.linalg.norm(dg, axis=1)
    return dict(n=int(err.size),
                rpe_pct=float(100 * np.median(err) / max(np.median(ng), 1e-9)),
                rpe_rms=float(np.sqrt(np.mean(err ** 2))),
                med_err=float(np.median(err)),
                med_gt=float(np.median(ng)),
                ratio=float(np.sum(np.linalg.norm(de, axis=1)) /
                            max(np.sum(ng), 1e-9)))


def _align_and_rpe(t_e, p_e, gt, t0, t1, win, fs, shift=0.0, min_disp=0.05):
    """Align an estimate segment to truth and score it over `win` windows.

    Every metric is computed twice: over all windows, and over the windows
    where the aircraft actually moved (true displacement >= min_disp). A
    percentage error against a 1 cm denominator is a noise measurement, not
    an accuracy measurement, and hovering flights are almost all such windows.
    """
    grid = np.arange(max(t0, gt["t"][0]) + 0.2,
                     min(t1, gt["t"][-1]) - 0.2, 1.0 / fs)
    if grid.size < int(3 * fs):
        return None
    E, vE = resample(t_e + shift, p_e, grid, 0.5)
    G, vG = resample(gt["t"], gt["p"], grid, 0.2)
    v = vE & vG & np.isfinite(E).all(1) & np.isfinite(G).all(1)
    if v.sum() < int(3 * fs):
        return None
    al_s = umeyama(E[v], G[v], with_scale=True)
    al_r = umeyama(E[v], G[v], with_scale=False)
    k = int(round(win * fs))
    if k >= v.size - 1:
        return None
    pair = v[:-k] & v[k:]
    if pair.sum() < 20:
        return None
    d_e = (E[k:] - E[:-k])[pair]
    d_g = (G[k:] - G[:-k])[pair]
    mov = np.linalg.norm(d_g, axis=1) >= min_disp
    res = {}
    for name, al in (("scaled", al_s), ("as_runs", al_r)):
        de = al["s"] * (al["R"] @ d_e.T).T
        res[name] = _rpe_stats(de, d_g)
        res[name + "_moving"] = (_rpe_stats(de[mov], d_g[mov])
                                 if mov.sum() >= 20 else None)
    return dict(n=int(v.sum()), dur=float(grid[v][-1] - grid[v][0]),
                fitted_scale=al_s["s"], align_yaw_deg=al_s["yaw_deg"],
                align_rms=al_s["rms"], rigid_rms=al_r["rms"],
                med_gt_disp=res["scaled"]["med_gt"],
                moving_frac=float(mov.mean()), n_moving=int(mov.sum()),
                **res, _align=al_s,
                _t0=float(grid[v][0]), _t1=float(grid[v][-1]))


def sec_estimator(D, rep, ctx, args):
    rep.h("7  ESTIMATOR ACCURACY, PER VO EPOCH")
    gt = _gt_frame(ctx)
    if gt is None or not have(D, "localisation.pose", "p"):
        rep.p("skipped")
        return
    t_e = tsel(D, "localisation.pose", "hdr")
    p_e = np.asarray(D["localisation.pose"]["p"], float)
    q_e = np.asarray(D["localisation.pose"]["q"], float)
    t_e, p_e, q_e = mono(t_e, p_e, q_e)

    segs = epochs(D, ctx, args)
    if not segs:
        segs = [dict(epoch=0, t0=float(t_e[0]), t1=float(t_e[-1]))]
    rep.p(f"{len(segs)} VO epochs over "
          f"{t_e[-1]-t_e[0]:.0f} s "
          f"(one every {(t_e[-1]-t_e[0])/max(len(segs),1):.0f} s)")
    rep.p()
    rep.p(f"{'epoch':>6s} {'t0':>7s} {'dur':>6s} {'fit s':>7s} {'yaw':>7s} "
          f"{'RPE%':>6s} {'runs%':>6s} {'RPEm%':>6s} {'runsm%':>7s} "
          f"{'mov%':>5s} {'|d_gt|m':>8s} {'s_state':>8s} {'shift':>6s}")
    rows = []
    aligns = []
    for sg in segs:
        if sg["t1"] - sg["t0"] < args.min_epoch:
            continue
        r = _align_and_rpe(t_e, p_e, gt, sg["t0"], sg["t1"], args.rpe_win,
                           args.fs, min_disp=args.min_disp)
        if r is None:
            continue

        # the shift scan uses the MOVING windows: a stationary window carries
        # no timing information, so including it only adds noise
        def score(x):
            m = x.get("scaled_moving")
            return (m or x["scaled"])["rpe_pct"]
        best = (score(r), 0.0, r)
        for sh in np.arange(-args.max_shift, args.max_shift + 1e-9, 0.05):
            rs = _align_and_rpe(t_e, p_e, gt, sg["t0"], sg["t1"],
                                args.rpe_win, args.fs, shift=sh,
                                min_disp=args.min_disp)
            if rs and score(rs) < best[0]:
                best = (score(rs), float(sh), rs)
        s_state = float("nan")
        if have(D, "localisation.status", "scale"):
            tst = tsel(D, "localisation.status", "hdr")
            sc = np.asarray(D["localisation.status"]["scale"], float)
            m = (tst >= sg["t0"]) & (tst <= sg["t1"])
            if m.sum():
                s_state = float(np.median(sc[m]))
        mv = r.get("scaled_moving")
        mvr = r.get("as_runs_moving")
        row = dict(epoch=sg["epoch"], t0=sg["t0"] - t_e[0],
                   dur=r["dur"], fitted_scale=r["fitted_scale"],
                   align_yaw_deg=r["align_yaw_deg"],
                   rpe_pct=r["scaled"]["rpe_pct"],
                   rpe_as_runs_pct=r["as_runs"]["rpe_pct"],
                   rpe_moving_pct=mv["rpe_pct"] if mv else float("nan"),
                   rpe_as_runs_moving_pct=(mvr["rpe_pct"] if mvr
                                           else float("nan")),
                   rpe_moving_rms=mv["rpe_rms"] if mv else float("nan"),
                   moving_frac=r["moving_frac"], n_moving=r["n_moving"],
                   med_gt_disp_moving=mv["med_gt"] if mv else float("nan"),
                   ratio=r["as_runs"]["ratio"],
                   ratio_moving=mvr["ratio"] if mvr else float("nan"),
                   med_gt_disp=r["med_gt_disp"],
                   scale_state=s_state,
                   best_shift=best[1], rpe_at_best_shift=best[0],
                   align_rms=r["align_rms"])
        rows.append(row)
        aligns.append((sg["t0"], sg["t1"], r["_align"]))
        rep.p(f"{row['epoch']:6d} {row['t0']:7.0f} {row['dur']:6.0f} "
              f"{row['fitted_scale']:7.3f} {row['align_yaw_deg']:+7.1f} "
              f"{row['rpe_pct']:6.1f} {row['rpe_as_runs_pct']:6.1f} "
              f"{row['rpe_moving_pct']:6.1f} "
              f"{row['rpe_as_runs_moving_pct']:7.1f} "
              f"{100*row['moving_frac']:5.1f} "
              f"{row['med_gt_disp_moving']:8.3f} "
              f"{row['scale_state']:8.3f} {row['best_shift']:+6.2f}")
    if not rows:
        rep.p("no epoch long enough to evaluate "
              f"(minimum {args.min_epoch:.0f} s)")
        return
    rep.p()
    rep.p("fit s   scale of the estimate against truth over the epoch. "
          "1.00 = correct.\n        s_state x fit s is the scale the filter "
          "should have been holding.\nRPE%    "
          f"{args.rpe_win:.0f} s relative position error after removing the "
          "fitted scale --\n        what the filter could achieve.\n"
          "runs%   same window, scale NOT removed -- what the controller "
          "actually got.\n"
          f"RPEm%   as RPE%, restricted to windows with at least "
          f"{args.min_disp*100:.0f} cm of true motion.\n"
          "runsm%  as runs%, same restriction. QUOTE THESE TWO. A percentage "
          "against a\n        1 cm denominator measures the noise floor, not "
          "accuracy.\n"
          "mov%    share of windows that qualify. Low means the flight was "
          "mostly\n        stationary and even the moving columns rest on "
          "few samples.\n"
          "shift   the time shift that minimises the MOVING RPE. If it is not "
          "near zero,\n        the pose stamp is biased and VO_DELAY is "
          "wrong by about that much.")
    arr = lambda k: np.array([r[k] for r in rows], float)
    dur = arr("dur")
    w = dur / dur.sum()
    rep.p()
    rep.p(f"duration-weighted, all windows:    RPE "
          f"{float(w@arr('rpe_pct')):.1f} %, as-runs "
          f"{float(w@arr('rpe_as_runs_pct')):.1f} %")
    rep.p(f"duration-weighted, moving windows: RPE "
          f"{float(w@arr('rpe_moving_pct')):.1f} %, as-runs "
          f"{float(w@arr('rpe_as_runs_moving_pct')):.1f} %, "
          f"median error {float(w@arr('rpe_moving_rms')):.3f} m")
    rep.p(f"fitted scale {float(w@arr('fitted_scale')):.3f}, "
          f"best shift {float(w@arr('best_shift')):+.2f} s, "
          f"moving windows {100*float(w@arr('moving_frac')):.0f} %")
    rep.p(f"epochs evaluated {len(rows)} of {len(segs)}; "
          f"longest {dur.max():.0f} s")
    if np.any(np.abs(arr("best_shift")) > 0.98 * args.max_shift):
        rep.p("WARNING: a best shift is pinned at +-max_shift; widen "
              "--max-shift before\n         reading the shift column.")
    ctx["aligns"] = aligns
    ctx["epoch_rows"] = rows
    rep.kv("estimator", epochs=rows,
           weighted_rpe=float(w @ arr("rpe_pct")),
           weighted_as_runs=float(w @ arr("rpe_as_runs_pct")),
           weighted_rpe_moving=float(w @ arr("rpe_moving_pct")),
           weighted_as_runs_moving=float(w @ arr("rpe_as_runs_moving_pct")),
           weighted_moving_frac=float(w @ arr("moving_frac")),
           weighted_scale=float(w @ arr("fitted_scale")),
           n_epochs=len(segs))

    # published yaw quality
    grid = np.arange(t_e[0], t_e[-1], 1.0 / args.fs)
    Ye, vY = resample(t_e, np.unwrap(quat_yaw(q_e)), grid, 0.5)
    G, vG = resample(gt["t"], gt["yaw"], grid, 0.2)
    m = vY & vG
    if m.sum() > 100:
        d = wrap_pi(Ye[m] - G[m])
        c, s = np.cos(d).mean(), np.sin(d).mean()
        sd = math.degrees(math.sqrt(max(-2 * math.log(max(math.hypot(c, s),
                                                          1e-12)), 0)))
        rep.p()
        rep.p(f"published EKF yaw minus mocap yaw: "
              f"{math.degrees(math.atan2(s,c)):+.1f} deg, sd {sd:.2f} deg")
        rep.p("  a small sd means yaw_source: ekf_pose is usable with a fixed "
              "offset;\n  the offset itself is per session.")
        rep.kv("estimator", ekf_yaw_offset_deg=math.degrees(math.atan2(s, c)),
               ekf_yaw_sd_deg=sd)


def sec_frame_drift(D, rep, ctx, args):
    """Is the odom->mocap map constant, or does it rotate during the flight?
    """
    rep.h("7b  FRAME STABILITY OVER TIME")
    gt = _gt_frame(ctx)
    if gt is None or not have(D, "localisation.pose", "p"):
        rep.p("skipped")
        return
    t_e, p_e, q_e = mono(tsel(D, "localisation.pose", "hdr"),
                         np.asarray(D["localisation.pose"]["p"], float),
                         np.asarray(D["localisation.pose"]["q"], float))
    yaw_e = np.unwrap(quat_yaw(q_e))
    lo = max(t_e[0], gt["t"][0])
    hi = min(t_e[-1], gt["t"][-1])
    if hi - lo < 2 * args.align_win:
        rep.p("overlap too short")
        return

    rows = []
    skipped = 0
    for w0 in np.arange(lo, hi - args.align_win, args.align_stride):
        w1 = w0 + args.align_win
        grid = np.arange(w0, w1, 1.0 / args.fs)
        E, vE = resample(t_e, p_e, grid, 0.5)
        G, vG = resample(gt["t"], gt["p"], grid, 0.2)
        Y, vY = resample(t_e, yaw_e, grid, 0.5)
        M, vM = resample(gt["t"], gt["yaw"], grid, 0.2)
        v = vE & vG
        if v.sum() < args.fs * 5:
            skipped += 1
            continue
        
        c = G[v].mean(0)
        extent = float(2.0 * np.max(np.linalg.norm(G[v] - c, axis=1)))
        if extent < args.align_min_extent:
            skipped += 1
            continue
        al = umeyama(E[v], G[v], with_scale=True)
        dy = float("nan")
        vv = v & vY & vM
        if vv.sum() > args.fs:
            d = wrap_pi(Y[vv] - M[vv])
            dy = math.degrees(math.atan2(np.sin(d).mean(), np.cos(d).mean()))
        rows.append(dict(t=float(w0 - lo), extent=extent, scale=al["s"],
                         yaw_deg=al["yaw_deg"], rms=al["rms"],
                         ekf_minus_mocap_yaw_deg=dy, n=int(v.sum())))
    if len(rows) < 3:
        rep.p(f"only {len(rows)} usable windows "
              f"({skipped} skipped for less than "
              f"{args.align_min_extent:.2f} m of extent) -- the aircraft did "
              f"not move enough for a\nwindowed fit. Nothing can be said "
              f"about frame drift from this flight.")
        rep.kv("frame_drift", n_windows=len(rows), n_skipped=skipped)
        return

    rep.p(f"{args.align_win:.0f} s windows every {args.align_stride:.0f} s, "
          f"{len(rows)} fitted, {skipped} skipped for extent below "
          f"{args.align_min_extent:.2f} m")
    rep.p()
    rep.p(f"{'t':>7s} {'extent':>7s} {'scale':>7s} {'pos yaw':>8s} "
          f"{'att yaw':>8s} {'diff':>7s} {'rms':>7s}")
    for r in rows:
        rep.p(f"{r['t']:7.0f} {r['extent']:7.2f} {r['scale']:7.3f} "
              f"{r['yaw_deg']:+8.1f} {r['ekf_minus_mocap_yaw_deg']:+8.1f} "
              f"{r['yaw_deg']-r['ekf_minus_mocap_yaw_deg']:+7.1f} "
              f"{r['rms']:7.3f}")

    t = np.array([r["t"] for r in rows])
    yw = np.unwrap(np.radians([r["yaw_deg"] for r in rows]))
    yw = np.degrees(yw)
    sc = np.array([r["scale"] for r in rows])
    ay = np.array([r["ekf_minus_mocap_yaw_deg"] for r in rows])
    A = np.column_stack([np.ones(t.size), t])
    slope_pos = float(np.linalg.lstsq(A, yw, rcond=None)[0][1] * 100)
    slope_sc = float(np.linalg.lstsq(A, sc, rcond=None)[0][1] * 100)
    ok = np.isfinite(ay)
    slope_att = (float(np.linalg.lstsq(A[ok], np.unwrap(np.radians(ay[ok])) *
                                       180 / np.pi, rcond=None)[0][1] * 100)
                 if ok.sum() > 2 else float("nan"))
    rep.p()
    rep.p(f"position-frame yaw: range {yw.max()-yw.min():.1f} deg, "
          f"sd {yw.std():.1f} deg, trend {slope_pos:+.2f} deg per 100 s")
    rep.p(f"published attitude yaw offset: "
          f"range {np.nanmax(ay)-np.nanmin(ay):.1f} deg, "
          f"sd {np.nanstd(ay):.1f} deg, trend {slope_att:+.2f} deg per 100 s")
    rep.p(f"fitted scale: range {sc.max()-sc.min():.3f}, sd {sc.std():.3f}, "
          f"trend {slope_sc:+.3f} per 100 s")
    rep.p(f"windowed alignment residual: median {np.median([r['rms'] for r in rows]):.3f} m")
    rep.p()
    rep.p("'pos yaw' is the odom->mocap rotation from WHERE THE AIRCRAFT "
          "WENT; 'att yaw'\nis the published EKF orientation against mocap "
          "yaw. They are not the same\nnumber -- they differ by the "
          "rigid-body mounting offset, which is the 'diff'\ncolumn and "
          "should match the constant in section 3. What matters is that all\n"
          "three are FLAT. A 'pos yaw' that walks while 'att yaw' holds means "
          "the\nposition frame is turning independently of the published "
          "orientation: a\nviewpoint planner would be pointed by one and "
          "moved by the other. It shows up\nin section 8 as a direction "
          "error no single rotation can absorb. A scale that\ntrends is `s` "
          "drifting inside one epoch.")
    rep.kv("frame_drift", windows=rows, yaw_range_deg=float(yw.max()-yw.min()),
           yaw_sd_deg=float(yw.std()), yaw_trend_deg_per_100s=slope_pos,
           att_yaw_trend_deg_per_100s=slope_att,
           scale_range=float(sc.max()-sc.min()),
           scale_trend_per_100s=slope_sc, n_windows=len(rows),
           n_skipped=skipped)


def _align_at(ctx, t):
    """Epoch alignment (estimate -> mocap) covering time t, or None."""
    for t0, t1, al in ctx.get("aligns", []):
        if t0 - 1.0 <= t <= t1 + 1.0:
            return al
    return None


def sec_tracking(D, rep, ctx, args):
    rep.h("8  SETPOINT TRACKING")
    if not have(D, "setpoint", "p"):
        rep.p("skipped: no setpoint topic")
        return
    gt = _gt_frame(ctx)
    ts = tsel(D, "setpoint", "recv")
    sp = np.asarray(D["setpoint"]["p"], float)
    sq = np.asarray(D["setpoint"]["q"], float)
    ts, sp, sq = mono(ts, sp, sq)
    syaw = quat_yaw(sq)

    step_i = [0]
    for i in range(1, ts.size):
        if (np.linalg.norm(sp[i] - sp[i - 1]) > args.step_min or
                abs(wrap_pi(syaw[i] - syaw[i - 1])) > math.radians(5)):
            step_i.append(i)
    rep.p(f"{ts.size} setpoint messages, {len(step_i)} distinct setpoints")

    t_e = tsel(D, "localisation.pose", "hdr")
    p_e = np.asarray(D["localisation.pose"]["p"], float)
    q_e = np.asarray(D["localisation.pose"]["q"], float)
    t_e, p_e, q_e = mono(t_e, p_e, q_e)
    yaw_e = np.unwrap(quat_yaw(q_e))

    # command activity and setpoint staleness, for the hold check below
    cmd_t = cmd_mag = None
    if have(D, "command.vel", "lin"):
        cmd_t, cmd_lin = mono(np.asarray(D["command.vel"]["t_recv"], float),
                              np.asarray(D["command.vel"]["lin"], float))
        cmd_mag = np.linalg.norm(cmd_lin, axis=1)

    en_t = en_v = None
    if have(D, "controller.enable", "data"):
        en_t = np.asarray(D["controller.enable"]["t_recv"], float)
        en_v = np.asarray(D["controller.enable"]["data"]).astype(bool)

    rows = []
    pairs_cmd, pairs_gt = [], []
    for k, i in enumerate(step_i):
        t0 = ts[i]
        t1 = ts[step_i[k + 1]] if k + 1 < len(step_i) else ts[-1]
        if t1 - t0 < args.step_min_dur:
            continue
        if en_t is not None:
            on, _, ok = zoh(en_t, en_v, np.array([t0]))
            if not (ok[0] and on[0]):
                continue
        tt = np.arange(t0, t1, 1.0 / args.fs)
        E, vE = resample(t_e, p_e, tt, 0.5)
        if vE.sum() < args.fs * 2:
            continue
        target = sp[i]
        err = np.linalg.norm(E - target, axis=1)
        e0 = err[vE][0] if vE.any() else float("nan")
        # settling in the estimator's own frame
        tol = max(args.settle_tol, args.deadband_xy)
        set_t = float("nan")
        below = vE & (err < tol)
        need = int(1.0 * args.fs)
        run = 0
        for j in range(tt.size):
            run = run + 1 if below[j] else 0
            if run >= need:
                set_t = tt[j - need + 1] - t0
                break
        tail = vE & (tt > t1 - min(5.0, (t1 - t0) * 0.4))
        ss = float(np.median(err[tail])) if tail.sum() > 5 else float("nan")
        dirv = target - E[vE][0]
        nrm = np.linalg.norm(dirv)
        over = float("nan")
        if nrm > 0.05:
            u = dirv / nrm
            proj = (E - E[vE][0]) @ u
            over = float(np.nanmax(np.where(vE, proj, np.nan)) - nrm)

        row = dict(t0=float(t0), dur=float(t1 - t0),
                   step_m=float(nrm), err0=float(e0),
                   settle_s=set_t, ss_err_ekf=ss, overshoot_ekf=over)

        # ---- the same step in truth -------------------------------------
        if gt is not None:
            al = _align_at(ctx, t0)
            G, vG = resample(gt["t"], gt["p"], tt, 0.2)
            if vG.sum() > args.fs * 2:
                g0 = G[vG][0]
                # achieved displacement over the whole segment
                d_gt = G[vG][-1] - g0
                d_cmd = target - E[vE][0]
                row["achieved_m"] = float(np.linalg.norm(d_gt))
                row["gain_true"] = (float(np.linalg.norm(d_gt) / nrm)
                                    if nrm > 0.05 else float("nan"))
                if nrm > args.step_min:
                    pairs_cmd.append(d_cmd)
                    pairs_gt.append(d_gt)
                if al is not None and nrm > args.step_min:
                    dc = al["R"] @ d_cmd
                    if np.linalg.norm(dc) > 1e-6 and \
                            np.linalg.norm(d_gt) > 1e-6:
                        cosang = float(np.dot(dc, d_gt) /
                                       (np.linalg.norm(dc) *
                                        np.linalg.norm(d_gt)))
                        row["dir_err_deg"] = float(
                            math.degrees(math.acos(np.clip(cosang, -1, 1))))
                        a2, b2 = dc[:2], d_gt[:2]
                        if np.linalg.norm(a2) > 1e-6 and \
                                np.linalg.norm(b2) > 1e-6:
                            row["dir_err_h_deg"] = float(math.degrees(wrap_pi(
                                math.atan2(b2[1], b2[0]) -
                                math.atan2(a2[1], a2[0]))))
                        u = dc / np.linalg.norm(dc)
                        row["gain_along_cmd"] = float(np.dot(d_gt, u) / nrm)
                        row["cross_track_m"] = float(
                            np.linalg.norm(d_gt - np.dot(d_gt, u) * u))
                if al is not None:
                    tgt_gt = al["s"] * (al["R"] @ target) + al["t"]
                    terr = np.linalg.norm(G - tgt_gt, axis=1)
                    row["ss_err_true"] = (float(np.median(terr[vG &
                                                              (tt > t1 - 5)]))
                                          if (vG & (tt > t1 - 5)).sum() > 5
                                          else float("nan"))
                # hold drift, after settling
                t_hold = t0 + (set_t + 1.0 if np.isfinite(set_t) else 5.0)
                hm = vG & (tt > t_hold)
                if hm.sum() > args.fs * 3:
                    th_q = tt[hm]
                    if cmd_t is not None:
                        cm, _, okc = zoh(cmd_t, cmd_mag, th_q)
                        row["hold_cmd_active_frac"] = float(
                            np.mean(okc & (cm > 1e-9)))
                    sp_age = th_q - zoh(ts, ts, th_q)[0]
                    stale = sp_age > args.setpoint_timeout
                    row["hold_sp_stale_frac"] = float(np.mean(stale))
                    gated = stale.copy()
                    if en_t is not None:
                        on, _, oke = zoh(en_t, en_v, th_q)
                        gated |= ~(oke & on.astype(bool))
                    if have(D, "localisation.status", "degraded"):
                        dg, _, okd = zoh(
                            np.asarray(D["localisation.status"]["t_recv"],
                                       float),
                            np.asarray(D["localisation.status"]["degraded"]
                                       ).astype(bool), th_q)
                        gated |= (okd & dg)
                    row["hold_gated_frac"] = float(np.mean(gated))
                if hm.sum() > args.fs * 3:
                    th = tt[hm] - tt[hm][0]
                    A = np.column_stack([np.ones(th.size), th])
                    coef, *_ = np.linalg.lstsq(A, G[hm], rcond=None)
                    drift = np.linalg.norm(coef[1][:2])
                    row["hold_s"] = float(th[-1])
                    row["hold_drift_mps"] = float(drift)
                    row["hold_excursion_m"] = float(
                        np.max(np.linalg.norm(G[hm] - G[hm][0], axis=1)))
        rows.append(row)

    if not rows:
        rep.p("no usable step segments "
              f"(need {args.step_min_dur:.0f} s and the controller enabled)")
        return

    fin = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    rep.p()
    rep.p(f"{len(rows)} step segments analysed, "
          f"{np.nansum(fin('dur')):.0f} s of armed flight")
    rep.p()
    rep.p("In the ESTIMATOR's own frame (what the controller believed):")
    rep.p(f"  settling time to {max(args.settle_tol,args.deadband_xy):.2f} m: "
          f"median {np.nanmedian(fin('settle_s')):.1f} s, "
          f"{int(np.isfinite(fin('settle_s')).sum())} of {len(rows)} settled")
    rep.p(f"  steady-state error: median "
          f"{np.nanmedian(fin('ss_err_ekf')):.3f} m, "
          f"p90 {pct(fin('ss_err_ekf'),90):.3f} m")
    rep.p(f"  overshoot: median {np.nanmedian(fin('overshoot_ekf')):.3f} m")
    if gt is not None:
        rep.p()
        rep.p("Against MOCAP (what actually happened):")
        rep.p(f"  achieved / commanded displacement: median "
              f"{np.nanmedian(fin('gain_true')):.3f}, "
              f"p10 {pct(fin('gain_true'),10):.3f}, "
              f"p90 {pct(fin('gain_true'),90):.3f}")
        rep.p(f"  true error against the mapped setpoint at rest: median "
              f"{np.nanmedian(fin('ss_err_true')):.3f} m, "
              f"p90 {pct(fin('ss_err_true'),90):.3f} m")
        rep.p(f"  direction error, commanded against achieved: median "
              f"{np.nanmedian(fin('dir_err_deg')):.1f} deg, "
              f"p90 {pct(fin('dir_err_deg'),90):.1f} deg")
        rep.p(f"  horizontal direction error, signed: median "
              f"{np.nanmedian(fin('dir_err_h_deg')):+.1f} deg, "
              f"sd {np.nanstd(fin('dir_err_h_deg')):.1f} deg")
        rep.p(f"  cross-track at the end of the step: median "
              f"{np.nanmedian(fin('cross_track_m')):.3f} m, "
              f"p90 {pct(fin('cross_track_m'),90):.3f} m")
        rep.p("  a signed direction error with a large sd and a small mean is "
              "a frame that\n  turns during the flight (section 7b); a "
              "constant offset is a fixed datum\n  error and could be "
              "removed with yaw_offset_deg.")
        rep.p()
        rep.p(f"  drift while holding: median "
              f"{np.nanmedian(fin('hold_drift_mps'))*100:.2f} cm/s, "
              f"p90 {pct(fin('hold_drift_mps'),90)*100:.2f} cm/s")
        rep.p(f"  excursion while holding: median "
              f"{np.nanmedian(fin('hold_excursion_m')):.3f} m, "
              f"max {np.nanmax(fin('hold_excursion_m')):.3f} m")
        gat = fin("hold_gated_frac")
        stale = fin("hold_sp_stale_frac")
        act = fin("hold_cmd_active_frac")
        if np.isfinite(gat).any():
            closed = gat < 0.1
            open_ = gat > 0.5
            drift = fin("hold_drift_mps")
            rep.p(f"  of {int(np.isfinite(gat).sum())} holds: "
                  f"{int(closed.sum())} closed-loop (controller authorised "
                  f"throughout),\n    {int(open_.sum())} open-loop "
                  f"(gated for over half the window). Setpoint stale in "
                  f"{100*np.nanmedian(stale):.0f} % of hold time; "
                  f"commands non-zero in "
                  f"{100*np.nanmedian(act):.0f} %")
            for name, m in (("closed-loop", closed), ("open-loop", open_)):
                if m.sum() >= 2:
                    rep.p(f"    {name:11s} drift median "
                          f"{np.nanmedian(drift[m])*100:.2f} cm/s, "
                          f"excursion median "
                          f"{np.nanmedian(fin('hold_excursion_m')[m]):.3f} m "
                          f"over {int(m.sum())} holds")
            rep.p("  gated means disabled, degraded, or the setpoint "
                  "publisher stopped: the\n  aircraft was on DJI's own hold, "
                  "not this controller, and drift measured\n  there says "
                  "nothing about closed-loop performance. Zero commands "
                  "inside the\n  deadband are still closed-loop, which is "
                  "why the split is on authority, not\n  on command "
                  "magnitude.")
        rep.p("  the estimator reports a settled hold while the aircraft "
              "walks: that walk\n  is the dead-reckoned x, y drift being "
              "chased by the controller, and it is\n  the number a planner "
              "has to live with.")

    if len(pairs_cmd) >= 4:
        X = np.array(pairs_cmd)
        Y = np.array(pairs_gt)
        al = umeyama(X, Y, with_scale=True)
        Kh, psi, frac, n = complex_gain(Y[:, :2], X[:, :2])
        rep.p()
        rep.p(f"commanded-versus-achieved displacement, {len(X)} steps:")
        rep.p(f"  similarity fit: scale {al['s']:.3f}, "
              f"yaw {al['yaw_deg']:+.1f} deg, residual {al['rms']:.3f} m")
        rep.p(f"  horizontal only: {Kh:.3f} x, rotated "
              f"{math.degrees(psi):+.1f} deg, residual {100*frac:.0f} %")
        rep.p("  scale is the end-to-end magnitude fidelity: ask for a "
              "metre, get `scale`\n  metres. The yaw is NOT an error -- "
              "commanded displacements are in odom and\n  achieved ones are "
              "in mocap, so they differ by the frame rotation by\n  "
              "construction. Check it against the epoch alignment yaw in "
              "section 7; a\n  disagreement, or a residual much above the "
              "step accuracy, means no single\n  rotation fits every step "
              "and the per-step direction errors above are the\n  place to "
              "look.")
        rep.kv("tracking", displacement_scale=al["s"],
               displacement_yaw_deg=al["yaw_deg"],
               displacement_rms=al["rms"],
               displacement_h_gain=Kh,
               displacement_h_yaw_deg=math.degrees(psi), n_steps=len(X))

    ctx["steps"] = rows
    rep.kv("tracking", steps=rows)


def sec_control_law(D, rep, ctx, args):
    rep.h("9  CONTROL LAW RE-SIMULATION")
    need = ["command.vel", "localisation.pose", "setpoint"]
    if not all(g in D for g in need):
        rep.p("skipped")
        return
    tc = np.asarray(D["command.vel"]["t_recv"], float)
    lin = np.asarray(D["command.vel"]["lin"], float)
    ang = np.asarray(D["command.vel"]["ang"], float)
    tc, lin, ang = mono(tc, lin, ang)

    tp, pp, qp = mono(np.asarray(D["localisation.pose"]["t_recv"], float),
                      np.asarray(D["localisation.pose"]["p"], float),
                      np.asarray(D["localisation.pose"]["q"], float))
    yaw_ekf = quat_yaw(qp)
    tsp, spp, spq = mono(np.asarray(D["setpoint"]["t_recv"], float),
                         np.asarray(D["setpoint"]["p"], float),
                         np.asarray(D["setpoint"]["q"], float))
    yaw_sp = quat_yaw(spq)

    P, age_p, okp = zoh(tp, pp, tc)
    Ye, _, _ = zoh(tp, yaw_ekf, tc)
    S, age_s, oks = zoh(tsp, spp, tc)
    Ys, _, _ = zoh(tsp, yaw_sp, tc)

    yaw_used = Ye + math.radians(args.yaw_offset_deg)
    if args.yaw_source == "dji_attitude" and have(D, "attitude", "yaw"):
        ta, ya = mono(np.asarray(D["attitude"]["t_recv"], float),
                      -np.radians(np.asarray(D["attitude"]["yaw"], float)))
        A, _, _ = zoh(ta, ya, tc)
        yaw_used = A + math.radians(args.yaw_offset_deg)

    e = S - P
    c, s = np.cos(-yaw_used), np.sin(-yaw_used)
    ex = c * e[:, 0] - s * e[:, 1]
    ey = s * e[:, 0] + c * e[:, 1]
    db = np.hypot(ex, ey) < args.deadband_xy
    ex = np.where(db, 0.0, ex)
    ey = np.where(db, 0.0, ey)
    vx = args.kp_xy * ex
    vy = args.kp_xy * ey
    n = np.hypot(vx, vy)
    scale = np.where(n > args.v_max_xy, args.v_max_xy / np.maximum(n, 1e-9), 1.0)
    vx, vy = vx * scale, vy * scale
    ez = np.where(np.abs(e[:, 2]) < args.deadband_z, 0.0, e[:, 2])
    vz = np.clip(args.kp_z * ez, -args.v_max_z, args.v_max_z)
    eyaw = wrap_pi(Ys - yaw_used)
    eyaw = np.where(np.abs(eyaw) < math.radians(args.deadband_yaw_deg), 0.0,
                    eyaw)
    r = np.clip(args.kp_yaw * eyaw, -math.radians(args.yaw_rate_max_deg),
                math.radians(args.yaw_rate_max_deg))

    pred = np.column_stack([vx, vy, vz, r])
    # gating: reproduce only the conditions visible in the bag
    gate = np.zeros(tc.size, bool)
    gate |= ~okp | (age_p > args.pose_timeout)
    gate |= ~oks | (age_s > args.setpoint_timeout)
    if have(D, "controller.enable", "data"):
        en, _, oke = zoh(np.asarray(D["controller.enable"]["t_recv"], float),
                         np.asarray(D["controller.enable"]["data"]).astype(bool),
                         tc)
        gate |= ~oke | ~en
    if args.gate_on_degraded and have(D, "localisation.status", "degraded"):
        dg, age_d, okd = zoh(
            np.asarray(D["localisation.status"]["t_recv"], float),
            np.asarray(D["localisation.status"]["degraded"]).astype(bool), tc)
        gate |= ~okd | dg
    pred[gate] = 0.0

    # one-step slew from the previous recorded command, so errors do not
    # accumulate: this is a prediction check, not a re-run
    prev = np.zeros_like(pred)
    prev[1:, :3] = lin[:-1]
    prev[1:, 3] = ang[:-1, 2] / args.yaw_scale
    dt = np.diff(tc, prepend=tc[0] - 1.0 / args.rate_hz)
    lim = np.outer(dt, [args.slew_xy, args.slew_xy, args.slew_z,
                        math.radians(args.slew_yaw_deg)])
    pred = prev + np.clip(pred - prev, -lim, lim)

    act = np.column_stack([lin, ang[:, 2] / args.yaw_scale])
    d = act - pred
    ok = np.isfinite(d).all(1)
    rep.p(f"parameters assumed: kp_xy {args.kp_xy}, kp_z {args.kp_z}, "
          f"kp_yaw {args.kp_yaw}, v_max_xy {args.v_max_xy}, "
          f"v_max_z {args.v_max_z},\n  yaw_rate_max {args.yaw_rate_max_deg} "
          f"deg/s, yaw_source {args.yaw_source}, "
          f"yaw_offset {args.yaw_offset_deg} deg,\n  deadband "
          f"{args.deadband_xy}/{args.deadband_z} m / "
          f"{args.deadband_yaw_deg} deg, yaw_scale {args.yaw_scale}")
    rep.p(f"samples {int(ok.sum())}, gated by reconstruction "
          f"{100*gate.mean():.1f} %, actually zero "
          f"{100*np.mean(np.linalg.norm(act,axis=1)<1e-9):.1f} %")
    for i, name in enumerate(("vx", "vy", "vz", "yaw_rate")):
        rep.p(f"  {name:9s} residual median "
              f"{np.median(np.abs(d[ok, i])):.4f}, p95 "
              f"{pct(np.abs(d[ok,i]),95):.4f}, max "
              f"{np.max(np.abs(d[ok,i])):.4f}")
    rep.p("  a residual at the 1e-3 level is stamp and ordering jitter. "
          "Anything larger\n  means a parameter above is not what the node "
          "ran with, and the regression\n  below says which.")

    # recover the gains without assuming them
    lin_reg = {}
    step = np.abs(act - prev)
    slewing = np.any(step >= 0.98 * lim, axis=1)
    m = (~gate) & (~slewing) & \
        (np.hypot(ex, ey) > args.deadband_xy + 0.02) & \
        (np.hypot(act[:, 0], act[:, 1]) < 0.98 * args.v_max_xy)
    if m.sum() > 100:
        num = float(act[m, 0] @ ex[m] + act[m, 1] @ ey[m])
        den = float(ex[m] @ ex[m] + ey[m] @ ey[m])
        lin_reg["kp_xy"] = num / den if den > 0 else float("nan")
    mz = (~gate) & (~slewing) & (np.abs(ez) > args.deadband_z + 0.02) & \
         (np.abs(act[:, 2]) < 0.98 * args.v_max_z)
    if mz.sum() > 50:
        lin_reg["kp_z"] = float((act[mz, 2] @ ez[mz]) / (ez[mz] @ ez[mz]))
    my = (~gate) & (~slewing) & \
         (np.abs(eyaw) > math.radians(args.deadband_yaw_deg + 1)) & \
         (np.abs(act[:, 3]) < 0.98 * math.radians(args.yaw_rate_max_deg))
    if my.sum() > 50:
        lin_reg["kp_yaw"] = float((act[my, 3] @ eyaw[my]) /
                                  (eyaw[my] @ eyaw[my]))
    if lin_reg:
        rep.p("  regressed from the bag: " +
              ", ".join(f"{k} = {v:.3f}" for k, v in lin_reg.items()))
    rep.kv("control_law",
           residual_median=[float(np.median(np.abs(d[ok, i])))
                            for i in range(4)],
           gated_frac=float(gate.mean()), regressed=lin_reg)


def sec_health(D, rep, ctx, args):
    rep.h("10  HEALTH, VO AND VIDEO")
    out = {}
    if have(D, "localisation.status", "degraded"):
        t = tsel(D, "localisation.status", "recv")
        dg = np.asarray(D["localisation.status"]["degraded"]).astype(bool)
        rep.p(f"degraded: {100*dg.mean():.1f} % of "
              f"{dg.size} status messages")
        out["degraded_frac"] = float(dg.mean())
        if "flags" in D["localisation.status"]:
            from collections import Counter
            cnt = Counter()
            for f in D["localisation.status"]["flags"]:
                for x in str(f).split(";"):
                    if x:
                        cnt[x.split("=")[0]] += 1
            rep.p("  flags: " + ", ".join(f"{k} {v}" for k, v in
                                          cnt.most_common()))
            out["flags"] = dict(cnt)
        if "scale" in D["localisation.status"]:
            sc = np.asarray(D["localisation.status"]["scale"], float)
            sg = np.asarray(D["localisation.status"]["sigma_scale"], float)
            rep.p(f"  scale state: min {sc.min():.3f}, median "
                  f"{np.median(sc):.3f}, max {sc.max():.3f}; "
                  f"sigma_s/s median {np.median(sg/np.maximum(sc,1e-9)):.3f}")
            out["scale_min"] = float(sc.min())
            out["scale_max"] = float(sc.max())
            out["scale_med"] = float(np.median(sc))
    segs = epochs(D, ctx, args)
    if segs:
        dur = np.array([s["t1"] - s["t0"] for s in segs])
        total = segs[-1]["t1"] - segs[0]["t0"]
        rep.p(f"VO epochs: {len(segs)} in {total:.0f} s "
              f"(one per {total/len(segs):.0f} s); "
              f"longest {dur.max():.0f} s, median {np.median(dur):.0f} s")
        rep.p(f"  epochs longer than 60 s: {int((dur>60).sum())}")
        out["n_epochs"] = len(segs)
        out["epoch_dur_max"] = float(dur.max())
        out["epoch_dur_med"] = float(np.median(dur))
    if have(D, "vo.status", "pose_valid"):
        pv = np.asarray(D["vo.status"]["pose_valid"]).astype(bool)
        rep.p(f"VO pose_valid: {100*pv.mean():.1f} % of frames")
        out["pose_valid_frac"] = float(pv.mean())
        if "tracking_state" in D["vo.status"]:
            from collections import Counter
            cnt = Counter(str(x) for x in D["vo.status"]["tracking_state"])
            rep.p("  tracking states: " +
                  ", ".join(f"{k} {100*v/pv.size:.1f} %"
                            for k, v in cnt.most_common()))
    if have(D, "camera.image", "t_recv"):
        t = np.asarray(D["camera.image"]["t_recv"], float)
        d = np.diff(t)
        st = rate_stats(t)
        rep.p(f"camera frames: {st['hz']:.1f} Hz, p99 interval "
              f"{st['p99_dt']*1e3:.0f} ms, max {st['max_dt']:.2f} s")
        big = np.where(d > args.freeze)[0]
        rep.p(f"  freezes over {args.freeze:.1f} s: {big.size}, "
              f"total {float(d[big].sum()) if big.size else 0:.1f} s lost")
        out["camera"] = st
        out["n_freeze"] = int(big.size)
        out["freeze_lost_s"] = float(d[big].sum()) if big.size else 0.0
        if big.size and have(D, "command.vel", "lin"):
            tc = np.asarray(D["command.vel"]["t_recv"], float)
            lin = np.asarray(D["command.vel"]["lin"], float)
            moving = np.linalg.norm(lin, axis=1) > 0.02
            onset = tc[1:][moving[1:] & ~moving[:-1]]
            rep.p(f"  command onsets (zero -> moving): {onset.size}")
            if onset.size:
                rows = []
                for i in big[:20]:
                    dtn = onset - t[i]
                    before = dtn[dtn <= 0]
                    nearest = float(before.max()) if before.size else float("nan")
                    rows.append((t[i] - t[0], d[i], nearest))
                rep.p(f"    {'t_freeze':>9s} {'len_s':>7s} "
                      f"{'since last command onset':>26s}")
                for a, b, cc in rows:
                    rep.p(f"    {a:9.1f} {b:7.2f} {cc:26.2f}")
                rep.p("  a cluster of small negative values is the "
                      "first-motion freeze; values\n  spread randomly mean "
                      "the freeze is not command-triggered.")
                out["freeze_vs_command"] = [
                    dict(t=a, len=b, since_onset=cc) for a, b, cc in rows]
    rep.kv("health", **out)


# ================================================================ selftest ==


def _synth(seed=0):
    """Synthetic flight with every constant injected, in the npz layout.
    """
    rng = np.random.default_rng(seed)
    T = dict(K_VEL=0.87, lag_att=0.038, lag_vel=0.073, lag_alt=0.058,
             video_latency=0.42, vo_delay_applied=0.40,
             psi_vel_deg=40.0,        # DJI ENU against the room
             alpha=1.25,              # true metres per estimated metre
             phi_deg=15.0,            # odom yaw against mocap
             mount_yaw_deg=20.0,      # rigid body against the nose
             plant_K=1.10, plant_Kz=1.01, plant_Kyaw=0.763, plant_tau=0.28,
             kp_xy=0.6, kp_z=0.5, kp_yaw=0.8, yaw_scale=1.31,
             v_max_xy=0.5, v_max_z=0.3, yaw_rate_max=math.radians(15.0),
             db_xy=0.02, db_z=0.02, db_yaw=math.radians(2.0),
             slew_xy=1.0, slew_z=1.0, slew_yaw=math.radians(60.0))
    dur, fs = 400.0, 100.0
    n = int(dur * fs)
    dt = 1.0 / fs
    t0 = 1000.0
    t = t0 + np.arange(n) * dt
    phi = math.radians(T["phi_deg"])
    alpha = T["alpha"]
    R_om = np.array([[math.cos(phi), -math.sin(phi), 0.0],
                     [math.sin(phi), math.cos(phi), 0.0],
                     [0.0, 0.0, 1.0]])          # odom -> mocap, before scale

    def to_odom(p):
        return (R_om.T @ p) / alpha

    # ---- setpoint messages at 5 Hz, stepping every 20 s ------------------
    t_sp = np.arange(t0 + 4.0, t0 + dur - 2.0, 0.2)
    sp_p = np.zeros((t_sp.size, 3))
    sp_y = np.zeros(t_sp.size)
    cur = np.array([0.0, 0.0, 1.2])
    cy = 0.0
    for i, tt in enumerate(t_sp):
        if i == 0 or (tt - t_sp[0]) // 20.0 != (t_sp[i - 1] - t_sp[0]) // 20.0:
            cur = cur + rng.uniform(-0.5, 0.5, 3) * np.array([1.0, 1.0, 0.4])
            cur[2] = float(np.clip(cur[2], 0.8, 1.8))
            cy = float(wrap_pi(cy + rng.uniform(-0.5, 0.5)))
        sp_p[i] = cur
        sp_y[i] = cy

    # ---- simulation ------------------------------------------------------
    p_true = np.zeros((n, 3))
    v_true = np.zeros((n, 3))
    head = np.zeros(n)                # aircraft heading in the mocap frame
    wz_true = np.zeros(n)
    p_true[0] = [0.4, -0.2, 1.2]
    t_vo = np.arange(t0 + 1.0, t0 + dur - 1.0, 0.05)     # frame times
    vo_i = 0
    ekf_t_recv, ekf_p, ekf_yaw, ekf_frame_t = [], [], [], []
    cmd_t, cmd_v, cmd_r = [], [], []
    last_cmd = np.zeros(4)
    ctrl_dt = 0.05
    t_next_ctrl = t0 + 2.0
    for i in range(1, n):
        now = t[i]
        # publish an EKF pose for every frame whose latency has elapsed
        while vo_i < t_vo.size and t_vo[vo_i] + T["video_latency"] <= now:
            k = int((t_vo[vo_i] - t0) * fs)
            ekf_frame_t.append(t_vo[vo_i])
            ekf_t_recv.append(t_vo[vo_i] + T["video_latency"])
            ekf_p.append(to_odom(p_true[k]) + rng.normal(0, 0.008, 3))
            ekf_yaw.append(head[k] - phi)
            vo_i += 1
        # controller tick
        if now >= t_next_ctrl and ekf_p:
            t_next_ctrl += ctrl_dt
            p_e = np.asarray(ekf_p[-1])
            y_e = ekf_yaw[-1]
            j = np.searchsorted(t_sp, now, side="right") - 1
            j = max(j, 0)
            e = sp_p[j] - p_e
            c, sn = math.cos(-y_e), math.sin(-y_e)
            ex = c * e[0] - sn * e[1]
            ey = sn * e[0] + c * e[1]
            if math.hypot(ex, ey) < T["db_xy"]:
                ex = ey = 0.0
            vx, vy = T["kp_xy"] * ex, T["kp_xy"] * ey
            nn = math.hypot(vx, vy)
            if nn > T["v_max_xy"]:
                vx *= T["v_max_xy"] / nn
                vy *= T["v_max_xy"] / nn
            ez = e[2] if abs(e[2]) > T["db_z"] else 0.0
            vz = float(np.clip(T["kp_z"] * ez, -T["v_max_z"], T["v_max_z"]))
            ey_ = wrap_pi(sp_y[j] - y_e)
            if abs(ey_) < T["db_yaw"]:
                ey_ = 0.0
            r = float(np.clip(T["kp_yaw"] * ey_, -T["yaw_rate_max"],
                              T["yaw_rate_max"]))
            want = np.array([vx, vy, vz, r])
            lim = np.array([T["slew_xy"], T["slew_xy"], T["slew_z"],
                            T["slew_yaw"]]) * ctrl_dt
            cmd = last_cmd + np.clip(want - last_cmd, -lim, lim)
            last_cmd = cmd
            cmd_t.append(now)
            cmd_v.append(cmd[:3].copy())
            cmd_r.append(cmd[3])
        # plant, with transport delay
        if cmd_t:
            k = np.searchsorted(np.asarray(cmd_t), now - T["plant_tau"],
                                side="right") - 1
            u = np.asarray(cmd_v[k]) if k >= 0 else np.zeros(3)
            ur = cmd_r[k] if k >= 0 else 0.0
        else:
            u, ur = np.zeros(3), 0.0
        cb, sb = math.cos(head[i - 1]), math.sin(head[i - 1])
        v = np.array([T["plant_K"] * (cb * u[0] - sb * u[1]),
                      T["plant_K"] * (sb * u[0] + cb * u[1]),
                      T["plant_Kz"] * u[2]]) + rng.normal(0, 0.004, 3)
        w = T["plant_Kyaw"] * (ur * T["yaw_scale"]) + rng.normal(0, 0.002)
        v_true[i] = v
        wz_true[i] = w
        p_true[i] = p_true[i - 1] + v * dt
        head[i] = head[i - 1] + w * dt

    yaw_mocap = head + math.radians(T["mount_yaw_deg"])

    def q_yaw(y):
        y = np.asarray(y, float)
        return np.column_stack([np.zeros_like(y), np.zeros_like(y),
                                np.sin(y / 2), np.cos(y / 2)])

    def samp(ts, arr):
        arr = np.atleast_2d(arr.T).T if arr.ndim > 1 else arr
        if arr.ndim == 1:
            return np.interp(ts, t, arr)
        return np.column_stack([np.interp(ts, t, arr[:, k])
                                for k in range(arr.shape[1])])

    D = {}
    D["mocap"] = dict(t_recv=t + 0.006, t_hdr=t, p=p_true, q=q_yaw(yaw_mocap))

    tt = np.arange(t0 + 1, t0 + dur - 1, 1 / 9.0)
    ya = -np.degrees(samp(tt - T["lag_att"], yaw_mocap)) + 30.0
    D["attitude"] = dict(
        t_recv=tt + 0.012, t_hdr=tt,
        roll=np.degrees(0.02 * np.sin(2 * np.pi * tt / 7.0)),
        pitch=np.degrees(0.02 * np.cos(2 * np.pi * tt / 5.0)),
        yaw=ya)
    psi = math.radians(T["psi_vel_deg"])
    vs = samp(tt - T["lag_vel"], v_true) * T["K_VEL"]
    c, sn = math.cos(psi), math.sin(psi)
    v_enu = np.column_stack([c * vs[:, 0] - sn * vs[:, 1],
                             sn * vs[:, 0] + c * vs[:, 1], vs[:, 2]])
    D["speed_vector"] = dict(t_recv=tt + 0.012, t_hdr=tt,
                             v=np.column_stack([v_enu[:, 1], v_enu[:, 0],
                                                -v_enu[:, 2]]))
    D["relative_altitude"] = dict(
        t_recv=tt + 0.012, t_hdr=tt,
        altitude=samp(tt - T["lag_alt"], p_true[:, 2]) + 0.15)
    D["gimbal_joint_attitude"] = dict(
        t_recv=tt + 0.012, t_hdr=tt, roll=np.zeros_like(tt),
        pitch=np.full(tt.size, 30.0), yaw=np.zeros_like(tt))

    # VO: unscaled, in its own frame, stamped at decode time
    p_vo = samp(t_vo, p_true) / 3.5
    D["vo.pose"] = dict(t_recv=t_vo + T["video_latency"] + 0.004,
                        t_hdr=t_vo + T["video_latency"],
                        p=p_vo, q=q_yaw(samp(t_vo, head)))
    ep = np.floor((t_vo - t_vo[0]) / 90.0).astype(int)
    D["vo.status"] = dict(t_recv=t_vo + T["video_latency"] + 0.004,
                          t_hdr=t_vo + T["video_latency"],
                          vo_epoch=ep,
                          n_map_points=np.full(t_vo.size, 300),
                          pose_valid=np.ones(t_vo.size, bool),
                          tracking_state=np.array(["OK"] * t_vo.size))

    ekf_frame_t = np.asarray(ekf_frame_t)
    ekf_hdr = ekf_frame_t + T["video_latency"] - T["vo_delay_applied"]
    D["localisation.pose"] = dict(
        t_recv=np.asarray(ekf_t_recv) + 0.001, t_hdr=ekf_hdr,
        p=np.asarray(ekf_p), q=q_yaw(np.asarray(ekf_yaw)),
        cov=np.zeros((ekf_hdr.size, 36)))
    D["localisation.status"] = dict(
        t_recv=np.asarray(ekf_t_recv) + 0.001, t_hdr=ekf_hdr,
        scale=np.full(ekf_hdr.size, 3.5),
        sigma_scale=np.full(ekf_hdr.size, 0.1),
        alt_bias=np.zeros(ekf_hdr.size),
        sigma_alt_bias=np.zeros(ekf_hdr.size),
        cov_scale_bias=np.zeros(ekf_hdr.size),
        degraded=np.zeros(ekf_hdr.size, bool),
        state=np.array(["OK"] * ekf_hdr.size),
        flags=np.array([""] * ekf_hdr.size))

    cmd_t = np.asarray(cmd_t)
    D["command.vel"] = dict(
        t_recv=cmd_t + 0.002, t_hdr=cmd_t, lin=np.asarray(cmd_v),
        ang=np.column_stack([np.zeros(cmd_t.size), np.zeros(cmd_t.size),
                             np.asarray(cmd_r) * T["yaw_scale"]]))
    D["setpoint"] = dict(t_recv=t_sp, t_hdr=t_sp, p=sp_p, q=q_yaw(sp_y))
    D["controller.enable"] = dict(t_recv=np.array([t0 + 1.5]),
                                  data=np.array([True]))
    ti = t_vo + T["video_latency"]
    D["camera.image"] = dict(t_recv=ti, t_hdr=ti,
                             size=np.full(ti.size, 20000))
    return D, T


def selftest(args):
    D, T = _synth()
    rep = Report()
    args.yaw_source = "ekf_pose"
    args.kp_xy, args.kp_z, args.kp_yaw = 0.6, 0.5, 0.8
    args.v_max_xy, args.v_max_z = 0.5, 0.3
    args.yaw_rate_max_deg = 15.0
    run(D, rep, args)
    print(rep.text())
    got = rep.data
    print("\n" + "=" * 78)
    print("SELF-TEST: injected against recovered")
    print("=" * 78)
    checks = []

    def chk(name, want, val, tol):
        val = float(val) if val is not None else float("nan")
        ok = np.isfinite(val) and abs(val - want) <= tol
        checks.append((name, ok))
        print(f"  {name:36s} want {want:8.3f}  got {val:8.3f}  "
              f"tol {tol:5.3f}  {'ok' if ok else 'FAIL'}")

    lags = got.get("alignment", {}).get("lags", {})
    ref = got.get("alignment", {}).get("attitude_ref_lag", float("nan"))
    g = lambda k, f="lag": lags.get(k, {}).get(f, float("nan"))
    chk("attitude lag vs mocap", T["lag_att"], g("attitude"), 0.03)
    chk("velocity lag, vs anchor", T["lag_vel"] - T["lag_att"],
        g("velocity") - ref, 0.03)
    chk("altitude lag, vs anchor", T["lag_alt"] - T["lag_att"],
        g("altitude") - ref, 0.03)
    chk("VO_DELAY, mocap-free", T["video_latency"] - T["lag_att"],
        g("vo_vs_attitude"), 0.05)
    chk("VO_DELAY, mocap route", T["video_latency"] - T["lag_att"],
        g("vo_vs_mocap") - ref, 0.05)
    chk("VO_DELAY applied", T["vo_delay_applied"],
        got.get("stamp_chain", {}).get("vo_delay_applied", float("nan")), 0.01)
    chk("pose age believed", T["vo_delay_applied"] + 0.03,
        got.get("stamp_chain", {}).get("pose_age_stamp_med", float("nan")),
        0.03)
    chk("pose age true", T["video_latency"] + 0.03,
        got.get("summary", {}).get("pose_age_true", float("nan")), 0.05)
    v = got.get("velocity", {})
    chk("K_VEL (complex fit)", T["K_VEL"], v.get("K_VEL_complex"), 0.02)
    chk("K_VEL (path ratio)", T["K_VEL"], v.get("K_VEL_path"), 0.03)
    chk("DJI velocity yaw datum, deg", T["psi_vel_deg"],
        v.get("yaw_datum_deg"), 3.0)
    pl = got.get("plant", {})
    chk("plant gain, horizontal", T["plant_K"], pl.get("K_horizontal"), 0.06)
    chk("body offset, deg", -T["mount_yaw_deg"], pl.get("body_offset_deg"),
        4.0)
    chk("command latency", T["plant_tau"], pl.get("latency"), 0.06)
    chk("plant gain, vz", T["plant_Kz"], pl.get("K_vz"), 0.08)
    chk("plant gain, yaw", T["plant_Kyaw"], pl.get("K_yaw_vs_published"), 0.08)
    est = got.get("estimator", {})
    chk("fitted scale (alpha)", T["alpha"], est.get("weighted_scale"), 0.03)
    shifts = [r["best_shift"] for r in est.get("epochs", [])]
    chk("RPE-minimising shift", 0.0,
        float(np.median(shifts)) if shifts else float("nan"), 0.1)
    tr = got.get("tracking", {})
    chk("commanded->achieved scale", T["alpha"],
        tr.get("displacement_scale"), 0.08)
    chk("commanded->achieved yaw, deg", T["phi_deg"],
        tr.get("displacement_yaw_deg"), 3.0)
    fd = got.get("frame_drift", {})
    chk("frame yaw drift, deg/100 s", 0.0,
        fd.get("yaw_trend_deg_per_100s", float("nan")), 0.5)
    chk("frame yaw sd, deg", 0.0, fd.get("yaw_sd_deg", float("nan")), 1.5)
    steps_csv = tr.get("steps", [])
    de = [r.get("dir_err_deg", np.nan) for r in steps_csv]
    chk("step direction error, deg", 0.0,
        float(np.nanmedian(de)) if de else float("nan"), 6.0)
    cl = got.get("control_law", {})
    res = cl.get("residual_median", [np.nan] * 4)
    chk("control-law residual, vx", 0.0, res[0], 5e-3)
    chk("control-law residual, yaw", 0.0, res[3], 5e-3)
    chk("regressed kp_xy", T["kp_xy"],
        cl.get("regressed", {}).get("kp_xy", float("nan")), 0.05)

    n_ok = sum(1 for _, o in checks if o)
    print(f"\n{n_ok}/{len(checks)} checks passed")
    bad = [n for n, o in checks if not o]
    if bad:
        print("failed: " + ", ".join(bad))
    return 0 if not bad else 1


# ==================================================================== main ==


def sec_summary(D, rep, ctx, args):
    rep.h("11  SUMMARY AND LOOP-DELAY BUDGET")
    g = rep.data
    lags = g.get("alignment", {}).get("lags", {})
    ref = g.get("alignment", {}).get("attitude_ref_lag", float("nan"))
    applied = g.get("stamp_chain", {}).get("vo_delay_applied", float("nan"))
    meas = lags.get("vo_vs_attitude", {}).get("lag", float("nan"))
    if not np.isfinite(meas) and np.isfinite(ref):
        meas = lags.get("vo_vs_mocap", {}).get("lag", float("nan")) - ref
    believed = g.get("stamp_chain", {}).get("pose_age_stamp_med", float("nan"))
    tau = g.get("plant", {}).get("latency", float("nan"))
    K = g.get("plant", {}).get("K_horizontal", float("nan"))

    rep.p(f"VO delay measured   {meas:.3f} s")
    rep.p(f"VO_DELAY applied    {applied:.3f} s   "
          f"(error {meas-applied:+.3f} s on every pose stamp)")
    age_true = believed + (meas - applied)
    rep.p(f"pose age, believed  {believed:.3f} s")
    rep.p(f"pose age, true      {age_true:.3f} s")
    rep.p(f"command to motion   {tau:.3f} s")
    Ltot = age_true + tau
    rep.p(f"total loop delay L  {Ltot:.3f} s")
    if np.isfinite(Ltot) and np.isfinite(K) and Ltot > 0 and K > 0:
        kp60 = (30.0 / 57.3) / (Ltot * K)
        kp45 = (45.0 / 57.3) / (Ltot * K)
        rep.p(f"  with plant gain {K:.2f} and position as the integrator, "
              f"60 deg phase margin\n  gives kp_xy <= {kp60:.2f}; "
              f"45 deg gives {kp45:.2f}. Predicting the pose forward over "
              f"its\n  age would remove {age_true:.2f} s of this.")
        rep.kv("summary", kp_xy_max_60deg=kp60, kp_xy_max_45deg=kp45)
    rep.kv("summary", vo_delay_measured=meas, vo_delay_applied=applied,
           pose_age_true=age_true, cmd_latency=tau, loop_delay=Ltot)

    est = g.get("estimator", {})
    tr = g.get("tracking", {})
    rep.p()
    rep.p("headline numbers")
    rep.p(f"  estimator, achievable   RPE {est.get('weighted_rpe', float('nan')):.1f} % "
          f"over {args.rpe_win:.0f} s windows, scale removed")
    rep.p(f"  estimator, as it runs   RPE "
          f"{est.get('weighted_as_runs', float('nan')):.1f} %, "
          f"fitted scale {est.get('weighted_scale', float('nan')):.3f}")
    rep.p(f"  command fidelity        "
          f"{tr.get('displacement_scale', float('nan')):.3f} x commanded, "
          f"{tr.get('displacement_yaw_deg', float('nan')):+.1f} deg rotated")
    steps = ctx.get("steps", [])
    if steps:
        f = lambda k: np.array([r.get(k, np.nan) for r in steps], float)
        rep.p(f"  true error at rest      "
              f"{np.nanmedian(f('ss_err_true')):.3f} m median, "
              f"{pct(f('ss_err_true'),90):.3f} m p90")
        rep.p(f"  drift while holding     "
              f"{np.nanmedian(f('hold_drift_mps'))*100:.2f} cm/s median, "
              f"excursion up to {np.nanmax(f('hold_excursion_m')):.2f} m")
    rep.p()
    rep.p("read it as: the controller is only as good as the frame it closes "
          "in. The\nscale row says how much of the error is a wrong metre; "
          "the yaw row says how\nmuch is a wrong direction; the drift row "
          "says what happens when both are\nnominally zero and nothing "
          "observes x and y.")


def run(D, rep, args):
    ctx = {}
    sec_inventory(D, rep, ctx)
    sec_stamp_chain(D, rep, ctx, args)
    sec_mocap(D, rep, ctx, args)
    sec_alignment(D, rep, ctx, args)
    sec_velocity(D, rep, ctx, args)
    sec_plant(D, rep, ctx, args)
    sec_estimator(D, rep, ctx, args)
    sec_frame_drift(D, rep, ctx, args)
    sec_tracking(D, rep, ctx, args)
    sec_control_law(D, rep, ctx, args)
    sec_health(D, rep, ctx, args)
    sec_summary(D, rep, ctx, args)
    return ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--steps", default=None, help="per-step CSV")
    ap.add_argument("--t0", type=float, default=None,
                    help="seconds from the start of the file")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--max-lag", type=float, default=1.5)
    ap.add_argument("--highpass", type=float, default=2.0,
                    help="seconds; removes slow trends before correlating")
    ap.add_argument("--rpe-win", type=float, default=1.0)
    ap.add_argument("--align-win", type=float, default=60.0,
                    help="seconds; windowed frame fit in section 7b")
    ap.add_argument("--align-stride", type=float, default=20.0)
    ap.add_argument("--align-min-extent", type=float, default=0.3,
                    help="metres; a window with less spread cannot identify "
                         "a rotation")
    ap.add_argument("--min-disp", type=float, default=0.05,
                    help="metres of true motion for a window to count as "
                         "moving")
    ap.add_argument("--max-shift", type=float, default=0.6,
                    help="seconds; range of the RPE time-shift scan")
    ap.add_argument("--min-epoch", type=float, default=15.0)
    ap.add_argument("--v-low", type=float, default=0.15)
    ap.add_argument("--freeze", type=float, default=0.5)
    ap.add_argument("--step-min", type=float, default=0.1,
                    help="metres; smaller setpoint changes are not a step")
    ap.add_argument("--step-min-dur", type=float, default=5.0)
    ap.add_argument("--settle-tol", type=float, default=0.05)
    # controller parameters, for section 9 only
    ap.add_argument("--kp-xy", type=float, default=0.6)
    ap.add_argument("--kp-z", type=float, default=0.5)
    ap.add_argument("--kp-yaw", type=float, default=0.8)
    ap.add_argument("--v-max-xy", type=float, default=0.5)
    ap.add_argument("--v-max-z", type=float, default=0.3)
    ap.add_argument("--yaw-rate-max-deg", type=float, default=15.0)
    ap.add_argument("--deadband-xy", type=float, default=0.02)
    ap.add_argument("--deadband-z", type=float, default=0.02)
    ap.add_argument("--deadband-yaw-deg", type=float, default=2.0)
    ap.add_argument("--slew-xy", type=float, default=1.0)
    ap.add_argument("--slew-z", type=float, default=1.0)
    ap.add_argument("--slew-yaw-deg", type=float, default=60.0)
    ap.add_argument("--yaw-scale", type=float, default=1.31)
    ap.add_argument("--yaw-source", default="ekf_pose",
                    choices=["ekf_pose", "dji_attitude"])
    ap.add_argument("--yaw-offset-deg", type=float, default=0.0)
    ap.add_argument("--rate-hz", type=float, default=20.0)
    ap.add_argument("--pose-timeout", type=float, default=0.5)
    ap.add_argument("--setpoint-timeout", type=float, default=5.0)
    ap.add_argument("--gate-on-degraded", action="store_true", default=True)
    ap.add_argument("--no-gate-on-degraded", dest="gate_on_degraded",
                    action="store_false")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest(a))
    if not a.npz:
        ap.error("give an npz, or --selftest")

    D = load(a.npz)
    if a.t0 is not None or a.t1 is not None:
        D = trim(D, a.t0, a.t1)
    rep = Report()
    ctx = run(D, rep, a)
    print(rep.text())

    if a.json:
        def default(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        with open(a.json, "w") as f:
            json.dump(rep.data, f, indent=1, default=default)
        print(f"\nwrote {a.json}")
    if a.steps and ctx.get("steps"):
        keys = sorted({k for r in ctx["steps"] for k in r})
        with open(a.steps, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in ctx["steps"]:
                w.writerow(r)
        print(f"wrote {a.steps}")


if __name__ == "__main__":
    main()
