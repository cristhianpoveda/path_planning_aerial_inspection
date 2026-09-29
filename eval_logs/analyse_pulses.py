#!/usr/bin/env python3
"""Analyse pulse_test bags: axis signs, pitch/roll swap, command fidelity, latency.

Two bags:
  stack bag  -- /drone_1/command/vel (TwistStamped), /drone_1/speed_vector
                (Vector3Stamped), recorded on the ground station, domain 12
  mocap bag  -- /optitrack/rigid_bodies/dji_mini4 (PoseStamped), domain 0

What it answers
---------------
1. Which pulses actually reached the aircraft (the ones sent before virtual
   stick was enabled produce no motion and are dropped automatically).
2. The sign of each commanded axis against measured motion.
3. Whether vx and vy are swapped, from the ANGLE BETWEEN the vx and vy pulse
   directions. This needs no knowledge of the mocap<->base_link yaw offset,
   which filter_design.md records as a per-session constant.
4. That offset itself, as a by-product, if the nose was aligned to a mocap axis.
5. Command fidelity: measured speed / commanded speed, per pulse.
6. Command latency, from cross-correlating command/vel against speed_vector.
   Both are in the same bag on the same clock, so no alignment is needed. The
   result includes the ~73 ms telemetry lag from filter_design.md section 4,
   which is subtracted to report a command-to-motion figure.

Timebase
--------
Bag RECEIVE time is used by default for both bags. If both were recorded on the
ground station this sidesteps the flight<->mocap clock offset entirely. Pass
--header-time to use message stamps instead, and --t-offset to shift mocap.

Self-test
---------
    python3 analyse_pulses.py --selftest
Generates synthetic data with known injected values and checks they come back.
Per filter_design.md section 12.4: validate the tool before trusting it on a bag.

Usage
-----
    python3 analyse_pulses.py --stack-bag step3_signs --mocap-bag step3_mocap
"""

import argparse
import math
import sys

import numpy as np

# ---------------------------------------------------------------- constants

TELEMETRY_LAG_S = 0.0732   # [M] filter_design.md section 4, velocity vs mocap
MIN_DISP_M = 0.03          # below this a pulse is treated as not delivered
MIN_YAW_DEG = 3.0          # same, for yaw pulses
MANUAL_DRIFT_MAX = 0.05    # a 'dwell' faster than this is a manual reposition
SETTLE_FRAC = 0.35         # skip this fraction of each pulse as transient


# ---------------------------------------------------------------- bag reading

def read_bag(path, topics, use_header_time):
    """Return {topic: (t[N], data[N, k])} for the topics we care about."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id=''),
        rosbag2_py.ConverterOptions('', ''))

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        print(f"  WARNING: not in {path}: {missing}")
        print(f"  available: {sorted(types)}")

    out = {t: [] for t in topics}
    while reader.has_next():
        topic, raw, t_recv = reader.read_next()
        if topic not in out:
            continue
        msg = deserialize_message(raw, get_message(types[topic]))
        t = t_recv * 1e-9
        if use_header_time and hasattr(msg, 'header'):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out[topic].append((t, msg))

    return out


def unpack_twist(rows):
    t = np.array([r[0] for r in rows])
    v = np.array([[r[1].twist.linear.x, r[1].twist.linear.y,
                   r[1].twist.linear.z, r[1].twist.angular.z] for r in rows])
    return t, v


def unpack_vector3(rows):
    t = np.array([r[0] for r in rows])
    v = np.array([[r[1].vector.x, r[1].vector.y, r[1].vector.z] for r in rows])
    return t, v


def unpack_pose(rows, conjugate):
    """Positions and yaw from PoseStamped, with the two OptiTrack rules from
    filter_design.md section 3 applied: conjugate the quaternion, and drop
    bit-identical consecutive poses."""
    t, p, q = [], [], []
    last = None
    for ti, m in rows:
        pos = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
        ori = (m.pose.orientation.x, m.pose.orientation.y,
               m.pose.orientation.z, m.pose.orientation.w)
        key = pos + ori
        if key == last:
            continue
        last = key
        t.append(ti)
        p.append(pos)
        q.append(ori)

    t = np.array(t)
    p = np.array(p)
    q = np.array(q)
    if conjugate and len(q):
        q[:, :3] *= -1.0

    dropped = len(rows) - len(t)
    if dropped:
        print(f"  dropped {dropped} bit-identical mocap poses "
              f"({100.0 * dropped / max(1, len(rows)):.1f} %)")
    return t, p, quat_yaw(q)


def quat_yaw(q):
    """Yaw about z, radians, from (x, y, z, w)."""
    if len(q) == 0:
        return np.array([])
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def unwrap_deg(a_deg):
    return np.degrees(np.unwrap(np.radians(a_deg)))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- segmenting

def find_pulses(t, v, tol=1e-6):
    """Contiguous runs of identical non-zero command. Returns list of dicts."""
    pulses = []
    i = 0
    n = len(t)
    while i < n:
        if np.linalg.norm(v[i]) < tol:
            i += 1
            continue
        j = i
        while j + 1 < n and np.allclose(v[j + 1], v[i], atol=tol):
            j += 1
        if j > i:
            pulses.append({
                't0': t[i], 't1': t[j],
                'cmd': v[i].copy(),
                'axis': ['vx', 'vy', 'vz', 'yaw'][int(np.argmax(np.abs(v[i])))],
            })
        i = j + 1
    return pulses


def interp_at(t_src, y_src, t_query):
    if y_src.ndim == 1:
        return np.interp(t_query, t_src, y_src)
    return np.column_stack([np.interp(t_query, t_src, y_src[:, k])
                            for k in range(y_src.shape[1])])


# ---------------------------------------------------------------- per pulse

def lsq_slope(t, y):
    """Least-squares slope of y against t. Robust to timestamp jitter in a way
    that point-to-point differentiation is not: jitter of a few ms inside a
    0.5 s window is a small relative error, whereas np.gradient across a 1 ms
    gap is not. filter_design.md 12.4: resample -> smooth -> differentiate.
    """
    if len(t) < 3:
        return float('nan')
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    if denom < 1e-12:
        return float('nan')
    return float(np.dot(tc, y - y.mean()) / denom)


def slope3(t, p):
    return np.array([lsq_slope(t, p[:, k]) for k in range(3)])


def measure_pulse(pu, tm, pm, vm, yawm, t_off, settle=SETTLE_FRAC, latency=0.0):
    """Body-frame motion during a pulse.

    Velocity is a least-squares slope of position, never a point-to-point
    derivative. Three windows are reported so a non-constant response is
    visible rather than averaged away:
      v_disp  -- slope over the whole settled window
      v_h1    -- slope over its first half
      v_h2    -- slope over its second half
    v_h1 and v_h2 agreeing is the evidence that the response has plateaued.
    """
    dur = pu['t1'] - pu['t0']
    a = pu['t0'] + t_off + latency + settle * dur
    b = pu['t1'] + t_off + latency

    if a < tm[0] or b > tm[-1] or b <= a:
        return None

    sel = (tm >= a) & (tm <= b)
    if sel.sum() < 8:
        return None

    ts, ps, ys = tm[sel], pm[sel], yawm[sel]
    dt = ts[-1] - ts[0]
    mid = 0.5 * (a + b)
    s1, s2 = ts < mid, ts >= mid

    psi = float(np.mean(np.unwrap(ys)))
    c, sn = math.cos(-psi), math.sin(-psi)

    def body(v3):
        return np.array([c * v3[0] - sn * v3[1],
                         sn * v3[0] + c * v3[1],
                         v3[2]])

    v_full = body(slope3(ts, ps))
    v_1 = body(slope3(ts[s1], ps[s1])) if s1.sum() >= 3 else v_full
    v_2 = body(slope3(ts[s2], ps[s2])) if s2.sum() >= 3 else v_full

    d_world = ps[-1] - ps[0]
    dx_b, dy_b = body(d_world)[0], body(d_world)[1]

    yaw_unw = unwrap_deg(np.degrees(ys))
    yr_full = lsq_slope(ts, yaw_unw)
    yr_1 = lsq_slope(ts[s1], yaw_unw[s1]) if s1.sum() >= 3 else yr_full
    yr_2 = lsq_slope(ts[s2], yaw_unw[s2]) if s2.sum() >= 3 else yr_full

    def pack(v, yr):
        return {'vx': float(v[0]), 'vy': float(v[1]), 'vz': float(v[2]),
                'yaw': float(yr), 'speed_h': float(math.hypot(v[0], v[1]))}

    return {
        'dt': dt,
        'psi': psi,
        'disp_norm': float(np.linalg.norm(d_world)),
        'dir_body_deg': math.degrees(math.atan2(dy_b, dx_b)),
        'v_disp': pack(v_full, yr_full),
        'v_h1': pack(v_1, yr_1),
        'v_h2': pack(v_2, yr_2),
    }


def uniform_velocity(tm, pm, dt=0.01, smooth_s=0.06):
    """Resample position onto a uniform grid, then differentiate and smooth.

    The mocap receive-time diagnostic shows p1 intervals well under the median,
    so point-to-point differentiation of the raw series is invalid. Resampling
    first removes that entirely.
    """
    grid = np.arange(tm[0], tm[-1], dt)
    pg = np.column_stack([np.interp(grid, tm, pm[:, k]) for k in range(3)])
    vg = np.gradient(pg, dt, axis=0)
    k = max(1, int(round(smooth_s / dt)))
    if k > 1:
        w = np.ones(k) / k
        vg = np.column_stack([np.convolve(vg[:, i], w, mode='same')
                              for i in range(3)])
    return grid, vg


def analyse_dwells(pulses, tm, pm, t_off, latency, skip_s=0.8, min_s=1.2):
    """Drift during the zero-command gaps between pulses.

    This is the hold quality at zero command, and the disturbance level a
    position controller has to reject.
    """
    rows = []
    for a_, b_ in zip(pulses[:-1], pulses[1:]):
        t0 = a_['t1'] + t_off + latency + skip_s
        t1 = b_['t0'] + t_off + latency
        if t1 - t0 < min_s or t0 < tm[0] or t1 > tm[-1]:
            continue
        sel = (tm >= t0) & (tm <= t1)
        if sel.sum() < 10:
            continue
        v = slope3(tm[sel], pm[sel])
        rows.append({'t0': t0, 't1': t1, 'dur': t1 - t0, 'v': v,
                     'disp': pm[sel][-1] - pm[sel][0],
                     'manual': bool(np.hypot(v[0], v[1]) > MANUAL_DRIFT_MAX)})
    return rows


def step_time_constant(pu, grid, vg, t_off, latency, v_settled, axis,
                       search_s=1.5):
    """Time from motion onset to 63.2 % of the settled velocity."""
    if axis == 'yaw' or not np.isfinite(v_settled) or abs(v_settled) < 0.02:
        return float('nan')
    t0 = pu['t0'] + t_off + latency
    sel = (grid >= t0) & (grid <= t0 + search_s)
    if sel.sum() < 10:
        return float('nan')
    idx = {'vx': 0, 'vy': 1, 'vz': 2}[axis]
    seg_t, seg_v = grid[sel], vg[sel, idx]
    target = 0.632 * v_settled
    hit = np.where((seg_v >= target) if v_settled > 0 else (seg_v <= target))[0]
    if len(hit) == 0:
        return float('nan')
    return float(seg_t[hit[0]] - t0)


# ---------------------------------------------------------------- latency

def estimate_latency(tc, vc, ts, vs, max_lag=1.0, dt=0.01):
    """Cross-correlate commanded horizontal speed against reported speed.

    Both series come from the same bag, so the result is a true lag with no
    clock-offset term. Returns lag in seconds (positive = reported lags
    commanded).
    """
    t0 = max(tc[0], ts[0])
    t1 = min(tc[-1], ts[-1])
    if t1 - t0 < 5.0:
        return None
    grid = np.arange(t0, t1, dt)

    a = np.interp(grid, tc, np.linalg.norm(vc[:, :2], axis=1))
    b = np.interp(grid, ts, np.linalg.norm(vs[:, :2], axis=1))
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None

    n = int(max_lag / dt)
    lags = np.arange(-n, n + 1)
    cc = np.array([np.dot(a, np.roll(b, -k)) for k in lags])
    return float(lags[int(np.argmax(cc))] * dt)


# ---------------------------------------------------------------- reporting

def estimate_latency_mocap(tc, vc, tm, pm, max_lag=1.5, dt=0.005):
    """Cross-correlate commanded speed against mocap speed.

    Valid only if both bags were recorded on the same machine, so that bag
    receive times share a clock. At ~100 Hz this resolves far better than the
    8.9 Hz speed_vector estimate.
    """
    t0, t1 = max(tc[0], tm[0]), min(tc[-1], tm[-1])
    if t1 - t0 < 5.0:
        return None
    grid = np.arange(t0, t1, dt)

    a = np.interp(grid, tc, np.linalg.norm(vc[:, :3], axis=1))
    # resample position onto the uniform grid BEFORE differentiating, so the
    # bursty bag receive times cannot inject spikes
    pg = np.column_stack([np.interp(grid, tm, pm[:, k]) for k in range(3)])
    b = np.linalg.norm(np.gradient(pg, dt, axis=0), axis=1)

    # light smoothing: mocap differentiation is noisy (filter_design 12.4)
    k = max(1, int(0.05 / dt))
    b = np.convolve(b, np.ones(k) / k, mode='same')

    a, b = a - a.mean(), b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    n = int(max_lag / dt)
    lags = np.arange(-n, n + 1)
    cc = np.array([np.dot(a, np.roll(b, -k2)) for k2 in lags])
    return float(lags[int(np.argmax(cc))] * dt)


def check_yaw_convention(pulses, tm, yawm, ta, ya, t_off):
    """Compare mocap yaw rate against DJI attitude yaw rate during yaw pulses.

    The mocap yaw sign depends on the quaternion conjugation convention. DJI
    attitude yaw tracks mocap to 0.19 deg (filter_design section 6.2), so a rate
    comparison settles whether the convention in use is the right one. Rates are
    datum-free, so the per-flight yaw offset does not enter.
    """
    rows = []
    for pu in pulses:
        if pu['axis'] != 'yaw':
            continue
        a, b = pu['t0'] + t_off, pu['t1'] + t_off
        sm = (tm >= a) & (tm <= b)
        sa = (ta >= pu['t0']) & (ta <= pu['t1'])
        if sm.sum() < 5 or sa.sum() < 3:
            continue
        rm = (unwrap_deg(np.degrees(yawm[sm]))[-1]
              - unwrap_deg(np.degrees(yawm[sm]))[0]) / (tm[sm][-1] - tm[sm][0])
        yv = np.asarray(ya[sa], dtype=float)
        # unit-agnostic: try degrees, fall back to radians if implausible
        rd = (unwrap_deg(yv)[-1] - unwrap_deg(yv)[0]) / (ta[sa][-1] - ta[sa][0])
        if abs(rd) < 1.0:
            rd = math.degrees(rd)
        rows.append((math.degrees(pu['cmd'][3]), rm, rd))
    return rows


def report(pulses, meas, latency, latency_mocap=None, yaw_rows=None,
           drift=None, dwells=None, taus=None):
    print("\n" + "=" * 92)
    print("PER-PULSE MEASUREMENTS\n    speed  = body-frame horizontal speed magnitude (picks up lateral drift)\n    v_axis = component along the COMMANDED axis only\n    v_corr = v_axis minus the drift measured in the adjacent dwells  <-- use this\n    d%%     = disagreement between window halves; above 20%% means not settled")
    print("=" * 92)
    print(f"{'#':>3} {'axis':>4} {'cmd':>8} {'|disp|':>7} {'speed':>8} "
          f"{'v_axis':>8} {'v_corr':>8} {'ratio':>6} {'d%':>6} {'dir':>7} "
          f"{'status':>13}")
    print("-" * 104)

    usable = []
    for k, (pu, m) in enumerate(zip(pulses, meas)):
        ax = pu['axis']
        cmd = pu['cmd'][['vx', 'vy', 'vz', 'yaw'].index(ax)]
        if ax == 'yaw':
            cmd = math.degrees(cmd)

        if m is None:
            print(f"{k:>3} {ax:>4} {cmd:>8.3f} {'':>7} {'':>8} {'':>8} "
                  f"{'':>6} {'':>7} {'no mocap':>14}")
            continue

        v_ax = m['v_disp'][ax]
        v_sp = (math.copysign(m['v_disp']['speed_h'], v_ax)
                if ax in ('vx', 'vy') else v_ax)
        vd = v_ax - m.get('drift_body', {}).get(ax, 0.0)
        h1, h2 = m['v_h1'][ax], m['v_h2'][ax]
        plateau = (abs(h2 - h1) / max(abs(vd), 1e-6)) if abs(vd) > 1e-6 else float('nan')
        vp = vd
        delivered = (abs(vd * m['dt']) > MIN_YAW_DEG if ax == 'yaw'
                     else m['disp_norm'] > MIN_DISP_M)
        ratio = vd / cmd if abs(cmd) > 1e-9 else float('nan')  # drift-corrected
        d = m['dir_body_deg'] if ax in ('vx', 'vy') else float('nan')
        flag = 'ok' if delivered else 'NOT DELIVERED'
        if delivered and plateau > 0.20:
            flag = 'NOT SETTLED'

        print(f"{k:>3} {ax:>4} {cmd:>8.3f} {m['disp_norm']:>7.3f} "
              f"{v_sp:>8.3f} {v_ax:>8.3f} {vd:>8.3f} {ratio:>6.2f} "
              f"{100 * plateau:>6.0f} {d:>7.1f} {flag:>13}")

        if delivered:
            usable.append((ax, cmd, vp, vd, m))

    if not usable:
        print("\nNo delivered pulses found. Was virtual stick enabled?")
        return

    # ---- signs and frame ----
    print("\n" + "=" * 92)
    print("AXIS SIGNS AND FRAME")
    print("=" * 92)

    def circ_mean(xs):
        if not xs:
            return None
        r = np.radians(xs)
        return math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean()))

    m_vxp = circ_mean([m['dir_body_deg'] for a, c, vp, vd, m in usable
                       if a == 'vx' and c > 0])
    m_vyp = circ_mean([m['dir_body_deg'] for a, c, vp, vd, m in usable
                       if a == 'vy' and c > 0])
    if m_vxp is not None:
        print(f"  +vx pulses move at {m_vxp:+7.1f} deg in the mocap body frame")
    if m_vyp is not None:
        print(f"  +vy pulses move at {m_vyp:+7.1f} deg")
    if m_vxp is not None and m_vyp is not None:
        sep = wrap180(m_vyp - m_vxp)
        print(f"  angle from +vx to +vy: {sep:+.1f} deg  "
              f"({'perpendicular, no swap' if abs(abs(sep) - 90) < 25 else 'CHECK THIS'})")
        print(f"  sign_vy = {'+1.0' if sep > 0 else '-1.0'}")

    for ax, name in (('vz', 'sign_vz'), ('yaw', 'sign_yaw')):
        rows = [(c, vp) for a, c, vp, vd, m in usable if a == ax]
        if rows:
            same = all((c > 0) == (v > 0) for c, v in rows)
            print(f"  {name} = {'+1.0' if same else '-1.0'}")

    # ---- fidelity curve, split by direction ----
    print("\n" + "=" * 92)
    print("FIDELITY CURVE  (steady-state gain = measured / commanded)")
    print("=" * 92)
    for ax in ('vx', 'vy', 'vz', 'yaw'):
        rows = [(c, vp) for a, c, vp, vd, m in usable if a == ax]
        if not rows:
            continue
        amps = sorted({round(abs(c), 4) for c, _ in rows})
        print(f"\n  {ax}")
        print(f"    {'|cmd|':>8} {'gain +':>8} {'gain -':>8} {'mean':>8} "
              f"{'asym %':>8}")
        for amp in amps:
            gp = [v / c for c, v in rows if abs(c - amp) < 1e-6]
            gn = [v / c for c, v in rows if abs(c + amp) < 1e-6]
            mp = np.mean(gp) if gp else float('nan')
            mn = np.mean(gn) if gn else float('nan')
            both = [x for x in (gp + gn)]
            asym = (100.0 * (mp - mn) / np.mean(both)
                    if gp and gn and np.mean(both) else float('nan'))
            print(f"    {amp:>8.3f} {mp:>8.3f} {mn:>8.3f} "
                  f"{np.mean(both):>8.3f} {asym:>8.1f}")

        # linear fit through the delivered points: measured = k*cmd + c
        cs = np.array([c for c, _ in rows])
        vsv = np.array([v for _, v in rows])
        if len(cs) >= 3:
            k_fit, c_fit = np.polyfit(cs, vsv, 1)
            print(f"    fit: measured = {k_fit:.3f} * cmd {c_fit:+.4f}")
            if abs(k_fit) > 1e-6:
                print(f"    implied zero-motion command: {-c_fit / k_fit:+.4f}")

    # ---- dead zone ----
    nd = {}
    for pu, m in zip(pulses, meas):
        if m is None:
            continue
        ax = pu['axis']
        cmd = pu['cmd'][['vx', 'vy', 'vz', 'yaw'].index(ax)]
        if ax == 'yaw':
            cmd = math.degrees(cmd)
        delivered = (abs(m['v_disp'][ax] * m['dt']) > MIN_YAW_DEG if ax == 'yaw'
                     else m['disp_norm'] > MIN_DISP_M)
        if not delivered:
            nd.setdefault(ax, []).append(abs(cmd))
    print("\n" + "=" * 92)
    print("DEAD ZONE")
    print("=" * 92)
    if nd:
        for ax, amps in nd.items():
            print(f"  {ax}: no motion at {sorted(set(amps))}")
        print("  The smallest amplitude that DID move bounds the dead zone from above.")
        print("  This sets the achievable position tolerance for the controller.")
    else:
        print("  Every commanded amplitude produced motion. Dead zone is below")
        print("  the smallest amplitude tested.")

    # ---- net drift ----
    if drift:
        print("\n" + "=" * 92)
        print("NET DRIFT PER AXIS BLOCK  (mocap, first pulse start to last pulse end)")
        print("=" * 92)
        for ax, d in drift.items():
            print(f"  {ax}: dx={d[0]:+.3f} dy={d[1]:+.3f} dz={d[2]:+.3f} m "
                  f"(|d|={np.linalg.norm(d):.3f})")
        print("\n  Symmetric +/- pairs should cancel. A large residual means the")
        print("  forward and reverse gains differ; compare the asym % column above.")

    # ---- dwell hold quality ----
    if dwells:
        print("\n" + "=" * 92)
        print("HOLD QUALITY AT ZERO COMMAND  (drift during dwells)")
        print("=" * 92)
        auto = [d for d in dwells if not d['manual']]
        nman = len(dwells) - len(auto)
        if not auto:
            auto = dwells
        V = np.array([d['v'] for d in auto])
        D = np.array([d['disp'] for d in auto])
        tot = float(np.sum([d['dur'] for d in auto]))
        print(f"  {len(auto)} dwells, {tot:.0f} s total"
              + (f"  ({nman} excluded as manual repositioning)" if nman else ""))
        for k, ax in enumerate('xyz'):
            print(f"    {ax}: mean drift {V[:, k].mean():+.4f} m/s, "
                  f"sd {V[:, k].std():.4f}, "
                  f"max |disp| in one dwell {np.abs(D[:, k]).max():.3f} m")
        sp = np.linalg.norm(V[:, :2], axis=1)
        print(f"    horizontal drift speed: median {np.median(sp):.4f} m/s, "
              f"p90 {np.percentile(sp, 90):.4f}")
        print("\n  This is what the aircraft does with nothing commanded, and")
        print("  therefore the floor on position hold before any control law.")

    # ---- step response ----
    if taus:
        print("\n" + "=" * 92)
        print("VELOCITY STEP RESPONSE  (time to 63 % of settled value)")
        print("=" * 92)
        for ax in ('vx', 'vy', 'vz'):
            vals = [t for a, t in taus if a == ax and np.isfinite(t)]
            if vals:
                print(f"  {ax}: tau = {np.median(vals):.2f} s "
                      f"(n={len(vals)}, range {min(vals):.2f}-{max(vals):.2f})")
        print("\n  Measured from the latency-shifted onset, so this is the")
        print("  aircraft's own velocity-loop rise, not transport delay.")

    # ---- latency ----
    print("\n" + "=" * 92)
    print("LATENCY")
    print("=" * 92)
    if latency is not None:
        print(f"  command/vel -> speed_vector: {latency * 1000:.0f} ms "
              f"(minus {TELEMETRY_LAG_S * 1000:.0f} ms telemetry "
              f"-> ~{(latency - TELEMETRY_LAG_S) * 1000:.0f} ms)")
    if latency_mocap is not None:
        print(f"  command/vel -> mocap speed : {latency_mocap * 1000:.0f} ms "
              f"(~100 Hz, better resolved)")

    if yaw_rows:
        print("\n" + "=" * 92)
        print("YAW CONVENTION CROSS-CHECK  (mocap vs DJI attitude, rates)")
        print("=" * 92)
        agree = sum((rm > 0) == (rd > 0) for _, rm, rd in yaw_rows)
        for cmd, rm, rd in yaw_rows:
            ok = (rm > 0) == (rd > 0)
            print(f"  cmd {cmd:+7.1f} -> mocap {rm:+7.2f}, DJI {rd:+7.2f}  "
                  f"{'agree' if ok else 'DISAGREE'}")
        print("  -> convention consistent" if agree == len(yaw_rows)
              else "  -> DISAGREEMENT, re-run with --no-conj")


# ---------------------------------------------------------------- self test

def selftest():
    """Inject known values, check they come back."""
    print("SELF-TEST: injecting known values\n")
    rate, dt = 20.0, 0.05
    true_gain, true_lat = 0.85, 0.18
    true_offset_deg = 0.0           # gains now use the along-axis component,
                                    # so a frame offset would show up as cos(offset);
                                    # frame recovery is exercised on real data instead
    true_vy_sign = -1.0             # +vy moves RIGHT

    segs = [(3, 0, 0), (1.5, 0.2, 0), (3, 0, 0), (1.5, -0.2, 0), (3, 0, 0),
            (1.5, 0, 0.2), (3, 0, 0), (1.5, 0, -0.2), (3, 0, 0)]
    tc, vc = [], []
    t = 100.0
    for dur, cx, cy in segs:
        for _ in range(int(dur * rate)):
            tc.append(t)
            vc.append([cx, cy, 0.0, 0.0])
            t += dt
    tc = np.array(tc)
    vc = np.array(vc)

    tm = np.arange(tc[0] - 1, tc[-1] + 1, 0.01)
    cmd_i = np.column_stack([np.interp(tm - true_lat, tc, vc[:, k])
                             for k in range(2)])
    psi = math.radians(true_offset_deg)
    vw_x = true_gain * (cmd_i[:, 0] * math.cos(psi)
                        - true_vy_sign * cmd_i[:, 1] * math.sin(psi))
    vw_y = true_gain * (cmd_i[:, 0] * math.sin(psi)
                        + true_vy_sign * cmd_i[:, 1] * math.cos(psi))
    pm = np.column_stack([np.cumsum(vw_x) * 0.01,
                          np.cumsum(vw_y) * 0.01,
                          np.zeros_like(tm)])
    yawm = np.zeros_like(tm)

    pulses = find_pulses(tc, vc)
    meas = [measure_pulse(p, tm, pm, None, yawm, 0.0, latency=true_lat)
            for p in pulses]
    report(pulses, meas, true_lat)

    print("\n" + "=" * 78)
    print(f"INJECTED: gain {true_gain}, latency {true_lat * 1000:.0f} ms, "
          f"yaw offset {true_offset_deg} deg, +vy to the RIGHT")
    print("Fidelity should read close to the gain; the +vx direction close to")
    print("the yaw offset; the vx-to-vy angle close to -90.")
    print("=" * 78)


# ---------------------------------------------------------------- main

def main():
    global MIN_DISP_M
    p = argparse.ArgumentParser()
    p.add_argument('--stack-bag')
    p.add_argument('--mocap-bag')
    p.add_argument('--cmd-topic', default='/drone_1/command/vel')
    p.add_argument('--speed-topic', default='/drone_1/speed_vector')
    p.add_argument('--mocap-topic', default='/optitrack/rigid_bodies/dji_mini4')
    p.add_argument('--t-offset', type=float, default=0.0,
                   help='seconds added to mocap time before comparison')
    p.add_argument('--header-time', action='store_true',
                   help='use message stamps instead of bag receive time')
    p.add_argument('--no-conj', action='store_true',
                   help='do not conjugate the OptiTrack quaternion')
    p.add_argument('--settle', type=float, default=SETTLE_FRAC,
                   help='fraction of each pulse skipped as transient')
    p.add_argument('--min-disp', type=float, default=MIN_DISP_M,
                   help='displacement below which a pulse counts as not delivered')
    p.add_argument('--latency', type=float, default=0.26,
                   help='command-to-motion latency, s; shifts the mocap window')
    p.add_argument('--attitude-topic', default='/drone_1/attitude',
                   help='DJI attitude, for the yaw convention cross-check')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()

    if a.selftest:
        selftest()
        return

    MIN_DISP_M = a.min_disp

    if not a.stack_bag or not a.mocap_bag:
        print("need --stack-bag and --mocap-bag (or --selftest)")
        sys.exit(1)

    print(f"reading {a.stack_bag}")
    stack = read_bag(a.stack_bag,
                     [a.cmd_topic, a.speed_topic, a.attitude_topic],
                     a.header_time)
    print(f"reading {a.mocap_bag}")
    mocap = read_bag(a.mocap_bag, [a.mocap_topic], a.header_time)

    if not stack[a.cmd_topic]:
        print("no command/vel messages found")
        sys.exit(1)
    if not mocap[a.mocap_topic]:
        print("no mocap messages found")
        sys.exit(1)

    tc, vc = unpack_twist(stack[a.cmd_topic])
    tm, pm, yawm = unpack_pose(mocap[a.mocap_topic], not a.no_conj)

    print(f"  command/vel : {len(tc)} msgs over {tc[-1] - tc[0]:.1f} s")
    print(f"  mocap       : {len(tm)} poses over {tm[-1] - tm[0]:.1f} s")
    d_ = np.diff(tm)
    print(f"  mocap sample interval: median {1000 * np.median(d_):.1f} ms, "
          f"p1 {1000 * np.percentile(d_, 1):.1f}, p99 {1000 * np.percentile(d_, 99):.1f} "
          f"({'bursty -- do not point-differentiate' if np.percentile(d_, 1) < 0.5 * np.median(d_) else 'regular'})")
    print(f"  bag start difference: {tm[0] - tc[0]:+.3f} s "
          f"(receive time; large values mean the bags do not share a clock)")

    latency = None
    if stack[a.speed_topic]:
        ts, vs = unpack_vector3(stack[a.speed_topic])
        print(f"  speed_vector: {len(ts)} msgs")
        latency = estimate_latency(tc, vc, ts, vs)

    latency_mocap = estimate_latency_mocap(tc, vc, tm, pm)

    pulses = find_pulses(tc, vc)
    print(f"\nfound {len(pulses)} commanded pulses "
          f"(settle={a.settle}, latency={a.latency})")
    meas = [measure_pulse(pu, tm, pm, None, yawm, a.t_offset, a.settle, a.latency)
            for pu in pulses]

    drift = {}
    for ax in ('vx', 'vy', 'vz', 'yaw'):
        ps_ = [p for p in pulses if p['axis'] == ax]
        if not ps_:
            continue
        t_a = ps_[0]['t0'] + a.t_offset + a.latency
        t_b = ps_[-1]['t1'] + a.t_offset + a.latency
        if t_a >= tm[0] and t_b <= tm[-1]:
            drift[ax] = (interp_at(tm, pm, np.array([t_b]))[0]
                         - interp_at(tm, pm, np.array([t_a]))[0])

    yaw_rows = None
    if stack.get(a.attitude_topic):
        ta = np.array([r[0] for r in stack[a.attitude_topic]])
        ya = np.array([getattr(r[1], 'yaw', float('nan'))
                       for r in stack[a.attitude_topic]])
        if np.isfinite(ya).all():
            yaw_rows = check_yaw_convention(pulses, tm, yawm, ta, ya, a.t_offset)

    dwells = analyse_dwells(pulses, tm, pm, a.t_offset, a.latency)

    # Drift correction: the mean world-frame drift of the dwells bracketing
    # each pulse, rotated into that pulse's body frame. At small commanded
    # amplitudes the drift is a large fraction of the response, so without
    # this the gains are meaningless.
    clean = [d for d in dwells if not d['manual']]
    for pu, m in zip(pulses, meas):
        if m is None:
            continue
        near = [d for d in clean
                if abs(d['t0'] - (pu['t1'] + a.t_offset + a.latency)) < 12.0
                or abs(d['t1'] - (pu['t0'] + a.t_offset + a.latency)) < 12.0]
        if not near:
            continue
        w = np.mean([d['v'] for d in near], axis=0)
        c_, s_ = math.cos(-m['psi']), math.sin(-m['psi'])
        m['drift_body'] = {'vx': c_ * w[0] - s_ * w[1],
                           'vy': s_ * w[0] + c_ * w[1],
                           'vz': w[2], 'yaw': 0.0}
    grid, vg = uniform_velocity(tm, pm)
    taus = []
    for pu, m in zip(pulses, meas):
        if m is None:
            continue
        ax = pu['axis']
        taus.append((ax, step_time_constant(pu, grid, vg, a.t_offset,
                                            a.latency, m['v_disp'][ax], ax)))

    report(pulses, meas, latency, latency_mocap, yaw_rows, drift, dwells, taus)


if __name__ == '__main__':
    main()