#!/usr/bin/env python3
"""npz -> registration, scale reading, drift, validation.  Pure numpy/scipy.

  python3 reg_solve.py --npz reg.npz --out-dir out/

Chain (T_A_B = pose of B in A):

  T_opti_odom = T_opti_board . T_board_cam . T_cam_base . T_base_odom
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import correlate, savgol_filter

import reg_board

# Board pose in the optitrack frame, kalibr convention, as a constant.
T_OPTI_BOARD = np.array([
    [-0.025663, 0.000000, -0.999671,  6.013222],
    [-0.999671, 0.000000,  0.025663, -1.405717],
    [0.000000,  1.000000,  0.000000,  0.537352],
    [0.000000,  0.000000,  0.000000,  1.000000]])

HOVER_SPEED = 0.08      # m/s, on the PnP camera track
HOVER_MIN_S = 3.0       # s
HOVER_MAX_DISP = 0.15   # m; a slow drift is not a hover


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def orthonormalise(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    out = T.copy()
    out[:3, :3] = R
    return out


def quat_pos_to_T(q, p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = p
    return T


def mean_T(Ts):
    """Chordal rotation mean, median translation."""
    R = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).mean()
    out = np.eye(4)
    out[:3, :3] = R.as_matrix()
    out[:3, 3] = np.median(np.array([T[:3, 3] for T in Ts]), axis=0)
    return out


def spread(Ts, T0):
    p = np.array([T[:3, 3] for T in Ts])
    ang = np.array([np.linalg.norm(
        Rotation.from_matrix(T0[:3, :3].T @ T[:3, :3]).as_rotvec())
        for T in Ts])
    return dict(n=len(Ts),
                pos_sd_mm=(np.std(p, axis=0) * 1e3).round(1).tolist(),
                pos_sd_norm_mm=round(float(np.std(
                    np.linalg.norm(p - p.mean(0), axis=1)) * 1e3), 1),
                rot_sd_deg=round(float(np.degrees(np.std(ang))), 3),
                rot_max_deg=round(float(np.degrees(ang.max())), 3))


def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


# ------------------------------------------------------------------ health
def health(d, t0, t1):
    t, s, sig = d["status_t"], d["status_s"], d["status_sig"]
    if len(t) == 0:
        return {"note": "no localisation/status in the npz -- "
                        "drone_interfaces was not on the path at extract time"}
    m = (t >= t0) & (t <= t1)
    if not m.any():
        return {"note": "no status samples over the detection window"}
    ratio = sig[m] / np.maximum(s[m], 1e-9)
    flags = [f for f in d["status_flags"][m] if f]
    rep = dict(
        s_median=round(float(np.median(s[m])), 4),
        s_min=round(float(s[m].min()), 4),
        s_max=round(float(s[m].max()), 4),
        sigma_s_over_s_median=round(float(np.median(ratio)), 3),
        sigma_s_over_s_max=round(float(ratio.max()), 3),
        degraded_frac=round(float(d["status_degraded"][m].mean()), 3),
        flag_counts={f: int(sum(f in x for x in flags))
                     for f in ("SIGMA_S_MAX", "S_CLAMPED", "VO_GAP",
                               "TRANSPORT_GAP")})
    rep["usable"] = bool(rep["sigma_s_over_s_median"] < 0.10)
    return rep


# ------------------------------------------------------------------ hovers
def segment_hovers(t, p):
    """Hover runs on the PnP camera track: metric, drift-free, independent of
    the filter -- which matters when the filter's scale may be wrong."""
    if len(t) < 21:
        return []
    w = min(21, len(t) // 2 * 2 + 1)
    ps = np.column_stack([savgol_filter(p[:, i], w, 2) for i in range(3)])
    v = np.gradient(ps, t, axis=0)
    sp = np.linalg.norm(v, axis=1)
    still = sp < HOVER_SPEED
    runs, i = [], 0
    while i < len(still):
        if not still[i]:
            i += 1
            continue
        
        j = i
        while (j < len(still) and still[j]
               and np.linalg.norm(ps[j] - ps[i]) < HOVER_MAX_DISP):
            j += 1
        if t[j - 1] - t[i] >= HOVER_MIN_S:
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


# ------------------------------------------------------- clock alignment
def resample(t, v, grid):
    return np.interp(grid, t, v)


def clock_offset(d, max_lag=3.0, rate=20.0):
    """Speed-profile cross-correlation: DJI speed_vector against mocap.
    Resample -> smooth -> differentiate, per filter_design.md 12.4."""
    sp, mo = d["speed"], d["mocap_drone"]
    if len(sp) < 10 or len(mo) < 10:
        return 0.0, 0.0
    t0 = max(sp[0, 0], mo[0, 0])
    t1 = min(sp[-1, 0], mo[-1, 0])
    g = np.arange(t0, t1, 1.0 / rate)
    a = resample(sp[:, 0], np.linalg.norm(sp[:, 1:4], axis=1), g)
    pos = np.column_stack([resample(mo[:, 0], mo[:, i], g) for i in (1, 2, 3)])
    w = 21
    pos = np.column_stack([savgol_filter(pos[:, i], w, 2) for i in range(3)])
    b = np.linalg.norm(np.gradient(pos, g, axis=0), axis=1)
    a = a - a.mean()
    b = b - b.mean()
    c = correlate(b, a, mode="full")
    lags = np.arange(-len(a) + 1, len(b)) / rate
    k = np.abs(lags) <= max_lag
    lag = float(lags[k][np.argmax(c[k])])
    peak = float(c[k].max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return lag, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--t0", type=float, help="restrict to stamps >= this "
                    "(absolute, or seconds from the first detection)")
    ap.add_argument("--t1", type=float)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = np.load(a.npz, allow_pickle=True)
    rep = {}

    t = d["det_t_cap"]
    T_cam_board = d["T_cam_board"]
    T_base_cam = d["T_base_cam"]
    T_odom_base = d["T_odom_base"]
    if a.t0 is not None or a.t1 is not None:
        base = t[0] if (a.t0 or a.t1 or 0) < 1e6 else 0.0
        lo = base + a.t0 if a.t0 is not None else -np.inf
        hi = base + a.t1 if a.t1 is not None else np.inf
        k = (t >= lo) & (t <= hi)
        print(f"window {lo:.1f}..{hi:.1f}: {k.sum()} of {len(t)} frames")
        t, T_cam_board = t[k], T_cam_board[k]
        T_base_cam, T_odom_base = T_base_cam[k], T_odom_base[k]
        d = {kk: np.asarray(d[kk]) for kk in d.files}
        for key in ("det_n", "det_rms"):
            d[key] = d[key][k]
        # the pose track must be windowed too, or the validation scores this
        # window's transform against the whole flight
        pw = d["pose"]
        if len(pw):
            d["pose"] = pw[(pw[:, 0] >= lo) & (pw[:, 0] <= hi)]
    print(f"{len(t)} detected frames, {t[-1]-t[0]:.1f} s span, "
          f"tags/frame median {np.median(d['det_n']):.0f}, "
          f"reproj rms median {np.median(d['det_rms']):.2f} px")
    rep["detection"] = dict(
        frames=int(len(t)),
        tags_median=float(np.median(d["det_n"])),
        reproj_rms_px_median=round(float(np.median(d["det_rms"])), 3),
        reproj_rms_px_p95=round(float(np.percentile(d["det_rms"], 95)), 3))

    # -------------------------------------------------- filter health FIRST
    rep["filter_health"] = health(d, t[0], t[-1])
    print("\nfilter health over the detection window:")
    for k, v in rep["filter_health"].items():
        print(f"  {k}: {v}")
    if not rep["filter_health"].get("usable", False):
        print("\n  *** sigma_s/s exceeds the 0.10 engagement gate. The "
              "registration below\n      is still computed, but the odom "
              "frame it registers is not metric.\n")

    # ------------------------------------------------------- board in opti
    T_opti_board = orthonormalise(T_OPTI_BOARD)
    rep["board"] = dict(source="constant, reg_fit_board.py --vertical",
                        origin=T_opti_board[:3, 3].round(4).tolist(),
                        normal=T_opti_board[:3, 2].round(4).tolist(),
                        fit_residual_mm=32.2)

    # -------------------------------------------- per-frame registration
    T_opti_odom = np.array([
        T_opti_board @ inv(orthonormalise(T_cam_board[i]))
        @ inv(T_base_cam[i]) @ inv(T_odom_base[i])
        for i in range(len(t))])
    # base_link in opti, from the board only -- independent of the filter
    p_base_opti = np.array([
        (T_opti_board @ inv(orthonormalise(T_cam_board[i]))
         @ inv(T_base_cam[i]))[:3, 3] for i in range(len(t))])
    p_cam_board = np.array([inv(T_cam_board[i])[:3, 3] for i in range(len(t))])
    p_base_odom = T_odom_base[:, :3, 3]

    # ---------------------------------------- convention sanity, scalar only
    md = d["mocap_drone"]
    dm = np.array([np.interp(t, md[:, 0], md[:, i]) for i in (1, 2, 3)]).T
    r_survey = np.linalg.norm(dm - T_opti_board[:3, 3], axis=1)
    r_observed = np.linalg.norm(p_base_opti - T_opti_board[:3, 3], axis=1)
    resid = r_survey - r_observed
    rep["convention_check"] = dict(
        range_resid_median_m=round(float(np.median(resid)), 4),
        range_resid_sd_m=round(float(np.std(resid)), 4),
        verdict=("consistent" if abs(np.median(resid)) < 0.15
                 else "INCONSISTENT -- axis convention or origin is wrong"))
    print(f"\nconvention check: range residual "
          f"{np.median(resid):+.3f} +/- {np.std(resid):.3f} m "
          f"-> {rep['convention_check']['verdict']}")

    # ------------------------------------------------------------- hovers
    runs = segment_hovers(t, p_cam_board)
    print(f"\n{len(runs)} hover segments:")
    hovers = []
    for (i, j) in runs:
        Ts = [T_opti_odom[k] for k in range(i, j)]
        T0 = mean_T(Ts)
        h = dict(t0=round(float(t[i]), 2), t1=round(float(t[j - 1]), 2),
                 dur=round(float(t[j - 1] - t[i]), 1),
                 stand_off_m=round(float(np.median(
                     np.linalg.norm(p_cam_board[i:j], axis=1))), 3),
                 T=T0, spread=spread(Ts, T0),
                 p_board=np.median(p_base_opti[i:j], axis=0),
                 p_odom=np.median(p_base_odom[i:j], axis=0))
        hovers.append(h)
        print(f"  {h['t0']:.1f}-{h['t1']:.1f}s ({h['dur']:.0f}s) "
              f"standoff {h['stand_off_m']:.2f} m  "
              f"spread {h['spread']['pos_sd_norm_mm']:.1f} mm / "
              f"{h['spread']['rot_sd_deg']:.2f} deg")
    rep["hovers"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in h.items()} for h in hovers]

    # ------------------------------------------------- traverse scale reading
    scales = []
    for m in range(len(hovers) - 1):
        db = np.linalg.norm(hovers[m + 1]["p_board"] - hovers[m]["p_board"])
        do = np.linalg.norm(hovers[m + 1]["p_odom"] - hovers[m]["p_odom"])
        if do > 1e-6:
            scales.append(dict(pair=[m, m + 1],
                               board_m=round(float(db), 4),
                               odom_m=round(float(do), 4),
                               s_corr=round(float(db / do), 4)))
    rep["traverse_scale"] = scales
    if scales:
        v = np.array([s["s_corr"] for s in scales])
        rep["traverse_scale_summary"] = dict(
            median=round(float(np.median(v)), 4),
            sd=round(float(np.std(v)), 4), n=len(v))
        print(f"\ntraverse scale correction: "
              f"{np.median(v):.4f} +/- {np.std(v):.4f} over {len(v)} traverses")
        for s in scales:
            print(f"  hover {s['pair'][0]}->{s['pair'][1]}: "
                  f"board {s['board_m']:.3f} m vs odom {s['odom_m']:.3f} m "
                  f"-> {s['s_corr']:.4f}")

    # -------------------------------------------------------------- drift
    if len(hovers) >= 2:
        A, B = hovers[0]["T"], hovers[-1]["T"]
        rel = inv(A) @ B
        rep["drift_first_to_last"] = dict(
            dt_s=round(hovers[-1]["t0"] - hovers[0]["t0"], 1),
            translation_m=rel[:3, 3].round(4).tolist(),
            norm_m=round(float(np.linalg.norm(rel[:3, 3])), 4),
            horizontal_m=round(float(np.linalg.norm(rel[:2, 3])), 4),
            yaw_deg=round(yaw_deg(rel[:3, :3]), 3))
        print(f"\nregistration drift over "
              f"{rep['drift_first_to_last']['dt_s']:.0f} s: "
              f"{rep['drift_first_to_last']['horizontal_m']:.3f} m "
              f"horizontal, {rep['drift_first_to_last']['yaw_deg']:+.2f} deg yaw")

    # ------------------------------------------------------- the answer
    T_reg = hovers[0]["T"] if hovers else mean_T(list(T_opti_odom))
    rep["T_opti_odom"] = T_reg.round(6).tolist()
    q = Rotation.from_matrix(T_reg[:3, :3]).as_quat()

    # ---------------------------------------------------------- validation
    lag, peak = clock_offset(d)
    rep["clock"] = dict(mocap_minus_drone_s=round(lag, 4),
                        corr_peak=round(peak, 3))
    print(f"\nclock offset (mocap - drone): {lag:+.3f} s, peak {peak:.3f}")

    pose = d["pose"]
    if len(pose) and len(md):
        pe = np.array([np.interp(pose[:, 0] + lag, md[:, 0], md[:, i])
                       for i in (1, 2, 3)]).T
        ok = ((pose[:, 0] + lag >= md[0, 0]) & (pose[:, 0] + lag <= md[-1, 0]))
        est = (T_reg[:3, :3] @ pose[:, 1:4].T).T + T_reg[:3, 3]
        err = np.linalg.norm(est[ok] - pe[ok], axis=1)
        rep["validation_vs_mocap"] = dict(
            n=int(ok.sum()),
            median_m=round(float(np.median(err)), 4),
            p95_m=round(float(np.percentile(err, 95)), 4),
            max_m=round(float(err.max()), 4))
        print(f"validation against mocap: median {np.median(err):.3f} m, "
              f"p95 {np.percentile(err, 95):.3f} m over {ok.sum()} samples")

    with open(os.path.join(a.out_dir, "registration.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    with open(os.path.join(a.out_dir, "registration.yaml"), "w") as f:
        f.write("# optitrack_map -> map (map->odom is identity)\n")
        f.write("registration:\n  translation: [%.6f, %.6f, %.6f]\n"
                % tuple(T_reg[:3, 3]))
        f.write("  rotation_xyzw: [%.8f, %.8f, %.8f, %.8f]\n" % tuple(q))
        if scales:
            f.write("  scale_correction: %.5f\n"
                    % rep["traverse_scale_summary"]["median"])
    print(f"\nwrote {a.out_dir}/registration.json and registration.yaml")

    try:
        plot(d, t, rep, a.out_dir)
    except Exception as e:
        print(f"(plots skipped: {e})")


def plot(d, t, rep, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    st, s, sig = d["status_t"], d["status_s"], d["status_sig"]
    if len(st) == 0:
        return
    ax[0].plot(st - t[0], s, lw=0.8)
    ax[0].set_ylabel("s")
    ax[0].set_yscale("log")
    ax[1].plot(st - t[0], sig / np.maximum(s, 1e-9), lw=0.8)
    ax[1].axhline(0.10, color="r", ls="--", lw=0.8)
    ax[1].set_ylabel("sigma_s / s")
    ax[1].set_yscale("log")
    ax[2].plot(t - t[0], d["det_n"], ".", ms=1)
    ax[2].set_ylabel("tags detected")
    ax[2].set_xlabel("s")
    for h in rep["hovers"]:
        for x in ax:
            x.axvspan(h["t0"] - t[0], h["t1"] - t[0], color="g", alpha=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "health.png"), dpi=110)
    print(f"wrote {out_dir}/health.png")


if __name__ == "__main__":
    main()
