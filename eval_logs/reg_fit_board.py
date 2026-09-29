#!/usr/bin/env python3
"""Solve T_opti_board from the data, and check the survey against it.

  python3 reg_fit_board.py --npz reg.npz

    p_base_opti  =  T_opti_board . p_base_kalibr
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import reg_board as B
import reg_solve as S


def umeyama(src, dst, with_scale=False):
    """src, dst: (N,3).  Returns (T, scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(A)
    S_ = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_[2, 2] = -1
    R = U @ S_ @ Vt
    c = 1.0
    if with_scale:
        var = ((src - mu_s) ** 2).sum() / len(src)
        c = float(np.trace(np.diag(D) @ S_) / var)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - c * R @ mu_s
    return T, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vertical", action="store_true",
                    help="4-DOF fit: board hangs flat on a wall, so board-up "
                         "is world-up; solve yaw + translation only")
    ap.add_argument("--free", action="store_true",
                    help="solve all 6 DOF; badly conditioned on a small "
                         "point cloud, use only as a diagnostic")
    ap.add_argument("--max-speed", type=float, default=0.15,
                    help="drop frames where mocap speed exceeds this (m/s); "
                         "a 50 ms clock residual at 1 m/s is 50 mm")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)

    t = d["det_t_cap"]
    # base_link expressed in the kalibr board frame -- no mocap, no survey
    p_kal = np.array([
        (S.inv(S.orthonormalise(d["T_cam_board"][i])) @ S.inv(d["T_base_cam"][i]))[:3, 3]
        for i in range(len(t))])

    lag, peak = S.clock_offset(d)
    md = d["mocap_drone"]
    p_opt = np.array([np.interp(t + lag, md[:, 0], md[:, i]) for i in (1, 2, 3)]).T
    ok = (t + lag >= md[0, 0]) & (t + lag <= md[-1, 0])
    if a.max_speed > 0 and len(md) > 20:
        from scipy.signal import savgol_filter
        g = np.arange(md[0, 0], md[-1, 0], 0.01)
        pos = np.column_stack([savgol_filter(np.interp(g, md[:, 0], md[:, i]),
                                             21, 2) for i in (1, 2, 3)])
        sp = np.linalg.norm(np.gradient(pos, g, axis=0), axis=1)
        ok &= np.interp(t + lag, g, sp) < a.max_speed
    p_kal, p_opt = p_kal[ok], p_opt[ok]
    print(f"clock offset {lag:+.3f} s (peak {peak:.3f}), "
          f"{len(p_kal)} paired frames after the speed gate")

    # conditioning
    sv = np.linalg.svd(p_kal - p_kal.mean(0), compute_uv=False) / np.sqrt(len(p_kal))
    print(f"point-cloud spread (sd along principal axes): "
          f"{sv[0]:.3f} {sv[1]:.3f} {sv[2]:.3f} m")
    if sv[2] < 0.05:
        print("  !! the smallest axis is under 5 cm -- the rotation about it is "
              "weakly\n     constrained and the fit below may be arbitrary in "
              "that direction")

    # ---------------------------------------------------------- 4-DOF fit
    if a.vertical:
        from scipy.optimize import least_squares

        def R_of(psi):
            n = np.array([np.cos(psi), np.sin(psi), 0.0])   # kalibr +z
            up = np.array([0.0, 0.0, 1.0])                  # kalibr +y
            return np.column_stack([np.cross(up, n), up, n])

        def resid(x):
            return ((R_of(x[0]) @ p_kal.T).T + x[1:] - p_opt).ravel()

        best = None
        for psi0 in np.linspace(-np.pi, np.pi, 12, endpoint=False):
            t0 = p_opt.mean(0) - R_of(psi0) @ p_kal.mean(0)
            r = least_squares(resid, np.r_[psi0, t0])
            if best is None or r.cost < best.cost:
                best = r
        psi = best.x[0]
        T_v = np.eye(4)
        T_v[:3, :3] = R_of(psi)
        T_v[:3, 3] = best.x[1:]
        e = np.linalg.norm(resid(best.x).reshape(-1, 3), axis=1)
        print(f"\n4-DOF vertical fit: residual median {np.median(e)*1e3:.1f} mm  "
              f"p95 {np.percentile(e,95)*1e3:.1f} mm")
        print(f"  yaw {np.degrees(psi):+.2f} deg   normal {np.round(T_v[:3,2],4)}")
        print(f"  kalibr origin (bottom-left of tag 0) {np.round(T_v[:3,3],4)}")
        ctr = (T_v @ np.array([3*B.PITCH, 3*B.PITCH, 0, 1.0]))[:3]
        top = (T_v @ np.array([0.0, 6*B.PITCH - B.TAG_GAP, 0, 1.0]))[:3]
        print(f"  board centre {np.round(ctr,4)}   top edge z {top[2]:.3f} m")
        print("\nT_opti_board (kalibr) -- store this as the survey constant, "
              "the rigid body is not tracked:")
        for row in T_v:
            print("   [" + ", ".join(f"{v: .6f}" for v in row) + "],")
        runs = S.segment_hovers(t[ok], p_kal)
        cen = [p_opt[i:j].mean(0) for i, j in runs]
        ken = [p_kal[i:j].mean(0) for i, j in runs]
        print(f"\nmocap vs tag displacement between hovers ({len(runs)} found):")
        for m_ in range(len(runs) - 1):
            print(f"  {m_}->{m_+1}: mocap {np.linalg.norm(cen[m_+1]-cen[m_]):.3f} m"
                  f"   tags {np.linalg.norm(ken[m_+1]-ken[m_]):.3f} m")
        return

    # ------------------------------------------------------- constrained fit
    mb0 = d["mocap_board"]
    T_opti_rb0 = S.mean_T([S.quat_pos_to_T(r[4:8], r[1:4]) for r in mb0])
    origin = T_opti_rb0[:3, :3] @ S.T_RB_USER[:3, 3] + T_opti_rb0[:3, 3]
    p_usr = (B.T_USER_KALIBR[:3, :3] @ p_kal.T).T + B.T_USER_KALIBR[:3, 3]
    M = (p_opt - origin).T @ p_usr
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R_opti_user = U @ D @ Vt
    res_c = np.linalg.norm((R_opti_user @ p_usr.T).T + origin - p_opt, axis=1)
    print(f"\nconstrained (surveyed translation, rotation only): "
          f"residual median {np.median(res_c)*1e3:.1f} mm  "
          f"p95 {np.percentile(res_c,95)*1e3:.1f} mm")
    R_rb_user = T_opti_rb0[:3, :3].T @ R_opti_user
    T_rb_user_c = np.eye(4)
    T_rb_user_c[:3, :3] = R_rb_user
    T_rb_user_c[:3, 3] = S.T_RB_USER[:3, 3]
    T_opti_user = np.eye(4)
    T_opti_user[:3, :3] = R_opti_user
    T_opti_user[:3, 3] = origin
    T_ob_c = T_opti_user @ B.T_USER_KALIBR
    n, up = T_ob_c[:3, 2], T_ob_c[:3, 1]
    print(f"  board normal {np.round(n,4)} "
          f"({np.degrees(np.arcsin(abs(n[2]))):.1f} deg from horizontal)")
    print(f"  board-up     {np.round(up,4)} "
          f"({np.degrees(np.arccos(np.clip(up[2],-1,1))):.1f} deg from vertical)")
    print(f"  user origin  {np.round(origin,4)}   "
          f"board centre {np.round((T_ob_c @ np.array([3*B.PITCH,3*B.PITCH,0,1.0]))[:3],4)}")
    dRc = Rot.from_matrix(S.orthonormalise(S.T_RB_USER)[:3, :3].T @ R_rb_user)
    print(f"  vs the supplied rotation: {np.degrees(np.linalg.norm(dRc.as_rotvec())):.2f} deg")
    print("\nT_rb_user, constrained fit -- paste this into reg_solve.py:")
    for row in T_rb_user_c:
        print("   [" + ", ".join(f"{v: .6f}" for v in row) + "],")

    if not a.free:
        runs = S.segment_hovers(t[ok], p_kal)
        cen = [p_opt[i:j].mean(0) for i, j in runs]
        ken = [p_kal[i:j].mean(0) for i, j in runs]
        print(f"\nmocap displacement between hover segments ({len(runs)} found):")
        for m_ in range(len(runs) - 1):
            print(f"  {m_}->{m_+1}: mocap {np.linalg.norm(cen[m_+1]-cen[m_]):.3f} m"
                  f"   tags {np.linalg.norm(ken[m_+1]-ken[m_]):.3f} m")
        return

    for with_scale in (False, True):
        T, c = umeyama(p_kal, p_opt, with_scale)
        res = np.linalg.norm((c * (T[:3, :3] @ p_kal.T).T + T[:3, 3]) - p_opt, axis=1)
        tag = "with scale" if with_scale else "rigid     "
        print(f"\n{tag}: residual median {np.median(res)*1e3:7.1f} mm  "
              f"p95 {np.percentile(res,95)*1e3:7.1f} mm" +
              (f"   scale {c:.4f}" if with_scale else ""))
        if not with_scale:
            T_fit = T

    n, up = T_fit[:3, 2], T_fit[:3, 1]
    print(f"\nfitted board:  normal {np.round(n,4)}  "
          f"({np.degrees(np.arcsin(abs(n[2]))):.1f} deg from horizontal)")
    print(f"               board-up {np.round(up,4)}  "
          f"({np.degrees(np.arccos(np.clip(up[2],-1,1))):.1f} deg from vertical)")
    print(f"               kalibr origin {np.round(T_fit[:3,3],4)}")

    # what the survey constant would have to be
    mb = d["mocap_board"]
    T_opti_rb = S.mean_T([S.quat_pos_to_T(r[4:8], r[1:4]) for r in mb])
    T_rb_kal = S.inv(T_opti_rb) @ T_fit
    T_rb_user = T_rb_kal @ B.T_KALIBR_USER
    print("\nimplied T_rb_user (rigid body -> tag frame, OpenCV convention):")
    for row in T_rb_user:
        print("   [" + ", ".join(f"{v: .6f}" for v in row) + "],")

    dR = Rot.from_matrix(S.orthonormalise(S.T_RB_USER)[:3, :3].T
                         @ T_rb_user[:3, :3])
    print(f"\nvs the supplied matrix: rotation differs by "
          f"{np.degrees(np.linalg.norm(dR.as_rotvec())):.2f} deg, "
          f"translation by {np.linalg.norm(T_rb_user[:3,3]-S.T_RB_USER[:3,3])*1e3:.1f} mm")
    print(f"  axis of the difference: {np.round(dR.as_rotvec()/max(np.linalg.norm(dR.as_rotvec()),1e-9),3)}")

    # how far did the drone actually move, per mocap, between the board epochs?
    runs = S.segment_hovers(t[ok], p_kal)
    print(f"\nmocap displacement between hover segments ({len(runs)} found):")
    cen = [p_opt[i:j].mean(0) for i, j in runs]
    ken = [p_kal[i:j].mean(0) for i, j in runs]
    for m in range(len(runs) - 1):
        print(f"  {m}->{m+1}: mocap {np.linalg.norm(cen[m+1]-cen[m]):.3f} m   "
              f"tags {np.linalg.norm(ken[m+1]-ken[m]):.3f} m")


if __name__ == "__main__":
    main()
