#!/usr/bin/env python3
"""Acceptance checks for model.py and viewpoints.py.

    python3 -m drone_navigation.planning.check_planning arena.yaml

Checks the two properties planner_design.md 4 rests on, and then that the
candidate set is actually feasible for the arena. No ROS.
"""

import sys

import numpy as np

from .geometry import GeometricMap
from . import model as M
from . import viewpoints as V

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def check_model(q):
    print("quality model")
    # 10: at 1.5 m the GSD is 1.048 mm/px and 2-3 px is resolvable
    g = M.gsd(1.5, 0.0, q) * 1e3
    wm = M.w_min(1.5, 0.0, 0.5, q) * 1e3
    check("GSD at 1.5 m matches 10", abs(g - 1.048) < 0.02, f"{g:.3f} mm/px")
    check("w_min at 1.5 m inside the 2.1-3.1 mm measured in 10",
          2.1 <= wm <= 3.1, f"{wm:.2f} mm")

    # 4.3: Q_def must peak inside the band and be concave there
    try:
        d0, q2 = M.assert_concave_near_peak(q)
        check("Q_def peaks inside the DoF band and is concave there", True,
              f"peak {d0:.3f} m, Q'' {q2:.2f}")
    except AssertionError as e:
        check("Q_def peaks inside the DoF band and is concave there", False,
              str(e))
        return

    # and the concavity has to actually bite: E[Q] < Q(E[d]) at the peak
    nom = float(M.q_def(d0, 0.0, 0.5, 0.002, q))
    exp = M.expected_q(d0, 0.0, 0.5, 0.002, 0.14, q)
    check("E[Q] < Q(E[d]) at the peak", exp < nom,
          f"{exp:.3f} vs {nom:.3f}, penalty {exp - nom:+.3f}")

    band = M.concave_band(q)
    lo, hi = M.effective_band(q)
    print(f"  concave region {band[0]:.2f}-{band[1]:.2f} m; declared band "
          f"[{q.d_min}, {q.d_max}] -> using [{lo:.2f}, {hi:.2f}]"
          + ("  (capped)" if hi < q.d_max - 1e-9 else ""))
    ks = [M.curvature_at(d, q) for d in np.linspace(lo, hi, 25)]
    check("Q_def is concave everywhere in the band actually used",
          all(k < 0 for k in ks), f"max Q'' = {max(ks):+.3f}")

    # 4.4: the scale term must dominate and vary strongly across the arena
    Sig = np.diag([0.07 ** 2] * 3)
    n = np.array([1.0, 0.0, 0.0])
    near = M.sigma_d(n, [0.5, 0, 0], [0, 0, 0], Sig, 0.10)
    far = M.sigma_d(n, [4.0, 0, 0], [0, 0, 0], Sig, 0.10)
    check("sigma_d varies several-fold across the arena", far / near > 3.0,
          f"{near:.3f} m near origin, {far:.3f} m at 4 m, "
          f"ratio {far / near:.1f}x")


def check_gauss_hermite(q):
    print("Gauss-Hermite quadrature")
    check("weights sum to 1", abs(M.GH_WEIGHTS.sum() - 1.0) < 1e-12,
          f"{M.GH_WEIGHTS.sum():.12f}")
    # exact for polynomials up to degree 9; check against Monte Carlo on Q
    rng = np.random.default_rng(0)
    d0, sd = 1.3, 0.15
    mc = float(np.mean(M.q_def(d0 + sd * rng.standard_normal(400000),
                               0.0, 0.5, 0.002, q)))
    gh = M.expected_q(d0, 0.0, 0.5, 0.002, sd, q)
    check("5-node quadrature agrees with 400k-sample Monte Carlo",
          abs(gh - mc) < 5e-3, f"GH {gh:.4f} vs MC {mc:.4f}")


def check_candidates(gmap, q):
    print("candidates")
    print(f"  standoffs: {np.round(V.standoff_set(q), 3).tolist()}")
    try:
        cands, rejected = V.generate(gmap, q)
    except ValueError as e:
        check("every target has at least one feasible candidate", False, str(e))
        return None
    check("every target has at least one feasible candidate", True,
          f"{len(cands)} candidates, {len(rejected)} rejected")

    cov = V.coverage_table(cands, gmap)
    for name, lst in cov.items():
        check(f"target {name} is covered", len(lst) > 0, f"{len(lst)} candidates")

    for c in cands:
        check(f"candidate {c.name} is in free space",
              gmap.is_point_valid(c.position))
        check(f"candidate {c.name} sees its own patch", c.target in c.covers)

    R = np.array([V.camera_frame(c.view_dir) for c in cands])
    orth = all(np.allclose(r @ r.T, np.eye(3), atol=1e-9) for r in R)
    dets = all(abs(np.linalg.det(r) - 1.0) < 1e-9 for r in R)
    check("camera frames are orthonormal and right-handed", orth and dets)

    multi = [c.name for c in cands if len(c.covers) > 1]
    print(f"  candidates covering more than one target: {len(multi)}"
          + (f"  {multi}" if multi else "  -- set-cover has nothing to do, "
             "so the covering set is one candidate per target"))
    return cands


def check_scoring(gmap, q, cands):
    print("scoring")
    Sigma_pp = np.diag([0.07 ** 2] * 3)
    nav = gmap.arena.centre
    tot_n = tot_e = 0.0
    worst = ("", 0.0)
    for c in cands:
        nom = V.score(c, gmap, q, 0.5, 0.10, Sigma_pp, nav, nominal=True)
        exp = V.score(c, gmap, q, 0.5, 0.10, Sigma_pp, nav)
        for k in nom:
            tot_n += nom[k]
            tot_e += exp[k]
            if exp[k] - nom[k] < worst[1]:
                worst = (f"{c.name}/{k}", exp[k] - nom[k])
    check("E[Q] is below Q(E[d]) in aggregate", tot_e < tot_n,
          f"{tot_e:.3f} vs {tot_n:.3f}, penalty {tot_e - tot_n:+.3f}")
    print(f"  largest single penalty: {worst[0]} {worst[1]:+.3f}")
    print("\n  candidate         target      Q(E[d])    E[Q]   penalty")
    for c in cands:
        nom = V.score(c, gmap, q, 0.5, 0.10, Sigma_pp, nav, nominal=True)
        exp = V.score(c, gmap, q, 0.5, 0.10, Sigma_pp, nav)
        for k in nom:
            print(f"  {c.name:16s} {k:10s} {nom[k]:8.3f} {exp[k]:7.3f} "
                  f"{exp[k] - nom[k]:+8.3f}")


def check_forecast():
    print("localisation forecast")
    try:
        from .forecast import Forecast, ForecastConfig, v_low_true, v_min_true
        from drone_localisation.ekf.params import EkfParams
    except ImportError as e:
        check("drone_localisation imports without ROS", False, str(e)[:120])
        return
    check("drone_localisation imports without ROS", True)
    import sys as _sys
    check("importing the filter core did not pull in rclpy",
          "rclpy" not in _sys.modules)

    p = EkfParams()
    cfg = ForecastConfig()
    print(f"  V_MIN {v_min_true(p):.3f} m/s true, V_LOW {v_low_true(p):.3f} m/s true, "
          f"estimate_scale {p.estimate_scale}")

    def run(v, secs=60.0):
        f = Forecast(p, cfg).reset(sigma_s0=0.10 * cfg.s_true)
        for _ in range(int(secs / cfg.dt_vo)):
            f.step(np.array([v, 0.0, 0.0]), cfg.dt_vo)
        return f

    hover = run(0.0)
    below = run(v_low_true(p) - 0.02)
    above = run(v_low_true(p) + 0.15)
    check("sigma_s grows at hover", hover.sigma_s > 0.10 * cfg.s_true,
          f"{0.10 * cfg.s_true:.3f} -> {hover.sigma_s:.4f}")
    check("sigma_s does not shrink below V_LOW",
          below.sigma_s >= 0.10 * cfg.s_true,
          f"{below.sigma_s:.4f} at {v_low_true(p) - 0.02:.2f} m/s")
    check("sigma_s collapses above V_LOW", above.sigma_s < 0.2 * below.sigma_s,
          f"{above.sigma_s:.4f} at {v_low_true(p) + 0.15:.2f} m/s, "
          f"{below.sigma_s / above.sigma_s:.0f}x lower")
    check("Sigma_pp stays positive semi-definite",
          np.all(np.linalg.eigvalsh(above.Sigma_pp) >= -1e-12),
          f"eigenvalues {np.linalg.eigvalsh(above.Sigma_pp).round(6).tolist()}")
    
    # zero innovation must leave the state untouched
    f = Forecast(p, cfg).reset(sigma_s0=0.30)
    s0, b0 = f.core.x.s, f.core.x.b
    for _ in range(300):
        f.step(np.array([0.7, 0.0, 0.0]), cfg.dt_vo)
    check("the state does not drift under zero innovation",
          abs(f.core.x.s - s0) < 1e-9 and abs(f.core.x.b - b0) < 1e-9,
          f"ds {f.core.x.s - s0:+.2e}, db {f.core.x.b - b0:+.2e}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "arena_example.yaml"
    q = M.QualityParams()
    check_model(q)
    check_gauss_hermite(q)
    check_forecast()
    gmap = GeometricMap.from_yaml(path)
    cands = check_candidates(gmap, q)
    if cands:
        check_scoring(gmap, q, cands)
    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
