#!/usr/bin/env python3
"""check_forecast.py -- the two free checks on scale_forecast.py.

Run inside the ROS container, from wherever scale_forecast.py lives:

    python3 check_forecast.py --smoke        # do the signatures match filter.py?
    python3 check_forecast.py --invariance   # is sigma_s/s independent of s?
    python3 check_forecast.py --tour         # does sigma_s move over a tour?
    python3 check_forecast.py --all

No bags, no hardware. `planner_design.md` §8 item 1.
"""
import argparse
import sys

import numpy as np


def smoke():
    """Every call scale_forecast makes into the filter, one at a time.

    scale_forecast.py was written against a reading of filter.py, not against
    its docstrings. If a signature moved, this says which one.
    """
    from drone_localisation.ekf.params import EkfParams
    from drone_localisation.ekf import filter as ekf

    p = EkfParams(T_HOLD=0.0)
    ok = True

    def chk(label, fn):
        nonlocal ok
        try:
            fn()
            print(f"  ok    {label}")
        except Exception as e:
            ok = False
            print(f"  FAIL  {label}: {type(e).__name__}: {e}")

    print("filter.py surface used by scale_forecast:")
    chk("State(params)", lambda: ekf.State(p))
    chk("IDX_S", lambda: ekf.IDX_S)
    chk("EkfCore(params, state=...)",
        lambda: ekf.EkfCore(p, state=ekf.State(p)))
    chk("Increment(dp_v, dl, dt, Sigma_v, t)",
        lambda: ekf.Increment(dp_v=np.zeros(3), dl=np.zeros(3), dt=0.03,
                              Sigma_v=np.eye(3) * 1e-6, t=0.0))

    core = ekf.EkfCore(p, state=ekf.State(p))
    inc = ekf.Increment(dp_v=np.array([0.01, 0, 0]), dl=np.zeros(3), dt=0.03,
                        Sigma_v=np.eye(3) * 1e-6, t=0.0)
    chk("core.propagate(inc)", lambda: core.propagate(inc))
    chk("core.s_ref", lambda: getattr(core, "s_ref"))
    chk("core.update_velocity(v_enu, inc, omega_yaw, t)",
        lambda: core.update_velocity(np.array([0.4, 0, 0]), inc,
                                     omega_yaw=0.0, t=0.0))
    chk("core.update_altitude(z, vz, t)",
        lambda: core.update_altitude(1.5, vz=0.0, t=0.0))
    chk("core.update_attitude(...)",
        lambda: core.update_attitude(np.eye(3), np.eye(3), np.eye(3),
                                     a_h=0.0, t=0.0))
    chk("x.sigma_s", lambda: core.x.sigma_s)
    chk("x.R_nv", lambda: core.x.R_nv)

    print("\nany FAIL above means scale_forecast.py needs its call sites "
          "updated\nbefore the other two checks mean anything.")
    return ok


def straight(p0, p1, speed, Segment):
    return Segment(p0=np.asarray(p0, float), p1=np.asarray(p1, float),
                   speed=speed)


def invariance():
    """sigma_s/s should not depend on the s we plan at.
    """
    from scale_forecast import Segment, check_scale_invariance
    from drone_localisation.ekf.params import EkfParams

    p = EkfParams(T_HOLD=0.0)
    segs = [straight([0, 0, 1.5], [3, 0, 1.5], 0.5, Segment),
            straight([3, 0, 1.5], [3, 3, 1.5], 0.5, Segment),
            straight([3, 3, 1.5], [0, 0, 1.5], 0.5, Segment)]

    out = check_scale_invariance(p, segs, s_values=(0.5, 1.0, 2.0, 3.5, 5.0))
    print("planning s -> final sigma_s/s")
    for k, v in out.items():
        if k != "max_rel_spread":
            print(f"  s = {k:<5} sigma_s/s = {v:.5f}")
    spread = out["max_rel_spread"]
    print(f"\nmax relative spread {spread:.4f}")
    if spread < 0.05:
        print("PASS -- invariant to within 5 %. Plan at s_ref = 1.0.")
    else:
        print("FAIL -- NOT invariant. q_s is absolute and dominates.")
        print("       Either plan at the s the flight is expected to have")
        print("       (unknowable), or make q_s relative: q_s_eff = q_s_rel*s^2.")
    return spread < 0.05


def tour():
    """Does sigma_s move enough over a 2-3 minute tour to matter?
    """
    from scale_forecast import ScaleForecaster, Segment, Waypoint
    from drone_localisation.ekf.params import EkfParams

    p = EkfParams(T_HOLD=0.0)

    # a 5-viewpoint cycle in a 5 x 4 m arena, ~2.5 min at 0.4 m/s
    pts = [[0.0, 0.0, 1.5], [2.0, 0.5, 1.5], [3.5, 2.0, 1.5],
           [2.0, 3.0, 1.5], [0.5, 2.0, 1.5], [0.0, 0.0, 1.5]]

    print(f"{'speed':>6} {'dur_s':>7} {'sig0':>7} {'sig_end':>8} "
          f"{'min':>7} {'max':>7} {'range%':>7}")
    for speed in (0.3, 0.4, 0.6, 0.8):
        segs = [straight(pts[i], pts[i + 1], speed, Segment)
                for i in range(len(pts) - 1)]
        wps = [Waypoint(index=i, label=f"vp{i}") for i in range(len(segs))]
        r = ScaleForecaster(p, s_ref=1.0).run(segs, wps)
        rel = r.sigma_s_rel
        rng = 100.0 * (rel.max() - rel.min()) / max(rel.mean(), 1e-9)
        print(f"{speed:6.1f} {r.t[-1]:7.1f} {rel[0]:7.4f} {rel[-1]:8.4f} "
              f"{rel.min():7.4f} {rel.max():7.4f} {rng:7.1f}")
        print(f"       per waypoint: "
              + "  ".join(f"{k}={v:.3f}" for k, v in r.at_waypoint.items()))
        print(f"       feasible={r.feasible} {r.reason}")

    print("\nRead it as: if 'range%' is a few percent at every speed, sigma_s")
    print("is effectively constant over a tour. Predictions 7.4.2 and 7.4.3")
    print("are then not observable and should be withdrawn -- 7.4.1, which")
    print("depends only on |n_hat' p|, still holds.")
    print("If the speeds separate, speed IS a live decision variable.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--invariance", action="store_true")
    ap.add_argument("--tour", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not any([a.smoke, a.invariance, a.tour, a.all]):
        ap.print_help()
        return

    if a.smoke or a.all:
        print("=" * 68 + "\nSMOKE\n" + "=" * 68)
        if not smoke() and a.all:
            sys.exit("fix the call sites before continuing")
    if a.invariance or a.all:
        print("\n" + "=" * 68 + "\nSCALE INVARIANCE\n" + "=" * 68)
        invariance()
    if a.tour or a.all:
        print("\n" + "=" * 68 + "\nSIGMA_S OVER A TOUR\n" + "=" * 68)
        tour()


if __name__ == "__main__":
    main()
