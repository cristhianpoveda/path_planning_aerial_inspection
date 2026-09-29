#!/usr/bin/env python3
"""Planner CLI. No ROS.

    python3 -m drone_navigation.planning --arena arena.yaml --out tour.yaml
    python3 -m drone_navigation.planning --arena arena.yaml \\
        --registration registration.yaml --w 0.75 --out tour.yaml
    python3 -m drone_navigation.planning --arena arena.yaml --sweep

--sweep traces the Pareto front for both variants, which is the comparison
planner_design.md 6.1 asks for: COUPLED and DECOUPLED differ in one thing
only, whether sigma_d is zero.
"""

import argparse
import time

import numpy as np

from . import io as IO
from . import model as M
from . import solve as S
from . import viewpoints as V
from .forecast import ForecastConfig, v_low_true
from .geometry import GeometricMap
from .roadmap import Roadmap


def build(a):
    gmap = GeometricMap.from_yaml(a.arena, a.safety_margin, a.arena_clearance)
    q = M.QualityParams()
    cands, rejected = V.generate(gmap, q, a.standoffs)
    road = Roadmap(gmap, a.samples, a.radius, a.seed).build(verbose=a.verbose)
    centre, normal = IO.load_board(a.arena)
    board = (IO.board_viewpoint(centre, normal, a.board_standoff)
             if centre is not None else None)
    if board is not None and not gmap.is_point_valid(board):
        raise SystemExit(
            f"the board hover at {board.round(3).tolist()} is not free space. "
            "The tour opens and closes there (3.1 step 7), so this has to be "
            "reachable: move an obstacle or change --board-standoff.")
    s, ratio, epoch = IO.load_registration(a.registration)
    fcfg = ForecastConfig(s_true=s, dwell_s=a.dwell)
    return gmap, q, cands, road, board, fcfg, s, ratio, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", required=True)
    ap.add_argument("--registration", help="registration.yaml from the node")
    ap.add_argument("--out", default="tour.yaml")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--decoupled", action="store_true",
                    help="score at the nominal pose, i.e. sigma_d = 0")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--nav-origin", type=float, nargs=3,
                    help="EKF init point in arena coordinates. 4.4: the scale "
                         "term is proportional to distance from HERE along "
                         "the viewing normal, so this drives selection. "
                         "Defaults to the arena centre, which is a guess.")
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--radius", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standoffs", type=int, default=3)
    ap.add_argument("--board-standoff", type=float, default=1.5)
    ap.add_argument("--dwell", type=float, default=0.0)
    ap.add_argument("--safety-margin", type=float, default=0.35)
    ap.add_argument("--arena-clearance", type=float, default=0.5)
    ap.add_argument("--speeds", type=float, nargs="+",
                    default=[0.30, 0.45, 0.60, 0.90])
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from drone_localisation.ekf.params import EkfParams
    params = EkfParams()

    gmap, q, cands, road, board, fcfg, s, ratio, epoch = build(a)
    nav = np.asarray(a.nav_origin, float) if a.nav_origin else gmap.arena.centre

    print(f"# arena      {gmap}")
    print(f"# candidates {len(cands)} over {len(gmap.targets)} targets")
    print(f"# optics     [T] band {M.effective_band(q)[0]:.2f}-"
          f"{M.effective_band(q)[1]:.2f} m, peak {M.peak_depth(q):.2f} m")
    print(f"# scale      s = {s:.3f}, sigma_s/s = {ratio:.3f}"
          + (f", vo_epoch {epoch}" if epoch is not None else ""))
    print(f"# V_LOW      {v_low_true(params):.3f} m/s true; speeds "
          f"{a.speeds} straddle it")
    print(f"# nav origin {np.round(nav, 2).tolist()}"
          + ("" if a.nav_origin else "  [T] arena centre, not the real one"))

    base = dict(speeds=tuple(a.speeds), max_iter=a.max_iter,
                image_dwell_s=a.dwell, sigma_s_ratio_0=ratio,
                verbose=a.verbose)
    T_ref = S.reference_time(gmap, road, cands, q, params,
                             S.SolveConfig(**base), fcfg, board)
    print(f"# T_ref      {T_ref:.1f} s  (fastest tour, DECOUPLED, held fixed)")

    if a.sweep:
        print("\n variant     w      J        Qbar     That    T(s)   speeds")
        for coupled in (False, True):
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                cfg = S.SolveConfig(w=w, coupled=coupled, **base)
                p = S.plan(gmap, road, cands, q, params, cfg, fcfg, board,
                           nav, T_ref)
                print(f" {'COUPLED  ' if coupled else 'DECOUPLED'} {w:<5} "
                      f"{p.J:+.4f}  {p.Qbar:.4f}  {p.That:.3f}  {p.time_s:5.1f}"
                      f"  {p.speeds}"
                      + ("" if p.converged else "  NOT CONVERGED"))
        return

    cfg = S.SolveConfig(w=a.w, coupled=not a.decoupled, **base)
    t0 = time.time()
    p = S.plan(gmap, road, cands, q, params, cfg, fcfg, board, nav, T_ref)
    print(f"\n# solved in {time.time() - t0:.1f} s, {p.iterations} iterations"
          + ("" if p.converged else "  NOT CONVERGED -- 6.2 requires this to "
             "be reported"))
    print(f"# J {p.J:+.4f}  Qbar {p.Qbar:.4f}  That {p.That:.3f}  "
          f"T {p.time_s:.1f} s")
    for i, name in enumerate(p.order):
        print(f"#   {i}: {name:16s} v={p.speeds[i]:.2f} m/s  "
              f"sigma_s={p.sigma_s.get(name, float('nan')):.4f}")
    print("# per-target E[Q]: "
          + ", ".join(f"{k} {v:.3f}" for k, v in sorted(p.quality.items())))

    IO.write_tour(a.out, p, gmap, q, meta=dict(
        arena=a.arena, registration=a.registration, w=a.w,
        coupled=not a.decoupled, vo_epoch=epoch, s=s, sigma_s_ratio_0=ratio,
        nav_origin=[float(x) for x in nav], T_ref=float(T_ref),
        speeds=list(a.speeds)))
    print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
