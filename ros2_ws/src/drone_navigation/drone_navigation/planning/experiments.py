#!/usr/bin/env python3
"""Conditions under which coupled beats decoupled.

    python3 -m drone_navigation.planning.experiments arena.yaml
"""

import argparse
import sys

import numpy as np

from . import io as IO
from . import model as M
from . import solve as S
from . import viewpoints as V
from .forecast import ForecastConfig, v_low_true
from .geometry import GeometricMap
from .roadmap import Roadmap

REGIMES = {
    "transit allowed": (0.30, 0.45, 0.60, 0.90),
    "inspection only": (0.20, 0.30, 0.38, 0.45),
}


def compare(gmap, road, cands, q, params, board, nav, speeds, ratio, w,
            max_iter=2, beam=20):
    base = S.SolveConfig(max_iter=max_iter, beam=beam, speeds=speeds,
                         sigma_s_ratio_0=ratio, w=w)
    T_ref = S.reference_time(gmap, road, cands, q, params, base, None, board)
    Q_ref = S.reference_quality(gmap, road, cands, q, params, base, None,
                                board, nav, T_ref)
    out = {}
    for coupled in (True, False):
        cfg = S.SolveConfig(w=w, coupled=coupled, sigma_s_ratio_0=ratio,
                            max_iter=max_iter, beam=beam, speeds=speeds)
        p = S.plan(gmap, road, cands, q, params, cfg, None, board, nav,
                   T_ref, Q_ref)
        _, Qb, _, _ = S.score_true(p, gmap, q, cfg, nav, T_ref, Q_ref)
        out[coupled] = dict(Q=Qb, plan=p,
                            realised=float(np.mean(list(p.sigma_ratio.values()))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arena")
    ap.add_argument("--standoffs", type=int, default=5)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.10, 0.25, 0.40])
    ap.add_argument("--nav-origin", type=float, nargs=3)
    a = ap.parse_args()

    from drone_localisation.ekf.params import EkfParams
    params = EkfParams()
    gmap = GeometricMap.from_yaml(a.arena)
    q = M.QualityParams()
    cands, _ = V.generate(gmap, q, a.standoffs)
    road = Roadmap(gmap, a.samples, 0.9, seed=2).build(verbose=False)
    centre, normal = IO.load_board(a.arena)
    board = IO.board_viewpoint(centre, normal, 1.5) if centre is not None else None
    nav = np.asarray(a.nav_origin, float) if a.nav_origin else gmap.arena.lo + 0.6

    print(f"# {len(cands)} candidates, {a.standoffs} standoffs, w = {a.w}")
    print(f"# nav origin {np.round(nav, 2).tolist()}")
    print(f"# V_LOW = {v_low_true(params):.3f} m/s true")
    print(f"# Q_def peak {M.peak_depth(q):.3f} m, curvature "
          f"{M.curvature_at(M.peak_depth(q), q):+.2f}")
    for lbl, speeds in REGIMES.items():
        print(f"\n{lbl}  speeds={speeds}")
        print("  sigma_s/s_0   COUPLED  DECOUPLED     gain    realised")
        for ratio in a.ratios:
            r = compare(gmap, road, cands, q, params, board, nav, speeds,
                        ratio, a.w)
            g = r[True]["Q"] - r[False]["Q"]
            print(f"     {ratio:.2f}       {r[True]['Q']:.4f}   "
                  f"{r[False]['Q']:.4f}   {g:+.4f}   {r[True]['realised']:.3f}"
                  + ("   <-- coupling pays" if g > 0.002 else ""))
    print("\n# 'realised' is the mean sigma_s/s the forecast reports along the")
    print("# tour. Where it collapses to ~0.03 the two planners are identical")
    print("# by construction, not by coincidence: sigma_d is then dominated by")
    print("# the position term, which both planners see.")


if __name__ == "__main__":
    main()
