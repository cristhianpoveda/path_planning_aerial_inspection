"""Candidate viewpoints, coverage and scoring.

Viewpoints are decision variables here. Orientation is not a decision.

Set-cover is non trivial. A candidate generated for target A
may also image target B, at non-zero incidence and a different depth. Without
that, the minimum covering set is just one candidate per target and the
set-cover step does nothing. Each candidate therefore reports every target it
can actually see, subject to the 6.4 filters.
"""

from dataclasses import dataclass, field

import numpy as np

from . import model as M

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Candidate:
    name: str
    position: np.ndarray
    view_dir: np.ndarray          # unit, camera optical axis, world frame
    target: str                   # the patch it was generated for
    standoff: float
    covers: dict = field(default_factory=dict)   # target name -> (d, beta)

    @property
    def yaw_deg(self):
        return float(np.degrees(np.arctan2(self.view_dir[1], self.view_dir[0])))

    @property
    def pitch_deg(self):
        """Gimbal pitch, negative looking down."""
        return float(np.degrees(np.arcsin(np.clip(self.view_dir[2], -1, 1))))


def camera_frame(view_dir):
    """Rows are x_c (right), y_c (down), z_c (optical axis).

    y_c is world-down projected orthogonal to the axis, which is what a
    gimbal that only pitches and yaws produces: the horizon stays level in
    the image, so there is no roll to guess at.
    """
    z = np.asarray(view_dir, float)
    z = z / np.linalg.norm(z)
    if abs(z @ UP) > 0.999:
        raise ValueError("a straight up or down view has no defined roll")
    y = -UP + (UP @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z])


def projects_inside(p_cam, view_dir, p_world, q: M.QualityParams, margin_px=40):
    """Is p_world inside the image, with a margin?"""
    R = camera_frame(view_dir)
    v = R @ (np.asarray(p_world, float) - np.asarray(p_cam, float))
    if v[2] <= 1e-6:
        return False
    u = q.f_x * v[0] / v[2] + q.c_x
    w = q.f_y * v[1] / v[2] + q.c_y
    return bool(margin_px <= u <= q.width - margin_px
                and margin_px <= w <= q.height - margin_px)


def incidence(p_cam, target):
    """Angle between the viewing ray and the patch normal, radians."""
    d = np.asarray(p_cam, float) - target.centre
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.pi
    return float(np.arccos(np.clip((d / n) @ target.normal, -1.0, 1.0)))

BAND_TOL = 1e-6


def standoff_set(q: M.QualityParams, n=3, include_peak=True):
    """The n standoffs to try per patch.
    """
    lo, hi = M.effective_band(q)
    if not include_peak or n < 2:
        return np.linspace(lo, hi, n)
    d0 = M.peak_depth(q)
    rest = np.linspace(lo, hi, n - 1)
    return np.sort(np.concatenate([[d0], rest]))


def generate(gmap, q: M.QualityParams, n_standoffs=3, margin_px=40,
             include_peak=True, verbose=False):
    """Candidates for every target, with their coverage sets.
    """
    standoffs = standoff_set(q, n_standoffs, include_peak)
    d_lo, d_hi = M.effective_band(q)
    beta_max = np.radians(q.beta_max_deg)
    cands, rejected = [], []
    for t in gmap.targets:
        for d in standoffs:
            p = t.centre + d * t.normal
            if not gmap.is_point_valid(p):
                rejected.append((t.name, d, "outside arena or in an obstacle"))
                continue
            if not gmap.has_line_of_sight(p, t.centre, t.normal):
                rejected.append((t.name, d, "own patch occluded"))
                continue
            c = Candidate(name=f"{t.name}@{d:.2f}", position=p,
                          view_dir=-t.normal, target=t.name, standoff=float(d))
            for u in gmap.targets:
                dd = float(np.linalg.norm(p - u.centre))
                if not (d_lo - BAND_TOL <= dd <= d_hi + BAND_TOL):
                    continue
                b = incidence(p, u)
                if b > beta_max:
                    continue
                if not gmap.has_line_of_sight(p, u.centre, u.normal):
                    continue
                if not projects_inside(p, c.view_dir, u.centre, q, margin_px):
                    continue
                c.covers[u.name] = (dd, b)
            if not c.covers:
                rejected.append((t.name, d, "covers nothing, not even itself"))
                continue
            cands.append(c)
    covered = {n for c in cands for n in c.covers}
    uncovered = [t.name for t in gmap.targets if t.name not in covered]
    if verbose:
        print(f"candidates: {len(cands)} kept, {len(rejected)} rejected")
        for r in rejected:
            print(f"  rejected {r[0]} @ {r[1]:.2f} m: {r[2]}")
    if uncovered:
        # set-cover is infeasible, so say so here rather than let the solver
        # return an empty tour with no explanation
        raise ValueError(
            f"no feasible candidate images {uncovered}. Either the standoff "
            f"band [{d_lo:.2f}, {d_hi:.2f}] does not fit between the patch and "
            "the arena clearance, or the patch is occluded from every "
            "standoff. Move the target, widen the band, or raise "
            "n_standoffs.")
    return cands, rejected


def score(cand, gmap, q: M.QualityParams, v_perp, sigma_s_over_s,
          Sigma_pp, nav_origin, nominal=False):
    """E[Q_def] per covered target at this candidate.

    `nominal=True` returns Q(E[d]) instead, which is what a planner that
    evaluates quality at the nominal pose computes. The difference between
    the two is the contribution this planner claims.
    """
    by_name = {t.name: t for t in gmap.targets}
    out = {}
    for name, (d, beta) in cand.covers.items():
        t = by_name[name]
        w_t = t.w_t
        if nominal:
            out[name] = float(M.q_def(d, beta, v_perp, w_t, q))
        else:
            sd = M.sigma_d(t.normal, cand.position, nav_origin,
                           Sigma_pp, sigma_s_over_s)
            out[name] = M.expected_q(d, beta, v_perp, w_t, sd, q)
    return out


def coverage_table(cands, gmap):
    """target name -> list of candidate names that cover it."""
    out = {t.name: [] for t in gmap.targets}
    for c in cands:
        for name in c.covers:
            out[name].append(c.name)
    return out
