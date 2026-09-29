"""
scale_forecast.py -- offline forward covariance prediction for the planner.

Answers one question: if the drone flies this trajectory at these speeds, what
is sigma_s / s when it arrives at each viewpoint?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from drone_localisation.ekf import filter as ekf
from drone_localisation.ekf.filter import IDX_S, EkfCore, Kind, State
from drone_localisation.ekf.params import EkfParams


# --------------------------------------------------------------------------
# trajectory input
# --------------------------------------------------------------------------

@dataclass
class Segment:
    """One straight leg flown at constant speed.

    p0, p1 : metres in the nav frame
    speed  : m/s, commanded ground speed
    yaw_rate : rad/s, |omega| about the vertical -- feeds the OMEGA_GATE branch
               of R_speed and (if kappa is ever non-zero) g_slew
    """
    p0: np.ndarray
    p1: np.ndarray
    speed: float
    yaw_rate: float = 0.0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.p1) - np.asarray(self.p0)))

    @property
    def duration(self) -> float:
        return self.length / max(self.speed, 1e-6)

    def direction(self) -> np.ndarray:
        d = np.asarray(self.p1, float) - np.asarray(self.p0, float)
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else np.zeros(3)


@dataclass
class Waypoint:
    """A point on the trajectory the planner wants a covariance reading at."""
    index: int          # index into the segment list, at that segment's END
    label: str = ""

NMapPointsFn = Callable[[np.ndarray, np.ndarray], float]

# Probability per second that VO rebuilds its map, as a function of the same
# geometry. Until fitted, return a constant.
RebuildHazardFn = Callable[[np.ndarray, np.ndarray], float]


def constant_n_map_points(n: float = 250.0) -> NMapPointsFn:
    return lambda p, d: n


def constant_hazard(rate_per_s: float = 0.0) -> RebuildHazardFn:
    return lambda p, d: rate_per_s


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------

@dataclass
class ForecastResult:
    t: np.ndarray                  # (N,) seconds
    sigma_s: np.ndarray            # (N,)
    s: np.ndarray                  # (N,) nominal scale (constant unless rescaled)
    sigma_s_rel: np.ndarray        # (N,) sigma_s / s
    at_waypoint: dict              # label -> sigma_s_rel on arrival
    feasible: bool
    reason: str = ""
    n_rescales: int = 0
    n_epochs: int = 0
    events: list = field(default_factory=list)   # (t, str)


# --------------------------------------------------------------------------
# the forecaster
# --------------------------------------------------------------------------

class ScaleForecaster:
    """Drives EkfCore over a planned trajectory and reports sigma_s.

    Parameters
    ----------
    params        : EkfParams, the SAME object the flight config uses.
    n_map_points  : geometry -> predicted map-point count.
    rebuild_hazard: geometry -> rebuild rate per second.
    s_ref         : scale to plan at. sigma_s_rel should be insensitive to this
                    (see `check_scale_invariance`); 1.0 is the sane default
                    because true `s` is unknowable before flight.
    dt            : integration step. NOMINAL_DT matches the VO rate.
    seed          : rebuild sampling. None -> expected-value mode (no rebuilds
                    sampled; hazard applied as a continuous P_ss inflation).
    """

    def __init__(self,
                 params: EkfParams,
                 n_map_points: NMapPointsFn = None,
                 rebuild_hazard: RebuildHazardFn = None,
                 s_ref: float = 1.0,
                 dt: float = None,
                 seed: int | None = None):
        self.p = params
        self.n_map_points = n_map_points or constant_n_map_points(params.n_ref)
        self.rebuild_hazard = rebuild_hazard or constant_hazard(0.0)
        self.s_ref = float(s_ref)
        self.dt = float(dt if dt is not None else params.NOMINAL_DT)
        self.rng = np.random.default_rng(seed) if seed is not None else None

    # -- initial state -----------------------------------------------------

    def _initial_state(self) -> State:
        """Post-initialisation state, per ekf_node._try_init.

        p, b and the attitude block take their init values; P_ss takes the
        (0.10*s)^2 floor, which dominates the sd/sqrt(N) term in practice
        (filter_design.md 7: 84% per-sample scatter over ~60 samples).
        """
        x = State(self.p)
        x.s = self.s_ref
        x.P[IDX_S, IDX_S] = (0.10 * self.s_ref) ** 2
        return x

    # -- synthesising one increment ---------------------------------------

    def _increment(self, seg: Segment, dt: float, core: EkfCore) -> ekf.Increment:
        """Build the Increment the frontend would have built for this step.

        dp_v is the UNSCALED VO translation: metric motion divided by the
        nominal scale, which is what ORB-SLAM3 would report for a map of that
        scale.
        """
        v_metric = seg.direction() * seg.speed
        dp_v = v_metric * dt / max(core.x.s, 1e-6)

        n_pts = self.n_map_points(seg.p1, seg.direction())
        g_feat = float(np.clip(self.p.n_ref / max(n_pts, 1.0),
                               1.0, self.p.feat_cap))
        g_slew = 1.0 + self.p.kappa * abs(seg.yaw_rate)
        Sigma_v = self.p.Sigma_base * (g_feat * g_slew)

        return ekf.Increment(dp_v=dp_v, dl=np.zeros(3), dt=dt,
                             Sigma_v=Sigma_v, t=0.0)

    # -- the epoch event ---------------------------------------------------

    def _apply_epoch(self, core: EkfCore):
        """P_ss <- min(P_ss*4, (0.15*s)^2); the theta block is inflated too but
        does not reach s once 6.1 blocks the cross-covariance route.
        """
        x = core.x
        x.P[IDX_S, IDX_S] = min(x.P[IDX_S, IDX_S] * 4.0,
                                (0.15 * max(x.s, 1e-3)) ** 2)
        x.P[5, 5] = min(x.P[5, 5] * 4.0, self.p.P0_theta_xy)
        x.P[6, 6] = min(x.P[6, 6] * 4.0, self.p.P0_theta_xy)
        x.P[7, 7] = min(x.P[7, 7] * 4.0, self.p.P0_theta_z)

    def _apply_rescale(self, core: EkfCore):
        """Mirrors ekf_node._maybe_rescale.
        """
        x = core.x
        x.P[IDX_S, :] = 0.0
        x.P[:, IDX_S] = 0.0
        x.P[IDX_S, IDX_S] = (0.10 * x.s) ** 2

    # -- main loop ---------------------------------------------------------

    def run(self,
            segments: Sequence[Segment],
            waypoints: Sequence[Waypoint] = ()) -> ForecastResult:
        p = self.p
        core = EkfCore(p, state=self._initial_state())
        core.s_ref = self.s_ref

        ts, sig, ss = [], [], []
        events = []
        wp_at = {}
        wp_by_seg = {}
        for w in waypoints:
            wp_by_seg.setdefault(w.index, []).append(w)

        t = 0.0
        rescale_path = 0.0
        rescale_pending = False
        n_epochs = n_rescales = 0
        telemetry_period = 1.0 / 8.9        # [M] filter_design.md 3
        t_next_telemetry = telemetry_period

        for i, seg in enumerate(segments):
            n_steps = max(1, int(round(seg.duration / self.dt)))
            dt = seg.duration / n_steps
            v_dji = seg.speed * p.K_VEL      # what DJI would report

            for _ in range(n_steps):
                # ---- rebuild event -------------------------------------
                hz = self.rebuild_hazard(seg.p1, seg.direction())
                if hz > 0.0:
                    if self.rng is not None:
                        fired = self.rng.random() < hz * dt
                    else:
                        fired = False
                        cap = (0.15 * max(core.x.s, 1e-3)) ** 2
                        w = hz * dt
                        core.x.P[IDX_S, IDX_S] = (
                            (1 - w) * core.x.P[IDX_S, IDX_S] + w * cap)
                    if fired:
                        self._apply_epoch(core)
                        rescale_pending = True
                        rescale_path = 0.0
                        n_epochs += 1
                        events.append((t, "epoch"))

                # ---- propagate -----------------------------------------
                inc = self._increment(seg, dt, core)
                core.propagate(inc)

                # ---- rescale accumulation (ekf_node._maybe_rescale) ------
                # NOTE: accumulated at ANY speed, not gated on V_LOW.
                if rescale_pending:
                    rescale_path += seg.speed * dt
                    if rescale_path >= p.RESCALE_PATH_M:
                        self._apply_rescale(core)
                        rescale_pending = False
                        n_rescales += 1
                        events.append((t, "rescale"))

                # ---- telemetry updates at 8.9 Hz ------------------------
                t += dt
                if t >= t_next_telemetry:
                    t_next_telemetry += telemetry_period
                    self._telemetry_updates(core, seg, inc, dt)

                ts.append(t)
                sig.append(core.x.sigma_s)
                ss.append(core.x.s)

            for w in wp_by_seg.get(i, []):
                wp_at[w.label or f"wp{i}"] = core.x.sigma_s / max(core.x.s, 1e-9)

        ts = np.asarray(ts)
        sig = np.asarray(sig)
        ss = np.asarray(ss)
        rel = sig / np.maximum(ss, 1e-9)

        worst = float(rel.max()) if rel.size else float("inf")
        feasible = worst <= p.SIGMA_S_MAX
        reason = "" if feasible else f"sigma_s_rel peaks at {worst:.3f} > SIGMA_S_MAX"

        return ForecastResult(t=ts, sigma_s=sig, s=ss, sigma_s_rel=rel,
                              at_waypoint=wp_at, feasible=feasible,
                              reason=reason, n_rescales=n_rescales,
                              n_epochs=n_epochs, events=events)

    # -- the updates -------------------------------------------------------

    def _telemetry_updates(self, core: EkfCore, seg: Segment,
                           inc: ekf.Increment, dt: float):
        """Velocity, altitude and attitude arrive in one packet at 8.9 Hz.
        """
        p = self.p

        # -- velocity: the only channel that moves s ----------------------
        speed = seg.speed * p.K_VEL
        if speed >= p.V_MIN:
            v_enu = seg.direction() * speed
            core.update_velocity(v_enu, inc, omega_yaw=seg.yaw_rate, t=0.0)

        # -- altitude ------------------------------------------------------
        # h(x) = p_z + b, so y = 0 with a consistent state.
        core.update_altitude(float(core.x.p[2] + core.x.b),
                             vz=0.0, t=0.0)

        # -- attitude ------------------------------------------------------
        R_nb = core.x.R_nv
        core.update_attitude(R_nb, np.eye(3), np.eye(3), a_h=0.0, t=0.0)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_scale_invariance(params: EkfParams,
                           segments: Sequence[Segment],
                           s_values=(1.0, 3.5)) -> dict:
    """sigma_s/s should be independent of the s we plan at.

    H[:, s] = u_w and u_w is proportional to 1/s, so P_ss should scale as s^2
    and the ratio should be invariant. But q_s is absolute (q_s*dt added to
    P_ss regardless of s), so the invariance may break. If it does, planning
    needs a relative q_s -- true `s` is set by an arbitrary VO map scale and
    is unknowable before flight.
    """
    out = {}
    for s in s_values:
        f = ScaleForecaster(params, s_ref=s)
        r = f.run(segments)
        out[s] = float(r.sigma_s_rel[-1])
    vals = list(out.values())
    out["max_rel_spread"] = (max(vals) - min(vals)) / max(min(vals), 1e-12)
    return out
