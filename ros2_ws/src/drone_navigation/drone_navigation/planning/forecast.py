"""sigma_s and Sigma_pp along a candidate tour.

Drives the actual EkfCore.
"""

from dataclasses import dataclass

import numpy as np

try:
    from drone_localisation.ekf.filter import (EkfCore, Increment, IDX_P,
                                               IDX_S, State)
    from drone_localisation.ekf.params import EkfParams
except ImportError as e:                                    # pragma: no cover
    raise ImportError(
        "the planner needs drone_localisation on the path: it drives the real "
        "EkfCore so the forecast is consistent with the filter by "
        "construction. filter.py and params.py import "
        f"only numpy, so no ROS dependency comes with it. Original error: {e}")


@dataclass
class ForecastConfig:
    dt_vo: float = 0.0333          # [M] EkfParams.NOMINAL_DT
    dt_telemetry: float = 0.10     # [M] DJI telemetry period
    s_true: float = 3.0            # [T] VO scale the forecast assumes
    sigma_vo_mm: float = 1.5       # [T] per-increment VO noise, mm
    dwell_s: float = 3.0           # hover time at each viewpoint
    z_nominal: float = 1.4         # m, used only for the altitude update


class Forecast:
    """Replays a tour through EkfCore and reports sigma_s at each viewpoint."""

    def __init__(self, params: EkfParams = None, cfg: ForecastConfig = None):
        self.p = params or EkfParams()
        self.cfg = cfg or ForecastConfig()
        self.reset()

    def reset(self, sigma_s0=None):
        self.core = EkfCore(self.p, State(self.p))
        self.core.x.R_nv = np.eye(3)
        self.core.x.s = self.cfg.s_true
        self.core.s_ref = self.cfg.s_true
        self.core.x.p = np.array([0.0, 0.0, self.cfg.z_nominal])
        if self.p.estimate_scale <= 0.5:
            self.core.hold_scale()
        elif sigma_s0 is not None:
            self.core.x.P[IDX_S, IDX_S] = float(sigma_s0) ** 2
        self._telemetry_debt = 0.0
        return self

    # ------------------------------------------------------------- readouts
    @property
    def sigma_s(self):
        return self.core.x.sigma_s

    @property
    def sigma_s_ratio(self):
        return self.core.x.sigma_s / max(self.core.x.s, 1e-9)

    @property
    def Sigma_pp(self):
        return self.core.x.P[IDX_P, IDX_P].copy()

    # --------------------------------------------------------------- stepping
    def _increment(self, v_n, dt):
        """A VO increment consistent with moving at v_n for dt.

        R_nv is identity here.
        """
        dp_n = np.asarray(v_n, float) * dt
        sig = (self.cfg.sigma_vo_mm * 1e-3) ** 2
        return Increment(dp_v=dp_n / self.core.x.s,
                         dl=np.zeros(3), dt=dt,
                         Sigma_v=np.eye(3) * sig / self.core.x.s ** 2,
                         t=0.0)

    def step(self, v_n, dt):
        """Advance dt at nav-frame velocity v_n, applying updates in turn."""
        inc = self._increment(v_n, dt)
        self.core.propagate(inc)

        self._telemetry_debt += dt
        if self._telemetry_debt + 1e-12 < self.cfg.dt_telemetry:
            return
        self._telemetry_debt = 0.0

        # attitude: R_v_c = R_b_c = I and R_dji = R_nv gives a zero residual
        self.core.update_attitude(self.core.x.R_nv, np.eye(3), np.eye(3),
                                  a_h=0.0)
        # altitude: z = p_z + b
        self.core.update_altitude(self.core.x.p[2] + self.core.x.b,
                                  vz=float(v_n[2]))
        # velocity: v_enu = K_VEL * h, so z = v_enu / K_VEL == h
        h, _ = self.core.velocity_prediction(inc)
        self.core.update_velocity(self.p.K_VEL * h, inc)

    def fly(self, path, speed, dwell=None):
        """Fly a polyline at `speed`, then hover for `dwell` seconds.

        Returns sigma_s at arrival, i.e. after the dwell, which is when the
        image is taken.
        """
        pts = np.asarray(path, float)
        dwell = self.cfg.dwell_s if dwell is None else dwell
        for a, b in zip(pts[:-1], pts[1:]):
            d = float(np.linalg.norm(b - a))
            if d < 1e-9:
                continue
            v = (b - a) / d * speed
            n = max(1, int(np.ceil(d / speed / self.cfg.dt_vo)))
            for _ in range(n):
                self.step(v, self.cfg.dt_vo)
        for _ in range(int(round(dwell / self.cfg.dt_vo))):
            self.step(np.zeros(3), self.cfg.dt_vo)
        return self.sigma_s

    def over_tour(self, legs, sigma_s0=None):
        """legs: iterable of (name, path, speed). Returns a list of dicts."""
        self.reset(sigma_s0)
        out = []
        t = 0.0
        for name, path, speed in legs:
            pts = np.asarray(path, float)
            length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            self.fly(pts, speed)
            t += length / speed + self.cfg.dwell_s
            out.append(dict(name=name, t=t, length=length, speed=speed,
                            sigma_s=self.sigma_s,
                            sigma_s_ratio=self.sigma_s_ratio,
                            Sigma_pp=self.Sigma_pp))
        return out


def v_low_true(p: EkfParams):
    """True ground speed at which the velocity update stops being inflated.
    """
    return p.V_LOW / p.K_VEL


def v_min_true(p: EkfParams):
    return p.V_MIN / p.K_VEL
