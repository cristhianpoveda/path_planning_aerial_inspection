"""
filter.py -- error-state EKF core. Propagation + the three updates.

Imports numpy and so3 and NOTHING else. No ROS, no clock reads, no I/O. That
is what makes the finite-difference tests possible and what makes bag replay
bit-reproducible (filter_design.md 10).

State (filter_design.md 2), 8 error states:

    dx = [ dp_x, dp_y, dp_z, ds, db, dtheta_x, dtheta_y, dtheta_z ]
           0     1     2     3   4   5         6         7

The nominal (p, s, b, R_bar_n_v) lives outside the covariance. `dtheta` is
identically zero at every linearisation point because the reset runs after
every update, so the Jacobians in filter_design.md 5 are exact as written
rather than first-order.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import so3

IDX_P = slice(0, 3)
IDX_S = 3
IDX_B = 4
IDX_TH = slice(5, 8)
NX = 8

G = 9.81


class Kind(Enum):
    ATTITUDE = "attitude"
    ALTITUDE = "altitude"
    VELOCITY = "velocity"


@dataclass
class Increment:
    """One conditioned VO increment (filter_design.md 4.1 step 6).

    dp_v and dl are NEVER summed: dp_v is unscaled (VO units), dl is metric.
    Keeping them separate is a type-level guard against that mistake.
    """
    dp_v: np.ndarray        # (3,) unscaled VO translation increment, in `v`
    dl: np.ndarray          # (3,) metric lever-arm term, in `n`
    dt: float               # s
    Sigma_v: np.ndarray     # (3,3) increment noise in `v`, already Sigma_base*g
    t: float = 0.0          # effective timestamp of t1


@dataclass
class Innovation:
    """Logged unconditionally for every update -- this log IS the tuning loop
    (filter_design.md 10)."""
    kind: Kind
    t: float
    y: np.ndarray
    S: np.ndarray
    nis: float
    accepted: bool
    R_used: np.ndarray = field(default=None)
    h: np.ndarray = field(default=None)
    note: str = ""


# --------------------------------------------------------------------- frames
def ned_to_enu(v):
    """[M] DJI velocity is NED, magnetic-north referenced. The mapping to the
    ENU nav frame is (x, y, z) -> (y, x, -z), confirmed on every bag by
    axis_perm [1,0,2] with signs [+,+,-].

    Note the FULL 3x3 map is a proper rotation (det = +1) -- both NED and ENU
    are right-handed. What is a reflection is its HORIZONTAL 2x2 block,
    det = -1, which is why the fitted det_horizontal is negative on every bag.
    The two facts are consistent: the x-y swap contributes -1 and the z flip
    another -1.

    The residual yaw between magnetic north and the room is NOT removed here;
    it is dtheta_z, estimated by the filter (filter_design.md 2).
    """
    v = np.asarray(v, float).reshape(3)
    return np.array([v[1], v[0], -v[2]])


def sigma_rp(params, a_h):
    """Acceleration-scheduled roll/pitch noise (filter_design.md 5.1).

    A quadrotor accelerates by tilting, so the specific force stays along the
    thrust axis and a gravity-referenced tilt estimate under-reports by close
    to the whole tilt angle. Measured slope 0.098 s^2/m against 1/g = 0.102.

    a_h must come from differentiated DJI velocity, NEVER from DJI's own tilt:
    that reads near zero exactly when the error is largest.
    """
    return float(np.hypot(params.sigma_rp0, params.accel_slope * float(a_h)))


def symmetrise(P):
    return 0.5 * (P + P.T)


def _spd_solve(S, B):
    """Solve S X = B for symmetric positive-definite S, by Cholesky.

    S is at most 3x3 (filter_design.md 5.0), so this is never a bottleneck.
    Never form an explicit inverse.
    """
    L = np.linalg.cholesky(S)
    Y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, Y)


class State:
    """Nominal state plus error covariance."""

    def __init__(self, params):
        self.p = np.zeros(3)
        self.s = 1.0
        self.b = params.b_prior_mean
        self.R_nv = np.eye(3)               # nominal R_bar_n_v
        self.P = np.zeros((NX, NX))
        self.P[IDX_P, IDX_P] = np.eye(3) * params.P0_pos
        self.P[IDX_S, IDX_S] = params.P0_scale
        self.P[IDX_B, IDX_B] = params.b_prior_sigma ** 2
        self.P[5, 5] = params.P0_theta_xy
        self.P[6, 6] = params.P0_theta_xy
        self.P[7, 7] = params.P0_theta_z

    def copy(self):
        out = State.__new__(State)
        out.p = self.p.copy()
        out.s = float(self.s)
        out.b = float(self.b)
        out.R_nv = self.R_nv.copy()
        out.P = self.P.copy()
        return out

    @property
    def sigma_s(self):
        return float(np.sqrt(max(self.P[IDX_S, IDX_S], 0.0)))

    @property
    def sigma_b(self):
        return float(np.sqrt(max(self.P[IDX_B, IDX_B], 0.0)))

    @property
    def cov_s_b(self):
        """filter_design.md 7 says watch this, not just the marginals --
        altitude reaches s only through it."""
        return float(self.P[IDX_S, IDX_B])


class EkfCore:

    def __init__(self, params, state=None):
        self.p = params
        self.x = state if state is not None else State(params)
        self.n_rejected = {k: 0 for k in Kind}
        self.n_s_clamped = 0
        # Velocity rejections have three distinct causes that n_rejected
        # lumps together. They mean different things: V_MIN is by design and
        # unbiased, NIS is outlier rejection, and a missing covering increment
        # is a coverage problem that may correlate with speed.
        self.n_vel_reason = {"below_V_MIN": 0, "NIS": 0, "applied": 0}
        # Reference scale for R_vo. R MUST NOT depend on the state being
        # estimated: with R_vo proportional to s^2, a downward correction
        # shrinks R, which raises the gain, which amplifies the next downward
        # correction. s walked 3.61 -> 1.87 on F9_02 while sigma_s shrank.
        self.s_ref = float(self.x.s)
        # [M] DJI attitude yaw and DJI velocity NED use DIFFERENT yaw datums:
        # measured 51.84 deg +- 3.81 on F9_02, constant in time (fitted slope
        # +0.013 against bearing). filter_design.md gives the filter one nav
        # frame, so no single R_n_v satisfies both updates and the tighter R
        # wins. This is that offset, estimated at init and held.
        self.psi_v = 0.0
        self.n_s_limited = 0

    # ================================================================ predict
    def process_noise(self, inc):
        p = self.p
        Q = np.zeros((NX, NX))
        Rn = self.x.R_nv
        # s^2 because the increment is unscaled: its noise scales with s too
        Q[IDX_P, IDX_P] = (self.x.s ** 2) * (Rn @ inc.Sigma_v @ Rn.T)
        Q[IDX_S, IDX_S] = p.q_s * inc.dt
        Q[IDX_B, IDX_B] = p.q_b * inc.dt
        Q[5, 5] = p.q_theta_xy * inc.dt
        Q[6, 6] = p.q_theta_xy * inc.dt
        Q[7, 7] = p.q_theta_z * inc.dt
        return Q

    def transition(self, inc):
        u = self.x.R_nv @ inc.dp_v
        F = np.eye(NX)
        F[IDX_P, IDX_S] = u
        F[IDX_P, IDX_TH] = -self.x.s * so3.hat(u)
        return F, u

    def propagate(self, inc):
        """p+ = p + s*u - dl ; s, b, dtheta unchanged.

        dl is subtracted: it is the camera's motion relative to base_link, so
        removing it converts camera motion to body motion.
        """
        F, u = self.transition(inc)
        Q = self.process_noise(inc)
        self.x.p = self.x.p + self.x.s * u - inc.dl
        self.x.P = symmetrise(F @ self.x.P @ F.T + Q)
        return F

    def dead_reckon(self, v_n, dt):
        """Propagate on DJI velocity when VO is unavailable (6).

        v_n must already be in the nav frame and K_VEL-corrected. Q_p is
        inflated hard: at inspection speed the velocity being integrated is
        itself below V_MIN and unreliable (5.3).
        """
        self.x.p = self.x.p + np.asarray(v_n, float) * dt
        Q = np.zeros((NX, NX))
        Q[IDX_P, IDX_P] = np.eye(3) * (self.p.R_speed_h * self.p.V_INFL) * dt
        Q[IDX_S, IDX_S] = self.p.q_s * dt
        Q[IDX_B, IDX_B] = self.p.q_b * dt
        Q[5, 5] = self.p.q_theta_xy * dt
        Q[6, 6] = self.p.q_theta_xy * dt
        Q[7, 7] = self.p.q_theta_z * dt
        self.x.P = symmetrise(self.x.P + Q)

    # ================================================================= update
    def _apply_update(self, kind, y, H, R, gate, t=0.0, note="", h=None):
        """Joseph-form update with a chi-square innovation gate.

        Rejection is counted, not silent: filter_design.md 5.1 requires it for
        attitude, where +-20 deg transients during aggressive manoeuvres are
        outliers rather than Gaussian tails and inflating R will not stop them
        corrupting dtheta.
        """
        y = np.atleast_1d(np.asarray(y, float))
        H = np.atleast_2d(np.asarray(H, float))
        R = np.atleast_2d(np.asarray(R, float))

        S = symmetrise(H @ self.x.P @ H.T + R)
        try:
            nis = float(y @ _spd_solve(S, y))
        except np.linalg.LinAlgError:
            self.n_rejected[kind] += 1
            return Innovation(kind, t, y, S, float("inf"), False, R, h,
                              "S not positive definite")

        if nis > gate:
            self.n_rejected[kind] += 1
            if kind is Kind.VELOCITY:
                self.n_vel_reason["NIS"] += 1
            return Innovation(kind, t, y, S, nis, False, R, h,
                              (note + " NIS gate").strip())

        PHt = self.x.P @ H.T
        K = _spd_solve(S, PHt.T).T                  # K = P H^T S^-1
        if kind is not Kind.VELOCITY:
            # Only VELOCITY observes `s`. Altitude and attitude must NOT pin
            # it. Their H does not touch `s`, but the propagation term
            # F[IDX_P, IDX_S] = u builds a p_z-s cross-covariance and the gain
            # then moves `s` through it -- entangled with `b`, which is
            # exactly why filter_design.md 12 calls velocity the primary scale
            # cue. Attitude leaks the same way via P_s_theta. Measured on
            # F9_02: cov_s_b climbed 1e-5 -> 2.9e-3 and `s` slid 3.61 -> 3.19
            # over 75 s of inspection-speed flight, while the velocity update
            # was correctly gated out by V_LOW. Joseph form is valid for ANY
            # gain, so zeroing this row keeps P symmetric and PSD.
            K[IDX_S, :] = 0.0
        dx = K @ y

        if kind is Kind.VELOCITY and abs(dx[IDX_S]) > self.p.S_STEP_MAX * self.x.s:
            # One update may not rewrite `s`. [M] On F6, two updates 100 ms
            # apart moved `s` 2.778 -> 1.032 -> 0.247 and then to the 1e-3
            # clamp, freezing position for 23 s of a 43 s flight. Scale is a
            # per-flight constant estimated from many samples (7); a single
            # innovation carrying that much authority means P_ss is wide and
            # the sample is an outlier, not that the scale changed.
            dx = dx.copy()
            dx[IDX_S] = np.sign(dx[IDX_S]) * self.p.S_STEP_MAX * self.x.s
            self.n_s_limited += 1

        self.x.p = self.x.p + dx[IDX_P]
        self.x.s = float(self.x.s + dx[IDX_S])
        # `s` is a scale factor: negative flips the direction of every
        # propagated increment. Observed at s = -0.9998 on F9_02. Counted
        # separately -- the update was APPLIED then clamped, not rejected,
        # and any update can move s through cross-covariance.
        if self.x.s < 1e-3:
            self.x.s = 1e-3
            self.n_s_clamped += 1
        self.x.b = float(self.x.b + dx[IDX_B])

        IKH = np.eye(NX) - K @ H
        self.x.P = symmetrise(IKH @ self.x.P @ IKH.T + K @ R @ K.T)

        # Reset AFTER every update, not once per packet: this is what keeps
        # dtheta == 0 at every linearisation point (filter_design.md 1).
        self.reset_error_state(dx[IDX_TH])
        if kind is Kind.VELOCITY:
            self.n_vel_reason["applied"] += 1
        return Innovation(kind, t, y, S, nis, True, R, h, note)

    # ---------------------------------------------------------------- 5.1
    def attitude_residual(self, R_n_b_dji, R_v_c, R_b_c):
        """z = Log(R_dji @ R_bar_n_b^-1), the observed misalignment in `n`."""
        R_bar_n_b = self.x.R_nv @ R_v_c @ R_b_c.T
        return so3.log(R_n_b_dji @ R_bar_n_b.T), R_bar_n_b

    def update_attitude(self, R_n_b_dji, R_v_c, R_b_c, a_h=0.0, t=0.0):
        """Observes all three components of dtheta.

        h(dtheta) = dtheta, so H = +I3 and y = z - h(x) = z, since dtheta is
        zero after every reset.

        SIGN TRAP (filter_design.md 5.1): y = z - h(x) implies dy/ddtheta =
        -H, so finite-differencing the RESIDUAL returns -I3 and will "confirm"
        the wrong sign. H = +I3 is correct: K = P H^T S^-1 is then positive,
        dtheta+ = K z, and the reset moves the nominal TOWARD R_dji.
        """
        z, _ = self.attitude_residual(R_n_b_dji, R_v_c, R_b_c)
        H = np.zeros((3, NX))
        H[:, IDX_TH] = np.eye(3)
        srp = sigma_rp(self.p, a_h)
        R = np.diag([srp ** 2, srp ** 2, self.p.sigma_yaw ** 2])
        return self._apply_update(Kind.ATTITUDE, z, H, R,
                                  self.p.NIS_GATE_ATT, t)

    # ---------------------------------------------------------------- 5.2
    def update_altitude(self, z_alt, vz=0.0, t=0.0):
        """h(x) = p_z + b. Height above takeoff, terrain-blind.

        [M] The gain is unity; the direction-dependent k correction inherited
        from the old altitude_agl key does not apply to KeyAltitude.
        """
        H = np.zeros((1, NX))
        H[0, 2] = 1.0
        H[0, IDX_B] = 1.0
        vertical = abs(float(vz)) > self.p.VZ_INFL
        R = np.array([[self.p.R_alt_vert if vertical else self.p.R_alt_hover]])
        y = np.array([float(z_alt) - (self.x.p[2] + self.x.b)])
        return self._apply_update(Kind.ALTITUDE, y, H, R,
                                  self.p.NIS_GATE_ALT, t,
                                  "vertical" if vertical else "hover")

    # ---------------------------------------------------------------- 5.3
    def velocity_prediction(self, inc):
        """h(x) = s*u_w - dl/dt, with u_w = R_bar_n_v @ (dp_v/dt)."""
        w_v = inc.dp_v / inc.dt
        u_w = self.x.R_nv @ w_v
        return self.x.s * u_w - inc.dl / inc.dt, u_w

    def update_velocity(self, v_enu, inc, omega_yaw=0.0, t=0.0):
        """Observes s along u_w and dtheta_z perpendicular to it.

        v_enu is the ENU-converted DJI velocity, NOT yet K_VEL-corrected.
        The correction is applied here together with its covariance, because
        dividing a measurement without dividing its covariance misstates R
        (filter_design.md 12).

        [M] LOW-SPEED GATE. gain_horizontal falls 0.93 -> 0.82 -> 0.77 as speed
        drops from 1.0 m/s to hover, while gain_z holds: the 0.1 m/s quantum
        acts as a dead zone. This is the inspection regime, so `s` is refined
        on transits and held through passes.
        """
        p = self.p
        v_enu = np.asarray(v_enu, float).reshape(3)
        
        speed = float(np.linalg.norm(v_enu))

        if speed < p.V_MIN:
            self.n_rejected[Kind.VELOCITY] += 1
            self.n_vel_reason["below_V_MIN"] += 1
            return Innovation(Kind.VELOCITY, t, np.zeros(3), np.eye(3),
                              0.0, False, None, None, "below V_MIN")

        h, u_w = self.velocity_prediction(inc)
        H = np.zeros((3, NX))
        # Both sides must carry speed information. V_LOW gates the
        # MEASUREMENT; this gates the PREDICTION. [M] On F6 the VO increment
        # was near zero (87 % of |h| below 0.02 m/s) while DJI occasionally
        # read 0.3-0.4 m/s, so the innovation was the whole measurement and
        # K_s = P_ss*u_w*y/S drove `s` by ~0.5 per update. `s` hit the 1e-3
        # clamp at t=5 s and stayed there for 23 s of a 43 s flight, freezing
        # position: measured |dp_est|/|dp_gt| = 0.003 over 1 s windows.
        if (p.estimate_scale > 0.5 and speed >= p.V_LOW
                and float(np.linalg.norm(u_w)) >= p.V_VO_MIN):
            H[:, IDX_S] = u_w
        H[:, IDX_TH] = -self.x.s * so3.hat(u_w)

        base = np.diag([p.R_speed_h, p.R_speed_h, p.R_speed_v])
        note = ""
        if speed < p.V_LOW:
            base = base * p.V_INFL
            note = "low-speed inflated"
        if abs(float(omega_yaw)) > p.OMEGA_GATE:
            # omega_b x r_imu is dropped, not modelled: the residual is lateral
            # and aliases onto dtheta_z rather than averaging out (5.3).
            base = base + np.eye(3) * p.R_speed_rot
            note = (note + " yaw-rate inflated").strip()

        z = v_enu / p.K_VEL
        R = base / (p.K_VEL ** 2)
        # The prediction h = s*u_w is built from the VO increment, which is a
        # MEASURED quantity with noise Sigma_v -- but that noise enters only Q,
        # never S, so the update treats the increment as exact. Scaled by
        # s ~ 3.5 and divided by dt ~ 0.03 it is LARGER than R_speed itself:
        # 0.11 m/s against 0.057. Added on the prediction side, so it is not
        # divided by K_VEL and not multiplied by V_INFL.
        # s_ref, not the live s, for the same reason R_vo uses it: R must not
        # depend on the state being estimated.
        Rn = self.x.R_nv
        R = R + (self.s_ref ** 2) * (Rn @ inc.Sigma_v @ Rn.T) / (inc.dt ** 2)
        return self._apply_update(Kind.VELOCITY, z - h, H, R,
                                  p.NIS_GATE_VEL, t, note, h=h)

    def hold_scale(self):
        """Hold `s` exactly, by removing it from the covariance.

        `estimate_scale = 0` alone does NOT hold `s`: it only zeroes the
        velocity update's scale column. The altitude update still reaches `s`
        through the p_z-s cross-covariance that propagation creates via
        F[IDX_P, IDX_S] = u, entangled with `b`. Measured on `mixed`: an s
        seeded EXACTLY at truth (1.37) was dragged to 0.9608 in 80 s, for
        3.578 m of position error.

        With P_ss = 0 and q_s = 0 the row stays identically zero forever:
        propagation gives P_ss+ = P_ss + q_s*dt = 0 and P_ps+ = P_ps +
        u*P_ss = 0, and K[s, :] = (P H^T S^-1)[s, :] = 0, so no update can
        move it.
        """
        self.x.P[IDX_S, :] = 0.0
        self.x.P[:, IDX_S] = 0.0

    # ================================================================== misc
    def reset_error_state(self, dth):
        """Fold dtheta into the nominal and zero it (filter_design.md 1)."""
        dth = np.asarray(dth, float)
        if float(dth @ dth) > 0.0:
            self.x.R_nv = so3.normalise(so3.exp(dth) @ self.x.R_nv)

    def body_attitude(self, R_v_c, R_b_c):
        """R_n_b for output (8): composed at publish time, not a state."""
        return self.x.R_nv @ R_v_c @ R_b_c.T

    def snapshot(self):
        return {
            "p": self.x.p.copy(),
            "s": float(self.x.s),
            "b": float(self.x.b),
            "R_nv": self.x.R_nv.copy(),
            "P": self.x.P.copy(),
            "sigma_s": self.x.sigma_s,
            "sigma_b": self.x.sigma_b,
            "cov_s_b": self.x.cov_s_b,
            "n_rejected": {k.value: v for k, v in self.n_rejected.items()},
            "s_clamp": self.n_s_clamped,
        }
