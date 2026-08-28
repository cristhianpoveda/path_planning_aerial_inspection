"""
frontend.py -- everything between the ROS topics and the EKF core.

Owns time: buffering, interpolation, dedupe, re-stamping and ordering. The core
(filter.py) owns algebra and never reads a clock. Keeping the two apart is what
makes the finite-difference tests possible and what makes bag replay
reproducible (filter_design.md 10).

Four pieces:
  AttitudeBuffer   19 Hz gimbal ring, SLERP interpolation, no ZOH fallback
  AccelEstimator   |a_h| for the R_att schedule, from differentiated velocity
  EventQueue       per-topic value dedupe + velocity re-stamp + stamp order
  IncrementBuilder filter_design.md 4.1 in full
  Scheduler        VO-clocked drive loop, and the 6.1 reset policy
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Flag, auto

import heapq
import itertools

import numpy as np

from . import so3
from .filter import Increment, Kind, ned_to_enu

# --------------------------------------------------------------- event kinds
VELOCITY, ALTITUDE, ATTITUDE = 0, 1, 2
_KIND_NAME = {VELOCITY: "velocity", ALTITUDE: "altitude", ATTITUDE: "attitude"}


class Flags(Flag):
    NONE = 0
    OK = auto()
    NOT_READY = auto()          # first NOT_INITIALIZED, never been ready
    VO_DISCONTINUITY = auto()
    VO_LOST = auto()
    REANCHORED = auto()
    GATE_V_FAIL = auto()
    GATE_R_FAIL = auto()
    DT_FAIL = auto()
    EPOCH_CHANGE = auto()


# ORB-SLAM3 Tracking::eTrackingState, mirrored from slam_node.cpp.
# LOST is never externally observable: it is set and overwritten inside one
# Track() call, so the real sequence is OK -> RECENTLY_LOST -> NOT_INITIALIZED
# (filter_design.md 6). Key on pose_valid, not on the string.
_VALID = {"OK", "OK_KLT"}
_UNINIT = {"NOT_INITIALIZED", "NO_IMAGES_YET", "SYSTEM_NOT_READY"}


@dataclass
class Pose:
    R: np.ndarray
    p: np.ndarray


@dataclass
class VoStatus:
    tracking_state: str = "OK"
    vo_epoch: int = 0
    n_map_points: int = 250
    pose_valid: bool = True


@dataclass
class Event:
    kind: int
    t: float                 # EFFECTIVE time (velocity is re-stamped)
    t_raw: float             # as published, kept for delay re-identification
    value: np.ndarray


# ============================================================ gimbal buffer
class AttitudeBuffer:
    """19 Hz gimbal attitude, SLERPed to arbitrary times.

    filter_design.md 4.1 step 4: true interpolation, no ZOH fallback. The
    gimbal runs faster than the camera, so any image time is bracketed within
    <= 53 ms and interpolation is always available in normal operation.
    """

    def __init__(self, maxlen=4000):
        self._t = deque(maxlen=maxlen)
        self._q = deque(maxlen=maxlen)

    def push(self, t, R):
        q = so3.R_to_quat(R)
        if self._t and t <= self._t[-1]:
            return False                     # out of order, drop
        self._t.append(float(t))
        self._q.append(q)
        return True

    def __len__(self):
        return len(self._t)

    def at(self, t):
        """Interpolated rotation at t, or None if t is not bracketed."""
        n = len(self._t)
        if n == 0:
            return None
        if t < self._t[0] or t > self._t[-1]:
            return None
        ts = np.fromiter(self._t, float, n)
        j = int(np.searchsorted(ts, t))
        if j == 0:
            return so3.quat_to_R(self._q[0])
        t0, t1 = ts[j - 1], ts[j]
        if t1 - t0 <= 0.0:
            return so3.quat_to_R(self._q[j])
        u = (t - t0) / (t1 - t0)
        return so3.quat_to_R(so3.slerp(self._q[j - 1], self._q[j], u))

    def omega(self, t0, t1):
        """Mean angular-rate magnitude over [t0, t1], for g_slew."""
        R0, R1 = self.at(t0), self.at(t1)
        if R0 is None or R1 is None or t1 <= t0:
            return 0.0
        return float(so3.angle(R0.T @ R1) / (t1 - t0))


# ========================================================= accel estimation
class AccelEstimator:
    """|a_h| for the R_att schedule (filter_design.md 5.1).

    From differentiated DJI velocity over ~0.3 s, NEVER from DJI's own tilt:
    that reads near zero exactly when the error is largest, because a quadrotor
    accelerating holds the specific force along the thrust axis.
    """

    def __init__(self, window_s=0.30):
        self.window_s = float(window_s)
        self._buf = deque(maxlen=32)

    def push(self, t, v_enu):
        self._buf.append((float(t), np.asarray(v_enu, float)))

    def value(self):
        if len(self._buf) < 2:
            return 0.0
        t_now = self._buf[-1][0]
        t0, v0 = self._buf[0]
        for t, v in self._buf:
            if t_now - t <= self.window_s:
                t0, v0 = t, v
                break
        dt = t_now - t0
        if dt <= 1e-3:
            return 0.0
        a = (self._buf[-1][1] - v0) / dt
        return float(np.linalg.norm(a[:2]))


# =============================================================== event queue
class EventQueue:
    """Stamp-ordered telemetry queue with per-topic value dedupe.

    Dedupe rationale (filter_design.md 5.0): every DJI source is fully
    quantised. In hover altitude repeats one 0.1 m bin ~10x/s; treating those
    as independent samples shrinks sigma_pz by sqrt(N) and parks p_z on the bin
    centre. dji_node dedupes at PACKET level, but pktTMonoNs advances on
    ATTITUDE change, so a fresh packet can still carry an unchanged altitude.

    R_alt and R_speed are unaffected: both were fitted as per-sample residual
    variance, which does not depend on how many samples are used.
    """

    def __init__(self, params):
        self.p = params
        self._q = []
        self._seq = itertools.count()
        self._last = {}                      # kind -> (value, t_accepted)
        self._t_processed = -np.inf
        self.n_deduped = {VELOCITY: 0, ALTITUDE: 0, ATTITUDE: 0}
        self.n_late = {VELOCITY: 0, ALTITUDE: 0, ATTITUDE: 0}

    def push(self, kind, t_raw, value):
        value = np.atleast_1d(np.asarray(value, float))
        # Velocity is a PURE DELAY, so relabel rather than filter. [M] the
        # delay is ~10 ms, small enough to be near-irrelevant, but the
        # mechanism costs nothing and NEES may prefer a non-zero value.
        t = t_raw - self.p.VEL_DELAY if kind == VELOCITY else t_raw

        if t <= self._t_processed:
            self.n_late[kind] += 1           # re-stamped into a closed interval
            return False

        prev = self._last.get(kind)
        if prev is not None:
            prev_val, t_accepted = prev
            if np.array_equal(value, prev_val):   # values sit on a fixed grid
                if (t - t_accepted) < self.p.T_HOLD:
                    self.n_deduped[kind] += 1
                    return False
                t_accepted = t                    # hold expired, admit
            else:
                t_accepted = t
        else:
            t_accepted = t

        self._last[kind] = (value, t_accepted)
        heapq.heappush(self._q, (t, next(self._seq),
                                 Event(kind, t, float(t_raw), value)))
        return True

    def drain_until(self, t_end):
        out = []
        while self._q and self._q[0][0] <= t_end:
            _, _, ev = heapq.heappop(self._q)
            self._t_processed = ev.t
            out.append(ev)
        return out

    def __len__(self):
        return len(self._q)


# ========================================================= increment builder
@dataclass
class Context:
    """What the builder needs from the core and from telemetry."""
    R_nv: np.ndarray
    s_hat: float
    sigma_s: float
    v_dji_mag: float = 0.0
    R_n_b_dji: np.ndarray = None


class IncrementBuilder:
    """filter_design.md 4.1, in full."""

    def __init__(self, params, att_buf):
        self.p = params
        self.att = att_buf
        self._anchor = None                  # (t, Pose)
        self._epoch = None
        self._ever_ready = False
        self._last_dji = None                # (t, R) for GATE_R
        self.n_discont = 0
        self.n_flags = {}
        self.dbg = deque(maxlen=2000)

    def reset_anchor(self):
        self._anchor = None

    def _discont(self, flags):
        """Count discontinuities by cause. 139 of them on F3_02 with no
        breakdown was undiagnosable."""
        self.n_discont += 1
        k = str(flags).replace("Flags.", "")
        self.n_flags[k] = self.n_flags.get(k, 0) + 1
        return None, flags

    def push(self, t, pose, status, ctx):
        """Returns (Increment | None, Flags)."""
        p = self.p

        # ---- 1. epoch guard, BEFORE pairing -----------------------------
        if self._epoch is None:
            self._epoch = status.vo_epoch
        elif status.vo_epoch != self._epoch:
            self._epoch = status.vo_epoch
            self._anchor = (t, pose) if status.pose_valid else None
            return self._discont(Flags.VO_DISCONTINUITY | Flags.EPOCH_CHANGE)

        # ---- 2. tracking state ------------------------------------------
        if not status.pose_valid:
            # On loss, clear the anchor entirely: the next pose may belong to
            # a different map (filter_design.md 4.1 step 3).
            self._anchor = None
            if not self._ever_ready and status.tracking_state in _UNINIT:
                return None, Flags.NOT_READY
            return None, Flags.VO_LOST
        self._ever_ready = True

        if self._anchor is None:
            self._anchor = (t, pose)
            return None, Flags.REANCHORED

        t0, pose0 = self._anchor
        dt = t - t0

        # ---- 3a. frame-period guard --------------------------------------
        if not (p.DT_MIN <= dt <= p.DT_MAX):
            self._anchor = (t, pose)
            return self._discont(Flags.VO_DISCONTINUITY | Flags.DT_FAIL)

        # ---- 4. pair -----------------------------------------------------
        # dT_c = T_v_c(t0)^-1 T_v_c(t1); dp_v = R_v_c(t0) @ dp_c, which is
        # just the difference of VO positions in `v`.
        dp_v = pose.p - pose0.p
        dR_c = pose0.R.T @ pose.R

        # ---- 3b. plausibility gates, once s is known ---------------------
        flags = Flags.OK
        if ctx.sigma_s / max(ctx.s_hat, 1e-6) < p.SIGMA_S_OK_REL and \
                ctx.v_dji_mag > p.V_MIN:
            v_vo = ctx.s_hat * float(np.linalg.norm(dp_v)) / dt
            if abs(v_vo - ctx.v_dji_mag / p.K_VEL) > \
                    p.GATE_V + p.GATE_V_REL * ctx.v_dji_mag:
                self._anchor = (t, pose)
                return self._discont(Flags.VO_DISCONTINUITY | Flags.GATE_V_FAIL)

        if ctx.R_n_b_dji is not None and self._last_dji is not None:
            t_prev, R_prev = self._last_dji
            if abs(t_prev - t0) < p.NOMINAL_DT:
                dR_dji = R_prev.T @ ctx.R_n_b_dji
                # VO relative rotation is a CAMERA rotation; the gimbal makes
                # camera and body rotation differ, so compare loosely.
                if abs(so3.angle(dR_c) - so3.angle(dR_dji)) > p.GATE_R:
                    self._anchor = (t, pose)
                    return self._discont(Flags.VO_DISCONTINUITY | Flags.GATE_R_FAIL)

        # ---- 5. gimbal lookup and lever arm ------------------------------
        R_bc0, R_bc1 = self.att.at(t0), self.att.at(t)
        if R_bc0 is None or R_bc1 is None:
            self._anchor = (t, pose)
            return self._discont(Flags.VO_DISCONTINUITY | Flags.REANCHORED)

        # R_n_b = R_bar_n_v @ R_v_c @ R_b_c^-1, using the filter's own frame
        # estimate -- not truth. dl is metric; dp_v is not. Never summed.
        R_nb0 = ctx.R_nv @ pose0.R @ R_bc0.T
        R_nb1 = ctx.R_nv @ pose.R @ R_bc1.T
        dl = (R_nb1 - R_nb0) @ p.r_l

        # ---- 6. noise gain (4.3) -----------------------------------------
        g_feat = float(np.clip(p.n_ref / max(status.n_map_points, 1),
                               1.0, p.feat_cap))
        g_slew = 1.0 + p.kappa * self.att.omega(t0, t)
        Sigma_v = p.Sigma_base * (g_feat * g_slew)

        self._anchor = (t, pose)
        if ctx.v_dji_mag > 0.3:
            # s implied by THIS increment alone, from DJI velocity. Should
            # equal check_scale's per-increment value (median 3.91 on F9_02).
            s_implied = (ctx.v_dji_mag / p.K_VEL) / (float(np.linalg.norm(dp_v)) / dt)
            self.dbg.append((t, dt, float(np.linalg.norm(dp_v)),
                             ctx.v_dji_mag, ctx.s_hat, ctx.sigma_s, s_implied))
        return Increment(dp_v=dp_v, dl=dl, dt=dt, Sigma_v=Sigma_v, t=t), flags


# =================================================================== scheduler
class Scheduler:
    """VO-clocked drive loop.

    Arrival order is not stamp order: camera frames are stamped ~200 ms in the
    past (VIDEO_LATENCY) while telemetry is near-live, so telemetry for time T
    is in hand well before the VO increment covering T. Processing is therefore
    driven by VO: when the increment for (t0, t1] is complete, propagate to t1
    and drain every telemetry event with an effective stamp in that window, in
    stamp order.

    That gives determinism without a fixed lag: given the same SET of messages
    the processing order is identical regardless of arrival order, which is the
    requirement in filter_design.md 10.
    """

    def __init__(self, params, core):
        self.p = params
        self.core = core
        self.att = AttitudeBuffer()
        self.q = EventQueue(params)
        self.builder = IncrementBuilder(params, self.att)
        self.accel = AccelEstimator(params.ACC_SMOOTH_S)
        self.innovations = []
        # Velocity must pair with the increment whose interval CONTAINS its
        # effective stamp (filter_design.md 5.3), not merely the newest one.
        # Pairing against the newest let a stale increment through whenever a
        # DT_FAIL intervened -- 179 of them on F9_02 -- and `s` collapsed.
        self._last_inc = None
        self._inc_t0 = None          # start of _last_inc's interval
        self._last_pose = None
        self._last_vz = 0.0
        self._last_v_enu = np.zeros(3)
        self._last_R_dji = None
        self._t_last_vo = None
        self._saw_uninit_since_epoch = False
        self.ready = False
        self.on_processed = None
        # VO stamps are decode_time - VIDEO_LATENCY, i.e. deliberately in the
        # past, while gimbal stamps are near-live. If VO trails the newest
        # gimbal sample the buffer must be long enough to still hold the
        # bracketing pair; if it LEADS, no buffer length helps.
        self._lag_min = float("inf")
        self._lag_max = float("-inf")
        self._pending = deque()
        self._no_inc_speeds = deque(maxlen=2000)
        self._applied_speeds = deque(maxlen=2000)
        self.needs_reinit = False

    # ---- ingress ------------------------------------------------------
    def on_gimbal(self, t, R_b_c):
        self.att.push(t, R_b_c)
        self._flush_pending()

    def on_attitude(self, t, rpy):
        self.q.push(ATTITUDE, t, rpy)

    def on_altitude(self, t, z):
        self.q.push(ALTITUDE, t, [z])

    def on_velocity(self, t, v_ned):
        self.q.push(VELOCITY, t, v_ned)

    def on_vo(self, t, pose, status):
        """Defer until the gimbal buffer brackets t.

        [M] VO stamps run from -0.245 s to +0.160 s relative to the newest
        gimbal sample. The negative side is VIDEO_LATENCY and a long buffer
        covers it; the positive side is arrival order between two
        subscriptions, and no buffer length helps -- the sample simply has not
        arrived yet. Holding the frame costs one gimbal period (~72 ms) and
        preserves 4.1 step 4's "no ZOH fallback".
        """
        t = t - self.p.VO_DELAY
        if len(self._pending) > 100:
            self._pending.popleft()
        self._pending.append((t, pose, status))
        return self._flush_pending()

    def _flush_pending(self):
        flags = Flags.NONE
        while self._pending and len(self.att) and \
                self._pending[0][0] <= self.att._t[-1]:
            t, pose, status = self._pending.popleft()
            flags = self._process_vo(t, pose, status)
        return flags

    # ---- VO tick ------------------------------------------------------
    def _process_vo(self, t, pose, status):

        if len(self.att):
            lag = t - self.att._t[-1]
            self._lag_min = min(self._lag_min, lag)
            self._lag_max = max(self._lag_max, lag)
            
        if status.tracking_state in _UNINIT:
            self._saw_uninit_since_epoch = True

        ctx = Context(R_nv=self.core.x.R_nv, s_hat=self.core.x.s,
                      sigma_s=self.core.x.sigma_s,
                      v_dji_mag=float(np.linalg.norm(self._last_v_enu)),
                      R_n_b_dji=self._last_R_dji)
        inc, flags = self.builder.push(t, pose, status, ctx)

        if Flags.EPOCH_CHANGE in flags:
            self._apply_reset_policy()

        if inc is not None:
            self.core.propagate(inc)
            self._last_inc = inc
            self._inc_t0 = inc.t - inc.dt
            self.ready = True
        elif self.ready and (Flags.VO_LOST in flags or inc is None):
            # Propagation gap, not a rejected measurement: dead-reckon on DJI
            # velocity (filter_design.md 6). K_VEL-corrected, nav frame.
            if self._t_last_vo is not None and t > self._t_last_vo:
                self.core.dead_reckon(self._last_v_enu / self.p.K_VEL,
                                      t - self._t_last_vo)

        self._last_pose = pose
        self._t_last_vo = t

        for ev in self.q.drain_until(t):
            self._apply(ev, pose)

        if self.on_processed is not None:
            self.on_processed(t, pose)
            
        return flags

    # ---- event application --------------------------------------------
    def _apply(self, ev, pose):
        core = self.core
        if ev.kind == ATTITUDE:
            R_b_c = self.att.at(ev.t)
            if R_b_c is None:
                return
            self._last_R_dji = so3.rpy_to_R(*ev.value)
            # R_v_c at the VO tick, not at ev.t: the VO pose is only published
            # at frame times. At 10 Hz the error over one interval is small,
            # and 5.1 already accepts a 25 ms attitude/velocity skew.
            inn = core.update_attitude(self._last_R_dji, pose.R, R_b_c,
                                       a_h=self.accel.value(), t=ev.t)
            self.innovations.append(inn)

        elif ev.kind == ALTITUDE:
            inn = core.update_altitude(float(ev.value[0]), vz=self._last_vz,
                                       t=ev.t)
            self.innovations.append(inn)

        elif ev.kind == VELOCITY:
            v_enu = ned_to_enu(ev.value)
            self._last_v_enu = v_enu
            self._last_vz = float(v_enu[2])
            self.accel.push(ev.t, v_enu)
            if self._last_inc is not None and self._inc_t0 is not None \
                    and self._inc_t0 <= ev.t <= self._last_inc.t:
                inn = core.update_velocity(v_enu, self._last_inc, t=ev.t)
                self.innovations.append(inn)
                self._applied_speeds.append(float(np.linalg.norm(v_enu)))
            else:
                self.q.n_late[VELOCITY] += 1
                self._no_inc_speeds.append(float(np.linalg.norm(v_enu)))

    # ---- 6.1 ------------------------------------------------------------
    def _apply_reset_policy(self):
        """Sim(3) correction vs map rebuild, distinguished from data the EKF
        already has (filter_design.md 6.1).

        Monocular loop closure and merge use Sim(3) with FREE SCALE, so a
        correction can rescale the map even with the state at OK -- hence
        inflating P_ss rather than trusting the mean. A rebuild replaces the
        frame and scale outright.

        p and b survive both: they live in the nav frame and are unaffected by
        anything VO does.
        """
        x = self.core.x
        if self._saw_uninit_since_epoch:
            # NOT P0_scale. [M] On F9_02 the rebuilt map reproduced the old
            # map's scale (implied truth 3.85 vs 3.79), so `s` is uncertain,
            # not unknown. At sigma_s/s = 28% the propagation term
            # F[IDX_P, IDX_S] = u builds cross-covariance fast, and the
            # ALTITUDE update then drags `s` through it: measured 3.61 -> 2.29
            # over 70 s with cov_s_b growing 1e-5 -> 1e-2, while the velocity
            # update was correctly blocked by the V_LOW gate.
            x.P[3, 3] = min(x.P[3, 3] * 4.0, (0.15 * max(x.s, 1e-3)) ** 2)
            x.P[5, 5] = self.p.P0_theta_xy
            x.P[6, 6] = self.p.P0_theta_xy
            x.P[7, 7] = self.p.P0_theta_z
            if self._last_R_dji is not None and self._last_pose is not None:
                R_bc = self.att.at(self._t_last_vo) if self._t_last_vo else None
                if R_bc is not None:
                    x.R_nv = so3.normalise(
                        self._last_R_dji @ R_bc @ self._last_pose.R.T)
            self._saw_uninit_since_epoch = False
            self.needs_reinit = True
        else:
            # Same cap as the rebuild branch. [M] Uncapped, this fired at
            # t=85 on F9_02 and took sigma_s from 0.12 to 0.48 (13% relative),
            # which re-opened the p_z-s cross-covariance: cov_s_b climbed to
            # 2.9e-3 and altitude dragged `s` from 3.61 to 3.19 over 75 s.
            # P0_scale is an INIT prior, not a mid-flight one.
            x.P[3, 3] = min(x.P[3, 3] * 4.0, (0.15 * max(x.s, 1e-3)) ** 2)
            x.P[5, 5] = min(x.P[5, 5] * 4.0, self.p.P0_theta_xy)
            x.P[6, 6] = min(x.P[6, 6] * 4.0, self.p.P0_theta_xy)
            x.P[7, 7] = min(x.P[7, 7] * 4.0, self.p.P0_theta_z)
            # Sim(3) loop closure uses FREE SCALE in monocular, so `s` can be
            # wrong by the correction factor even with the state at OK.
            # Inflating P_ss is not enough on its own -- with no motion there
            # is nothing to shrink it back. Ask for a re-measure too.
            self.needs_reinit = True

    # ---- diagnostics -----------------------------------------------------
    def stats(self):
        return {
            "deduped": {_KIND_NAME[k]: v for k, v in self.q.n_deduped.items()},
            "late": {_KIND_NAME[k]: v for k, v in self.q.n_late.items()},
            "vo_discontinuities": self.builder.n_discont,
            "queued": len(self.q),
            "gimbal_buffered": len(self.att),
            "discont_by_cause": dict(self.builder.n_flags),
                        "vo_lag_s": (round(self._lag_min, 4), round(self._lag_max, 4)),
        }
