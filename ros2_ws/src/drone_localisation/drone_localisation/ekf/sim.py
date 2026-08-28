"""
sim.py -- synthetic scenario generator for validating the EKF core.

ROS-free and bag-free. Generates a known trajectory, a known VO scale and a
known non-gravity-aligned VO frame, then produces measurements that reproduce
the behaviours actually measured on the aircraft:

  * DJI velocity carries K_VEL ~ 0.91 and is quantised at 0.1 m/s, so it has a
    dead zone that makes it unreliable below ~0.3 m/s (filter_design.md 5.3).
  * DJI altitude is quantised at 0.1 m and carries a per-flight bias `b` that
    RANDOM-WALKS at q_b -- barometric drift, 0.2-0.4 m over 175 s (4.2).
  * DJI roll/pitch UNDER-REPORTS tilt by ~atan(a_h/g), because a quadrotor
    accelerates by tilting and the specific force stays along the thrust axis
    (5.1). This is generated as a real error, not as inflated noise, which is
    the point: the filter must survive a biased measurement, not a noisy one.
  * VO is unscaled and lives in an arbitrary, non-gravity-aligned frame.

The world frame here IS the nav frame `n`: gravity-aligned ENU with its yaw
datum taken from DJI attitude at t0 (filter_design.md 9). The room-to-magnetic
offset is therefore an evaluation concern, not a filter one, and does not
appear.
"""

from dataclasses import dataclass, field

import numpy as np

from . import so3
from .filter import Increment, ned_to_enu, G


@dataclass
class SimConfig:
    dt: float = 0.10                 # s, telemetry and VO period [M] 9.93 Hz
    s_true: float = 1.37             # VO scale to be recovered
    b_true: float = 0.06             # m, altitude bias at t=0
    q_b: float = 6e-4                # m^2/s, barometric random walk [M]
    twist_deg: float = 6.0           # non-gravity-alignment of the VO frame
    K_VEL: float = 0.91              # [M]
    sigma_alt: float = 0.02          # m, pre-quantisation sensor noise
    sigma_vel: float = 0.02          # m/s, pre-quantisation
    sigma_rp0: float = 0.012         # rad [M] quiet floor
    sigma_yaw: float = 0.011         # rad [M]
    quantise: bool = True
    vo_noise: float = 2e-4           # m per increment, in VO units
    seed: int = 0
    r_l: np.ndarray = field(
        default_factory=lambda: np.array([0.117, 0.0, -0.030]))


@dataclass
class Step:
    t: float
    inc: Increment
    R_dji: np.ndarray                # measured body attitude, nav frame
    z_alt: float                     # measured altitude
    v_enu: np.ndarray                # measured velocity, ENU, NOT K_VEL-corrected
    a_h: float                       # |horizontal accel| for R_att scheduling
    vz: float                        # for R_alt scheduling
    # ---- truth, for assertions only ----
    p_true: np.ndarray
    b_true: float
    R_n_b: np.ndarray
    R_v_c: np.ndarray
    R_b_c: np.ndarray
    p_v_c: np.ndarray = None         # absolute VO position, for the frontend
    v_ned: np.ndarray = None         # raw DJI velocity, before ned_to_enu
    rpy_dji: tuple = None            # raw DJI attitude, as published


def _quant(x, q):
    return np.round(np.asarray(x, float) / q) * q


def trajectory(profile, t):
    """Return (p, yaw) for the chosen profile. p is base_link in `n`."""
    if profile == "hover":
        return np.zeros((len(t), 3)) + np.array([0.0, 0.0, 1.5]), np.zeros(len(t))

    if profile == "vertical":
        # 0.5 -> 3.0 m, three cycles at increasing rate (F2)
        z = 1.75 + 1.25 * np.sin(2 * np.pi * t / 40.0)
        p = np.column_stack([np.zeros_like(t), np.zeros_like(t), z])
        return p, np.zeros(len(t))

    if profile == "box":
        # 3 m square at ~0.8 m/s with sharp corners (F3 take 2)
        per = 20.0
        u = (t % per) / per
        x = np.interp(u, [0, .25, .5, .75, 1], [0, 3, 3, 0, 0])
        y = np.interp(u, [0, .25, .5, .75, 1], [0, 0, 3, 3, 0])
        p = np.column_stack([x, y, np.full_like(t, 1.5)])
        return p, np.zeros(len(t))

    if profile == "inspection":
        # slow sideways sweep at 0.3 m/s with yaw framing pauses (F6)
        y = 0.3 * t
        yaw = np.radians(20.0) * np.sin(2 * np.pi * t / 30.0)
        p = np.column_stack([np.full_like(t, 1.5), y, np.full_like(t, 1.5)])
        return p, yaw

    if profile == "mixed":
        # transit then inspect: the regime filter_design.md 7 says `s` is
        # estimated on transits and HELD through passes
        p, yaw = trajectory("box", t)
        slow = t > t[-1] * 0.5
        p[slow] = p[slow][0] + np.column_stack(
            [np.zeros(slow.sum()), 0.3 * (t[slow] - t[slow][0]),
             np.zeros(slow.sum())])
        return p, yaw

    raise ValueError(profile)


def simulate(profile, duration=60.0, cfg=None):
    """Generate a list of Step, one per telemetry/VO tick."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(0.0, duration, cfg.dt)
    n = len(t)

    p, yaw = trajectory(profile, t)

    # velocity and acceleration by central differences on the dense grid
    v = np.gradient(p, cfg.dt, axis=0)
    a = np.gradient(v, cfg.dt, axis=0)
    a_h = np.linalg.norm(a[:, :2], axis=1)

    # body attitude: a quadrotor tilts INTO its acceleration
    R_n_b = []
    for k in range(n):
        tilt = np.array([a[k, 1], -a[k, 0], 0.0]) / G      # small-angle tilt
        R_n_b.append(so3.exp(tilt) @ so3.exp(np.array([0.0, 0.0, yaw[k]])))
    R_n_b = np.array(R_n_b)

    # gimbal holds the camera near-level in `n`; R_b_c is what remains
    R_b_c = np.array([R_n_b[k].T for k in range(n)])

    # the VO frame: deliberately NOT gravity-aligned
    tw = cfg.twist_deg
    axis = np.array([0.6, -0.8, 0.3])
    axis = axis / np.linalg.norm(axis)
    R_n_v = so3.exp(axis * np.radians(tw))

    # camera position in `n`, and the VO-frame camera pose (unscaled)
    p_c = np.array([p[k] + R_n_b[k] @ cfg.r_l for k in range(n)])
    R_v_c = np.array([R_n_v.T @ (R_n_b[k] @ R_b_c[k]) for k in range(n)])
    p_v_c = np.array([(R_n_v.T @ (p_c[k] - p_c[0])) / cfg.s_true
                      for k in range(n)])

    # altitude bias random walk [M] barometric drift
    b = np.empty(n)
    b[0] = cfg.b_true
    for k in range(1, n):
        b[k] = b[k - 1] + rng.normal(scale=np.sqrt(cfg.q_b * cfg.dt))

    steps = []
    for k in range(1, n):
        # ---- VO increment (what the frontend will build) ----
        dp_v = (p_v_c[k] - p_v_c[k - 1]) + rng.normal(scale=cfg.vo_noise, size=3)
        dl = (R_n_b[k] - R_n_b[k - 1]) @ cfg.r_l
        inc = Increment(dp_v=dp_v, dl=dl, dt=cfg.dt,
                        Sigma_v=np.eye(3) * cfg.vo_noise ** 2, t=t[k])

        # ---- attitude: the measured under-reporting mechanism ----
        # DJI sees the specific force, which stays along the thrust axis, so
        # its tilt estimate is short by ~|a_h|/g about the perpendicular axis.
        err = np.array([0.0, 0.0, 1.0])
        e_vec = np.cross(err, np.array([a[k, 0], a[k, 1], 0.0])) / G
        noise = np.array([rng.normal(scale=cfg.sigma_rp0),
                          rng.normal(scale=cfg.sigma_rp0),
                          rng.normal(scale=cfg.sigma_yaw)])
        R_dji = so3.exp(e_vec) @ R_n_b[k] @ so3.exp(noise)

        # ---- altitude ----
        z_alt = p[k, 2] + b[k] + rng.normal(scale=cfg.sigma_alt)
        if cfg.quantise:
            z_alt = float(_quant(z_alt, 0.1))

        # ---- velocity: K_VEL, then NED, then quantise ----
        v_ned = ned_to_enu(v[k]) * cfg.K_VEL          # involution: enu<->ned
        v_ned = v_ned + rng.normal(scale=cfg.sigma_vel, size=3)
        if cfg.quantise:
            v_ned = _quant(v_ned, 0.1)
        v_enu = ned_to_enu(v_ned)

        steps.append(Step(t=t[k], inc=inc, R_dji=R_dji, z_alt=z_alt,
                          v_enu=v_enu, a_h=float(a_h[k]), vz=float(v[k, 2]),
                          p_true=p[k].copy(), b_true=float(b[k]),
                          R_n_b=R_n_b[k].copy(), R_v_c=R_v_c[k].copy(),
                          R_b_c=R_b_c[k].copy(),
                          p_v_c=p_v_c[k].copy(), v_ned=v_ned.copy(),
                          rpy_dji=so3.R_to_rpy(R_dji)))

    return steps, {"s_true": cfg.s_true, "R_n_v": R_n_v, "p": p, "b": b, "t": t}


def init_core(core, step0, s_guess=1.0, p0=None):
    """Initialise per filter_design.md 9.

    R_bar_n_v = R_n_b^dji(t0) @ R_b_c(t0) @ R_v_c(t0)^-1 -- this is what
    gravity-aligns the nav frame. Note it inherits the attitude error at t0,
    which is exactly what dtheta then has to clean up.

    p0: `n` has its origin at takeoff, but these scenarios start already
    airborne, so the filter is handed the true starting position. Getting this
    wrong makes the very first altitude residual ~1.5 m, which trips the NIS
    gate and then NEVER recovers -- p_z is only observed by altitude, so a
    rejected altitude stream is self-sustaining. Worth remembering for the
    real init gate (9): if p_z and b start inconsistent with the first
    measurement, the filter locks itself out.
    """
    core.x.R_nv = so3.normalise(step0.R_dji @ step0.R_b_c @ step0.R_v_c.T)
    core.x.p = np.zeros(3) if p0 is None else np.asarray(p0, float).copy()
    core.x.s = s_guess
    # Same enforcement point as ekf_node._try_init: if `s` is held, it must be
    # held in the covariance, not merely in the velocity Jacobian.
    if core.p.estimate_scale <= 0.5:
        core.hold_scale()
    return core


def run(core, steps, use_velocity=True, use_altitude=True, use_attitude=True):
    """Drive the core in stamp order. Returns per-step diagnostics."""
    log = []
    for st in steps:
        core.propagate(st.inc)
        if use_attitude:
            core.update_attitude(st.R_dji, st.R_v_c, st.R_b_c, a_h=st.a_h,
                                 t=st.t)
        if use_altitude:
            core.update_altitude(st.z_alt, vz=st.vz, t=st.t)
        if use_velocity:
            core.update_velocity(st.v_enu, st.inc, t=st.t)
        log.append({"t": st.t, "s": core.x.s, "b": core.x.b,
                    "sigma_s": core.x.sigma_s, "sigma_b": core.x.sigma_b,
                    "cov_s_b": core.x.cov_s_b,
                    "p": core.x.p.copy(), "p_true": st.p_true.copy(),
                    "b_true": st.b_true})
    return log


def run_frontend(sched, steps, gimbal_lead=0.02):
    """Drive a Scheduler from raw events, in ARRIVAL order.

    Telemetry and gimbal arrive before the VO tick they belong to, which is the
    real ordering: camera frames are stamped ~200 ms in the past while
    telemetry is near-live. The Scheduler must sort this out internally --
    that is the whole point of VO-clocking (filter_design.md 10).
    """
    from .frontend import Pose, VoStatus

    log = []
    for st in steps:
        # gimbal runs faster than the camera; give the buffer a bracketing
        # sample on each side so `at()` never has to extrapolate
        sched.on_gimbal(st.t - gimbal_lead, st.R_b_c)
        sched.on_gimbal(st.t + gimbal_lead, st.R_b_c)
        sched.on_attitude(st.t, np.asarray(st.rpy_dji, float))
        sched.on_altitude(st.t, st.z_alt)
        sched.on_velocity(st.t, st.v_ned)
        sched.on_vo(st.t, Pose(R=st.R_v_c.copy(), p=st.p_v_c.copy()),
                    VoStatus())
        if sched._last_inc is not None and st.t < 1.0:
            print(f"t={st.t:.2f} inc_t={sched._last_inc.t:.2f} "
                  f"|dp_v| got={np.linalg.norm(sched._last_inc.dp_v):.5f} "
                  f"ref={np.linalg.norm(st.inc.dp_v):.5f}")
        log.append({"t": st.t, "s": sched.core.x.s, "b": sched.core.x.b,
                    "sigma_s": sched.core.x.sigma_s,
                    "p": sched.core.x.p.copy(), "p_true": st.p_true.copy(),
                    "b_true": st.b_true})
    print(f"\n  discont_by_cause = {dict(sched.builder.n_flags)}")
    print(f"  n_discont = {sched.builder.n_discont}  "
          f"applied_vel = {sched.core.n_vel_reason}")
    return log
