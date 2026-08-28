"""
params.py -- the boundary between ROS and the ROS-free filter core.

`filter.py` imports nothing but numpy and so3, which is what makes the
finite-difference tests possible and what makes bag replay deterministic. This
module carries values across that boundary and contains no numbers of its own:
defaults live in ekf_node.py's declare_parameter calls, per conventions.md 3.

Every default quoted in the docstrings below is [M] measured -- see
filter_design.md 11. They are repeated here only as documentation.
"""

from dataclasses import dataclass, field, fields

import numpy as np


@dataclass(frozen=True)
class EkfParams:
    # ---- visual odometry
    # [M] VO stamps are decode-time with NO latency subtracted --
    # camera_decoder_node sets msg.header.stamp = now() and the
    # "minus VIDEO_LATENCY" in data_flow.md was never implemented. Measured
    # against DJI attitude by yaw-rate cross-correlation, and independently
    # against mocap: 0.345 (F9_02), 0.418 (F3_02), 0.430 (F6c), each constant
    # to ~20 ms within its own flight. Session-dependent, so this nominal
    # leaves ~+-45 ms rather than the 400 ms it replaces.
    VO_DELAY: float = 0.40                  # s, applied in the frontend
    # ---- geometry (filter_design.md 1) --------------------------------
    r_l: np.ndarray = field(                # base_link -> camera_optical_frame
        default_factory=lambda: np.array([0.117, 0.0, -0.030]))
    r_imu: np.ndarray = field(              # base_link -> DJI IMU
        default_factory=lambda: np.array([0.07, 0.0, 0.0]))

    # ---- velocity (5.3) ----------------------------------------------
    # [M] Measured against mocap by integrated path ratio on F9_02 and F3_02,
    # windows >= 0.4 m/s: 0.875 and 0.856, +1-1.5% for the method's residual
    # low bias. Independently, ground truth requires init `s` = 3.78 on F9_02,
    # and s is proportional to 1/K_VEL, giving 0.873. Two independent routes,
    # 0.5% apart. The old 0.91 came from regression against differentiated
    # mocap, which filter_design.md 10 records as attenuating fitted gains.
    K_VEL: float = 0.87                     # z = K_VEL * v_true
    VEL_DELAY: float = 0.035                # s, applied in the frontend
    R_speed_h: float = 0.0011               # m^2/s^2
    R_speed_v: float = 0.00054              # m^2/s^2
    V_MIN: float = 0.15                     # m/s, below -> reject the update
    V_LOW: float = 0.40                     # m/s, below -> inflate R_speed
    V_VO_MIN: float = 0.05 # m/s of VO-implied speed below which the scale column is dead
    S_STEP_MAX: float = 0.15                # max fractional change in `s` from one velocity update
    V_INFL: float = 4.0                     # inflation factor below V_LOW
    OMEGA_GATE: float = np.radians(30.0)    # rad/s, yaw-rate R_speed switch
    R_speed_rot: float = 0.006              # m^2/s^2 above OMEGA_GATE

    # ---- altitude (5.2) ----------------------------------------------
    R_alt_hover: float = 0.00087            # m^2 (sigma 0.029 m)
    R_alt_vert: float = 0.0104              # m^2 (sigma 0.102 m)
    VZ_INFL: float = 0.15                   # m/s, switch to R_alt_vert
    b_prior_mean: float = 0.02              # m
    b_prior_sigma: float = 0.10             # m

    # ---- attitude (5.1) ----------------------------------------------
    sigma_rp0: float = 0.012                # rad, quiet floor
    accel_slope: float = 1.0 / 9.81         # s^2/m, NOT tuned -- physics
    sigma_yaw: float = 0.011                # rad
    ACC_SMOOTH_S: float = 0.30              # s, window for |a_h| scheduling
    # NIS gates. NOT the textbook chi2 quantiles: chi2 assumes a Gaussian
    # residual, but altitude and velocity are quantisation-dominated
    # (R_alt_hover 0.00087 is 95% quantisation floor), so their residuals are
    # uniform-dominated with hard edges at one quantum. A 0.1 m altitude bin
    # step is 3.4 sigma -> NIS ~ 11.8, and chi2(1,0.99) = 6.63 would reject
    # exactly the informative samples while keeping the uninformative repeats.
    # Sized instead to pass one full quantum and still catch gross outliers
    # (a 0.5 m altitude jump is NIS ~ 300).
    NIS_GATE_ATT: float = 11.34             # chi2(3, 0.99) -- attitude is Gaussian
    # Velocity's H scales with the state being estimated (H[:,s] = u_w), so S
    # is larger for negative innovations than positive ones. A tight NIS gate
    # therefore culls asymmetrically: on F9_02 accepted innovations averaged
    # -0.035 while rejected averaged +0.079, though the full population was
    # unbiased at +0.032. `s` then walked down 3.61 -> 2.26. Loosened to do
    # outlier rejection only.
    NIS_GATE_VEL: float = 60.0
    NIS_GATE_ALT: float = 16.0              # ~4 sigma, one altitude quantum

    # ---- process noise (4.2) -----------------------------------------
    q_s: float = 0.0                        # s is held per flight
    # `s` is HELD per flight, not estimated. The velocity update's scale
    # Jacobian is H[:,s] = u_w = R_bar_n_v @ w_v, so estimating `s` requires a
    # correct R_bar_n_v. With the attitude channel accepting only 33-59% of
    # updates, estimation drove s to 2.0 against a true 3.73, while init from
    # 60 samples gives 3.61 (3.2% error). Restore estimation once attitude is
    # fixed. Tests set this to 1.0 to exercise the estimating path.
    estimate_scale: float = 0.0
    q_b: float = 6e-4                       # m^2/s, barometric drift [M]
    q_theta_xy: float = 1e-6                # [T] rad^2/s
    q_theta_z: float = 4e-6                 # rad^2/s [M] from DJI yaw drift

    # ---- prediction noise gain (4.3) ---------------------------------
    # [T] VO increment noise, in `v`, per increment. Sizing rule, NOT a free
    # knob: the vertical element competes with q_b for the altitude residual.
    # If s^2 * Sigma_base >> q_b * dt (6e-5), p_z absorbs the barometric drift
    # instead of b, and the reported altitude drifts ~0.1 m over 80 s even
    # though p_z + b stays correct to 12 mm. Keep s^2 * Sigma_base well BELOW
    # q_b * dt. ORB-SLAM3 per-frame translation noise is millimetre-scale, so
    # (1-3 mm)^2 is the right order.
    Sigma_base: np.ndarray = field(
        default_factory=lambda: np.diag([1e-6, 1e-6, 1e-6]))
    n_ref: float = 250.0
    feat_cap: float = 5.0
    kappa: float = 0.0                      # g_slew coefficient [T]

    # ---- increment gating (4.1) --------------------------------------
    NOMINAL_DT: float = 0.0333              # [M] median vo/pose interval, F3_02
    # Absolute increment bounds. NOT multiples of NOMINAL_DT: slam_node drops
    # stale frames to stay on fresh images, so the interval genuinely varies
    # (p5 0.0139, median 0.0333, p95 0.1110, max 2.708 on F3_02). A
    # +-multiplier wide enough to span that would stop catching real gaps.
    DT_MIN: float = 0.010                   # s, numerical floor
    DT_MAX: float = 0.30                   # s, above this the increment
                                            # spans a dropped-frame gap
    GATE_V: float = 0.15                     # m/s [T]
    GATE_R: float = np.radians(15.0)        # rad [T]
    GATE_V_REL: float = 0.30

    # ---- admission (5.0) ---------------------------------------------
    T_HOLD: float = 1.0                     # s, dedupe hold timeout
    N_RESCALE: int = 20
    RESCALE_PATH_M: float = 2.0 # metres of DJI path to integrate before re-measuring s

    # ---- init and health (9, 6) --------------------------------------
    SIGMA_S_OK: float = 0.05
    SIGMA_S_MAX: float = 0.20
    SIGMA_S_OK_REL: float = 0.05
    VO_TIMEOUT: float = 0.5                 # s
    TRANSPORT_GAP: float = 0.5              # s
    P0_pos: float = 1e-4                    # m^2
    P0_scale: float = 1.0                   # wide until estimated
    P0_theta_xy: float = np.radians(5.0) ** 2
    P0_theta_z: float = np.radians(60.0) ** 2   # room<->magnetic spans 118-146 deg
    INIT_ALT_MIN: float = 0.30              # m
    INIT_TRANSLATION_S: float = 12.0        # s of parallax before ready
    INIT_PATH_M: float = 3.0                # m of metric travel during scale collection; replaces the time gate

    # Fields carried as ROS double-array parameters rather than scalars.
    _ARRAY_FIELDS = ("r_l", "r_imu")

    @classmethod
    def declare(cls, node):
        """Declare every field on the node, using this dataclass's defaults.

        conventions.md 3: declare in-node with a default, override from
        config/<node>/params.yaml. The defaults live HERE and nowhere else, so
        the YAML never has to restate a measured constant it might drift from.
        """
        d = cls()
        for f in fields(cls):
            v = getattr(d, f.name)
            if f.name == "Sigma_base":
                node.declare_parameter("Sigma_base_diag",
                                       [float(x) for x in np.diag(v)])
            elif f.name in cls._ARRAY_FIELDS:
                node.declare_parameter(f.name, [float(x) for x in v])
            else:
                node.declare_parameter(f.name, float(v))

    @classmethod
    def from_node(cls, node):
        """Build from an rclpy node that has already called declare().

        This is the ONLY ROS coupling the core's parameters have, and it
        contains no defaults of its own.
        """
        vals = {}
        for f in fields(cls):
            if f.name == "Sigma_base":
                diag = node.get_parameter("Sigma_base_diag").value
                vals[f.name] = np.diag(np.asarray(diag, float))
            elif f.name in cls._ARRAY_FIELDS:
                vals[f.name] = np.asarray(
                    node.get_parameter(f.name).value, float)
            else:
                vals[f.name] = float(node.get_parameter(f.name).value)
        return cls(**vals)
