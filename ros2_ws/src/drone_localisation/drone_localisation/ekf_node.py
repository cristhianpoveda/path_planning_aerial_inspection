#!/usr/bin/env python3
"""
ekf_node.py -- the ROS wrapper around the localisation EKF.

Owns subscriptions, parameters, publishing and tf. Owns NO estimation: the
core (filter.py) and the frontend (frontend.py) import nothing from rclpy,
which is what makes the finite-difference tests possible and what makes bag
replay reproducible (filter_design.md 10).

Topics in (namespace drone_1, per conventions.md 1):
    vo/pose                 geometry_msgs/PoseStamped          ~10 Hz
    vo/status               drone_interfaces/VoStatus          ~10 Hz
    attitude                drone_interfaces/AttitudeStamped    10 Hz
    relative_altitude       drone_interfaces/RelativeAltitudeStamped  10 Hz
    speed_vector            geometry_msgs/Vector3Stamped        10 Hz
    gimbal_joint_attitude   drone_interfaces/AttitudeStamped    19 Hz

Topics out:
    localisation/pose       geometry_msgs/PoseWithCovarianceStamped
    localisation/status     drone_interfaces/LocalisationStatus
    /tf                     odom -> base_link ONLY (7: p_x, p_y are unobserved,
                            so this filter cannot claim global drift correction)

QoS is RELIABLE with depth >= 100 everywhere. SensorDataQoS is best-effort and
must not be used here: the scheduler sorts by stamp internally, so processing
order is deterministic given the same SET of messages -- which reduces the
replay requirement to "do not drop messages" (filter_design.md 10).
"""

import csv
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped,
                               Vector3Stamped, TransformStamped)
from tf2_ros import TransformBroadcaster

from drone_interfaces.msg import (AttitudeStamped, RelativeAltitudeStamped,
                                  VoStatus as VoStatusMsg, LocalisationStatus)

from drone_localisation.ekf import so3
from drone_localisation.ekf.params import EkfParams
from drone_localisation.ekf.filter import EkfCore, Kind, ned_to_enu, IDX_P, IDX_TH, State
from drone_localisation.ekf.frontend import Scheduler, Pose, VoStatus, Flags
from dataclasses import fields

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST, depth=200)

R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class EkfNode(Node):

    def __init__(self):
        super().__init__("ekf_node")

        EkfParams.declare(self)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("innovation_log", "")     # "" disables
        self._init_dump = self.declare_parameter("init_dump", "").value
        self.p = EkfParams.from_node(self)
        self.get_logger().info(
            "PARAMS " + " ".join(
                f"{f.name}={getattr(self.p, f.name)}"
                for f in fields(EkfParams)
                if np.isscalar(getattr(self.p, f.name))))

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.core = EkfCore(self.p)
        self.sched = Scheduler(self.p, self.core)
        self.sched.on_processed = self._on_processed

        self._n_vo = 0
        self._n_pub = 0
        self._n_no_gimbal = 0

        # ---- initialisation state (filter_design.md 9) ----
        self.initialised = False
        self._airborne_since = None
        self._init_samples = []          # (|v_dji|, |dp_v|/dt) pairs
        self._init_v = []
        self._init_path = 0.0
        self._psi_samples = []
        self._last_status = None
        self._last_pose_msg = None
        self._t_last_telemetry = None
        self._excited_s = 0.0
        self._t_last_vel = None
        self._last_alt = None
        self._rescale_pending = False
        self._rescale_dji = 0.0
        self._rescale_vo = 0.0
        self._rescale_n = 0

        # ---- publishers ----
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, "localisation/pose", QOS)
        self.pub_status = self.create_publisher(
            LocalisationStatus, "localisation/status", QOS)
        self.tf_bc = TransformBroadcaster(self) \
            if self.get_parameter("publish_tf").value else None

        # ---- subscriptions ----
        self.create_subscription(PoseStamped, "vo/pose", self.on_vo_pose, QOS)
        self.create_subscription(VoStatusMsg, "vo/status", self.on_vo_status, QOS)
        self.create_subscription(AttitudeStamped, "attitude",
                                 self.on_attitude, QOS)
        self.create_subscription(RelativeAltitudeStamped, "relative_altitude",
                                 self.on_altitude, QOS)
        self.create_subscription(Vector3Stamped, "speed_vector",
                                 self.on_velocity, QOS)
        self.create_subscription(AttitudeStamped, "gimbal_joint_attitude",
                                 self.on_gimbal, QOS)

        # ---- innovation log: filter_design.md 10 calls this the tuning loop ----
        self._csv = None
        path = self.get_parameter("innovation_log").value
        if path:
            self._csv_file = open(path, "w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["t", "kind", "nis", "accepted", "note",
                                "y0", "y1", "y2", "s", "sigma_s", "b",
                                "sigma_b", "cov_s_b", "h0", "h1", "h2"])

        self.create_timer(1.0, self.on_diagnostics)
        self.get_logger().info("ekf_node ready; waiting for telemetry")

    def _on_processed(self, t, pose):
        """Called by the scheduler once a VO frame is actually processed --
        which, with deferral, may be from a gimbal callback rather than a VO
        one."""
        if not self.initialised:
            return
        self._log_innovations()
        self._publish(t, pose)

    # ================================================================ ingress
    def on_gimbal(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        roll = math.radians(msg.roll)
        # [M] DJI gimbal pitch arrives as an unsigned 16-bit count in 0.1 deg
        # units, so small negative pitch reads as ~6553.6 - |pitch|. This is
        # the `pitch_needs_unwrap` flag the characterisation script already
        # carried; camera_decoder corrects it, the raw topic does not.
        pitch_deg = msg.pitch
        if pitch_deg > 3276.8:
            pitch_deg -= 6553.6
        # [M] pitch and yaw are negated relative to ROS convention; roll is
        # not. Verified against camera_decoder's gimbal_base -> camera_link
        # transform on F3_02: roll matches to 0.00 deg, pitch and yaw match
        # only after negation.
        pitch = -math.radians(pitch_deg)
        yaw = -math.radians(msg.yaw)
        self.sched.on_gimbal(t, so3.rpy_to_R(roll, pitch, yaw) @ R_LINK_OPTICAL)

    def on_attitude(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self._t_last_telemetry = t
        self.sched.on_attitude(t, np.radians([msg.roll, msg.pitch, msg.yaw]))

    def on_altitude(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self._t_last_telemetry = t
        z = float(msg.altitude)
        self._last_alt = z
        self.sched.on_altitude(t, z)
        # airborne gate: KeyAltitude reads exactly 0.000 when grounded, but
        # ONLY before the first takeoff after power-up [M]; after a prior
        # flight it reads an arbitrary offset. Gate on height, not on zero.
        if z > self.p.INIT_ALT_MIN:
            if self._airborne_since is None:
                self._airborne_since = t
        else:
            self._airborne_since = None

    def on_velocity(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self._t_last_telemetry = t
        v_ned = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.sched.on_velocity(t, v_ned)
        # Accumulated time above V_LOW, not elapsed time since first exceeding
        # it. F6c drifted past 0.4 m/s once, waited 12 s, and initialised from
        # marginal samples -- giving a bad `s` rather than refusing.
        if self._t_last_vel is not None and np.linalg.norm(v_ned) > self.p.V_LOW:
            self._excited_s += min(t - self._t_last_vel, 0.5)
        self._t_last_vel = t

    def on_vo_status(self, msg):
        self._last_status = VoStatus(
            tracking_state=msg.tracking_state,
            vo_epoch=int(msg.vo_epoch),
            n_map_points=int(msg.n_map_points),
            pose_valid=bool(msg.pose_valid))
        self._maybe_step()

    def on_vo_pose(self, msg):
        self._last_pose_msg = msg
        self._maybe_step()

    # ============================================================== VO clock
    def _maybe_step(self):
        """Fire once per frame, when pose and status for the SAME stamp are
        both in hand. slam_node publishes them with identical stamps."""
        if self._last_pose_msg is None or self._last_status is None:
            return
        self._n_vo += 1
        msg = self._last_pose_msg
        t = stamp_to_sec(msg.header.stamp)
        pose = Pose(R=so3.quat_to_R([msg.pose.orientation.x,
                                     msg.pose.orientation.y,
                                     msg.pose.orientation.z,
                                     msg.pose.orientation.w]),
                    p=np.array([msg.pose.position.x, msg.pose.position.y,
                                msg.pose.position.z]))
        status = self._last_status
        self._last_pose_msg = None
        self._last_status = None

        if self.sched.needs_reinit:
            self.sched.needs_reinit = False
            if self.initialised:
                # Do NOT drop out of `initialised`. A rebuild invalidates `s`
                # alone -- p and b live in the nav frame (6.1) -- and dropping
                # out stops publishing entirely. On F9_02 that cost 130 s of
                # the flight, because the cold-start path then waits
                # INIT_TRANSLATION_S for parallax the map no longer needs, and
                # the aircraft was in a slow pass: 0.0 s of excitation
                # accumulated in the 90 s that followed.
                self._rescale_dji = 0.0
                self._rescale_vo = 0.0
                self._rescale_n = 0
                self._rescale_pending = True
                self._init_samples = []
                self._init_path = 0.0
                self._psi_samples = []
                self.get_logger().warn(
                    f"VO map rebuild -- s={self.core.x.s:.3f} untrusted, "
                    f"re-measuring opportunistically; publishing continues")
            else:
                self.get_logger().info(
                    "epoch change before init -- ignored, map still forming")

        if not self.initialised:
            self._try_init(t, pose, status)
            return

        self.sched.on_vo(t, pose, status)
        self._maybe_rescale()

    def _init_wait(self, why):
        """Say WHY init has not completed. Without this, a stalled init is
        indistinguishable from a node that simply stopped publishing -- which
        is what the 74.3 s pose bag looked like."""
        self.get_logger().info(
            f"init waiting: {why}  ["
            f"excited={self._excited_s:.1f}/{self.p.INIT_TRANSLATION_S:.0f}s "
            f"samples={len(self._init_samples)}/60 "
            f"alt={self._last_alt if self._last_alt is not None else float('nan'):.2f} "
            f"airborne={self._airborne_since is not None} "
            f"att_buf={len(self.sched.att)}]",
            throttle_duration_sec=2.0)
    
    def _maybe_rescale(self):
        """Re-measure `s` after a rebuild by INTEGRATED PATH RATIO.

        s = (path_dji / K_VEL) / path_vo, accumulated over RESCALE_PATH_M of
        travel. Integration averages the 0.1 m/s quantisation out, so this
        completes at inspection speed where the instantaneous v > V_LOW gate
        never fires: on F9_02 the aircraft accumulated 0.0 s above V_LOW in
        the 90 s after its rebuild.

        Samples below V_MIN are skipped: there DJI reads exactly zero while VO
        still moves, which would bias `s` low. The residual dead-zone bias
        above V_MIN is real but small against a 3.8x reset error.
        """
        if not self._rescale_pending:
            return
        inc = self.sched._last_inc
        if inc is None:
            return
        v = float(np.linalg.norm(self.sched._last_v_enu))
        if v <= self.p.V_LOW:
            return
        self._rescale_dji += v * inc.dt
        self._rescale_vo += float(np.linalg.norm(inc.dp_v))
        self._rescale_n += 1
        if self._rescale_dji < self.p.RESCALE_PATH_M or self._rescale_vo <= 1e-6:
            return

        s_new = (self._rescale_dji / self.p.K_VEL) / self._rescale_vo
        s_old = float(self.core.x.s)
        self.core.x.s = s_new
        self.core.s_ref = s_new
        # `s` is replaced, not corrected: its old cross-covariances are stale.
        self.core.x.P[3, :] = 0.0
        self.core.x.P[:, 3] = 0.0
        if self.p.estimate_scale > 0.5:
            self.core.x.P[3, 3] = float((0.10 * s_new) ** 2)
        self._rescale_pending = False
        self.get_logger().warn(
            f"rescaled after rebuild: s {s_old:.3f} -> {s_new:.3f} "
            f"from {self._rescale_dji:.2f} m of path, {self._rescale_n} incs")

    # ======================================================= initialisation
    def _try_init(self, t, pose, status):
        """filter_design.md 9.

        Gated on: telemetry flowing, aircraft airborne, VO healthy, and enough
        translation for monocular parallax. Scale is ESTIMATED, not assumed,
        from K_VEL^-1 |v_dji| / (|dp_v|/dt) over an excited window, requiring
        |v_dji| > V_LOW so the quantisation dead zone does not bias it.
        """
        # Keep the frontend warm from the FIRST frame. The queue has to drain
        # so that _last_R_dji, _last_v_enu and _last_inc get populated --
        # otherwise the guards below wait forever on values that only this
        # call can produce. Core updates before init are harmless: the state
        # is overwritten wholesale at the bottom of this method.
        self.sched.on_vo(t, pose, status)

        if not status.pose_valid:
            self._init_wait(f"VO not valid ({status.tracking_state})")
            return
        if self._airborne_since is None:
            self._init_wait("not airborne")
            return
        
        # Collect FIRST, and unconditionally. This used to sit BELOW the
        # excitation gate, so no sample could be taken until 12 s of motion
        # above V_LOW had already accumulated -- the two requirements ran in
        # SERIES when they describe the same thing. [M] F6 reached
        # excited=12.1/12 s and landed with 17/60 samples; F6c and F9 never
        # reached 12 s and collected 0. Four of seven bags published nothing.
        inc = self.sched._last_inc
        v = float(np.linalg.norm(self.sched._last_v_enu))
        
        if inc is not None and v > self.p.V_LOW:
            vo_speed = float(np.linalg.norm(inc.dp_v)) / inc.dt
            if vo_speed > 1e-6:
                self._init_samples.append((v / self.p.K_VEL) / vo_speed)
                self._init_v.append(v)
                self._init_path += v * inc.dt
                self._psi_samples.append(
                    (inc.dp_v / inc.dt,
                     np.asarray(self.sched._last_v_enu, float).copy()))

        # Observability stated as PATH, not TIME. `s` needs metric travel, and
        # 60 samples spanning INIT_PATH_M of it IS that evidence.
        # INIT_TRANSLATION_S was sized for monocular parallax, which
        # ORB-SLAM3 has already had by the time it reports pose_valid -- the
        # gate was redundant with VO's own initialisation and unsatisfiable on
        # inspection profiles.
        if len(self._init_samples) < 60 or self._init_path < self.p.INIT_PATH_M:
            self._init_wait(
                f"collecting (v={v:.2f} m/s, "
                f"samples={len(self._init_samples)}/60, "
                f"path={self._init_path:.1f}/{self.p.INIT_PATH_M:.1f} m)")
            return

        R_bc = self.sched.att.at(t)
        if R_bc is None:
            self._init_wait("gimbal does not bracket the VO stamp")
            return
        if self.sched._last_R_dji is None:
            self._init_wait("no DJI attitude yet")
            return

        if self._init_dump:
            np.savetxt(self._init_dump,
                       np.column_stack([np.array(self._init_v),
                                        np.array(self._init_samples)]),
                       header="v_dji s_sample", comments="")
            self.get_logger().info(
                f"init samples dumped to {self._init_dump} "
                f"({len(self._init_samples)} rows)")
            
        s0 = float(np.median(self._init_samples))
        self.core.x = State(self.p)
        self.core.x.s = s0
        self.core.s_ref = s0
        # `n` has its origin at TAKEOFF, but 9 gates init on 12 s of
        # translation AFTER takeoff, so p = 0 makes p_z inconsistent with the
        # first altitude measurement by the whole height. h(x) = p_z + b then
        # NIS-rejects, and p_z is observed by nothing else, so the rejection is
        # self-sustaining. Measured in sim: 240 of ~800 altitude updates
        # rejected and a permanent 1.5 m z offset. Datum p_z from the
        # measurement; `b` keeps its prior.
        if self._last_alt is not None:
            self.core.x.p[2] = float(self._last_alt) - self.p.b_prior_mean
        # Do NOT keep P0_scale here: init just MEASURED s from 20 samples, and
        # throwing that away leaves sigma_s ~ 1.0 on a value of ~1.5, so the
        # Kalman gain on s is large enough for one innovation to swing it by
        # an order of magnitude -- or negative (observed at s = -0.9998).
        if self.p.estimate_scale > 0.5:
            sd = (float(np.std(self._init_samples))
                  / np.sqrt(len(self._init_samples)))
            self.core.x.P[3, 3] = float(max(sd ** 2, (0.10 * s0) ** 2))
        else:
            self.core.hold_scale()
        self.core.x.R_nv = so3.normalise(
            self.sched._last_R_dji @ R_bc @ pose.R.T)

        # [M] DJI attitude yaw and DJI velocity NED use DIFFERENT yaw datums.
        # Measured 51.84 deg +- 3.81 on F9_02, constant in time (fitted slope
        # +0.013 deg per deg of heading). The filter has ONE nav frame, so no
        # single R_n_v satisfies both updates -- the tighter R wins and the
        # other is permanently inconsistent. Give velocity its own offset.
        ang = []
        for w_v, v_e in self._psi_samples:
            u_w = self.core.x.R_nv @ w_v
            if np.hypot(u_w[0], u_w[1]) > 1e-6 and np.hypot(v_e[0], v_e[1]) > 1e-6:
                ang.append(np.arctan2(v_e[1], v_e[0])
                           - np.arctan2(u_w[1], u_w[0]))
        psi, psi_sd, n_psi = 0.0, float("nan"), len(ang)
        if n_psi >= 20:
            ang = np.array(ang)
            cbar, sbar = np.mean(np.cos(ang)), np.mean(np.sin(ang))
            psi = float(np.arctan2(sbar, cbar))
            R_ = min(float(np.hypot(sbar, cbar)), 1.0)
            psi_sd = float(np.sqrt(-2.0 * np.log(max(R_, 1e-12))))
        self.core.psi_v = psi

        self.initialised = True
        self.sched.innovations.clear()   # pre-init updates ran on a throwaway state
        self.core.n_rejected = {k: 0 for k in Kind}
        self.core.n_vel_reason = {"below_V_MIN": 0, "NIS": 0, "applied": 0}
        self.core.n_s_clamped = 0
        self.sched.builder.n_discont = 0
        self.sched.builder.n_flags = {}
        self.sched.q.n_late = {0: 0, 1: 0, 2: 0}
        self.get_logger().info(
            f"initialised: s = {s0:.4f} +- {self.core.x.sigma_s:.4f} "
            f"from {len(self._init_samples)} samples, "
            f"p_z datum {self.core.x.p[2]:+.3f} m, "
            f"psi_v {np.degrees(psi):+.1f} deg "
            f"(sd {np.degrees(psi_sd):.1f}, n={n_psi})")

    # =================================================================== out
    def _publish(self, t, pose):
        R_bc = self.sched.att.at(t)
        if R_bc is None:
            # Gimbal runs at 13.8 Hz against VO at ~19 Hz, so the VO stamp is
            # not always bracketed and interpolation is unavailable (4.1 step
            # 4 forbids a ZOH fallback). Count it -- silently dropping output
            # looked like a rate problem.
            self._n_no_gimbal += 1
            return
        self._n_pub += 1
        # Orientation is NOT a state: composed at publish time from the
        # filtered frame estimate and the latest VO and gimbal attitudes (8).
        R_n_b = self.core.body_attitude(pose.R, R_bc)
        q = so3.R_to_quat(R_n_b)
        x = self.core.x

        msg = PoseWithCovarianceStamped()
        # Stamp with the VO stamp, never now(): the pose is as old as the
        # frame it came from.
        msg.header.stamp = rclpy.time.Time(seconds=t).to_msg()
        msg.header.frame_id = self.odom_frame
        msg.pose.pose.position.x = float(x.p[0])
        msg.pose.pose.position.y = float(x.p[1])
        msg.pose.pose.position.z = float(x.p[2])
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

        # 6x6: P_pp top-left; P_theta + R_att bottom-right (documented proxy --
        # P_theta is only the FRAME error, VO's relative-attitude error is
        # unmodelled); P_p_theta off-diagonal, which is tracked and real.
        C = np.zeros((6, 6))
        C[:3, :3] = x.P[IDX_P, IDX_P]
        srp = self.p.sigma_rp0
        C[3:, 3:] = x.P[IDX_TH, IDX_TH] + np.diag(
            [srp ** 2, srp ** 2, self.p.sigma_yaw ** 2])
        C[:3, 3:] = x.P[IDX_P, IDX_TH]
        C[3:, :3] = C[:3, 3:].T
        msg.pose.covariance = C.reshape(36).tolist()
        self.pub_pose.publish(msg)

        if self.tf_bc is not None:
            tf = TransformStamped()
            tf.header = msg.header
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = float(x.p[0])
            tf.transform.translation.y = float(x.p[1])
            tf.transform.translation.z = float(x.p[2])
            tf.transform.rotation = msg.pose.pose.orientation
            self.tf_bc.sendTransform(tf)

        self._publish_status(msg.header)

    def _publish_status(self, header):
        x = self.core.x
        st = LocalisationStatus()
        st.header = header
        st.scale = float(x.s)
        st.sigma_scale = float(x.sigma_s)
        st.alt_bias = float(x.b)
        st.sigma_alt_bias = float(x.sigma_b)
        # 7: monitor the s-b off-diagonal, not just the marginals -- altitude
        # reaches s only through it.
        st.cov_scale_bias = float(x.cov_s_b)

        flags = []
        if x.sigma_s > self.p.SIGMA_S_MAX:
            flags.append("SIGMA_S_MAX")
        if x.s <= 1.01e-3:
            # `s` is at the floor, so p+ = p + 1e-3*u and position is frozen
            # by construction. [M] F6 sat here for 23 s of a 43 s flight and
            # |dp_est|/|dp_gt| measured 0.003 over 1 s windows. This is a hard
            # failure, distinct from a merely uncertain scale.
            flags.append("S_CLAMPED")
        if self.sched.builder.n_discont:
            flags.append(f"vo_discont={self.sched.builder.n_discont}")
        if self._t_last_telemetry is not None:
            gap = stamp_to_sec(header.stamp) - self._t_last_telemetry
            if gap > self.p.TRANSPORT_GAP:
                flags.append("TRANSPORT_GAP")
        if not self.sched.ready:
            flags.append("VO_GAP")

        # NOTE: transport gap is NOT reliably detectable today. Dedupe happens
        # at source, so a stationary drone and a dead link look identical on
        # the wire -- a grounded bag published at 1.5 Hz with a healthy link.
        # Fix is a fixed-rate liveness heartbeat in dji_node (6).

        st.degraded = bool(flags)
        st.flags = flags
        st.state = "DEGRADED" if flags else "OK"
        self.pub_status.publish(st)

    def _log_innovations(self):
        if self._csv is None:
            return
        x = self.core.x
        for inn in self.sched.innovations:
            y = np.atleast_1d(np.asarray(inn.y, float))
            y = np.pad(y, (0, 3 - len(y)))
            # h is None for the below-V_MIN early return and for attitude,
            # which has no prediction worth logging; pad to a fixed width so
            # the CSV stays rectangular.
            h = (np.zeros(3) if inn.h is None
                 else np.atleast_1d(np.asarray(inn.h, float)))
            h = np.pad(h, (0, 3 - len(h)))
            self._csv.writerow([
                f"{inn.t:.6f}", inn.kind.value, f"{inn.nis:.6f}",
                int(inn.accepted), inn.note,
                *[f"{v:.6f}" for v in y],
                f"{x.s:.6f}", f"{x.sigma_s:.6f}", f"{x.b:.6f}",
                f"{x.sigma_b:.6f}", f"{x.cov_s_b:.9f}",
                *[f"{v:.6f}" for v in h]])
        self.sched.innovations.clear()

    def on_diagnostics(self):
        s = self.sched.stats()
        nis = self.sched._no_inc_speeds
        aps = self.sched._applied_speeds
        no_inc_v = f"{np.median(np.fromiter(nis, float)):.2f}" if nis else "-"
        applied_v = f"{np.median(np.fromiter(aps, float)):.2f}" if aps else "-"
        self.get_logger().info(
            f"STATE init={self.initialised} s={self.core.x.s:.4f} "
            f"sig_s={self.core.x.sigma_s:.4f} b={self.core.x.b:+.3f} "
            f"pub={self._n_pub} vo={self._n_vo}",
            throttle_duration_sec=5.0)
        self.get_logger().info(
            f"HEALTH rej={ {k.value: v for k, v in self.core.n_rejected.items()} } "
            f"vel={self.core.n_vel_reason} no_inc={self.sched.q.n_late[0]} "
            f"causes={s['discont_by_cause']} "
            f"dedup(a/v)={s['deduped']['altitude']}/{s['deduped']['velocity']} "
            f"s_clamp={self.core.n_s_clamped} s_lim={self.core.n_s_limited}",
            throttle_duration_sec=5.0)

    def destroy_node(self):
        if self._csv is not None:
            self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
