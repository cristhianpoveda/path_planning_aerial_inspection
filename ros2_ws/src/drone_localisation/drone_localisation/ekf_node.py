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

VO_DISCONT_HOLD = 3.0


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
        self._n_discont_last = 0
        self._t_last_discont = None

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
        pitch_deg = msg.pitch
        if pitch_deg > 3276.8: # pitch is a signed integer but it is overflown by the app.
            pitch_deg -= 6553.6
        pitch = -math.radians(pitch_deg)
        yaw = -math.radians(msg.yaw)
        self.sched.on_gimbal(t, so3.rpy_to_R(roll, pitch, yaw) @ R_LINK_OPTICAL)

    def on_attitude(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self._t_last_telemetry = t
        self.sched.on_attitude(t, np.radians([msg.roll, msg.pitch, -msg.yaw]))

    def on_altitude(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        self._t_last_telemetry = t
        z = float(msg.altitude)
        self._last_alt = z
        self.sched.on_altitude(t, z)
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
        """
        self.sched.on_vo(t, pose, status)

        if not status.pose_valid:
            self._init_wait(f"VO not valid ({status.tracking_state})")
            return
        if self._airborne_since is None:
            self._init_wait("not airborne")
            return
        
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
        if self._last_alt is not None:
            self.core.x.p[2] = float(self._last_alt) - self.p.b_prior_mean
        if self.p.estimate_scale > 0.5:
            sd = (float(np.std(self._init_samples))
                  / np.sqrt(len(self._init_samples)))
            self.core.x.P[3, 3] = float(max(sd ** 2, (0.10 * s0) ** 2))
        else:
            self.core.hold_scale()
        self.core.x.R_nv = so3.normalise(
            self.sched._last_R_dji @ R_bc @ pose.R.T)
        
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
        self._n_discont_last = 0          # keep in step with the line above
        self._t_last_discont = None
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
            self._n_no_gimbal += 1
            return
        self._n_pub += 1
        R_n_b = self.core.body_attitude(pose.R, R_bc)
        q = so3.R_to_quat(R_n_b)
        x = self.core.x

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rclpy.time.Time(seconds=t).to_msg()
        msg.header.frame_id = self.odom_frame
        msg.pose.pose.position.x = float(x.p[0])
        msg.pose.pose.position.y = float(x.p[1])
        msg.pose.pose.position.z = float(x.p[2])
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

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
        st.cov_scale_bias = float(x.cov_s_b)

        flags = []
        if x.sigma_s > self.p.SIGMA_S_MAX * x.s:
            flags.append("SIGMA_S_MAX")
        if x.s <= 1.01e-3:
            flags.append("S_CLAMPED")
        if self._rescale_pending:
            flags.append("S_UNRESCALED")
        n_disc = self.sched.builder.n_discont
        if n_disc != self._n_discont_last:
            self._n_discont_last = n_disc
            self._t_last_discont = stamp_to_sec(header.stamp)
        if self._t_last_discont is not None:
            age = stamp_to_sec(header.stamp) - self._t_last_discont
            if age < 0.0:
                # stamps went backwards (bag restart under sim time)
                self._t_last_discont = None
            elif age < VO_DISCONT_HOLD:
                flags.append(f"vo_discont={n_disc}")
        if self._t_last_telemetry is not None:
            gap = stamp_to_sec(header.stamp) - self._t_last_telemetry
            if gap > self.p.TRANSPORT_GAP:
                flags.append("TRANSPORT_GAP")
        if not self.sched.ready:
            flags.append("VO_GAP")

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
