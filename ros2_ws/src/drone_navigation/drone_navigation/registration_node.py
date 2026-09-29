"""Links the odom frame to the arena frame from two AprilGrid board fixes.

    T_opti_odom = T_opti_board . T_board_cam . T_cam_base . T_base_odom

map -> odom is identity, so the solved transform is published as the static
optitrack_map -> map. Disable comparison_node's manual alignment when this
node runs; both publish the same edge and tf will not arbitrate.

    ros2 topic pub --once /drone_1/registration/trigger \\
        std_msgs/msg/String "{data: 'a'}"
    # traverse at least min_baseline_m, keeping the board in view
    ros2 topic pub --once /drone_1/registration/trigger \\
        std_msgs/msg/String "{data: 'b'}"

Watch /drone_1/registration/status for linked=true, then enable virtual
stick and the controller. 'reset' returns to IDLE.
"""

import json
import os
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, \
    QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster

from drone_interfaces.msg import LocalisationStatus, VoStatus

from drone_navigation.registration import board as bd
from drone_navigation.registration import transforms as tf3
from drone_navigation.registration.calib import load_calib

IDLE, COLLECT_A, WAIT_B, COLLECT_B, LINKED, FAILED = \
    "IDLE", "COLLECT_A", "WAIT_B", "COLLECT_B", "LINKED", "FAILED"

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST, depth=1)
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST, depth=1)


class RegistrationNode(Node):

    def __init__(self):
        super().__init__("registration_node")
        p = self.declare_parameters("", [
            ("image_topic", "camera/image/compressed"),
            ("vo_status_topic", "vo/status"),
            ("localisation_status_topic", "localisation/status"),
            ("trigger_topic", "registration/trigger"),
            ("status_topic", "registration/status"),
            ("camera_calib", ""),
            # T_opti_board, kalibr convention, row-major 4x4. The calib_board
            # rigid body is NOT tracked (one stored pose, sd exactly 0 over
            # 37854 samples); this constant comes from offline_fit_board.py.
            ("board_pose", [
                -0.024300, 0.000000, -0.999705, 5.998163,
                -0.999705, 0.000000, 0.024300, -1.407646,
                0.000000, 1.000000, 0.000000, 0.552746,
                0.000000, 0.000000, 0.000000, 1.000000]),
            ("arena_frame", "optitrack_map"),
            ("map_frame", "map"),
            ("odom_frame", "odom"),
            ("base_frame", "base_link"),
            ("camera_frame", "camera_optical_frame"),
            # per-frame gates
            ("min_tags", 15),
            ("max_reproj_px", 3.0),
            ("min_standoff_m", 1.5),
            ("max_standoff_m", 3.0),
            # per-fix gates. Flight hovers held 6.7-20.3 mm and 0.08-0.28 deg,
            # so these pass a good fix and fail a marginal one.
            ("collect_seconds", 5.0),
            ("min_frames", 30),
            ("max_pos_sd_mm", 30.0),
            ("max_rot_sd_deg", 0.5),
            # link gates. Epoch 6 baselines of 0.14-0.48 m scattered the scale
            # reading from 1.56 to 2.43; 1.5 m is the shortest that is stable.
            ("min_baseline_m", 1.5),
            ("scale_min", 0.90),
            ("scale_max", 1.10),
            ("max_fix_disagree_m", 0.10),
            ("max_fix_disagree_deg", 2.0),
            # planner_design.md 3.1 step 3: accept the run only if
            # sigma_s/s < 0.10 at engagement. The node enforces it here so a
            # run that cannot be planned never reports linked.
            ("max_sigma_s_ratio", 0.10),
            ("require_epoch_stable", True),
            ("forbidden_flags", ["S_UNRESCALED", "S_CLAMPED", "VO_GAP"]),
            ("publish_fix", "b"),
            ("output_path", "registration.yaml"),
        ])
        self.p = {q.name: q.value for q in p}

        calib = self.p["camera_calib"]
        if not calib or not os.path.exists(calib):
            raise RuntimeError(
                f"camera_calib is {calib!r}, which does not exist. Point it "
                "at the calibration yaml, e.g.\n  ros2 launch "
                "drone_navigation navigation.launch.py "
                "camera_calib:=/path/to/camera_calibration.yaml\n"
                "The node refuses to guess: a wrong focal length is "
                "invisible in every diagnostic except the standoff.")
        self.K, self.dist = load_calib(calib)
        self.get_logger().info(
            f"intrinsics fx={self.K[0, 0]:.2f} cx={self.K[0, 2]:.2f} "
            f"k1={self.dist[0]:.4f}")

        self.T_opti_board = tf3.orthonormalise(
            np.array(self.p["board_pose"], float).reshape(4, 4))
        self.detector = bd.make_detector()

        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.static_bc = StaticTransformBroadcaster(self)

        self.state = IDLE
        self.reason = ""
        self.fix = {}
        self.samples = []
        self.collect_until = 0.0
        self.epoch = None
        self.pose_valid = False
        self.tracking_state = ""
        self.flags = []
        self.n_stamp_fallback = 0
        self.last_frame_reject = ""
        self.scale_s = float("nan")
        self.sigma_s = float("nan")

        self.create_subscription(CompressedImage, self.p["image_topic"],
                                 self._on_image, SENSOR_QOS)
        self.create_subscription(String, self.p["trigger_topic"],
                                 self._on_trigger, 10)
        self.pub = self.create_publisher(String, self.p["status_topic"],
                                         LATCHED_QOS)
        self.create_subscription(VoStatus, self.p["vo_status_topic"],
                                 self._on_vo, SENSOR_QOS)
        self.create_subscription(LocalisationStatus,
                                 self.p["localisation_status_topic"],
                                 self._on_loc, SENSOR_QOS)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info("registration_node ready; trigger with 'a'")

    # ------------------------------------------------------------ callbacks
    def _on_vo(self, msg):
        self.epoch = int(msg.vo_epoch)
        self.pose_valid = bool(msg.pose_valid)
        self.tracking_state = msg.tracking_state

    def _on_loc(self, msg):
        self.flags = list(getattr(msg, "flags", []))
        self.scale_s = float(getattr(msg, "scale", float("nan")))
        self.sigma_s = float(getattr(msg, "sigma_scale", float("nan")))

    def _on_trigger(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == "reset":
            self.state, self.fix, self.samples, self.reason = IDLE, {}, [], ""
            self.get_logger().info("reset")
        elif cmd == "a" and self.state in (IDLE, FAILED, LINKED):
            self._begin(COLLECT_A)
        elif cmd == "b" and self.state == WAIT_B:
            self._begin(COLLECT_B)
        else:
            self.get_logger().warn(f"ignoring '{cmd}' in state {self.state}")

    def _begin(self, state):
        if self.epoch is None:
            self.state, self.reason = FAILED, "NO_VO_STATUS: nothing on " \
                f"{self.p['vo_status_topic']}"
            self.get_logger().error(self.reason)
            return
        if not self.pose_valid:
            self.state, self.reason = FAILED, \
                f"VO_NOT_VALID: tracking_state={self.tracking_state}"
            self.get_logger().error(self.reason)
            return
        bad = [f for f in self.flags if f in self.p["forbidden_flags"]]
        if bad:
            self.state, self.reason = FAILED, f"FLAGS: {','.join(bad)}"
            self.get_logger().error(self.reason)
            return
        self.state = state
        self.samples = []
        self.n_stamp_fallback = 0
        self.epoch_at_start = self.epoch
        self.collect_until = time.time() + float(self.p["collect_seconds"])
        self.get_logger().info(
            f"{state}: collecting {self.p['collect_seconds']:.1f} s, "
            f"epoch={self.epoch}")

    # --------------------------------------------------------------- vision
    def _on_image(self, msg):
        if self.state not in (COLLECT_A, COLLECT_B):
            return
        if time.time() > self.collect_until:
            self._finish_fix()
            return
        if not self.pose_valid:

            self.last_frame_reject = f"vo {self.tracking_state}"
            return

        gray = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        out = bd.detect_board(gray, self.detector, self.K, self.dist,
                              int(self.p["min_tags"]))
        if out is None:
            self.last_frame_reject = "no board"
            return
        T_cam_board, n_tags, rms = out
        if rms > self.p["max_reproj_px"]:
            self.last_frame_reject = f"reproj {rms:.1f} px"
            return
        standoff = float(np.linalg.norm(T_cam_board[:3, 3]))
        if not (self.p["min_standoff_m"] <= standoff
                <= self.p["max_standoff_m"]):
            self.last_frame_reject = f"standoff {standoff:.2f} m"
            return

        chain = self._lookup(msg.header.stamp)
        if chain is None:
            self.last_frame_reject = "no tf"
            return
        T_base_cam, T_odom_base = chain

        T_cam_board = tf3.orthonormalise(T_cam_board)
        T_opti_odom = (self.T_opti_board @ tf3.inv(T_cam_board)
                       @ tf3.inv(T_base_cam) @ tf3.inv(T_odom_base))
        p_board = (self.T_opti_board @ tf3.inv(T_cam_board)
                   @ tf3.inv(T_base_cam))[:3, 3]
        self.samples.append(dict(T=T_opti_odom, p_board=p_board,
                                 p_odom=T_odom_base[:3, 3].copy(),
                                 standoff=standoff, n_tags=n_tags, rms=rms,
                                 epoch=self.epoch))
        self.last_frame_reject = ""

    def _lookup(self, stamp):
        """tf at the frame stamp"""
        from rclpy.time import Time
        for t, fallback in ((Time.from_msg(stamp), False), (Time(), True)):
            try:
                a = self.buf.lookup_transform(
                    self.p["base_frame"], self.p["camera_frame"], t)
                b = self.buf.lookup_transform(
                    self.p["odom_frame"], self.p["base_frame"], t)
            except Exception:
                continue
            if fallback:
                self.n_stamp_fallback += 1
            return self._msg_to_T(a), self._msg_to_T(b)
        return None

    @staticmethod
    def _msg_to_T(m):
        tr, ro = m.transform.translation, m.transform.rotation
        return tf3.quat_pos_to_T((ro.x, ro.y, ro.z, ro.w), (tr.x, tr.y, tr.z))

    # ----------------------------------------------------------- evaluation
    def _finish_fix(self):
        name = "A" if self.state == COLLECT_A else "B"
        s = self.samples
        if len(s) < int(self.p["min_frames"]):
            return self._fail(f"FIX_{name}_FRAMES: {len(s)} < "
                              f"{self.p['min_frames']}")
        Ts = [x["T"] for x in s]
        T0 = tf3.mean_T(Ts)
        sp = tf3.spread(Ts, T0)
        if sp["pos_sd_mm"] > self.p["max_pos_sd_mm"]:
            return self._fail(f"FIX_{name}_SPREAD: {sp['pos_sd_mm']:.1f} mm")
        if sp["rot_sd_deg"] > self.p["max_rot_sd_deg"]:
            return self._fail(f"FIX_{name}_SPREAD: {sp['rot_sd_deg']:.2f} deg")
        epochs = {x["epoch"] for x in s}
        if self.p["require_epoch_stable"] and len(epochs) > 1:
            return self._fail(f"FIX_{name}_EPOCH: changed during collection")

        self.fix[name] = dict(
            T=T0, spread=sp, epoch=s[0]["epoch"],
            p_board=np.median([x["p_board"] for x in s], axis=0),
            p_odom=np.median([x["p_odom"] for x in s], axis=0),
            standoff=float(np.median([x["standoff"] for x in s])),
            n_tags=float(np.median([x["n_tags"] for x in s])),
            rms=float(np.median([x["rms"] for x in s])), n=len(s))
        self.get_logger().info(
            f"fix {name}: {len(s)} frames, spread {sp['pos_sd_mm']:.1f} mm / "
            f"{sp['rot_sd_deg']:.2f} deg, standoff {self.fix[name]['standoff']:.2f} m")

        if name == "A":
            self.state, self.reason = WAIT_B, "traverse, then trigger 'b'"
            self.get_logger().info(
                f"fix A accepted. Traverse >= {self.p['min_baseline_m']} m "
                "with the board in view, then trigger 'b'.")
        else:
            self._link()

    def _link(self):
        A, B = self.fix["A"], self.fix["B"]
        if self.p["require_epoch_stable"] and A["epoch"] != B["epoch"]:
            return self._fail(f"EPOCH_CHANGED: {A['epoch']} -> {B['epoch']}; "
                              "a rebuild invalidates the VO-to-nav rotation")
        base = float(np.linalg.norm(B["p_board"] - A["p_board"]))
        if base < self.p["min_baseline_m"]:
            return self._fail(f"BASELINE: {base:.2f} m < "
                              f"{self.p['min_baseline_m']} m")
        d_odom = float(np.linalg.norm(B["p_odom"] - A["p_odom"]))
        if d_odom < 1e-6:
            return self._fail("BASELINE: odom did not move")
        scale = base / d_odom
        dm, dd = tf3.disagreement(A["T"], B["T"])
        self.scale, self.baseline, self.dis = scale, base, (dm, dd)
        if not (self.p["scale_min"] <= scale <= self.p["scale_max"]):
            return self._fail(
                f"SCALE_FAIL: {scale:.3f} (board {base:.3f} m vs odom "
                f"{d_odom:.3f} m); s is wrong, registration is unusable")
        if dm > self.p["max_fix_disagree_m"] or dd > self.p["max_fix_disagree_deg"]:
            return self._fail(f"FIX_DISAGREE: {dm*1e3:.0f} mm / {dd:.2f} deg")
        ratio = (self.sigma_s / self.scale_s
                 if self.scale_s == self.scale_s and self.scale_s > 0
                 else float("nan"))
        if ratio == ratio and ratio >= self.p["max_sigma_s_ratio"]:
            return self._fail(
                f"GATE_SIGMA_S: {ratio:.3f} >= {self.p['max_sigma_s_ratio']}; "
                "the run cannot be planned (planner_design.md 3.1 step 3)")
        self.sigma_s_ratio = ratio

        fix = self.fix[self.p["publish_fix"].upper()]
        self._broadcast(fix["T"])
        self._write(fix, scale, base, dm, dd)
        self.state, self.reason = LINKED, ""
        self.get_logger().info(
            f"LINKED. scale {scale:.3f}, baseline {base:.2f} m, "
            f"fixes agree to {dm*1e3:.0f} mm / {dd:.2f} deg. "
            "Safe to enable virtual stick and the controller.")
        if abs(scale - 1.0) > 0.01:
            self.get_logger().warn(
                f"odom is mis-scaled by {scale:.3f} and the FILTER IS NOT "
                f"CORRECTED -- this is deliberate. s={self.scale_s:.3f} "
                f"should be read as {self.scale_s * scale:.3f}, and "
                f"sigma_s/s from localisation/status reads "
                f"{self.sigma_s_ratio:.3f} where the true ratio is "
                f"{self.sigma_s_ratio / scale:.3f}. Everything downstream of "
                f"engagement must use vo_scale_s_corrected from "
                f"{self.p['output_path']}, and anything gating on "
                f"localisation/status sigma_s is optimistic by this factor.")

    def _fail(self, reason):
        self.state, self.reason = FAILED, reason
        self.get_logger().error(reason)

    # -------------------------------------------------------------- outputs
    def _broadcast(self, T):
        q, t = tf3.T_to_quat_pos(T)
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.p["arena_frame"]
        m.child_frame_id = self.p["map_frame"]
        m.transform.translation.x, m.transform.translation.y, \
            m.transform.translation.z = map(float, t)
        m.transform.rotation.x, m.transform.rotation.y, \
            m.transform.rotation.z, m.transform.rotation.w = map(float, q)
        self.static_bc.sendTransform(m)

    def _write(self, fix, scale, base, dm, dd):
        """The arena -> odom conversion is a Sim(3), not a rigid transform.
        """
        T = fix["T"]
        q, _ = tf3.T_to_quat_pos(T)
        p_opti_fix = (T[:3, :3] @ fix["p_odom"]) + T[:3, 3]
        t_sim = p_opti_fix - scale * (T[:3, :3] @ fix["p_odom"])
        path = self.p["output_path"]
        with open(path, "w") as f:
            f.write("# written by registration_node\n")
            f.write(f"# {self.p['arena_frame']} -> {self.p['map_frame']} "
                    "(map -> odom is identity)\n")
            f.write("#\n")
            f.write("# THE FILTER IS NOT CORRECTED FOR SCALE. odom stays\n")
            f.write("# mis-scaled by 'scale' below; the follower undoes it in\n")
            f.write("# the arena -> odom conversion. Consequences:\n")
            f.write("#   - use vo_scale_s_corrected, not localisation/status\n")
            f.write("#     scale, for anything downstream of engagement\n")
            f.write("#   - sigma_s/s on localisation/status is optimistic by\n")
            f.write("#     this factor; vo_sigma_s_ratio_corrected is the\n")
            f.write("#     honest number\n")
            f.write("registration:\n")
            f.write("  # Sim(3): p_opti = scale * R * p_odom + translation\n")
            f.write("  translation: [%.6f, %.6f, %.6f]\n" % tuple(t_sim))
            f.write("  rotation_xyzw: [%.8f, %.8f, %.8f, %.8f]\n" % tuple(q))
            f.write(f"  scale: {scale:.6f}\n")
            f.write("  rigid_translation: [%.6f, %.6f, %.6f]"
                    "   # tf broadcast only\n" % tuple(T[:3, 3]))
            f.write(f"  vo_epoch: {self.fix['A']['epoch']}\n")
            f.write("  # filter state at engagement; planner_design.md 3.1\n")
            f.write("  # step 4 re-solves the tour at this s\n")
            f.write(f"  vo_scale_s: {self.scale_s:.6f}\n")
            f.write(f"  vo_sigma_s: {self.sigma_s:.6f}\n")
            f.write(f"  vo_sigma_s_ratio: {self.sigma_s_ratio:.5f}"
                    "   # as reported\n")
            f.write("  vo_sigma_s_ratio_corrected: "
                    f"{self.sigma_s_ratio / scale:.5f}\n")
            f.write(f"  vo_scale_s_corrected: {self.scale_s * scale:.6f}\n")
            f.write(f"  scale_ratio: {scale:.5f}\n")
            f.write(f"  baseline_m: {base:.4f}\n")
            f.write(f"  fix_disagree_m: {dm:.5f}\n")
            f.write(f"  fix_disagree_deg: {dd:.4f}\n")
            f.write(f"  stamp: {time.time():.3f}\n")
            for k in ("A", "B"):
                x = self.fix[k]
                f.write(f"  fix_{k.lower()}:\n"
                        f"    frames: {x['n']}\n"
                        f"    pos_sd_mm: {x['spread']['pos_sd_mm']:.2f}\n"
                        f"    rot_sd_deg: {x['spread']['rot_sd_deg']:.3f}\n"
                        f"    standoff_m: {x['standoff']:.3f}\n"
                        f"    reproj_px: {x['rms']:.2f}\n")
        self.get_logger().info(f"wrote {os.path.abspath(path)}")

    def _publish_status(self):
        d = dict(state=self.state, linked=self.state == LINKED,
                 reason=self.reason, epoch=self.epoch,
                 vo_pose_valid=self.pose_valid,
                 tracking_state=self.tracking_state, flags=self.flags,
                 collecting=len(self.samples) if self.state in
                 (COLLECT_A, COLLECT_B) else 0,
                 last_reject=self.last_frame_reject,
                 stamp_fallbacks=self.n_stamp_fallback)
        for k in ("A", "B"):
            if k in self.fix:
                x = self.fix[k]
                d[f"fix_{k.lower()}"] = dict(
                    frames=x["n"], pos_sd_mm=round(x["spread"]["pos_sd_mm"], 1),
                    rot_sd_deg=round(x["spread"]["rot_sd_deg"], 3),
                    standoff_m=round(x["standoff"], 3), epoch=x["epoch"])
        if self.state in (LINKED, FAILED) and hasattr(self, "scale"):
            d["scale_ratio"] = round(self.scale, 4)
            d["vo_scale_s"] = round(self.scale_s, 4)
            d["vo_sigma_s_ratio"] = (round(self.sigma_s_ratio, 4)
                                     if hasattr(self, "sigma_s_ratio") else None)
            d["baseline_m"] = round(self.baseline, 3)
            d["fix_disagree_mm"] = round(self.dis[0] * 1e3, 1)
            d["fix_disagree_deg"] = round(self.dis[1], 3)
        self.pub.publish(String(data=json.dumps(d)))


def main(args=None):
    rclpy.init(args=args)
    node = RegistrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
