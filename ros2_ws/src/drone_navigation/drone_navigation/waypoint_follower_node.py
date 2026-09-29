"""waypoint_follower_node -- flies tour.yaml through position_heading_controller.

    tour.yaml           (planner output, arena frame)
    registration.yaml   (Sim(3) arena <- odom)
    localisation/pose   (PoseWithCovarianceStamped, odom)   -- feedback
    localisation/status (LocalisationStatus)                -- health gate
    attitude            (AttitudeStamped)                   -- heading feedback
    <epoch_topic>       (<epoch_type>)                      -- vo_epoch watch
        ->
    setpoint            (PoseStamped, odom)
    controller/enable   (Bool)
    follower/state      (String)
    follower/arrived    (String)
    follower/start, follower/abort (std_srvs/Trigger)
"""
import math
from dataclasses import dataclass

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from drone_interfaces.msg import AttitudeStamped, LocalisationStatus

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)

IDLE, RUNNING, DONE, ABORTED = "IDLE", "RUNNING", "DONE", "ABORTED"


def quat_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_matrix(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pick(d, keys, where):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"{where}: none of {list(keys)} found in {sorted(d)}")


def _rotation(r) -> np.ndarray:
    a = np.asarray(r, float)
    if a.shape == (4,):
        R = quat_matrix(*(a / np.linalg.norm(a)))
    elif a.size == 9:
        R = a.reshape(3, 3)
    else:
        raise ValueError(f"rotation must be a quaternion [x,y,z,w] or 3x3, "
                         f"got shape {a.shape}")
    if (not np.allclose(R @ R.T, np.eye(3), atol=1e-3)
            or np.linalg.det(R) < 0.99):
        raise ValueError("registration rotation is not a proper rotation")
    return R


class Sim3:
    """p_arena = scale * R @ p_odom + t."""

    def __init__(self, scale, R, t):
        self.scale = float(scale)
        self.R = np.asarray(R, float)
        self.t = np.asarray(t, float).reshape(3)

    def to_odom(self, p) -> np.ndarray:
        return self.R.T @ (np.asarray(p, float) - self.t) / self.scale

    def yaw_to_odom(self, R_arena_body) -> float:
        h = self.R.T @ R_arena_body[:, 0]
        return math.atan2(h[1], h[0])


def load_registration(path):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    r = d.get("registration", d)
    sim = r.get("sim3", r)
    scale = float(_pick(sim, ("scale", "s"), path))
    if scale <= 0.0:
        raise ValueError(f"{path}: scale {scale} is not positive")
    R = _rotation(_pick(sim, ("rotation", "R", "quaternion"), path))
    t = _pick(sim, ("translation", "t"), path)
    return Sim3(scale, R, t), r.get("vo_epoch")


def load_tour(path):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    pts = d.get("points") or []
    if not pts:
        raise ValueError(f"{path} has no points")
    meta = d.get("meta") or {}
    plan = d.get("plan") or {}
    frame = (d.get("header") or {}).get("frame_id")
    epoch = meta.get("vo_epoch", plan.get("vo_epoch"))
    return frame, pts, epoch, meta, plan


@dataclass
class Leg:
    name: str
    target: object
    pts: np.ndarray
    yaw: float
    speed: float
    length: float


def _clean(pts) -> np.ndarray:
    out = [np.asarray(pts[0], float)]
    for q in pts[1:]:
        q = np.asarray(q, float)
        if np.linalg.norm(q - out[-1]) > 1e-6:
            out.append(q)
    if len(out) == 1:
        out.append(out[0].copy())
    return np.array(out)


def make_leg(point, sim: Sim3) -> Leg:
    goal = sim.to_odom(point["translation"])
    path = [sim.to_odom(x) for x in (point.get("path") or [])]
    if not path or np.linalg.norm(path[-1] - goal) > 1e-6:
        path.append(goal)
    pts = _clean(path)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return Leg(name=str(point.get("name")), target=point.get("target"),
               pts=pts, yaw=sim.yaw_to_odom(quat_matrix(*point["rotation"])),
               speed=float(point["linear_velocity"][0]), length=length)


class WaypointFollowerNode(Node):

    def __init__(self) -> None:
        super().__init__("waypoint_follower_node")

        self.declare_parameter("tour_path", "tour.yaml")
        self.declare_parameter("registration_path", "registration.yaml")
        self.declare_parameter("tour_frame", "optitrack_map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("rate_hz", 20.0)

        self.declare_parameter("kp_xy", 0.6)
        self.declare_parameter("v_max_xy", 0.5)

        self.declare_parameter("corner_tol", 0.15)
        self.declare_parameter("arrive_tol_xy", 0.10)
        self.declare_parameter("arrive_tol_z", 0.10)
        self.declare_parameter("arrive_tol_yaw_deg", 5.0)
        self.declare_parameter("settle_s", 1.0)
        self.declare_parameter("dwell_s", 0.0)
        self.declare_parameter("start_radius", 0.5)
        self.declare_parameter("approach_speed", 0.3)
        self.declare_parameter("close_loop", True)
        self.declare_parameter("leg_timeout_factor", 3.0)
        self.declare_parameter("leg_timeout_margin", 20.0)

        self.declare_parameter("yaw_source", "dji_attitude")
        self.declare_parameter("yaw_offset_deg", 0.0)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("attitude_timeout", 0.5)
        self.declare_parameter("gate_on_degraded", True)

        self.declare_parameter("require_epoch", True)
        self.declare_parameter("epoch_topic", "vo/status")
        self.declare_parameter("epoch_type", "")
        self.declare_parameter("epoch_field", "vo_epoch")

        self.declare_parameter("manage_enable", True)
        self.declare_parameter("autostart", False)

        g = lambda n: self.get_parameter(n).value
        self.tour_path = str(g("tour_path"))
        self.reg_path = str(g("registration_path"))
        self.tour_frame = str(g("tour_frame"))
        self.odom_frame = str(g("odom_frame"))
        self.rate_hz = float(g("rate_hz"))
        self.kp_xy = float(g("kp_xy"))
        self.v_max_xy = float(g("v_max_xy"))
        self.corner_tol = float(g("corner_tol"))
        self.tol_xy = float(g("arrive_tol_xy"))
        self.tol_z = float(g("arrive_tol_z"))
        self.tol_yaw = math.radians(float(g("arrive_tol_yaw_deg")))
        self.settle_s = float(g("settle_s"))
        self.dwell_s = float(g("dwell_s"))
        self.start_radius = float(g("start_radius"))
        self.approach_speed = float(g("approach_speed"))
        self.close_loop = bool(g("close_loop"))
        self.to_factor = float(g("leg_timeout_factor"))
        self.to_margin = float(g("leg_timeout_margin"))
        self.yaw_source = str(g("yaw_source"))
        self.yaw_offset = math.radians(float(g("yaw_offset_deg")))
        self.pose_timeout = float(g("pose_timeout"))
        self.att_timeout = float(g("attitude_timeout"))
        self.gate_degraded = bool(g("gate_on_degraded"))
        self.require_epoch = bool(g("require_epoch"))
        self.epoch_field = str(g("epoch_field"))
        self.manage_enable = bool(g("manage_enable"))
        self.autostart = bool(g("autostart"))

        if self.yaw_source not in ("dji_attitude", "ekf_pose"):
            self.get_logger().error(
                f"unknown yaw_source '{self.yaw_source}', using dji_attitude")
            self.yaw_source = "dji_attitude"
        if self.kp_xy <= 0.0:
            raise ValueError("kp_xy must be positive")

        self._pose = None
        self._att_yaw = None
        self._status = None
        self._epoch = None
        self._epoch_src = None

        self.state = IDLE
        self.legs = []
        self.ref_epoch = None
        self.i = self.k = 0
        self.s = 0.0
        self._settle_t = 0.0
        self._dwell_t = None
        self._leg_t = 0.0
        self._leg_limit = math.inf
        self._sp = None
        self._t_last = None
        self._t_try = -math.inf
        self._last_try_msg = ""
        self._hold_reason = ""

        self.create_subscription(PoseWithCovarianceStamped,
                                 "localisation/pose", self.on_pose, QOS)
        self.create_subscription(LocalisationStatus, "localisation/status",
                                 self.on_status, QOS)
        self.create_subscription(AttitudeStamped, "attitude",
                                 self.on_attitude, QOS)
        self._setup_epoch(str(g("epoch_topic")), str(g("epoch_type")))

        self.pub_sp = self.create_publisher(PoseStamped, "setpoint", QOS)
        self.pub_enable = self.create_publisher(Bool, "controller/enable", 10)
        self.pub_state = self.create_publisher(String, "follower/state", 10)
        self.pub_arrived = self.create_publisher(String, "follower/arrived", 10)

        self.create_service(Trigger, "follower/start", self.on_start_srv)
        self.create_service(Trigger, "follower/abort", self.on_abort_srv)

        self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.create_timer(5.0, self.on_diagnostics)

        self.get_logger().info(
            f"waypoint_follower up: tour={self.tour_path}, "
            f"registration={self.reg_path}, kp_xy={self.kp_xy}, "
            f"v_max_xy={self.v_max_xy}, yaw_source={self.yaw_source}, "
            f"epoch_source={self._epoch_src}, autostart={self.autostart}")
        self._publish_state()

    def _setup_epoch(self, topic, type_name) -> None:
        if type_name:
            try:
                from rosidl_runtime_py.utilities import get_message
                cls = get_message(type_name)
            except (AttributeError, ModuleNotFoundError, ValueError) as e:
                self.get_logger().error(
                    f"cannot import epoch_type '{type_name}': {e}")
                return
            self.create_subscription(cls, topic, self.on_epoch, QOS)
            self._epoch_src = topic
        elif self.epoch_field in LocalisationStatus.get_fields_and_field_types():
            self._epoch_src = "localisation/status"

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        self._pose = (self._now(), np.array([p.x, p.y, p.z]),
                      quat_yaw(msg.pose.pose.orientation))

    def on_attitude(self, msg: AttitudeStamped) -> None:
        self._att_yaw = (self._now(), -math.radians(msg.yaw))

    def on_status(self, msg: LocalisationStatus) -> None:
        self._status = (self._now(), msg)
        if self._epoch_src == "localisation/status":
            self._epoch = (self._now(), getattr(msg, self.epoch_field))

    def on_epoch(self, msg) -> None:
        if hasattr(msg, self.epoch_field):
            self._epoch = (self._now(), getattr(msg, self.epoch_field))

    def on_start_srv(self, req, resp):
        resp.success, resp.message = self._start()
        (self.get_logger().info if resp.success
         else self.get_logger().error)(f"start: {resp.message}")
        return resp

    def on_abort_srv(self, req, resp):
        if self.state in (RUNNING, DONE):
            self._abort("abort requested")
            resp.success, resp.message = True, "aborted"
        else:
            resp.success, resp.message = False, f"nothing to abort ({self.state})"
        return resp

    def _heading(self, now):
        if self.yaw_source == "dji_attitude":
            if self._att_yaw is None:
                return None, "no attitude"
            if now - self._att_yaw[0] > self.att_timeout:
                return None, f"attitude stale ({now - self._att_yaw[0]:.2f} s)"
            return wrap_pi(self._att_yaw[1] + self.yaw_offset), ""
        return wrap_pi(self._pose[2] + self.yaw_offset), ""

    def _healthy(self, now):
        if self._pose is None:
            return None, "no pose"
        if now - self._pose[0] > self.pose_timeout:
            return None, f"pose stale ({now - self._pose[0]:.2f} s)"
        yaw, why = self._heading(now)
        if yaw is None:
            return None, why
        if self.gate_degraded:
            if self._status is None:
                return None, "no localisation/status"
            if self._status[1].degraded:
                return None, f"localisation degraded {list(self._status[1].flags)}"
        return yaw, ""

    def _start(self):
        if self.state == RUNNING:
            return False, "already running"
        now = self._now()
        if self._pose is None or now - self._pose[0] > self.pose_timeout:
            return False, "no fresh localisation/pose"
        try:
            frame, pts, t_epoch, meta, plan = load_tour(self.tour_path)
            sim, r_epoch = load_registration(self.reg_path)
            first = make_leg(pts[0], sim)
            seq = [make_leg(p, sim) for p in
                   pts[1:] + ([pts[0]] if self.close_loop else [])]
        except (OSError, KeyError, ValueError, TypeError, IndexError,
                yaml.YAMLError) as e:
            return False, f"cannot load tour/registration: {e}"

        if frame != self.tour_frame:
            return False, f"tour frame '{frame}' is not '{self.tour_frame}'"
        if (t_epoch is not None and r_epoch is not None
                and str(t_epoch) != str(r_epoch)):
            return False, (f"tour planned for vo_epoch {t_epoch}, registration "
                           f"is for {r_epoch}; replan")
        ref = t_epoch if t_epoch is not None else r_epoch
        if ref is None:
            self.get_logger().warning(
                "no vo_epoch in tour or registration: a map rebuild will not "
                "be detected")
        elif self.require_epoch:
            if self._epoch_src is None:
                return False, ("no vo_epoch source; set epoch_type/epoch_topic "
                               "or require_epoch:=false")
            if self._epoch is None:
                return False, f"no vo_epoch received on {self._epoch_src}"
            if str(self._epoch[1]) != str(ref):
                return False, (f"live vo_epoch {self._epoch[1]} != planned "
                               f"{ref}; replan")
        if plan and not plan.get("converged", True):
            self.get_logger().warning("tour was not converged when planned")

        p_now = self._pose[1]
        gap = float(np.linalg.norm(first.pts[-1] - p_now))
        if gap > self.start_radius:
            return False, (f"{first.name} is {gap:.2f} m away; start within "
                           f"{self.start_radius} m of it")
        approach = Leg(first.name, first.target, _clean([p_now, first.pts[-1]]),
                       first.yaw, self.approach_speed, gap)

        for sp in sorted({round(l.speed, 3) for l in seq}):
            if sp > self.v_max_xy + 1e-9:
                self.get_logger().warning(
                    f"planned speed {sp} m/s exceeds v_max_xy {self.v_max_xy}; "
                    "legs will fly at v_max_xy, slower than forecast")

        self.legs = [approach] + seq
        self.ref_epoch = ref
        self.i = 0
        self.state = RUNNING
        self._t_last = None
        self._begin_leg()
        if self.manage_enable:
            self.pub_enable.publish(Bool(data=True))
        total = sum(l.length for l in self.legs)
        return True, (f"{len(self.legs)} legs, {total:.1f} m, vo_epoch {ref}, "
                      f"scale {sim.scale:.4f}, w={meta.get('w')}, "
                      f"coupled={meta.get('coupled')}")

    def _abort(self, reason) -> None:
        self.state = ABORTED
        self._sp = None
        if self.manage_enable:
            self.pub_enable.publish(Bool(data=False))
        self.get_logger().error(f"ABORTED: {reason}")
        self._publish_state(reason)

    def _begin_leg(self) -> None:
        leg = self.legs[self.i]
        self.k = 0
        self.s = 0.0
        self._settle_t = 0.0
        self._dwell_t = None
        self._leg_t = 0.0
        v = max(min(leg.speed, self.v_max_xy), 1e-3)
        self._leg_limit = (self.to_factor * leg.length / v + self.to_margin
                           + self.settle_s + self.dwell_s)
        self._sp = (leg.pts[0].copy(), leg.yaw)
        self.get_logger().info(
            f"leg {self.i + 1}/{len(self.legs)} -> {leg.name} "
            f"({leg.length:.2f} m at {leg.speed:.2f} m/s)")
        self._publish_state()

    def _next_leg(self) -> None:
        self.i += 1
        if self.i >= len(self.legs):
            last = self.legs[-1]
            self._sp = (last.pts[-1].copy(), last.yaw)
            self.state = DONE
            self.get_logger().info("tour complete, holding at final point")
            self._publish_state()
            return
        self._begin_leg()

    def _arrived(self, leg) -> None:
        self.pub_arrived.publish(String(data=f"{leg.name} {leg.target}"))
        self.get_logger().info(f"arrived {leg.name} (target {leg.target})")
        if self.dwell_s <= 0.0:
            self._next_leg()
        else:
            self._dwell_t = 0.0

    def _advance(self, yaw, dt) -> None:
        leg = self.legs[self.i]
        p = self._pose[1]
        self._leg_t += dt
        if self._leg_t > self._leg_limit:
            self._abort(f"leg to {leg.name} exceeded {self._leg_limit:.0f} s")
            return
        if self._dwell_t is not None:
            self._dwell_t += dt
            self._sp = (leg.pts[-1].copy(), leg.yaw)
            if self._dwell_t >= self.dwell_s:
                self._next_leg()
            return

        n_seg = len(leg.pts) - 1
        while (self.k < n_seg - 1
               and np.linalg.norm(p - leg.pts[self.k + 1]) < self.corner_tol):
            self.k += 1
            self.s = 0.0
        a, b = leg.pts[self.k], leg.pts[self.k + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            sp = b.copy()
        else:
            u = d / L
            self.s = max(self.s, float(np.clip((p - a) @ u, 0.0, L)))
            lead = min(leg.speed, self.v_max_xy) / self.kp_xy
            sp = a + u * min(self.s + lead, L)
        self._sp = (sp, leg.yaw)

        if self.k == n_seg - 1:
            e = leg.pts[-1] - p
            inside = (math.hypot(e[0], e[1]) < self.tol_xy
                      and abs(e[2]) < self.tol_z
                      and abs(wrap_pi(leg.yaw - yaw)) < self.tol_yaw)
            self._settle_t = self._settle_t + dt if inside else 0.0
            if self._settle_t >= self.settle_s:
                self._arrived(leg)

    def on_timer(self) -> None:
        now = self._now()
        dt = 0.0 if self._t_last is None else min(max(now - self._t_last, 0.0), 0.5)
        self._t_last = now

        if self.state in (RUNNING, DONE) and self.ref_epoch is not None \
                and self._epoch is not None \
                and str(self._epoch[1]) != str(self.ref_epoch):
            self._abort(f"vo_epoch changed {self.ref_epoch} -> {self._epoch[1]}")

        if self.state == RUNNING:
            yaw, why = self._healthy(now)
            if why != self._hold_reason:
                if why:
                    self.get_logger().warning(f"paused: {why}")
                else:
                    self.get_logger().info("resumed")
                self._hold_reason = why
            if yaw is not None:
                self._advance(yaw, dt)

        if self.state in (RUNNING, DONE) and self._sp is not None:
            self._publish_sp(*self._sp)

        if self.autostart and self.state == IDLE and now - self._t_try > 2.0:
            self._t_try = now
            ok, msg = self._start()
            if ok:
                self.get_logger().info(f"autostart: {msg}")
            elif msg != self._last_try_msg:
                self.get_logger().warning(f"autostart waiting: {msg}")
                self._last_try_msg = msg

    def _publish_sp(self, p, yaw) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_sp.publish(msg)

    def _publish_state(self, extra="") -> None:
        s = self.state
        if self.legs and self.state in (RUNNING, DONE):
            leg = self.legs[min(self.i, len(self.legs) - 1)]
            s += f" {min(self.i + 1, len(self.legs))}/{len(self.legs)} {leg.name}"
        if extra:
            s += f" | {extra}"
        self.pub_state.publish(String(data=s))

    def on_diagnostics(self) -> None:
        parts = [f"state={self.state}"]
        if self.state == RUNNING and self._pose is not None:
            leg = self.legs[self.i]
            e = leg.pts[-1] - self._pose[1]
            parts += [f"leg={self.i + 1}/{len(self.legs)} {leg.name}",
                      f"seg={self.k + 1}/{len(leg.pts) - 1}",
                      f"to_go={np.linalg.norm(e):.2f} m",
                      f"t={self._leg_t:.0f}/{self._leg_limit:.0f} s"]
            if self._hold_reason:
                parts.append(f"paused: {self._hold_reason}")
        if self._epoch is not None:
            parts.append(f"vo_epoch={self._epoch[1]}")
        self.get_logger().info(" | ".join(parts))
        self._publish_state()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointFollowerNode()
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
