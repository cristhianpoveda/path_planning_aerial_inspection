#!/usr/bin/env python3
"""Bag pair -> one npz.  ROS-dependent module.

  python3 reg_extract.py --bag checkerboard_01 --mocap checkerboard_01_mocap \
      --calib camera_calibration.yaml --out reg.npz

"""

import argparse
import sys

import cv2
import numpy as np
import yaml

import rclpy.serialization
import rosbag2_py
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message

import reg_board

CAM_TOPIC = "/drone_1/camera/image/compressed"
VO_TOPIC = "/drone_1/vo/pose"
POSE_TOPIC = "/drone_1/localisation/pose"
STATUS_TOPIC = "/drone_1/localisation/status"
SPEED_TOPIC = "/drone_1/speed_vector"
MOCAP_DRONE = "/optitrack/rigid_bodies/dji_mini4"
MOCAP_BOARD = "/optitrack/rigid_bodies/calib_board"

OPTICAL_FRAME = "camera_optical_frame"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"


def stamp_sec(s):
    return s.sec + s.nanosec * 1e-9


_WARNED = set()


def safe_msg(type_str, topic):
    """get_message that survives a missing custom interface package."""
    try:
        return get_message(type_str)
    except Exception as e:
        if topic not in _WARNED:
            _WARNED.add(topic)
            print(f"  !! cannot load {type_str} for {topic}: {e}")
            print(f"  !! skipping {topic}")
        return None


def reader(path):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    return r, types


def quat_pos_to_T(q, p):
    """q = (x, y, z, w)."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, 3] = p
    return T


def tf_to_T(t):
    tr, ro = t.transform.translation, t.transform.rotation
    return quat_pos_to_T((ro.x, ro.y, ro.z, ro.w), (tr.x, tr.y, tr.z))


def load_calib(path):
    """Accepts flat fx/fy/cx/cy/k1/k2/p1/p2 or ORB-SLAM 'Camera.fx' keys or an
    OpenCV camera_matrix/distortion_coefficients block."""
    if path is None:
        # planner_design.md: fx = 1431.85, 1920x1080, k1 = 0.072, k2 = -0.084
        K = np.array([[1431.85, 0, 960.0], [0, 1431.85, 540.0], [0, 0, 1.0]])
        return K, np.array([0.072, -0.084, 0.0, 0.0, 0.0])
    with open(path) as f:
        d = yaml.safe_load(f.read().replace("%YAML:1.0", "").replace("\t", " "))

    def get(*names, default=None):
        for n in names:
            if n in d:
                return d[n]
        return default

    if "camera_matrix" in d:
        K = np.array(d["camera_matrix"]["data"], float).reshape(3, 3)
        dist = np.array(d["distortion_coefficients"]["data"], float).ravel()
        return K, dist
    fx = float(get("fx", "Camera.fx", "Camera1.fx"))
    fy = float(get("fy", "Camera.fy", "Camera1.fy", default=fx))
    cx = float(get("cx", "Camera.cx", "Camera1.cx"))
    cy = float(get("cy", "Camera.cy", "Camera1.cy"))
    dist = np.array([float(get(f"k1", "Camera.k1", "Camera1.k1", default=0.0)),
                     float(get("k2", "Camera.k2", "Camera1.k2", default=0.0)),
                     float(get("p1", "Camera.p1", "Camera1.p1", default=0.0)),
                     float(get("p2", "Camera.p2", "Camera1.p2", default=0.0)),
                     float(get("k3", "Camera.k3", "Camera1.k3", default=0.0))])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--mocap", required=True)
    ap.add_argument("--calib")
    ap.add_argument("--out", default="reg.npz")
    ap.add_argument("--vo-delay", type=float, default=None,
                    help="seconds; default = measured from the bag")
    ap.add_argument("--min-tags", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    K, dist = load_calib(a.calib)
    print(f"intrinsics fx={K[0,0]:.2f} cx={K[0,2]:.2f} dist={dist[:2]}")

    # ------------------------------------------------------------ single pass
    buf = tf2_ros.Buffer(cache_time=Duration(seconds=3600))
    det = reg_board.make_detector()
    status, pose_ts, speed, vo_ts, img_ts, dets = [], [], [], [], [], []
    n_img = 0
    r, types = reader(a.bag)
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic in ("/tf", "/tf_static"):
            m = rclpy.serialization.deserialize_message(
                data, get_message(types[topic]))
            for tr in m.transforms:
                if topic == "/tf_static":
                    buf.set_transform_static(tr, "bag")
                else:
                    buf.set_transform(tr, "bag")
        elif topic == STATUS_TOPIC:
            mt = safe_msg(types[topic], topic)
            if mt is None:
                continue
            m = rclpy.serialization.deserialize_message(data, mt)
            status.append((stamp_sec(m.header.stamp), m.scale, m.sigma_scale,
                           m.alt_bias, float(m.degraded), ";".join(m.flags)))
        elif topic == POSE_TOPIC:
            m = rclpy.serialization.deserialize_message(
                data, get_message(types[topic]))
            p, o = m.pose.pose.position, m.pose.pose.orientation
            pose_ts.append((stamp_sec(m.header.stamp), p.x, p.y, p.z,
                            o.x, o.y, o.z, o.w))
        elif topic == SPEED_TOPIC:
            m = rclpy.serialization.deserialize_message(
                data, get_message(types[topic]))
            speed.append((stamp_sec(m.header.stamp),
                          m.vector.x, m.vector.y, m.vector.z))
        elif topic == VO_TOPIC:
            m = rclpy.serialization.deserialize_message(
                data, get_message(types[topic]))
            vo_ts.append(stamp_sec(m.header.stamp))
        elif topic == CAM_TOPIC:
            n_img += 1
            m = rclpy.serialization.deserialize_message(
                data, get_message(types[topic]))
            img_ts.append(stamp_sec(m.header.stamp))
            if (n_img - 1) % a.stride:
                continue
            gray = cv2.imdecode(np.frombuffer(m.data, np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            out = reg_board.detect_board(gray, det, K, dist, a.min_tags)
            if out is None:
                continue
            T_cb, n_tags, rms = out
            dets.append((img_ts[-1], T_cb, n_tags, rms))
            if len(dets) % 250 == 0:
                print(f"  {n_img} frames, {len(dets)} detected")
    del r
    print(f"tf loaded; status={len(status)} pose={len(pose_ts)} "
          f"speed={len(speed)} vo={len(vo_ts)}")
    print(f"frames={n_img} detected={len(dets)}")

    # --------------------------------------------------------- VO_DELAY
    img_ts = np.array(img_ts)
    vo_ts = np.array(sorted(vo_ts))
    if a.vo_delay is not None:
        vo_delay, how = a.vo_delay, "given"
    elif len(vo_ts) and len(img_ts):
        j = np.clip(np.searchsorted(img_ts, vo_ts), 0, len(img_ts) - 1)
        dd = img_ts[j] - vo_ts
        vo_delay = float(np.median(dd[np.abs(dd) < 2.0]))
        how = f"measured, IQR {np.subtract(*np.percentile(dd, [75, 25])):.4f}"
    else:
        vo_delay, how = 0.0, "fallback"
    print(f"VO_DELAY = {vo_delay:.4f} s ({how})")

    # ------------------------------------------------- tf, buffer complete
    rows = []
    n_tf_fail = 0
    for (t_img, T_cb, n_tags, rms) in dets:
        t_cap = t_img - vo_delay
        try:
            tb = buf.lookup_transform(BASE_FRAME, OPTICAL_FRAME,
                                      Time(seconds=t_cap))
            to = buf.lookup_transform(ODOM_FRAME, BASE_FRAME,
                                      Time(seconds=t_cap))
        except Exception:
            n_tf_fail += 1
            continue
        rows.append((t_img, t_cap, n_tags, rms,
                     T_cb, tf_to_T(tb), tf_to_T(to)))
    print(f"with_tf={len(rows)} (tf lookup failed on {n_tf_fail})")
    if not rows:
        sys.exit("no usable frames -- check topics, tf frames and calibration")

    # ---------------------------------------------------------------- mocap
    mo_d, mo_b = [], []
    r, types = reader(a.mocap)
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic not in (MOCAP_DRONE, MOCAP_BOARD):
            continue
        m = rclpy.serialization.deserialize_message(
            data, get_message(types[topic]))
        p, o = m.pose.position, m.pose.orientation
        row = (stamp_sec(m.header.stamp), p.x, p.y, p.z, o.x, o.y, o.z, o.w)
        (mo_d if topic == MOCAP_DRONE else mo_b).append(row)
    del r
    print(f"mocap drone={len(mo_d)} board={len(mo_b)}")

    np.savez_compressed(
        a.out,
        K=K, dist=dist, vo_delay=vo_delay,
        det_t_img=np.array([r_[0] for r_ in rows]),
        det_t_cap=np.array([r_[1] for r_ in rows]),
        det_n=np.array([r_[2] for r_ in rows]),
        det_rms=np.array([r_[3] for r_ in rows]),
        T_cam_board=np.array([r_[4] for r_ in rows]),
        T_base_cam=np.array([r_[5] for r_ in rows]),
        T_odom_base=np.array([r_[6] for r_ in rows]),
        status_t=np.array([s[0] for s in status]),
        status_s=np.array([s[1] for s in status]),
        status_sig=np.array([s[2] for s in status]),
        status_degraded=np.array([s[4] for s in status]),
        status_flags=np.array([s[5] for s in status], dtype=object),
        pose=np.array(pose_ts) if pose_ts else np.zeros((0, 8)),
        speed=np.array(speed) if speed else np.zeros((0, 4)),
        mocap_drone=np.array(mo_d) if mo_d else np.zeros((0, 8)),
        mocap_board=np.array(mo_b) if mo_b else np.zeros((0, 8)),
        allow_pickle=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
