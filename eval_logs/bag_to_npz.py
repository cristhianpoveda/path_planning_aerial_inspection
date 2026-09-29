#!/usr/bin/env python3
"""bag_to_npz.py -- pull the flight bag and the mocap bag into one .npz.

Run this inside the ROS container (it needs rosbag2_py and drone_interfaces).
Everything after this step is plain numpy, so the analysis can run anywhere.

    python3 bag_to_npz.py controller_03 controller_03_mocap -o controller_03.npz

Both the header stamp and the bag receive stamp are kept for every message.
They answer different questions:

  * header stamp  -- the time the publisher assigned to the measurement.
                     For telemetry that is the phone packet time mapped onto
                     the laptop clock; for localisation/pose it is the VO
                     stamp minus VO_DELAY; for command/vel it is the
                     controller's own clock.
  * receive stamp -- the recorder's clock. Common to every topic in one bag,
                     and, if both bags were recorded on the same machine,
                     common across the two bags as well. This is the only
                     timebase that is shared by construction rather than by
                     assumption, so it is what the clock-offset estimate is
                     checked against.

No message content is interpreted here beyond unpacking fields. Nothing is
renamed, negated or converted.
"""
import argparse
import os
import sys

import numpy as np


def _reader(path):
    import rosbag2_py
    storage_id = ""
    # rosbag2 picks the storage plugin from metadata.yaml when id is empty.
    so = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
    co = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr")
    r = rosbag2_py.SequentialReader()
    r.open(so, co)
    return r


def _stamp(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return float("nan")
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _pose(p, q):
    return ([p.position.x, p.position.y, p.position.z],
            [q.x, q.y, q.z, q.w])


def _extract(msg, mtype, rec):
    """Append one message's fields to the per-topic record dict."""
    rec.setdefault("t_hdr", []).append(_stamp(msg))

    if mtype.endswith("PoseStamped"):
        p, q = _pose(msg.pose, msg.pose.orientation)
        rec.setdefault("p", []).append(p)
        rec.setdefault("q", []).append(q)

    elif mtype.endswith("PoseWithCovarianceStamped"):
        p, q = _pose(msg.pose.pose, msg.pose.pose.orientation)
        rec.setdefault("p", []).append(p)
        rec.setdefault("q", []).append(q)
        rec.setdefault("cov", []).append(list(msg.pose.covariance))

    elif mtype.endswith("TwistStamped"):
        t = msg.twist
        rec.setdefault("lin", []).append([t.linear.x, t.linear.y, t.linear.z])
        rec.setdefault("ang", []).append([t.angular.x, t.angular.y, t.angular.z])

    elif mtype.endswith("Vector3Stamped"):
        v = msg.vector
        rec.setdefault("v", []).append([v.x, v.y, v.z])

    elif mtype.endswith("AttitudeStamped"):
        rec.setdefault("roll", []).append(float(msg.roll))
        rec.setdefault("pitch", []).append(float(msg.pitch))
        rec.setdefault("yaw", []).append(float(msg.yaw))

    elif mtype.endswith("RelativeAltitudeStamped"):
        rec.setdefault("altitude", []).append(float(msg.altitude))

    elif mtype.endswith("VoStatus"):
        rec.setdefault("vo_epoch", []).append(int(msg.vo_epoch))
        rec.setdefault("n_map_points", []).append(int(msg.n_map_points))
        rec.setdefault("pose_valid", []).append(bool(msg.pose_valid))
        rec.setdefault("tracking_state", []).append(str(msg.tracking_state))

    elif mtype.endswith("LocalisationStatus"):
        rec.setdefault("scale", []).append(float(msg.scale))
        rec.setdefault("sigma_scale", []).append(float(msg.sigma_scale))
        rec.setdefault("alt_bias", []).append(float(msg.alt_bias))
        rec.setdefault("sigma_alt_bias", []).append(float(msg.sigma_alt_bias))
        rec.setdefault("cov_scale_bias", []).append(float(msg.cov_scale_bias))
        rec.setdefault("degraded", []).append(bool(msg.degraded))
        rec.setdefault("state", []).append(str(msg.state))
        rec.setdefault("flags", []).append(";".join(list(msg.flags)))

    elif mtype.endswith("CompressedImage"):
        rec.setdefault("size", []).append(len(msg.data))

    elif mtype.endswith("msgs/msg/Bool"):
        rec.setdefault("data", []).append(bool(msg.data))

    else:
        return False
    return True


# topic (with the namespace stripped) -> group name in the npz
DEFAULT_MAP = {
    "vo/pose": "vo.pose",
    "vo/status": "vo.status",
    "localisation/pose": "localisation.pose",
    "localisation/status": "localisation.status",
    "command/vel": "command.vel",
    "command/stick": "command.stick",
    "setpoint": "setpoint",
    "controller/enable": "controller.enable",
    "speed_vector": "speed_vector",
    "relative_altitude": "relative_altitude",
    "attitude": "attitude",
    "gimbal_joint_attitude": "gimbal_joint_attitude",
    "camera/image/compressed": "camera.image",
}


def read_bag(path, namespace="/drone_1", mocap_group=None, mocap_topic=None):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r = _reader(path)
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    classes = {}
    out = {}
    unknown = set()

    while r.has_next():
        topic, raw, t_recv = r.read_next()
        mtype = types[topic]

        if mocap_topic is not None and topic == mocap_topic:
            group = mocap_group
        else:
            short = topic
            if namespace and short.startswith(namespace + "/"):
                short = short[len(namespace) + 1:]
            short = short.lstrip("/")
            group = DEFAULT_MAP.get(short)
        if group is None:
            unknown.add(topic)
            continue

        if mtype not in classes:
            classes[mtype] = get_message(mtype)
        msg = deserialize_message(raw, classes[mtype])

        rec = out.setdefault(group, {})
        rec.setdefault("t_recv", []).append(t_recv * 1e-9)
        if not _extract(msg, mtype, rec):
            unknown.add(topic)

    if unknown:
        print(f"  ignored topics: {sorted(unknown)}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight_bag")
    ap.add_argument("mocap_bag", nargs="?", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--namespace", default="/drone_1")
    ap.add_argument("--mocap-topic", default="/optitrack/rigid_bodies/dji_mini4")
    a = ap.parse_args()

    data = {}
    print(f"reading {a.flight_bag}", file=sys.stderr)
    for g, rec in read_bag(a.flight_bag, a.namespace).items():
        data[g] = rec
    if a.mocap_bag:
        print(f"reading {a.mocap_bag}", file=sys.stderr)
        for g, rec in read_bag(a.mocap_bag, a.namespace,
                               mocap_group="mocap",
                               mocap_topic=a.mocap_topic).items():
            data[g] = rec

    flat = {}
    for g, rec in data.items():
        for k, v in rec.items():
            arr = np.asarray(v)
            if arr.dtype.kind in "OUS":
                arr = arr.astype("U32")
            flat[f"{g}/{k}"] = arr
        print(f"  {g:24s} {len(rec['t_recv']):7d} msgs", file=sys.stderr)

    out = a.out or (os.path.basename(a.flight_bag.rstrip("/")) + ".npz")
    np.savez_compressed(out, **flat)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
