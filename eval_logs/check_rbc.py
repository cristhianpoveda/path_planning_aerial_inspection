#!/usr/bin/env python3
"""Which composition of gimbal RPY gives R_body_camera_optical?

R_n_b = R_n_v R_v_c R_b_c^-1 is unobservable without truth, but DJI attitude
IS a measurement of R_n_b. So for the correct R_b_c, the ANGLE of
    R_n_b^dji  vs  R_n_v R_v_c R_b_c^-1
must be consistent -- and since R_n_v is a fixed unknown, the correct
hypothesis is the one whose implied R_n_v has the SMALLEST spread over time.
"""
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from drone_localisation.ekf import so3

BAG = "F9_02"
OPT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])

def R_bc(g):
    """Corrected gimbal composition -- matches camera_decoder to 0.1 deg."""
    roll, pitch, yaw = float(g[0]), float(g[1]), float(g[2])
    if np.degrees(pitch) > 3276.8:
        pitch -= np.radians(6553.6)
    return so3.rpy_to_R(roll, -pitch, -yaw) @ OPT

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id=""),
       rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = ["/drone_1/vo/pose", "/drone_1/attitude",
        "/drone_1/gimbal_joint_attitude", "/drone_1/vo/status"]
r.set_filter(rosbag2_py.StorageFilter(topics=want))

vo, att, gim, ep = [], [], [], []
while r.has_next():
    top, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types[top]))
    t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if top.endswith("vo/pose"):
        q = m.pose.orientation
        vo.append((t, so3.quat_to_R([q.x, q.y, q.z, q.w])))
    elif top.endswith("gimbal_joint_attitude"):
        gim.append((t, np.radians([m.roll, m.pitch, m.yaw])))
    elif top.endswith("vo/status"):
        ep.append((t, int(m.vo_epoch), m.tracking_state, int(m.n_map_points)))
    else:
        att.append((t, np.radians([m.roll, m.pitch, m.yaw])))

print(f"vo={len(vo)} att={len(att)} gim={len(gim)}")

def nearest(seq, t):
    i = min(range(len(seq)), key=lambda k: abs(seq[k][0] - t))
    return seq[i] if abs(seq[i][0] - t) < 0.05 else None

f = R_bc
rows = []
for t, R_v_c in vo[::5]:
    a, g = nearest(att, t), nearest(gim, t)
    if a is None or g is None:
        continue
    rows.append((t, so3.log(so3.rpy_to_R(*a[1]) @ f(g[1]) @ R_v_c.T)))

t0 = rows[0][0]
ref = so3.exp(rows[0][1])
print(f"\n{'t(s)':>8} {'|R_n_v - R_n_v(0)| (deg)':>26}")
for t, v in rows[::10]:
    print(f"{t - t0:8.1f} {np.degrees(so3.angle(so3.exp(v) @ ref.T)):26.2f}")

print("\nvo/status transitions:")
prev = None
for t, e, s, n in ep:
    key = (e, s)
    if key != prev:
        print(f"  t={t - t0:7.1f}  epoch={e}  state={s}  map_points={n}")
        prev = key