#!/usr/bin/env python3
"""Compare our composed R_b_c against the /tf chain camera_decoder publishes."""
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from drone_localisation.ekf import so3

BAG = "F3_02"
OPT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id=""),
       rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(
    topics=["/tf", "/tf_static", "/drone_1/gimbal_joint_attitude"]))

static, dyn, gim = {}, [], []
while r.has_next():
    top, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types[top]))
    if top == "/drone_1/gimbal_joint_attitude":
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        gim.append((t, np.radians([m.roll, m.pitch, m.yaw])))
        continue
    for tr in m.transforms:
        q = tr.transform.rotation
        R = so3.quat_to_R([q.x, q.y, q.z, q.w])
        key = (tr.header.frame_id, tr.child_frame_id)
        if top == "/tf_static":
            static[key] = R
            print(f"static: {key[0]:24s} -> {key[1]:24s}")
        else:
            t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
            dyn.append((t, key, R))

print(f"\ndynamic transforms seen: {sorted({k for _, k, _ in dyn})}")
print(f"gimbal msgs: {len(gim)}   dynamic tf msgs: {len(dyn)}")

# the static factor my earlier chain silently skipped
for (fa, fb), R in static.items():
    if fb.endswith("gimbal_base"):
        print(f"\nbase_link -> gimbal_base rpy(deg): "
              f"{np.degrees(so3.R_to_rpy(R))}")
R_opt = next(R for (fa, fb), R in static.items()
             if fb.endswith("camera_optical_frame"))

rows = []
for t, key, R_dyn in dyn[::3]:
    j = min(range(len(gim)), key=lambda k: abs(gim[k][0] - t))
    if abs(gim[j][0] - t) > 0.05:
        continue
    g = np.degrees(gim[j][1])                       # roll, pitch, yaw
    d = np.degrees(so3.R_to_rpy(R_dyn))             # same, from /tf
    gc = gim[j][1].copy()
    if np.degrees(gc[1]) > 3276.8:            # unsigned 16-bit pitch wrap
        gc[1] -= np.radians(6553.6)
    gc[1] = -gc[1]                            # pitch negated vs ROS
    gc[2] = -gc[2]                            # yaw negated vs ROS
    err = np.degrees(so3.angle(so3.rpy_to_R(*gc) @ R_dyn.T))
    rows.append((t, g, d, err))

rows.sort(key=lambda r: r[3])
print(f"\n{'':>6} {'gimbal topic r/p/y':>30} {'/tf dynamic r/p/y':>30} {'err':>7}")
for lab, sel in (("BEST", rows[:6]), ("WORST", rows[-8:])):
    print(f"-- {lab}")
    for t, g, d, e in sel:
        print(f"{t % 1000:6.1f} {g[0]:9.2f}{g[1]:10.2f}{g[2]:10.2f} "
              f"{d[0]:9.2f}{d[1]:10.2f}{d[2]:10.2f} {e:7.2f}")
