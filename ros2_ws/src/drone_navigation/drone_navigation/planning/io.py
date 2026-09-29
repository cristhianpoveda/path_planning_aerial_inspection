"""Loading inputs, writing the trajectory. planner_design.md 3.

The output mirrors trajectory_msgs/MultiDOFJointTrajectory field for field, so
the follower's loader is a transcription with no schema of its own to drift.
3 chose that message to avoid a new definition and parallel arrays; writing a
file instead should not quietly reintroduce one.
"""

import numpy as np
import yaml

from .forecast import ForecastConfig

DEFAULT_S = 3.0            # [T] used when no registration.yaml is available


def load_registration(path, verbose=True):
    """Return (s, sigma_s_ratio, epoch) from registration.yaml.
    """
    if path is None:
        if verbose:
            print(f"# no registration.yaml; planning at the default "
                  f"s = {DEFAULT_S} [T]. The tour geometry is unaffected, but "
                  "sigma_s/s and therefore E[Q] are only as good as this "
                  "guess.")
        return DEFAULT_S, 0.10, None
    with open(path) as f:
        d = yaml.safe_load(f)["registration"]
    s = float(d.get("vo_scale_s_corrected", d.get("vo_scale_s", DEFAULT_S)))
    ratio = float(d.get("vo_sigma_s_ratio_corrected",
                        d.get("vo_sigma_s_ratio", 0.10)))
    return s, ratio, d.get("vo_epoch")


def load_board(arena_path):
    """Board pose and the hover viewpoint in front of it, or None."""
    with open(arena_path) as f:
        d = yaml.safe_load(f)
    if "board" not in d:
        return None, None
    b = d["board"]
    centre = np.asarray(b["centre"], float)
    normal = np.asarray(b["normal"], float)
    normal = normal / np.linalg.norm(normal)
    return centre, normal


def board_viewpoint(centre, normal, standoff):
    return np.asarray(centre, float) + float(standoff) * np.asarray(normal,
                                                                   float)


def forecast_config(s, **kw):
    return ForecastConfig(s_true=s, **kw)


def _quat_from_yaw_pitch(yaw, pitch):
    """(x, y, z, w) for yaw about z then pitch about the new y.

    Roll is zero by construction: the gimbal only pitches and yaws, so the
    horizon stays level in the image (viewpoints.camera_frame).
    """
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([-sy * sp, cy * sp, sy * cp, cy * cp])


def write_tour(path, plan, gmap, q, meta=None):
    """Write tour.yaml, mirroring MultiDOFJointTrajectory.
    """
    by_name = {c.name: c for c in plan.selected}
    pts, t = [], 0.0
    for i, name in enumerate(plan.order):
        c = by_name.get(name)
        v = plan.speeds[i] if i < len(plan.speeds) else plan.speeds[-1]
        leg = plan.legs[i]
        length = float(np.sum(np.linalg.norm(np.diff(leg[1], axis=0), axis=1)))
        t += length / v
        if c is None:                       # the board node
            pos = np.asarray(leg[1][-1], float)
            yaw = pitch = 0.0
            tgt = None
        else:
            pos = c.position
            yaw = np.radians(c.yaw_deg)
            pitch = np.radians(c.pitch_deg)
            tgt = c.target
        qx, qy, qz, qw = _quat_from_yaw_pitch(yaw, pitch)
        pts.append(dict(
            name=name, target=tgt,
            translation=[float(x) for x in pos],
            rotation=[float(qx), float(qy), float(qz), float(qw)],
            linear_velocity=[float(v), 0.0, 0.0],
            time_from_start=float(t),
            path=[[float(x) for x in p] for p in leg[1]],
            sigma_s=float(plan.sigma_s.get(name, float("nan")))))

    doc = dict(
        header=dict(frame_id="optitrack_map"),
        joint_names=["base_link"],
        points=pts,
        plan=dict(
            J=float(plan.J), Qbar=float(plan.Qbar), That=float(plan.That),
            time_s=float(plan.time_s), iterations=int(plan.iterations),
            converged=bool(plan.converged),
            quality={k: float(v) for k, v in plan.quality.items()}),
        meta=meta or {})
    with open(path, "w") as f:
        f.write("# Tour in the ARENA frame. The follower applies the inverse\n")
        f.write("# Sim(3) from registration.yaml:\n")
        f.write("#     p_odom = (1/scale) * R^T * (p_opti - translation)\n")
        f.write("# and must stop if vo/status.vo_epoch leaves plan.vo_epoch:\n")
        f.write("# a map rebuild invalidates the registration by up to 96 deg.\n")
        f.write("#\n")
        f.write("# Fields mirror trajectory_msgs/MultiDOFJointTrajectory.\n")
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    return doc
