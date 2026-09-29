"""SE(3) helpers shared by registration_node.py and the offline tools.

T_A_B means "pose of B expressed in A", so composition reads left to right:
T_A_C = T_A_B @ T_B_C.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def orthonormalise(T):
    """Nearest proper rotation. Surveyed and quaternion-derived matrices drift
    from orthonormal by ~1e-4, which compounds through a four-factor chain."""
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    out = np.asarray(T, dtype=float).copy()
    out[:3, :3] = R
    return out


def quat_pos_to_T(q, p):
    """q = (x, y, z, w)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = p
    return T


def T_to_quat_pos(T):
    return Rotation.from_matrix(T[:3, :3]).as_quat(), T[:3, 3]


def mean_T(Ts):
    """Chordal rotation mean, median translation. The median is deliberate:
    a single bad PnP shifts a mean but not a median."""
    R = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).mean()
    out = np.eye(4)
    out[:3, :3] = R.as_matrix()
    out[:3, 3] = np.median(np.array([T[:3, 3] for T in Ts]), axis=0)
    return out


def spread(Ts, T0):
    """Scatter of a set of transforms about T0, in mm and degrees."""
    p = np.array([T[:3, 3] for T in Ts])
    ang = np.array([np.linalg.norm(
        Rotation.from_matrix(T0[:3, :3].T @ T[:3, :3]).as_rotvec()) for T in Ts])
    return dict(
        n=len(Ts),
        pos_sd_mm=float(np.std(np.linalg.norm(p - p.mean(axis=0), axis=1)) * 1e3),
        rot_sd_deg=float(np.degrees(np.std(ang))),
        rot_max_deg=float(np.degrees(ang.max())))


def disagreement(Ta, Tb):
    """Translation (m) and rotation (deg) between two transforms."""
    rel = inv(Ta) @ Tb
    return (float(np.linalg.norm(rel[:3, 3])),
            float(np.degrees(np.linalg.norm(
                Rotation.from_matrix(rel[:3, :3]).as_rotvec()))))


def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
