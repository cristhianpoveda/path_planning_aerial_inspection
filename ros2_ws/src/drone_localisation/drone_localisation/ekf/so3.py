"""
so3.py -- SO(3) helpers for the localisation EKF.

Pure numpy. No ROS, no scipy. This module is the only place rotation
conventions live, so it is also the only place a convention bug can hide --
hence test_so3.py.

Conventions
-----------
* Rotation matrices are `R_a_b`, mapping a vector expressed in frame `b` into
  frame `a`:  v_a = R_a_b @ v_b.
* Quaternions are (x, y, z, w) -- ROS order, NOT (w, x, y, z).
* `exp` takes a rotation VECTOR (axis * angle, radians), not a unit axis.
* `log` returns a rotation vector in (-pi, pi].

The error-state convention in filter_design.md 1 is
    R_n_v = exp(hat(dtheta)) @ R_bar_n_v
i.e. the error rotation is applied on the LEFT, in the nav frame. Every
perturbation in the finite-difference tests must be applied the same way.
"""

import numpy as np

_EPS = 1e-12
# Below this angle the Taylor series is more accurate than the trig form.
_SMALL = 1e-8
# Within this of pi, take the symmetric-part branch in log(). Measured
# crossover between the two branches is ~1e-5 rad; see the comment in log().
_NEAR_PI = 1e-5


def hat(v):
    """Skew-symmetric matrix of a 3-vector: hat(a) @ b == cross(a, b)."""
    v = np.asarray(v, float).reshape(3)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def vee(S):
    """Inverse of hat(). Does not check that S is skew-symmetric."""
    S = np.asarray(S, float)
    return np.array([S[2, 1] - S[1, 2],
                     S[0, 2] - S[2, 0],
                     S[1, 0] - S[0, 1]]) * 0.5


def exp(phi):
    """Rodrigues: rotation vector -> rotation matrix."""
    phi = np.asarray(phi, float).reshape(3)
    th2 = float(phi @ phi)
    th = np.sqrt(th2)
    K = hat(phi)
    if th < _SMALL:
        # sin(th)/th -> 1 - th^2/6 ; (1-cos)/th^2 -> 1/2 - th^2/24
        a = 1.0 - th2 / 6.0
        b = 0.5 - th2 / 24.0
    else:
        a = np.sin(th) / th
        b = (1.0 - np.cos(th)) / th2
    return np.eye(3) + a * K + b * (K @ K)


def log(R):
    """Rotation matrix -> rotation vector, in (-pi, pi]. Stable near 0 and pi."""
    R = np.asarray(R, float)
    v = vee(R)                              # = sin(th) * axis
    s = float(np.linalg.norm(v))
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arctan2(s, c))

    if th < _SMALL:
        # th / sin(th) -> 1 + th^2/6
        return v * (1.0 + th * th / 6.0)

    if th < np.pi - _NEAR_PI:
        return v * (th / s)

    # Near pi, sin(th) -> 0 and `v` is a difference of O(1) matrix elements, so
    # it loses relative precision. Recover the axis from the SYMMETRIC part
    # instead. Writing th = pi - d:
    #     (R + R^T)/2 + I = (d^2/2) I + (2 - d^2/2) a a^T
    # so every column is parallel to `a` up to O(d^2). Using (R + I)/2 instead
    # leaves the O(d) skew term in and is ~4 orders worse at d = 1e-4
    # (measured: 6.6e-9 vs 1.1e-4). `v` still carries a usable sign.
    A = 0.5 * (R + R.T) + np.eye(3)
    k = int(np.argmax(np.diag(A)))
    axis = A[:, k]
    n = float(np.linalg.norm(axis))
    if n < _EPS:                            # degenerate; fall back
        return v * (th / max(s, _EPS))
    axis = axis / n
    if float(axis @ v) < 0.0:
        axis = -axis
    return axis * th


def angle(R):
    """Rotation angle of R, radians, in [0, pi]."""
    c = float(np.clip((np.trace(np.asarray(R, float)) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(c))


def normalise(R):
    """Nearest rotation matrix, by SVD. Call after repeated multiplication."""
    U, _, Vt = np.linalg.svd(np.asarray(R, float))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:             # reflection -> flip the last axis
        U[:, 2] *= -1.0
        Rn = U @ Vt
    return Rn


def quat_to_R(q):
    """(x, y, z, w) -> rotation matrix. Normalises q first."""
    q = np.asarray(q, float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    """Rotation matrix -> (x, y, z, w), with w >= 0."""
    R = np.asarray(R, float)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x, y, z = 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x, y, z = (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x, y, z = (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([x, y, z, w])
    if q[3] < 0.0:                          # q and -q are the same rotation
        q = -q
    return q / np.linalg.norm(q)


def slerp(q0, q1, u):
    """Shortest-arc SLERP between two (x,y,z,w) quaternions, u in [0, 1].

    Sign-aligns first: q and -q are the same rotation, and interpolating
    between q and -q linearly passes through zero. This bit is what the mocap
    analysis got wrong for weeks.
    """
    a = np.asarray(q0, float).reshape(4)
    b = np.asarray(q1, float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(a @ b)
    if d < 0.0:
        b = -b
        d = -d
    if d > 1.0 - 1e-9:                      # nearly parallel -> lerp
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(d, -1.0, 1.0))
    th = th0 * u
    s0 = np.sin(th0)
    q = (np.sin(th0 - th) / s0) * a + (np.sin(th) / s0) * b
    return q / np.linalg.norm(q)


def rpy_to_R(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) intrinsic -> R. Matches DJI attitude convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def R_to_rpy(R):
    """R -> (roll, pitch, yaw), ZYX. Gimbal-lock safe."""
    R = np.asarray(R, float)
    sp = -float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) > 1.0 - 1e-9:                # gimbal lock: roll and yaw degenerate
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, float(pitch), yaw


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi
