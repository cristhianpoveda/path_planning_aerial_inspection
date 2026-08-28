"""
test_so3.py -- exp/log round-trips, small angle, near pi, and SLERP sign.

Run:  pytest -q tests/test_so3.py
"""

import numpy as np
import pytest

from drone_localisation.ekf import so3

RNG = np.random.default_rng(0)


def rand_axis():
    a = RNG.normal(size=3)
    return a / np.linalg.norm(a)


# --------------------------------------------------------------------- exp/log
@pytest.mark.parametrize("th", [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.5, 1.5,
                                3.0, np.pi - 1e-6, np.pi - 1e-9])
def test_exp_log_roundtrip(th):
    for _ in range(20):
        phi = rand_axis() * th
        R = so3.exp(phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
        back = so3.log(R)
        # near pi the axis sign is genuinely ambiguous; compare rotations
        assert np.allclose(so3.exp(back), R, atol=1e-8)


def test_log_exp_roundtrip():
    for _ in range(200):
        R = so3.exp(rand_axis() * RNG.uniform(0, np.pi - 1e-6))
        assert np.allclose(so3.exp(so3.log(R)), R, atol=1e-10)


def test_exp_small_angle_matches_series():
    """The Taylor branch must agree with the trig branch across the switch."""
    for th in (1e-9, 1e-8 - 1e-12, 1e-8 + 1e-12, 1e-7):
        phi = rand_axis() * th
        R = so3.exp(phi)
        approx = np.eye(3) + so3.hat(phi)     # first order
        assert np.allclose(R, approx, atol=1e-14 + th * th)


def test_log_identity_is_zero():
    assert np.allclose(so3.log(np.eye(3)), np.zeros(3), atol=1e-15)


def test_log_near_pi_axes():
    """Exactly pi about each principal axis -- the branch that loses precision."""
    for k in range(3):
        a = np.zeros(3)
        a[k] = 1.0
        R = so3.exp(a * np.pi)
        v = so3.log(R)
        assert abs(np.linalg.norm(v) - np.pi) < 1e-6
        assert np.allclose(so3.exp(v), R, atol=1e-7)


def test_hat_vee():
    for _ in range(50):
        v = RNG.normal(size=3)
        assert np.allclose(so3.vee(so3.hat(v)), v)
        w = RNG.normal(size=3)
        assert np.allclose(so3.hat(v) @ w, np.cross(v, w))


# ------------------------------------------------------------------ quaternion
def test_quat_roundtrip():
    for _ in range(200):
        R = so3.exp(rand_axis() * RNG.uniform(0, np.pi - 1e-6))
        assert np.allclose(so3.quat_to_R(so3.R_to_quat(R)), R, atol=1e-10)


def test_quat_sign_invariance():
    """q and -q are the same rotation."""
    for _ in range(50):
        q = so3.R_to_quat(so3.exp(rand_axis() * RNG.uniform(0.1, 3.0)))
        assert np.allclose(so3.quat_to_R(q), so3.quat_to_R(-q), atol=1e-12)


def test_slerp_endpoints_and_sign():
    q0 = so3.R_to_quat(so3.exp(rand_axis() * 0.3))
    q1 = so3.R_to_quat(so3.exp(rand_axis() * 1.2))
    assert np.allclose(so3.quat_to_R(so3.slerp(q0, q1, 0.0)),
                       so3.quat_to_R(q0), atol=1e-10)
    assert np.allclose(so3.quat_to_R(so3.slerp(q0, q1, 1.0)),
                       so3.quat_to_R(q1), atol=1e-10)
    # Sign-flipped endpoint must give the SAME interpolant. Naive lerp between
    # q and -q passes through zero -- this is the mocap bug, in miniature.
    for u in (0.25, 0.5, 0.75):
        a = so3.quat_to_R(so3.slerp(q0, q1, u))
        b = so3.quat_to_R(so3.slerp(q0, -q1, u))
        assert np.allclose(a, b, atol=1e-10)


def test_slerp_constant_rate():
    """Midpoint of a SLERP is half the rotation."""
    q0 = so3.R_to_quat(np.eye(3))
    axis = rand_axis()
    q1 = so3.R_to_quat(so3.exp(axis * 1.0))
    mid = so3.quat_to_R(so3.slerp(q0, q1, 0.5))
    assert so3.angle(mid) == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------------------------------ rpy
def test_rpy_roundtrip():
    for _ in range(200):
        r = RNG.uniform(-np.pi, np.pi)
        p = RNG.uniform(-1.4, 1.4)          # away from gimbal lock
        y = RNG.uniform(-np.pi, np.pi)
        R = so3.rpy_to_R(r, p, y)
        rr, pp, yy = so3.R_to_rpy(R)
        assert np.allclose(so3.rpy_to_R(rr, pp, yy), R, atol=1e-10)


def test_normalise_fixes_drift():
    R = so3.exp(rand_axis() * 1.0)
    R_drifted = R + 1e-6 * RNG.normal(size=(3, 3))
    Rn = so3.normalise(R_drifted)
    assert np.allclose(Rn @ Rn.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(Rn) == pytest.approx(1.0, abs=1e-12)


def test_wrap():
    assert so3.wrap(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert so3.wrap(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert np.allclose(so3.wrap(np.array([0.0, 2 * np.pi, -2 * np.pi])), 0.0,
                       atol=1e-12)
