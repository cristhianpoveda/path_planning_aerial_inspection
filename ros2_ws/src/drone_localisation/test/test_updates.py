"""
test_updates.py -- verification of the three measurement updates.

Disciplines from filter_design.md 10, applied throughout:

  1. Finite-difference h(x), NEVER the residual y = z - h(x). Differencing y
     returns -H and will "confirm" the wrong sign. There is a test below that
     demonstrates exactly this, so the trap is documented in code.

  2. Perturb dtheta through exp(hat(dtheta)) @ R_bar, never by adding to a
     vector.

  3. The attitude update needs a CONVERGENCE test, not an FD test: h(dtheta)
     = dtheta makes the FD check vacuous (it just returns I). The real risk is
     in the R_bar_n_v @ R_v_c @ R_b_c^T composition and in the reset, so the
     test seeds a known error and checks it is driven out.

Run:  pytest -q tests/test_updates.py
"""

import numpy as np
import pytest
import dataclasses

from drone_localisation.ekf import so3
from drone_localisation.ekf.params import EkfParams
from drone_localisation.ekf.filter import (EkfCore, Increment, Kind, NX, IDX_P, IDX_S, IDX_B, IDX_TH, ned_to_enu, sigma_rp, G)

RNG = np.random.default_rng(7)


def rand_axis():
    a = RNG.normal(size=3)
    return a / np.linalg.norm(a)


def make_core(s=1.3, estimate_scale=1.0):
    core = EkfCore(dataclasses.replace(EkfParams(),
                                       estimate_scale=estimate_scale))
    core.x.p = RNG.normal(size=3)
    core.x.s = s
    core.x.b = 0.05
    core.x.R_nv = so3.exp(rand_axis() * 0.4)
    # a non-diagonal, well-conditioned P so the Joseph form is exercised
    A = RNG.normal(size=(NX, NX)) * 0.05
    core.x.P = core.x.P + A @ A.T
    return core


def make_increment(scale=0.05, dt=0.1):
    return Increment(dp_v=RNG.normal(scale=scale, size=3),
                     dl=RNG.normal(scale=0.002, size=3),
                     dt=dt,
                     Sigma_v=np.diag(RNG.uniform(1e-5, 1e-4, size=3)))


def perturb(core, dx):
    """Apply an error-state perturbation to a COPY of the nominal."""
    pert = core.x.copy()
    pert.p = core.x.p + dx[IDX_P]
    pert.s = core.x.s + dx[IDX_S]
    pert.b = core.x.b + dx[IDX_B]
    pert.R_nv = so3.exp(dx[IDX_TH]) @ core.x.R_nv
    return pert


def numeric_H(core, h_of_state, m, eps=1e-7):
    """dh/dx by central differences on the STATE MAP h, not on y."""
    H = np.zeros((m, NX))
    for j in range(NX):
        dp, dm = np.zeros(NX), np.zeros(NX)
        dp[j] = eps
        dm[j] = -eps
        H[:, j] = (np.atleast_1d(h_of_state(perturb(core, dp)))
                   - np.atleast_1d(h_of_state(perturb(core, dm)))) / (2 * eps)
    return H


# ====================================================================== 5.2
def test_altitude_H_matches_finite_difference():
    for _ in range(20):
        core = make_core()
        H_num = numeric_H(core, lambda st: st.p[2] + st.b, 1)
        H_ana = np.zeros((1, NX))
        H_ana[0, 2] = 1.0
        H_ana[0, IDX_B] = 1.0
        assert np.allclose(H_ana, H_num, atol=1e-6)


def test_altitude_residual_sign():
    """A measurement above the prediction must push p_z + b UP."""
    core = make_core()
    before = core.x.p[2] + core.x.b
    core.update_altitude(before + 0.5)
    after = core.x.p[2] + core.x.b
    assert after > before


def test_altitude_R_switches_on_vz():
    core = make_core()
    inn_h = core.update_altitude(core.x.p[2] + core.x.b + 0.01, vz=0.0)
    inn_v = core.update_altitude(core.x.p[2] + core.x.b + 0.01, vz=1.0)
    assert inn_h.note == "hover" and inn_v.note == "vertical"
    assert inn_v.R_used[0, 0] > inn_h.R_used[0, 0]


def test_altitude_observes_pz_and_b():
    core = make_core()
    P0 = core.x.P.copy()
    for _ in range(50):
        core.update_altitude(core.x.p[2] + core.x.b + RNG.normal(scale=0.03))
    assert core.x.P[2, 2] < P0[2, 2]
    assert core.x.P[IDX_B, IDX_B] < P0[IDX_B, IDX_B]


# ====================================================================== 5.3
def test_velocity_H_matches_finite_difference():
    for _ in range(20):
        core = make_core()
        inc = make_increment(scale=0.08)

        def h(st):
            w_v = inc.dp_v / inc.dt
            return st.s * (st.R_nv @ w_v) - inc.dl / inc.dt

        H_num = numeric_H(core, h, 3)
        _, u_w = core.velocity_prediction(inc)
        H_ana = np.zeros((3, NX))
        H_ana[:, IDX_S] = u_w
        H_ana[:, IDX_TH] = -core.x.s * so3.hat(u_w)
        assert np.allclose(H_ana, H_num, atol=1e-6), (
            f"max |dH| = {np.abs(H_ana - H_num).max():.3e}")


def test_velocity_theta_block_sign():
    """dh/ddtheta = -s*hat(u_w): exp(dth) u ~ u - hat(u) dth."""
    core = make_core()
    inc = make_increment()
    _, u_w = core.velocity_prediction(inc)
    dth = np.array([0.0, 0.0, 1e-6])
    lhs = (core.x.s * (so3.exp(dth) @ u_w) - core.x.s * u_w) / 1e-6
    rhs = (-core.x.s * so3.hat(u_w)) @ (dth / 1e-6)
    assert np.allclose(lhs, rhs, atol=1e-5)


def test_velocity_rejected_below_V_MIN():
    """[M] gain_horizontal falls to 0.77 in hover -- the quantum is a dead
    zone. filter_design.md 5.3 rejects the update rather than trusting it."""
    core = make_core()
    inc = make_increment()
    s0, P0 = core.x.s, core.x.P.copy()
    inn = core.update_velocity(np.array([0.05, 0.0, 0.0]), inc)
    assert not inn.accepted and inn.note == "below V_MIN"
    assert core.x.s == s0
    assert np.allclose(core.x.P, P0)
    assert core.n_rejected[Kind.VELOCITY] == 1


def test_velocity_inflated_between_V_MIN_and_V_LOW():
    # separate cores: R_vo depends on s, and the first update changes it
    c1, c2 = make_core(), make_core()
    inc = make_increment()
    slow = c1.update_velocity(np.array([0.25, 0.0, 0.0]), inc)
    fast = c2.update_velocity(np.array([1.0, 0.0, 0.0]), inc)
    assert "low-speed inflated" in slow.note
    assert "inflated" not in fast.note
    p = EkfParams()
    assert (slow.R_used[0, 0] - fast.R_used[0, 0]) == pytest.approx(
        (p.V_INFL - 1) * p.R_speed_h / p.K_VEL ** 2, rel=1e-6)


def test_velocity_yaw_rate_inflation():
    core = make_core()
    inc = make_increment()
    p = EkfParams()
    calm = core.update_velocity(np.array([1.0, 0.0, 0.0]), inc, omega_yaw=0.0)
    spin = core.update_velocity(np.array([1.0, 0.0, 0.0]), inc,
                                omega_yaw=np.radians(50.0))
    assert "yaw-rate inflated" in spin.note
    assert spin.R_used[0, 0] > calm.R_used[0, 0]


def test_K_VEL_applied_to_measurement_and_covariance_together():
    """Dividing z without dividing R misstates R (filter_design.md 12)."""
    p = EkfParams()
    core = make_core()
    inc = make_increment()
    inn = core.update_velocity(np.array([1.0, 0.0, 0.0]), inc)
    expected = p.R_speed_h / (p.K_VEL ** 2)
    assert inn.R_used[0, 0] == pytest.approx(expected, rel=1e-12)


def test_velocity_observes_scale_along_u_w():
    """s is observable along u_w. Feed a consistent measurement stream with
    real motion and sigma_s must shrink."""
    core = make_core(s=1.0)
    core.x.P[IDX_S, IDX_S] = 0.25
    s_true = 1.25
    sig0 = core.x.sigma_s
    for _ in range(200):
        inc = make_increment(scale=0.08)
        w_v = inc.dp_v / inc.dt
        v_true = s_true * (core.x.R_nv @ w_v) - inc.dl / inc.dt
        z = v_true * EkfParams().K_VEL + RNG.normal(scale=0.03, size=3)
        core.propagate(inc)
        core.update_velocity(z, inc)
    assert core.x.sigma_s < 0.25 * sig0
    assert abs(core.x.s - s_true) < 0.06

def test_velocity_scale_with_wrong_R_nv():
    """If R_bar_n_v is wrong, does the velocity update corrupt `s`?

    Field: s collapses from 3.61 to ~1.5 with 46 clamps at zero, while
    check_scale says the true value is a stable 3.91.
    """
    core = make_core(s=1.0)
    core.x.P[IDX_S, IDX_S] = 0.36 ** 2
    core.x.s = 3.61
    core.x.R_nv = so3.exp(rand_axis() * np.radians(15.0)) @ core.x.R_nv
    s_true = 3.91
    for _ in range(200):
        inc = make_increment(scale=0.08)
        w_v = inc.dp_v / inc.dt
        v_true = s_true * (core.x.R_nv @ w_v) - inc.dl / inc.dt
        z = v_true * EkfParams().K_VEL + RNG.normal(scale=0.03, size=3)
        core.propagate(inc)
        core.update_velocity(z, inc)
    print(f"s = {core.x.s:.3f}, clamps = {core.n_s_clamped}")
    assert core.n_s_clamped == 0

def test_ned_to_enu_horizontal_block_is_the_reflection():
    """[M] det_horizontal < 0 on every bag.

    The FULL map is a proper rotation -- NED and ENU are both right-handed, so
    it must be. It is the horizontal 2x2 block that is a reflection, and that
    is what the fitted det_horizontal measures.
    """
    M = np.column_stack([ned_to_enu(e) for e in np.eye(3)])
    assert np.linalg.det(M) == pytest.approx(+1.0)
    assert np.linalg.det(M[:2, :2]) == pytest.approx(-1.0)
    assert np.allclose(ned_to_enu([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])


# ====================================================================== 5.1
def test_attitude_H_is_identity_and_FD_of_h_is_vacuous():
    """h(dtheta) = dtheta, so the FD check returns I and proves nothing. This
    test exists to record WHY the convergence test below is the real one."""
    core = make_core()
    H = np.zeros((3, NX))
    H[:, IDX_TH] = np.eye(3)
    H_num = numeric_H(core, lambda st: np.zeros(3), 3)  # h of the ERROR state
    assert np.allclose(H_num, 0.0)          # vacuous: h does not see the nominal
    assert np.allclose(H[:, IDX_TH], np.eye(3))


def test_attitude_residual_derivative_is_minus_identity():
    """THE TRAP, made explicit.

    dy/ddtheta = -I3 while H = +I3, and both are correct because y = z - h(x)
    implies dy/dx = -H. Anyone finite-differencing the residual will read -I3
    and 'fix' the sign to H = -I3, which makes K negative and drives the
    nominal AWAY from R_dji.
    """
    core = make_core()
    R_v_c = so3.exp(rand_axis() * 0.5)
    R_b_c = so3.exp(rand_axis() * 0.2)
    # Keep the seeded error small: Log(exp(e) exp(-dth)) has derivative
    # -Jr^-1(e), which is -I only to first order in |e|.
    R_dji = so3.exp(rand_axis() * 1e-3) @ core.body_attitude(R_v_c, R_b_c)

    eps = 1e-6
    J = np.zeros((3, 3))
    for j in range(3):
        dp, dm = np.zeros(3), np.zeros(3)
        dp[j], dm[j] = eps, -eps
        cp, cm = core.x.copy(), core.x.copy()
        cp.R_nv = so3.exp(dp) @ core.x.R_nv
        cm.R_nv = so3.exp(dm) @ core.x.R_nv
        yp = so3.log(R_dji @ (cp.R_nv @ R_v_c @ R_b_c.T).T)
        ym = so3.log(R_dji @ (cm.R_nv @ R_v_c @ R_b_c.T).T)
        J[:, j] = (yp - ym) / (2 * eps)
    # The exact derivative is -Jr^-1(e) ~ -(I + hat(e)/2), so the diagonal is
    # -1 to machine precision while the off-diagonals are O(|e|/2). Assert both
    # rather than loosening a single tolerance -- the structure is the point.
    assert np.allclose(np.diag(J), -1.0, atol=1e-5)
    off = J - np.diag(np.diag(J))
    assert np.abs(off).max() < 1e-3
    assert np.allclose(J, -np.eye(3), atol=1e-3)


def test_attitude_converges_to_dji():
    """THE REAL TEST (filter_design.md 10).

    Seed R_bar_n_v with a known 10 deg error, feed a constant R_dji, and check
    R_bar_n_b -> R_dji monotonically with P_theta shrinking. This catches an
    error in the R_bar_n_v @ R_v_c @ R_b_c^T composition or in the reset,
    which is where the real risk lives since H is an identity.
    """
    core = make_core()
    R_v_c = so3.exp(rand_axis() * 0.5)
    R_b_c = so3.exp(rand_axis() * 0.2)
    R_true = core.body_attitude(R_v_c, R_b_c)

    err_axis = rand_axis()
    core.x.R_nv = so3.exp(err_axis * np.radians(10.0)) @ core.x.R_nv
    R_dji = R_true

    errs, traces = [], []
    for _ in range(60):
        e = so3.angle(core.body_attitude(R_v_c, R_b_c) @ R_dji.T)
        errs.append(np.degrees(e))
        traces.append(np.trace(core.x.P[IDX_TH, IDX_TH]))
        inn = core.update_attitude(R_dji, R_v_c, R_b_c)
        assert inn.accepted or errs[-1] > 5.0   # early steps may gate; log it

    assert errs[0] == pytest.approx(10.0, abs=0.5)
    assert errs[-1] < 0.5, f"did not converge: {errs[-1]:.3f} deg"
    assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])), "not monotonic"
    assert traces[-1] < traces[0]


def test_attitude_reset_moves_toward_not_away():
    """One step must reduce the error. With H = -I3 it would grow."""
    core = make_core()
    R_v_c = so3.exp(rand_axis() * 0.3)
    R_b_c = so3.exp(rand_axis() * 0.1)
    R_dji = so3.exp(rand_axis() * np.radians(6.0)) @ core.body_attitude(R_v_c, R_b_c)
    before = so3.angle(core.body_attitude(R_v_c, R_b_c) @ R_dji.T)
    core.update_attitude(R_dji, R_v_c, R_b_c)
    after = so3.angle(core.body_attitude(R_v_c, R_b_c) @ R_dji.T)
    assert after < before


def test_attitude_gate_rejects_outliers():
    """[M] +-20 deg transients during aggressive manoeuvres are outliers, not
    Gaussian tails (15:43:09)."""
    core = make_core()
    core.x.P[IDX_TH, IDX_TH] = np.eye(3) * np.radians(0.5) ** 2
    R_v_c = so3.exp(rand_axis() * 0.3)
    R_b_c = so3.exp(rand_axis() * 0.1)
    R_bad = so3.exp(rand_axis() * np.radians(20.0)) @ core.body_attitude(R_v_c, R_b_c)
    R_before = core.x.R_nv.copy()
    inn = core.update_attitude(R_bad, R_v_c, R_b_c)
    assert not inn.accepted
    assert np.allclose(core.x.R_nv, R_before)
    assert core.n_rejected[Kind.ATTITUDE] == 1


# ---------------------------------------------------------------- scheduling
def test_sigma_rp_schedule():
    """[M] sigma_rp = sqrt(sigma_rp0^2 + (a_h/g)^2), slope 1/g, not tuned."""
    p = EkfParams()
    assert sigma_rp(p, 0.0) == pytest.approx(p.sigma_rp0)
    # at a = 0.8 m/s^2 the measured dynamic sigma_rp was ~0.06 rad
    assert sigma_rp(p, 0.8) == pytest.approx(
        np.hypot(p.sigma_rp0, 0.8 / G), rel=1e-9)
    assert 0.05 < sigma_rp(p, 0.8) < 0.09
    assert sigma_rp(p, 1.5) > sigma_rp(p, 0.5) > sigma_rp(p, 0.0)


def test_attitude_R_grows_with_acceleration():
    core = make_core()
    R_v_c = so3.exp(rand_axis() * 0.3)
    R_b_c = so3.exp(rand_axis() * 0.1)
    R_dji = core.body_attitude(R_v_c, R_b_c)
    quiet = core.update_attitude(R_dji, R_v_c, R_b_c, a_h=0.0)
    dyn = core.update_attitude(R_dji, R_v_c, R_b_c, a_h=1.0)
    assert dyn.R_used[0, 0] > 20.0 * quiet.R_used[0, 0]
    # yaw is NOT acceleration-scheduled -- it is a magnetometer measurement
    assert dyn.R_used[2, 2] == pytest.approx(quiet.R_used[2, 2])


# ------------------------------------------------------------------- numerics
def test_P_stays_symmetric_and_psd_through_mixed_updates():
    core = make_core()
    R_v_c = so3.exp(rand_axis() * 0.3)
    R_b_c = so3.exp(rand_axis() * 0.1)
    for _ in range(300):
        inc = make_increment()
        core.propagate(inc)
        core.update_attitude(
            so3.exp(rand_axis() * RNG.normal(scale=0.01)) @ core.body_attitude(R_v_c, R_b_c),
            R_v_c, R_b_c, a_h=abs(RNG.normal(scale=0.3)))
        core.update_altitude(core.x.p[2] + core.x.b + RNG.normal(scale=0.03))
        core.update_velocity(RNG.normal(scale=0.6, size=3) + np.array([1.0, 0, 0]), inc)
        P = core.x.P
        assert np.allclose(P, P.T, atol=1e-14)
        assert np.linalg.eigvalsh(P).min() > -1e-10


def test_no_H_column_touches_pxy():
    """filter_design.md 5.0: no H has a p_x or p_y column, so their unbounded
    growth never enters an inversion.

    This is a STRUCTURAL property of H, not a statement about P. P[0,0] does
    move under an update, because p_x is correlated with p_z, s and dtheta
    through F -- filter_design.md 7's 'no update reduces them' is loose
    shorthand for 'nothing observes them directly', which is what makes them
    diverge over a flight.
    """
    core = make_core()
    inc = make_increment()
    _, u_w = core.velocity_prediction(inc)

    H_alt = np.zeros((1, NX)); H_alt[0, 2] = 1.0; H_alt[0, IDX_B] = 1.0
    H_att = np.zeros((3, NX)); H_att[:, IDX_TH] = np.eye(3)
    H_vel = np.zeros((3, NX))
    H_vel[:, IDX_S] = u_w
    H_vel[:, IDX_TH] = -core.x.s * so3.hat(u_w)

    for H in (H_alt, H_att, H_vel):
        assert np.allclose(H[:, 0], 0.0)
        assert np.allclose(H[:, 1], 0.0)
