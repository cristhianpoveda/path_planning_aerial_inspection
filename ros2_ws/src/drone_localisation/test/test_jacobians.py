"""
test_jacobians.py -- finite-difference verification of F (filter_design.md 4.2).

The rule from filter_design.md 10, and the reason this file exists before any
flight data is touched: a wrong sign in the dtheta column converges plausibly
and is wrong, and NEES will not reveal it because the result is
self-consistent.

Two disciplines, both non-negotiable:

  1. Finite-difference the PROPAGATION MAP, never a residual. For updates this
     means differencing h(x), never y = z - h(x); differencing y returns -H and
     will "confirm" the wrong sign.

  2. Perturb dtheta through exp(hat(dtheta)) @ R_bar, never by adding to a
     vector. The error state is a rotation, not three loose numbers.

Run:  pytest -q tests/test_jacobians.py
"""

import numpy as np
import pytest

from drone_localisation.ekf import so3
from drone_localisation.ekf.params import EkfParams
from drone_localisation.ekf.filter import EkfCore, Increment, State, NX, IDX_P, IDX_S, IDX_B, IDX_TH

RNG = np.random.default_rng(1)


def rand_axis():
    a = RNG.normal(size=3)
    return a / np.linalg.norm(a)


def make_core(seed_scale=1.3):
    p = EkfParams()
    core = EkfCore(p)
    core.x.p = RNG.normal(size=3)
    core.x.s = seed_scale
    core.x.b = 0.05
    core.x.R_nv = so3.exp(rand_axis() * 0.4)   # deliberately not identity
    return core


def make_increment(dt=0.1):
    return Increment(
        dp_v=RNG.normal(scale=0.05, size=3),
        dl=RNG.normal(scale=0.002, size=3),
        dt=dt,
        Sigma_v=np.diag(RNG.uniform(1e-5, 1e-4, size=3)),
    )


def propagate_perturbed(core, inc, dx):
    """Apply an error-state perturbation, propagate, return the output error.

    This is the map whose Jacobian F must equal. The nominal is perturbed the
    way the error state is DEFINED to perturb it:

        p     -> p + dp
        s     -> s + ds
        b     -> b + db
        R_n_v -> exp(hat(dtheta)) @ R_bar_n_v      <-- on the LEFT, in `n`
    """
    x0 = core.x
    pert = x0.copy()
    pert.p = x0.p + dx[IDX_P]
    pert.s = x0.s + dx[IDX_S]
    pert.b = x0.b + dx[IDX_B]
    pert.R_nv = so3.exp(dx[IDX_TH]) @ x0.R_nv

    # propagate the perturbed nominal by hand (no covariance needed)
    u_pert = pert.R_nv @ inc.dp_v
    p_out = pert.p + pert.s * u_pert - inc.dl
    s_out, b_out, R_out = pert.s, pert.b, pert.R_nv

    # propagate the unperturbed nominal
    u0 = x0.R_nv @ inc.dp_v
    p_nom = x0.p + x0.s * u0 - inc.dl
    s_nom, b_nom, R_nom = x0.s, x0.b, x0.R_nv

    # express the result as an error state again
    out = np.zeros(NX)
    out[IDX_P] = p_out - p_nom
    out[IDX_S] = s_out - s_nom
    out[IDX_B] = b_out - b_nom
    out[IDX_TH] = so3.log(R_out @ R_nom.T)
    return out


def numeric_F(core, inc, eps=1e-7):
    F = np.zeros((NX, NX))
    for j in range(NX):
        dp = np.zeros(NX)
        dp[j] = eps
        dm = np.zeros(NX)
        dm[j] = -eps
        F[:, j] = (propagate_perturbed(core, inc, dp)
                   - propagate_perturbed(core, inc, dm)) / (2.0 * eps)
    return F


# ------------------------------------------------------------------------ F
def test_F_matches_finite_difference():
    for _ in range(25):
        core = make_core()
        inc = make_increment()
        F_ana, _ = core.transition(inc)
        F_num = numeric_F(core, inc)
        assert np.allclose(F_ana, F_num, atol=1e-6), (
            f"max |dF| = {np.abs(F_ana - F_num).max():.3e}\n"
            f"analytic:\n{F_ana}\nnumeric:\n{F_num}")


def test_F_scale_column_is_u():
    """dp/ds = u = R_bar_n_v @ dp_v, exactly."""
    core = make_core()
    inc = make_increment()
    F, u = core.transition(inc)
    assert np.allclose(F[IDX_P, IDX_S], u)
    assert np.allclose(u, core.x.R_nv @ inc.dp_v)


def test_F_theta_block_sign():
    """dp/dtheta = -s * hat(u).

    The sign trap: exp(hat(dth)) @ u ~ u - hat(u) @ dth, so the derivative of
    position w.r.t. dtheta is NEGATIVE s*hat(u). Getting this backwards yields
    a filter that converges and is wrong.
    """
    core = make_core()
    inc = make_increment()
    F, u = core.transition(inc)
    assert np.allclose(F[IDX_P, IDX_TH], -core.x.s * so3.hat(u), atol=1e-12)

    # and confirm the sign directly from the definition
    dth = np.array([1e-6, 0.0, 0.0])
    lhs = (so3.exp(dth) @ u - u) / 1e-6
    rhs = -so3.hat(u) @ (dth / 1e-6)
    assert np.allclose(lhs, rhs, atol=1e-5)


def test_F_identity_blocks():
    core = make_core()
    F, _ = core.transition(make_increment())
    assert np.allclose(F[IDX_P, IDX_P], np.eye(3))
    assert F[IDX_S, IDX_S] == pytest.approx(1.0)
    assert F[IDX_B, IDX_B] == pytest.approx(1.0)
    assert np.allclose(F[IDX_TH, IDX_TH], np.eye(3))
    # s, b and dtheta are untouched by propagation
    assert np.allclose(F[IDX_S, :3], 0.0)
    assert np.allclose(F[IDX_B, :], np.eye(NX)[IDX_B, :])
    assert np.allclose(F[IDX_TH, IDX_P], 0.0)


# ------------------------------------------------------------------ propagate
def test_propagate_moves_position_by_scaled_increment():
    core = make_core()
    inc = make_increment()
    p0 = core.x.p.copy()
    u = core.x.R_nv @ inc.dp_v
    core.propagate(inc)
    assert np.allclose(core.x.p, p0 + core.x.s * u - inc.dl)


def test_propagate_leaves_s_b_R_alone():
    core = make_core()
    s0, b0, R0 = core.x.s, core.x.b, core.x.R_nv.copy()
    core.propagate(make_increment())
    assert core.x.s == s0
    assert core.x.b == b0
    assert np.allclose(core.x.R_nv, R0)


def test_covariance_stays_symmetric_and_psd():
    core = make_core()
    for _ in range(200):
        core.propagate(make_increment())
        P = core.x.P
        assert np.allclose(P, P.T, atol=1e-15)
        w = np.linalg.eigvalsh(P)
        assert w.min() > -1e-12, f"P lost PSD: min eig {w.min():.3e}"


def test_position_covariance_grows_without_bound():
    """filter_design.md 7: nothing observes p_x, p_y, so P_xx and P_yy grow
    without bound and MUST NOT be used as a degradation gate.

    Note the growth is NOT monotonic step to step. The first row of F is
    [I | u | 0 | -s*hat(u)], so P+[0,0] is the variance of a combination of
    dp_x, ds and dtheta -- a negative correlation between them can shrink it
    for a step. The claim being tested, and the one 7 relies on, is that no
    update reduces it, so it diverges over a flight.
    """
    core = make_core()
    p0 = core.x.P[0, 0]
    COV_MAX = 1.0                       # a plausible gate: sigma_x = 1 m
    fired_at = None
    for k in range(2000):
        core.propagate(make_increment())
        if fired_at is None and core.x.P[0, 0] > COV_MAX:
            fired_at = k
    assert core.x.P[0, 0] > 100.0 * p0
    # The design consequence: a COV_MAX gate fires partway through a perfectly
    # healthy flight. At ~10 Hz this is a couple of minutes.
    assert fired_at is not None and fired_at < 2000, (
        "P_xx failed to exceed a fixed gate -- 7's argument would not hold")


def test_dead_reckon_moves_and_inflates():
    core = make_core()
    p0 = core.x.p.copy()
    tr0 = np.trace(core.x.P[IDX_P, IDX_P])
    v = np.array([0.3, -0.1, 0.0])
    core.dead_reckon(v, 0.1)
    assert np.allclose(core.x.p, p0 + v * 0.1)
    assert np.trace(core.x.P[IDX_P, IDX_P]) > tr0


# ----------------------------------------------------------------- reset
def test_reset_moves_nominal_and_is_a_rotation():
    core = make_core()
    R0 = core.x.R_nv.copy()
    dth = np.array([0.01, -0.02, 0.005])
    core.reset_error_state(dth)
    assert np.allclose(core.x.R_nv, so3.exp(dth) @ R0, atol=1e-9)
    assert np.allclose(core.x.R_nv @ core.x.R_nv.T, np.eye(3), atol=1e-12)


def test_reset_composes_left_not_right():
    """The error rotation is applied in the NAV frame, on the left. Applying it
    on the right silently gives a different filter."""
    core = make_core()
    R0 = core.x.R_nv.copy()
    dth = np.array([0.0, 0.0, 0.3])
    core.reset_error_state(dth)
    assert not np.allclose(core.x.R_nv, R0 @ so3.exp(dth), atol=1e-6)
