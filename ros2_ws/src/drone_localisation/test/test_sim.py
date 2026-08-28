"""
test_sim.py -- step 4: end-to-end validation against synthetic VO.

filter_design.md 10: "Validate propagation with synthetic VO first. With real
images a VO error and a filter error are indistinguishable; with a known
injected scale and a known non-gravity-aligned twist they are not."

Every scenario injects s_true = 1.37 and a 6 deg VO-frame twist, and generates
measurements with the behaviours actually measured on the aircraft: K_VEL,
0.1 m and 0.1 m/s quantisation, barometric random walk on b, and roll/pitch
that UNDER-REPORTS tilt by ~atan(a_h/g).

Run:  pytest -q tests/test_sim.py
"""

import dataclasses

import numpy as np
import pytest
import dataclasses

from drone_localisation.ekf import so3
from drone_localisation.ekf import sim
from drone_localisation.ekf.params import EkfParams
from drone_localisation.ekf.filter import EkfCore, Kind

DUR = 80.0
S_TRUE = 1.37


def run_profile(profile, params=None, duration=DUR, s_guess=1.0, **kw):
    steps, truth = sim.simulate(profile, duration=duration)
    # These tests exercise the SCALE-ESTIMATING path. The node ships with
    # estimate_scale = 0 (s held per flight, filter_design.md 5.3 revised),
    # so the tests must turn it back on or every scale assertion is vacuous.
    if params is None:
        params = dataclasses.replace(params or EkfParams(), estimate_scale=1.0)
    core = EkfCore(params)
    sim.init_core(core, steps[0], s_guess=s_guess, p0=truth["p"][1])
    log = sim.run(core, steps, **kw)
    return core, log, truth


def scale_err(core):
    return core.x.s / S_TRUE - 1.0


# ==================================================================== recovery
def test_box_recovers_scale():
    """Real motion at ~0.8 m/s: the design case for observing `s`."""
    core, log, truth = run_profile("box")
    assert abs(scale_err(core)) < 0.03, f"s = {core.x.s:.4f}"
    assert core.x.sigma_s < 0.02


def test_box_recovers_the_vo_frame_twist():
    """R_bar_n_v must converge to the injected 6 deg non-gravity-aligned frame.

    This is what filter_design.md 2 exists for: an unmodelled tilt leaks
    horizontal motion into p_z, and the altitude update can only absorb that by
    biasing `s` -- self-consistently, so NEES would not reveal it.
    """
    core, log, truth = run_profile("box")
    err_deg = np.degrees(so3.angle(core.x.R_nv @ truth["R_n_v"].T))
    assert err_deg < 0.5, f"R_n_v error {err_deg:.3f} deg"


def test_box_position_error_is_small():
    core, log, truth = run_profile("box")
    err = np.linalg.norm(log[-1]["p"] - log[-1]["p_true"])
    path = np.sum(np.linalg.norm(np.diff(truth["p"], axis=0), axis=1))
    assert err < 0.15, f"{err:.3f} m over {path:.1f} m of path"


def test_b_tracks_the_barometric_random_walk():
    """[M] q_b is NOT zero: 0.2-0.4 m of drift over 175 s on both F2 takes.
    With q_b = 0 the filter would lock b at its prior and push the drift into
    p_z, and from there into `s` through the p_z-s cross-covariance."""
    core, log, truth = run_profile("box")
    assert abs(core.x.b - log[-1]["b_true"]) < 0.05, (
        f"b = {core.x.b:+.4f} vs truth {log[-1]['b_true']:+.4f}")
    drift = abs(truth["b"][-1] - truth["b"][0])
    assert drift > 0.03, "scenario did not actually drift; test is vacuous"


def test_b_would_not_track_with_q_b_zero():
    """The counterfactual, so the q_b finding is pinned by a test."""
    p = dataclasses.replace(EkfParams(), q_b=0.0, estimate_scale=1.0)
    core, log, _ = run_profile("box", params=p)
    assert abs(core.x.b - log[-1]["b_true"]) > 0.03


# =============================================================== observability
def test_scale_is_unobservable_in_hover():
    """filter_design.md 7: `s` is degenerate without motion. The filter must
    SAY so -- sigma_s staying wide is the correct behaviour, not a failure."""
    core, log, _ = run_profile("hover")
    assert core.x.sigma_s > 0.5, f"sigma_s = {core.x.sigma_s:.4f}"
    p = EkfParams()
    assert core.x.sigma_s > p.SIGMA_S_MAX, (
        "degradation gate would not fire on an unobservable scale")


def test_all_velocity_rejected_in_hover():
    core, log, _ = run_profile("hover")
    assert core.n_rejected[Kind.VELOCITY] > 700


def test_yaw_row_makes_dtheta_z_observable_without_translation():
    """filter_design.md 7: with the attitude yaw row enabled, dtheta_z is
    observed in hover -- exactly the "rotate to frame a defect" condition that
    was previously a hole."""
    core, log, truth = run_profile("hover")
    err_deg = np.degrees(so3.angle(core.x.R_nv @ truth["R_n_v"].T))
    assert err_deg < 1.0, f"R_n_v error {err_deg:.3f} deg with no translation"
    p_yaw = core.x.P[7, 7]
    assert p_yaw < EkfParams().P0_theta_z / 10.0


def test_altitude_alone_does_not_pin_scale():
    """Ablation: with velocity off, `s` must not converge. Only velocity's
    Jacobian touches s directly (filter_design.md 12); altitude reaches it
    through the p_z-s cross-covariance, entangled with b."""
    core, log, _ = run_profile("box", use_velocity=False)
    assert core.x.sigma_s > 0.1, f"sigma_s = {core.x.sigma_s:.4f}"


def test_s_b_covariance_is_tracked_and_nonzero():
    """filter_design.md 7: monitor the s-b off-diagonal, not just marginals."""
    core, log, _ = run_profile("vertical")
    assert abs(core.x.cov_s_b) > 1e-6


# ================================================== the low-speed scale problem
def test_pure_inspection_biases_scale_and_understates_it():
    """[M] The important negative result.

    At ~0.3 m/s the 0.1 m/s quantum is a dead zone, so the velocity error is a
    BIAS, not noise -- gain_horizontal measured 0.82 (F6) and 0.77 (F0) against
    0.93 at 1 m/s. Inflating R by V_INFL slows the filter down but cannot
    remove a bias, so `s` ends up wrong AND confident.

    This is why filter_design.md 7 says `s` is estimated on transits and held
    through passes, and why 9 gates readiness on sigma_s < SIGMA_S_OK.
    """
    core, log, _ = run_profile("inspection")
    err = abs(scale_err(core))
    assert err > 0.02, "scenario no longer reproduces the low-speed bias"
    # and the filter does not know it is wrong
    assert core.x.sigma_s < err * S_TRUE / 2.0, (
        "overconfidence has gone away -- revisit V_INFL")


def test_transit_then_inspect_fixes_scale():
    """The mitigation, end to end: a transit above V_LOW pins `s`, and it is
    then held through the slow pass."""
    core, log, _ = run_profile("mixed")
    assert abs(scale_err(core)) < 0.02, f"s = {core.x.s:.4f}"
    err = np.linalg.norm(log[-1]["p"] - log[-1]["p_true"])
    assert err < 0.15


def test_raising_V_MIN_to_V_LOW_is_worse_not_better():
    """Rejecting instead of inflating below V_LOW removes ALL scale
    information from a slow flight, which is worse than a biased estimate:
    12.8 m of position error against 1.2 m in the scenario as tuned.

    Recorded as a test so the tempting 'just reject harder' fix stays
    rejected.
    """
    strict = dataclasses.replace(EkfParams(), V_MIN=0.40, estimate_scale=1.0)
    c_strict, l_strict, _ = run_profile("inspection", params=strict)
    c_base, l_base, _ = run_profile("inspection")
    e_strict = np.linalg.norm(l_strict[-1]["p"] - l_strict[-1]["p_true"])
    e_base = np.linalg.norm(l_base[-1]["p"] - l_base[-1]["p_true"])
    assert e_strict > 3.0 * e_base
    assert c_strict.x.sigma_s > 0.5      # at least it says it does not know


# ======================================================================= gating
def test_quantisation_aware_altitude_gate_accepts_bin_steps():
    """A 0.1 m altitude bin step is 3.4 sigma. chi2(1, 0.99) = 6.63 would
    reject exactly the informative samples while keeping the uninformative
    repeats -- 22% rejection measured before the gate was widened."""
    core, log, _ = run_profile("box")
    assert core.n_rejected[Kind.ALTITUDE] == 0


def test_tight_altitude_gate_would_reject_informative_samples():
    """The counterfactual, pinned."""
    tight = dataclasses.replace(EkfParams(), NIS_GATE_ALT=6.63, estimate_scale=1.0)
    core, log, _ = run_profile("box", params=tight)
    assert core.n_rejected[Kind.ALTITUDE] > 100


def test_attitude_gate_catches_acceleration_transients_only_rarely():
    """The scheduled R should absorb the acceleration error, so rejections
    stay rare -- if this climbs, sigma_rp0 or the slope is wrong."""
    core, log, _ = run_profile("box")
    assert core.n_rejected[Kind.ATTITUDE] < 0.05 * len(log)


def test_unscheduled_sigma_rp_rejects_far_more():
    """Without the acceleration schedule, the under-reported tilt looks like a
    stream of outliers."""
    flat = dataclasses.replace(EkfParams(), accel_slope=0.0, estimate_scale=1.0)
    core, log, _ = run_profile("box", params=flat)
    base, _, _ = run_profile("box")
    assert core.n_rejected[Kind.ATTITUDE] > 3 * max(
        base.n_rejected[Kind.ATTITUDE], 1)


# ================================================================== consistency
def test_covariance_stays_psd_over_a_long_run():
    core, log, _ = run_profile("mixed", duration=150.0)
    P = core.x.P
    assert np.allclose(P, P.T, atol=1e-12)
    assert np.linalg.eigvalsh(P).min() > -1e-9


def test_pxy_diverges_over_a_long_run():
    """filter_design.md 7: nothing observes p_x, p_y directly, so a fixed
    covariance gate fires on a healthy flight.

    Must be a MOVING profile. In hover the VO increments are near zero, so the
    s^2 * R Sigma R^T term contributes almost nothing and P_xx grows only ~4x
    over 150 s -- the divergence is driven by scale and tilt uncertainty
    multiplying real increments, not by time passing.
    """
    core, log, _ = run_profile("box", duration=150.0)
    assert core.x.P[0, 0] + core.x.P[1, 1] > 4.0 * EkfParams().P0_pos


def test_deterministic():
    """Same inputs -> same outputs, bit for bit. Required for bag replay."""
    a, la, _ = run_profile("box")
    b, lb, _ = run_profile("box")
    assert a.x.s == b.x.s
    assert np.array_equal(a.x.p, b.x.p)
    assert np.array_equal(a.x.P, b.x.P)
