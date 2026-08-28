"""
test_scale_hold.py -- pins two findings that cost a debugging session each.

1. `estimate_scale = 0` does not hold `s` unless the covariance holds it too.
2. `p` must be datumed against the altitude measurement at init, because
   altitude is takeoff-relative and init happens mid-flight.
"""

import dataclasses

import numpy as np

from drone_localisation.ekf import sim
from drone_localisation.ekf.params import EkfParams
from drone_localisation.ekf.filter import EkfCore, Kind
from drone_localisation.ekf.frontend import Scheduler

S_TRUE = 1.37


def _run(params, s_guess=1.0, p0_mode="true", profile="mixed", duration=80.0):
    steps, truth = sim.simulate(profile, duration=duration)
    core = EkfCore(params)
    p0 = truth["p"][1] if p0_mode == "true" else np.zeros(3)
    sim.init_core(core, steps[0], s_guess=s_guess, p0=p0)
    if p0_mode == "datum":
        core.x.p[2] = steps[0].z_alt - params.b_prior_mean
    log = sim.run_frontend(Scheduler(params, core), steps)
    return core, log, truth


# ============================================================= the scale hold
def test_altitude_drags_s_when_only_the_jacobian_is_zeroed():
    """The defect, pinned. estimate_scale = 0 zeroes H[:, s] in the VELOCITY
    update only. Altitude still reaches `s` through the p_z-s cross-covariance
    that propagation creates, entangled with `b` -- exactly the path
    test_sim.test_altitude_alone_does_not_pin_scale says must not pin scale.

    Seeded at truth, so any movement is the leak and nothing else.
    """
    p = dataclasses.replace(EkfParams(), estimate_scale=0.0)
    core = EkfCore(p)
    steps, truth = sim.simulate("mixed", duration=80.0)
    core.x.R_nv = sim.so3.normalise(
        steps[0].R_dji @ steps[0].R_b_c @ steps[0].R_v_c.T)
    core.x.p = truth["p"][1].copy()
    core.x.s = S_TRUE                      # NOT hold_scale(): the old config
    sim.run_frontend(Scheduler(p, core), steps)
    assert abs(core.x.s / S_TRUE - 1.0) > 0.20, (
        "the leak has been fixed elsewhere -- delete this test, not the hold")


def test_hold_scale_makes_s_immovable():
    p = dataclasses.replace(EkfParams(), estimate_scale=0.0)
    core, log, _ = _run(p, s_guess=S_TRUE)
    assert core.x.s == S_TRUE              # bit-exact, not approx
    assert core.x.sigma_s == 0.0
    assert abs(core.x.cov_s_b) == 0.0


def test_hold_keeps_P_psd_and_the_gates_quiet():
    """P_ss = 0 must not make S singular or make sigma_s/s trip the 4.1
    plausibility gates into rejecting healthy increments."""
    p = dataclasses.replace(EkfParams(), estimate_scale=0.0)
    steps, truth = sim.simulate("mixed", duration=80.0)
    core = EkfCore(p)
    sim.init_core(core, steps[0], s_guess=S_TRUE, p0=truth["p"][1])
    sch = Scheduler(p, core)
    sim.run_frontend(sch, steps)
    assert np.linalg.eigvalsh(core.x.P).min() > -1e-12
    assert sch.builder.n_discont == 0


def test_a_held_s_is_only_as_good_as_its_seed():
    """The honest cost of holding: init measures s to ~3%, and 3% is then
    carried for the whole flight. Estimating beats it -- this is the number
    that says by how much."""
    p_hold = dataclasses.replace(EkfParams(), estimate_scale=0.0)
    p_est = dataclasses.replace(EkfParams(), estimate_scale=1.0)
    c_h, l_h, _ = _run(p_hold, s_guess=S_TRUE * 1.03)
    c_e, l_e, _ = _run(p_est, s_guess=1.0)
    e_h = np.linalg.norm(l_h[-1]["p"] - l_h[-1]["p_true"])
    e_e = np.linalg.norm(l_e[-1]["p"] - l_e[-1]["p_true"])
    assert e_h < 1.0, f"a 3% seed should still be usable: {e_h:.3f} m"
    assert e_e < e_h, "estimation should beat a 3%-seeded hold"


# ============================================================== the init datum
def test_zero_p_at_midflight_init_locks_out_altitude():
    """ekf_node._try_init does `x = State(p)`, which sets p = 0, but 9 gates
    init on 12 s of translation AFTER takeoff. Altitude is takeoff-relative,
    so the first residual is the whole height and NIS-rejects; p_z is observed
    by nothing else, so the rejection is self-sustaining."""
    p = dataclasses.replace(EkfParams(), estimate_scale=1.0)
    core, log, _ = _run(p, p0_mode="zero")
    assert core.n_rejected[Kind.ALTITUDE] > 100
    assert abs(log[-1]["p"][2] - log[-1]["p_true"][2]) > 1.0


def test_datuming_p_z_from_the_altitude_measurement_fixes_it():
    p = dataclasses.replace(EkfParams(), estimate_scale=1.0)
    core, log, _ = _run(p, p0_mode="datum")
    assert core.n_rejected[Kind.ALTITUDE] == 0
    assert abs(log[-1]["p"][2] - log[-1]["p_true"][2]) < 0.10