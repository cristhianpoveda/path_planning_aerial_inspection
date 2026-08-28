"""
test_frontend.py -- step 5: buffering, dedupe, re-stamping, ordering.

The frontend owns TIME. Every test here is about something the core cannot
see: interpolation, arrival order versus stamp order, repeated quantised
values, and the filter_design.md 6.1 reset policy.

Run:  pytest -q tests/test_frontend.py
"""

import dataclasses

import numpy as np
import pytest

from drone_localisation.ekf import so3
from drone_localisation.ekf import sim
from drone_localisation.ekf.params import EkfParams
from drone_localisation.ekf.filter import EkfCore, Kind, ned_to_enu
from drone_localisation.ekf.frontend import (AttitudeBuffer, AccelEstimator, EventQueue, IncrementBuilder, Scheduler, Context, Pose, VoStatus, Flags, VELOCITY, ALTITUDE, ATTITUDE)

RNG = np.random.default_rng(11)
S_TRUE = 1.37


def rand_axis():
    a = RNG.normal(size=3)
    return a / np.linalg.norm(a)


# ============================================================ AttitudeBuffer
def test_buffer_slerps_between_samples():
    buf = AttitudeBuffer()
    axis = rand_axis()
    buf.push(0.0, np.eye(3))
    buf.push(1.0, so3.exp(axis * 1.0))
    mid = buf.at(0.5)
    assert so3.angle(mid) == pytest.approx(0.5, abs=1e-9)


def test_buffer_refuses_to_extrapolate():
    """filter_design.md 4.1 step 4: true interpolation, no ZOH fallback."""
    buf = AttitudeBuffer()
    buf.push(1.0, np.eye(3))
    buf.push(2.0, so3.exp(rand_axis() * 0.3))
    assert buf.at(0.5) is None
    assert buf.at(2.5) is None
    assert buf.at(1.5) is not None


def test_buffer_drops_out_of_order():
    buf = AttitudeBuffer()
    assert buf.push(1.0, np.eye(3))
    assert not buf.push(0.9, np.eye(3))
    assert len(buf) == 1


def test_buffer_omega():
    buf = AttitudeBuffer()
    buf.push(0.0, np.eye(3))
    buf.push(0.5, so3.exp(np.array([0.0, 0.0, 0.25])))
    assert buf.omega(0.0, 0.5) == pytest.approx(0.5, rel=1e-6)


def test_buffer_handles_quaternion_sign_flips():
    """q and -q are the same rotation; naive lerp between them passes through
    zero. This is the mocap bug that cost weeks -- it must not recur here."""
    buf = AttitudeBuffer()
    R0 = so3.exp(rand_axis() * 0.2)
    R1 = so3.exp(rand_axis() * 0.4)
    buf.push(0.0, R0)
    buf.push(1.0, R1)
    mid = buf.at(0.5)
    assert np.allclose(mid @ mid.T, np.eye(3), atol=1e-10)
    assert so3.angle(mid @ R0.T) < so3.angle(R1 @ R0.T) + 1e-9


# =========================================================== AccelEstimator
def test_accel_estimator_recovers_constant_acceleration():
    est = AccelEstimator(window_s=0.3)
    for k in range(10):
        t = k * 0.1
        est.push(t, np.array([0.5 * t, 0.0, 0.0]))     # a_x = 0.5 m/s^2
    assert est.value() == pytest.approx(0.5, rel=0.05)


def test_accel_estimator_ignores_vertical():
    """Only the HORIZONTAL component tilts the aircraft."""
    est = AccelEstimator()
    for k in range(10):
        est.push(k * 0.1, np.array([0.0, 0.0, k * 0.1]))
    assert est.value() == pytest.approx(0.0, abs=1e-9)


# ================================================================ EventQueue
def test_queue_dedupes_repeated_quantised_values():
    """[M] In hover altitude repeats one 0.1 m bin ~10x/s. Treating those as
    independent samples shrinks sigma_pz by sqrt(N)."""
    q = EventQueue(EkfParams())
    admitted = sum(q.push(ALTITUDE, 0.1 * k, [1.5]) for k in range(20))
    assert admitted == 2                     # t=0 and one T_HOLD expiry at 1.0
    assert q.n_deduped[ALTITUDE] == 18


def test_queue_admits_on_value_change():
    q = EventQueue(EkfParams())
    assert q.push(ALTITUDE, 0.0, [1.5])
    assert not q.push(ALTITUDE, 0.1, [1.5])
    assert q.push(ALTITUDE, 0.2, [1.6])       # bin step -> informative
    assert q.push(ALTITUDE, 0.3, [1.5])


def test_queue_hold_timeout_readmits():
    p = dataclasses.replace(EkfParams(), T_HOLD=0.5)
    q = EventQueue(p)
    q.push(ALTITUDE, 0.0, [1.5])
    assert not q.push(ALTITUDE, 0.3, [1.5])
    assert q.push(ALTITUDE, 0.6, [1.5])       # prior sigma has grown; admit


def test_queue_restamps_velocity_only():
    p = dataclasses.replace(EkfParams(), VEL_DELAY=0.075)
    q = EventQueue(p)
    q.push(VELOCITY, 1.000, [1.0, 0.0, 0.0])
    q.push(ALTITUDE, 1.000, [1.5])
    evs = q.drain_until(2.0)
    vel = [e for e in evs if e.kind == VELOCITY][0]
    alt = [e for e in evs if e.kind == ALTITUDE][0]
    assert vel.t == pytest.approx(0.925)
    assert vel.t_raw == pytest.approx(1.000)  # raw kept for re-identification
    assert alt.t == pytest.approx(1.000)


def test_queue_drains_in_stamp_order_not_arrival_order():
    q = EventQueue(EkfParams())
    q.push(ALTITUDE, 5.0, [1.5])
    q.push(ATTITUDE, 1.0, [0.0, 0.0, 0.0])
    q.push(VELOCITY, 3.0, [1.0, 0.0, 0.0])
    ts = [e.t for e in q.drain_until(10.0)]
    assert ts == sorted(ts)


def test_queue_drops_events_re_stamped_into_a_closed_interval():
    """filter_design.md 5.0: never reprocess a closed interval."""
    p = dataclasses.replace(EkfParams(), VEL_DELAY=0.5)
    q = EventQueue(p)
    q.push(ALTITUDE, 1.0, [1.5])
    q.drain_until(1.0)
    assert not q.push(VELOCITY, 1.2, [1.0, 0.0, 0.0])   # t_eff = 0.7 < 1.0
    assert q.n_late[VELOCITY] == 1


# ========================================================= IncrementBuilder
def _ctx(core, v=1.0):
    return Context(R_nv=core.x.R_nv, s_hat=core.x.s, sigma_s=core.x.sigma_s,
                   v_dji_mag=v)


def _builder(params=None):
    p = params or EkfParams()
    buf = AttitudeBuffer()
    for k in range(200):
        buf.push(k * 0.05, np.eye(3))
    return IncrementBuilder(p, buf), p


def test_builder_reanchors_on_first_pose():
    b, _ = _builder()
    core = EkfCore(EkfParams())
    inc, fl = b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core))
    assert inc is None and Flags.REANCHORED in fl
    inc, fl = b.push(1.1, Pose(np.eye(3), np.array([0.05, 0, 0])),
                     VoStatus(), _ctx(core))
    assert inc is not None and Flags.OK in fl


def test_builder_epoch_guard_blocks_propagation():
    """[M] F6 recorded 9 epoch increments. Differencing across one yields an
    increment corresponding to no physical motion, and because VO is the
    propagation input there is no R to reject it with."""
    b, _ = _builder()
    core = EkfCore(EkfParams())
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core))
    b.push(1.1, Pose(np.eye(3), np.array([0.05, 0, 0])), VoStatus(), _ctx(core))
    inc, fl = b.push(1.2, Pose(np.eye(3), np.array([9.0, 0, 0])),
                     VoStatus(vo_epoch=1), _ctx(core))
    assert inc is None and Flags.VO_DISCONTINUITY in fl
    assert b.n_discont == 1


def test_builder_dt_guard():
    b, p = _builder()
    core = EkfCore(EkfParams())
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core))
    inc, fl = b.push(1.0 + 2.0 * p.DT_MAX,
                     Pose(np.eye(3), np.array([0.05, 0, 0])),
                     VoStatus(), _ctx(core))
    assert inc is None and Flags.DT_FAIL in fl


def test_builder_clears_anchor_on_loss():
    """On loss the next pose may be in a NEW map, so re-anchoring is not
    enough -- the anchor must be discarded entirely."""
    b, _ = _builder()
    core = EkfCore(EkfParams())
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core))
    inc, fl = b.push(1.1, Pose(np.eye(3), np.zeros(3)),
                     VoStatus(tracking_state="RECENTLY_LOST", pose_valid=False),
                     _ctx(core))
    assert inc is None and Flags.VO_LOST in fl
    assert b._anchor is None


def test_builder_not_ready_before_first_ok():
    b, _ = _builder()
    core = EkfCore(EkfParams())
    inc, fl = b.push(1.0, Pose(np.eye(3), np.zeros(3)),
                     VoStatus(tracking_state="NOT_INITIALIZED",
                              pose_valid=False), _ctx(core))
    assert Flags.NOT_READY in fl


def test_builder_gate_v_rejects_implausible_increment():
    b, p = _builder()
    core = EkfCore(EkfParams())
    core.x.P[3, 3] = (0.5 * p.SIGMA_S_OK) ** 2       # s is "known"
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core, v=1.0))
    inc, fl = b.push(1.1, Pose(np.eye(3), np.array([2.0, 0, 0])),
                     VoStatus(), _ctx(core, v=1.0))
    assert inc is None and Flags.GATE_V_FAIL in fl


def test_builder_gate_v_disabled_while_scale_unknown():
    """filter_design.md 9: the gates are disabled until sigma_s < SIGMA_S_OK.
    Gating on an unknown scale would reject good data."""
    b, _ = _builder()
    core = EkfCore(EkfParams())                       # P_ss = 1.0, wide
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core, v=1.0))
    inc, fl = b.push(1.1, Pose(np.eye(3), np.array([2.0, 0, 0])),
                     VoStatus(), _ctx(core, v=1.0))
    assert inc is not None


def test_builder_lever_arm_uses_the_estimated_frame():
    """dl must be built from R_bar_n_v, not from truth -- the frontend has no
    truth. Rotating the estimate must change dl."""
    b, p = _builder()
    core = EkfCore(EkfParams())
    buf = b.att
    buf._t.clear(); buf._q.clear()
    for k in range(50):
        buf.push(k * 0.05, so3.exp(np.array([0.0, 0.0, 0.4 * k * 0.05])))
    ctx_a = Context(R_nv=np.eye(3), s_hat=1.0, sigma_s=1.0)
    ctx_b = Context(R_nv=so3.exp(np.array([0.0, 0.0, 1.0])), s_hat=1.0,
                    sigma_s=1.0)
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), ctx_a)
    inc_a, _ = b.push(1.1, Pose(so3.exp(np.array([0, 0, 0.05])),
                                np.array([0.05, 0, 0])), VoStatus(), ctx_a)
    b.reset_anchor()
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), ctx_b)
    inc_b, _ = b.push(1.1, Pose(so3.exp(np.array([0, 0, 0.05])),
                                np.array([0.05, 0, 0])), VoStatus(), ctx_b)
    assert not np.allclose(inc_a.dl, inc_b.dl)


def test_builder_noise_gain_rises_when_features_drop():
    """[M] map_points swung 68-337 handheld. g_feat = n_ref / n_map_points."""
    b, p = _builder()
    core = EkfCore(EkfParams())
    b.push(1.0, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core))
    rich, _ = b.push(1.1, Pose(np.eye(3), np.array([0.05, 0, 0])),
                     VoStatus(n_map_points=300), _ctx(core))
    b.reset_anchor()
    b.push(1.2, Pose(np.eye(3), np.zeros(3)), VoStatus(), _ctx(core))
    poor, _ = b.push(1.3, Pose(np.eye(3), np.array([0.05, 0, 0])),
                     VoStatus(n_map_points=60), _ctx(core))
    assert poor.Sigma_v[0, 0] > 3.0 * rich.Sigma_v[0, 0]


# ================================================================= Scheduler
def run_sched(profile, params=None, duration=80.0):
    p = params if params is not None else dataclasses.replace(
        EkfParams(), estimate_scale=1.0)
    steps, truth = sim.simulate(profile, duration=duration)
    core = EkfCore(p)
    sim.init_core(core, steps[0], s_guess=1.0, p0=truth["p"][1])
    sch = Scheduler(p, core)
    log = sim.run_frontend(sch, steps)
    return sch, core, log, truth


def test_scheduler_end_to_end_recovers_scale():
    """Through dedupe, re-stamping and VO-clocked ordering, with events in
    ARRIVAL order.

    The direct-core equivalent (test_sim.test_transit_then_inspect_fixes_scale)
    gets pos err < 0.15 m and R_n_v err < 0.5 deg on this same profile. Any
    large gap here is the FRONTEND, in a setting where truth is known exactly.
    """
    sch, core, log, truth = run_sched("mixed")
    err = np.linalg.norm(log[-1]["p"] - log[-1]["p_true"])
    err_deg = np.degrees(so3.angle(core.x.R_nv @ truth["R_n_v"].T))
    print(f"\n  s = {core.x.s:.4f} (true {S_TRUE})  "
          f"R_n_v err = {err_deg:.2f} deg  pos err = {err:.3f} m")
    assert abs(core.x.s / S_TRUE - 1.0) < 0.5, f"s = {core.x.s:.4f}"
    assert err < 0.10, f"{err:.3f} m"


def test_scheduler_recovers_the_frame_twist():
    sch, core, log, truth = run_sched("box")
    assert np.degrees(so3.angle(core.x.R_nv @ truth["R_n_v"].T)) < 0.5


def test_scheduler_dedupes_heavily_in_hover():
    """The dedupe must actually bite, or sigma_pz is a fiction."""
    sch, core, log, _ = run_sched("hover")
    assert sch.q.n_deduped[ALTITUDE] > 300
    assert sch.q.n_deduped[VELOCITY] > 300


def test_scheduler_drains_the_queue_completely():
    sch, core, log, _ = run_sched("box")
    assert len(sch.q) == 0
    assert sch.stats()["late"]["velocity"] <= 5


def test_scheduler_is_deterministic():
    """Required for bag replay (filter_design.md 10)."""
    a, ca, _, _ = run_sched("box")
    b, cb, _, _ = run_sched("box")
    assert ca.x.s == cb.x.s
    assert np.array_equal(ca.x.P, cb.x.P)


def test_pz_and_b_split_is_governed_by_Sigma_base():
    """[M] Sizing rule, not a free knob.

    Altitude observes p_z + b, so the SUM is well determined and the SPLIT is
    set by which state has more process noise. If s^2 * Sigma_base >> q_b * dt,
    p_z absorbs the barometric drift and the reported altitude is wrong even
    though the residual is fine.
    """
    big = dataclasses.replace(EkfParams(), Sigma_base=np.eye(3) * 1e-4)
    _, c_big, l_big, _ = run_sched("mixed", params=big)
    _, c_ok, l_ok, _ = run_sched("mixed")

    for c, l in ((c_big, l_big), (c_ok, l_ok)):
        total = (c.x.b - l[-1]["b_true"]) + (l[-1]["p"][2] - l[-1]["p_true"][2])
        assert abs(total) < 0.05, "p_z + b should be observable in both cases"

    pz_big = abs(l_big[-1]["p"][2] - l_big[-1]["p_true"][2])
    pz_ok = abs(l_ok[-1]["p"][2] - l_ok[-1]["p_true"][2])
    assert pz_big > 5.0 * pz_ok


# ==================================================================== 6.1
def test_reset_policy_sim3_inflates_but_keeps_the_mean():
    """Monocular loop closure uses Sim(3) with FREE SCALE, so a correction can
    rescale the map with the state still at OK."""
    p = EkfParams()
    core = EkfCore(p)
    core.x.s = 1.4
    core.x.P[3, 3] = 0.01
    sch = Scheduler(p, core)
    sch._saw_uninit_since_epoch = False
    sch._apply_reset_policy()
    assert core.x.s == 1.4                    # mean kept
    assert core.x.P[3, 3] == pytest.approx(0.04)   # inflated


def test_reset_policy_rebuild_resets_scale():
    """A rebuild replaces the map frame AND scale outright, so holding the old
    s would apply the previous map's scale to a new one."""
    p = EkfParams()
    core = EkfCore(p)
    core.x.s = 1.4
    core.x.P[3, 3] = 0.01
    core.x.p = np.array([1.0, 2.0, 3.0])
    core.x.b = 0.07
    sch = Scheduler(p, core)
    sch._saw_uninit_since_epoch = True
    sch._apply_reset_policy()
    assert core.x.s == 1.0
    assert core.x.P[3, 3] == pytest.approx(p.P0_scale)
    # p and b survive both cases: they live in the nav frame
    assert np.allclose(core.x.p, [1.0, 2.0, 3.0])
    assert core.x.b == pytest.approx(0.07)
