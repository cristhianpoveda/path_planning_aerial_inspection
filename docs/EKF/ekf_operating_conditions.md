# Localisation — Operating Conditions

What the filter needs to work, what it cannot do, and the routine that puts it
in a healthy state before an inspection.

Design and constants: `filter_design.md`.

---

## 1. Measured envelope (pre-rebuild build)

Configuration of §2, RPE over 1 s windows against mocap, per `vo_epoch`
segment. All seven bags on the current build.

| bag | profile | RPE 1 s, as it runs | fitted scale | epochs | status |
|---|---|---|---|---|---|
| F9_02 s0 | transit, 87 s | **6.5 %** | 1.020 | 5 / 229 s | works |
| F9_02 s2 | inspection after transit, 87 s | **8.4 %** | 1.003 | | works |
| F3_02 | fast square, 2 segs | **12.5–13.8 %** | 0.985 ± 0.047 | 5 / 128 s | works |
| F8 | vertical squares | 15.8–73.6 % | 1.188 ± 0.111 | 16 / 191 s | map churn |
| F6 | slow sweep at a wall | 72.8 % | 1.023 | 9 / 43 s | VO does not translate |
| F6c | as F6, shorter | 71.4 % | 1.016 | 3 / 29 s | as F6 |
| F3_01 | slow square | — | — | 7 / 106 s | no segment ≥ 15 s |
| F9 | inspection, no transit | — | — | — | never initialises |

**Headline:** with a transit leg and a stable VO map, **6–14 % relative
position error over 1–2 s windows, scale correct to within 5 %.** Outside those
conditions the filter either refuses to start or reports DEGRADED.

The filter's scale is correct even where position is not: F6 1.023, F6c 1.016.
The failures in §4.1 are VO input quality, not estimation.

---

## 1.1 Measured envelope, current build

Two closed-loop flights, position loop closed on `localisation/pose`, scored
against mocap by `flight_analysis.py`. RPE over 1 s windows, **restricted to
windows carrying ≥ 5 cm of true motion** — see the note below.

| bag | profile | RPE 1 s, scale removed | as it runs | fitted scale | epochs | status |
|---|---|---|---|---|---|---|
| C01 | closed-loop steps, 695 s | **7.3 %** (0.028 m) | 6.1 % | 1.044 | **1 / 695 s** | works |
| C03 | closed-loop steps, 870 s | **12.3 %** (0.051 m) | 11.8 % | 1.008 | **1 / 829 s** | works |

**Headline: one VO epoch for the whole flight, scale correct to within 5 %, and
2.8–5.1 cm of relative position error over 1 s windows.** Downstream, that
produced 4–5 cm of true position error at a commanded setpoint
(`controller_design.md` §4.1).

> **Restrict RPE to moving windows.** [M] Median true displacement per 1 s
> window was 0.010 m (C03) and 0.009 m (C01); only 14 % and 10 % of windows
> carried ≥ 5 cm. Unrestricted, the same computation reads 33.9 % and 25.1 %,
> which measures the noise floor against a 1 cm denominator.
> `flight_analysis.py` prints both; quote `RPEm%` and `runsm%`.

**What was not observed on these two flights:** no VO map rebuild, no camera
freeze over 0.5 s (C03; C01 did not record the camera topic, but its `vo/pose`
gaps stayed under 0.29 s), no `S_CLAMPED`, no scale excursion. The map-churn and
under-translation failures of §4.1 remain the known boundary; these flights did
not approach it.

---

## 2. Launch configuration

The launch-file defaults now match the values every measurement in §1 was taken with, so a bare `ros2 launch` reproduces the tested configuration. The explicit form below is the same thing written out, for the record.

```bash
ros2 launch drone_localisation localisation.launch.py \
    estimate_scale:=1.0 \
    sigma_yaw:=0.011 \
    R_speed_h:=0.0011 \
    q_s:=1e-4 \
    T_HOLD:=0.0 \
    K_VEL:=0.87
```

Everything else comes from `EkfParams` defaults; the ones that matter are in
`filter_design.md` §13.

| parameter | value | why not the alternative |
|---|---|---|
| `estimate_scale` | 1.0 | 0 does not hold `s` unless `P_ss` is zeroed too |
| `sigma_yaw` | 0.011 | [M] DJI attitude yaw tracks mocap to 0.19°. Do not inflate to fix the residual in §4.4 — that models a bias as noise |
| `R_speed_h` | 0.0011 | [M] the increment-noise term in `R_speed_eff` carries the rest; median accepted NIS 4.88 → 1.82 |
| `q_s` | 1e-4 | with 0, `P_ss` decreases monotonically and `s` freezes |
| `T_HOLD` | 0.0 | [M] velocity dedupe cut accepted updates 279 → 50 and made the survivors *worse* (rms 0.145 → 0.208 m/s) |
| `K_VEL` | 0.87 | [M] two independent routes, 0.5 % apart. Was 0.91 |

[?] `config/ekf/params.yaml` is loaded before these overrides and has not been checked against them. It does not contain any of the filter tuning parameters. All of them are passed by the launch file.

**[M] C01 and C03 flew the launch-file defaults**, confirmed from the bags: the
`vo/pose` stamp minus `localisation/pose` stamp is exactly 0.400 s on 100 % and
99.9 % of frames. Two of those defaults are now known to be off, and neither has
been changed (`filter_design.md` §13): `VO_DELAY 0.40` under-subtracts by
40–100 ms, and `K_VEL 0.87` reads high for the flown speed range.

For bag replay add `use_sim_time:=true` and play with `--clock`. For analysis
add `innovation_log:=<path>.csv`, and record `/drone_1/localisation/pose` if
you intend to run `evaluate_pose.py`.

---

## 3. Requirements

### 3.1 To initialise

| requirement | value | why |
|---|---|---|
| Airborne | altitude > `INIT_ALT_MIN` | `b` cannot be datumed on the ground |
| VO tracking | `pose_valid` | scale samples need an increment |
| Transit | **≥ 3 m of path at ≥ 0.4 m/s** (`INIT_PATH_M`, `V_LOW`) | scale is observable only from velocity, and only above the 0.1 m/s quantisation dead zone |
| Gimbal | brackets the VO stamp | `R_b_c` is interpolated, never held |

[M] F9 fails this: it accumulates 1.5 m of the required 3 m across a whole
flight. The refusal is correct — no scale estimate exists to publish.

### 3.2 To keep working

- **Re-transit after every VO map rebuild.** A rebuild gives the new map an
  arbitrary scale. [M] On F9_02 the rebuilt map matched the old one to 1 %; on
  F8 the factors ran 1.13–1.46 across 16 rebuilds. `_maybe_rescale` re-measures
  opportunistically but needs `RESCALE_PATH_M` above `V_LOW` to complete.
- **Keep VO in textured, parallax-rich geometry, and watch the rebuild rate.**
  See §4.1.
- Publishing continues through a rebuild; only `s` is invalidated.

---

## 4. Limitations

### 4.1 VO input quality — the hard limit

Four of seven bags fail here, in two forms. Neither is a filter defect:
`p⁺ = p + s·u`, and no update can recover motion absent from `u` or a frame
re-anchored faster than it converges.

**Under-translation (F6, F6c).** Slow sweep parallel to a near, flat wall.
[M] Scale is correct (1.023, 1.016) and yaw excellent (0.61–0.63° sd), yet
position covers only 29–32 % of true motion over 1 s windows. VO path × `s`
totals 11.1 m against 19.7 m of mocap path on F6, and 87 % of VO-implied speeds
are below 0.02 m/s while DJI reads 0.3–0.4 m/s.

**Map churn (F8, F3_01).** [M] 16 rebuilds in 191 s and 7 in 106 s — one every
12–15 s. Each invalidates `s` and re-anchors `R̄_n_v`, and neither reconverges
before the next. F8's velocity-direction disagreement swings 4°–42° per 20 s
window (F9_02: 4.1° steady), which is the re-convergence transient, not a
constant offset. F3_01 has no segment long enough to evaluate at all.

> **[?] Two measures of F6's deficit disagree: 56 % on whole-flight path, 29 %
> on windowed displacement.** Path is a scalar sum inflated by per-increment
> noise (VO at 32 Hz, mocap decimated to 10 Hz); displacement is a vector sum
> reduced by directional jitter. Not resolved; it does not change the
> conclusion.

### 4.2 Scale is unobservable without transit

`s` is observed only by the velocity update, only above `V_LOW`. [M] Below it
DJI's gain droops to 0.77–0.82 against 0.93 at 1 m/s — a bias, not noise, so
inflating `R` cannot remove it. During a pass the scale column is disabled and
`s` is held.

### 4.3 `p_x`, `p_y` are dead-reckoned

Nothing observes horizontal position, so `P_xx` and `P_yy` grow monotonically
and absolute error grows without bound. **Evaluate and gate on RPE over short
windows, never on ATE or `tr(P_pp)`.**

### 4.4 Published orientation yaw — FIXED

The pre-rebuild defect (yaw residual mean −22.8°, sd 28.0°) was a **sign error
in the attitude update**, inherited from `sign_yaw = +1.0`. After the fix,
attitude rejection fell from ~99 % to a handful per flight, and [M] the
published orientation yaw against mocap has **sd 1.06° (C03) and 0.47° (C01)**
over the whole flight. Both a controller and a viewpoint planner can use it,
with a per-session constant offset — measured −39.0° and −49.5° on these two
sessions. **The offset is per session and must not be inherited.**

### 4.5 A dead link looks like a stationary aircraft

Telemetry is deduplicated at source, so `TRANSPORT_GAP` does not fire reliably.
Until `dji_node` emits a fixed-rate heartbeat, comms loss is detected only by
the phone-side watchdog.

[M] On the current build the flag does fire — 4 times (C03) and 13 times (C01) —
alongside telemetry gaps of up to 1.65 s and 1.75 s visible in the bag. That it
fires does not make it reliable: the mechanism that hides a dead link is
unchanged.

### 4.6 [?] `vo_discont` fires without a map rebuild

[M] 633 flags (C03) and 322 (C01) while `vo_epoch` never changed. These are
increment-builder rejections inside a healthy map, not rebuilds, and they are
what put the filter into DEGRADED for 4.6 % and 2.9 % of the flight. Not
attributed to a specific gate; the innovation log would say which.

---

## 5. Pre-inspection routine

1. **Take off** and climb above `INIT_ALT_MIN`. Hold briefly.
2. **Fly a transit leg** — ≥ 3 m of continuous travel at ≥ 0.4 m/s, in an area
   with texture and depth variation. Translation across the field of view, not
   toward it: parallax is what VO needs.
3. **Wait for `initialised:`** in the node log, or `localisation/status.state ==
   "OK"`.
4. **Check scale confidence** before trusting position: `sigma_scale / scale <
   SIGMA_S_OK` (0.05). At init it is typically 0.10 and tightens over the first
   transit.
5. **Begin the inspection pass.** `s` is held through it by design.
6. **On any `VO map rebuild` warning**, insert another transit leg before
   continuing, or accept that `s` is carrying the previous map's value. The
   node re-measures `s` opportunistically once ~2 m of path above `V_LOW` is
   available — [M] confirmed on F8 and F3_01.
7. **Abort the pass if rebuilds exceed ~1 per 30 s.** [M] At one per 12 s
   neither `s` nor the nav frame reconverges between them (§4.1).

**Abort or re-transit if** `localisation/status.degraded` is true. The flags
the node currently emits are `SIGMA_S_MAX`, `vo_discont=<n>`, `TRANSPORT_GAP`
(unreliable, §4.5) and `VO_GAP`.

[M] On C01 and C03 the routine worked as written: both initialised and held one
epoch for the whole flight, with `σ_s/s` at 0.031–0.035, well inside
`SIGMA_S_OK`. Step 6 never triggered — there were no rebuilds to react to.

**Operational note, not a filter matter.** [M] `setpoint_cmd.py` publishes only
while it runs, and `setpoint_timeout` is 5 s, so the gaps between runs (up to
44.7 s on C03) gate the controller off and leave the aircraft on DJI's own
position hold. On C03 the setpoint was stale for 51 % of hold time and only 3 of
19 holds were closed-loop throughout. Keep the publisher running for any hold
that is meant to be evaluated.

---

## 6. Open questions that bear on operation

- **[?] `VO_DELAY` is under-subtracted at 0.40 by 40–100 ms**, per-flight,
  measured three ways on C01 and C03 (`filter_design.md` §4.1). Every published
  pose stamp is early by that much: 1–3 cm at inspection speed, against 4–5 cm
  of measured position error.
- **[?] Replay does not reproduce live behaviour** (`filter_design.md` §12.7).
  A replay of C01 at its own parameters gave fitted scale 1.437 and 1920
  `SIGMA_S_MAX` flags where the live flight had none. Any parameter chosen
  offline needs a live flight to confirm.

- **[?] `psi_v` — the yaw offset between DJI velocity and the attitude-seeded
  nav frame — is per-flight**, measured 9.7° to 152.6° across five bags. It is
  absorbed by `δθ_z`; correcting the velocity signal for it was tried and made
  things worse. Source unidentified.
- **[?] Init `s` is noisy**: per-sample scatter 84 % relative, and two draws
  from the same bag differ by 10 %. The filter converges from any start while
  `estimate_scale = 1`, so this matters only for a held configuration.
- **[?] `VO_DELAY` is per-flight**, 0.345–0.430 s across three bags, constant to
  ~20 ms within each. The fixed 0.40 leaves ±45 ms.
- **[?] F9_02 segment 1** (22 s of 176) is unexplained: 2.2× excess motion with
  correct yaw. See `filter_design.md` §11.
- **[?] The published-orientation yaw defect (§4.4) is unresolved** and is
  `filter_design.md` §14 item 1. It matters to a viewpoint planner in a way it
  does not to a position controller.
