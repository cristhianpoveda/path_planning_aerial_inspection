# Position and Heading Controller — Design and Status

The outer-loop controller that drives the drone to a commanded waypoint
`y = (p_x, p_y, p_z, yaw)`, the app-side changes that made a metric velocity
command path possible, and its validated behaviour on both ground-truth and
onboard-estimator feedback.

---

## 1. Architecture

DJI's onboard controller already closes attitude and translational velocity at
high rate. The command interface, in advanced virtual stick, accepts a
body-frame velocity setpoint directly (`controller_plant_model.md`). So the
plant, from this controller's point of view, is a near-unity velocity servo
followed by an integrator:

```
v_body(s) / v_cmd(s) ≈ K · e^{-τ·s}        p = ∫ v
```

with `K` per-axis and near 1 (`controller_plant_model.md`) and `τ ≈ 0.28 s`
(§8). There is therefore **no inner velocity loop to design**. The controller is
four decoupled single-input loops mapping position and heading error onto a
body-frame velocity command:

| axis | error | law |
|---|---|---|
| forward / lateral | `p_ref − p̂` in body frame | P + saturation |
| vertical | `z_ref − z` | P, optional leaky I |
| heading | `yaw_ref − yaw` | P + saturation |

**Design decisions, each with its basis:**

- **P, not PID, on x and y.** `filter_design.md` §9: `p_x, p_y` are observed by
  no update, so `P_xx, P_yy` grow without bound. An integrator would wind up
  against estimator drift, not a real disturbance. Indoors there is no wind to
  justify one.
- **Fixed-rate publishing regardless of feedback arrival.** The phone-side
  watchdog (200 ms) is the dead-man switch; the controller must not be a second
  one. A missing pose yields a zero command, not silence.
- **Heading closes on the published EKF orientation** (`yaw_source: ekf_pose`),
  with `dji_attitude` retained as an alternative. This reverses the earlier
  decision: it was taken when the published yaw had sd 28°, which was a sign
  error and is fixed. [M] Published EKF yaw now tracks mocap to sd 0.5–1.1°
  (`ekf_operating_conditions.md` §4.4), and both flights of §4.1 used it.
- **Yaw command scaled by 1.31** (`= 1/0.763`) to invert the measured yaw-rate
  gain (`controller_plant_model.md` §4.4), applied in the controller so
  `command/vel` stays a physical-units interface.
- **Gains sized by loop-shaping on the identified plant**, not trial and error.
  `controller_plant_model.md` §9, now with a measured pose age: `L ≈ 0.79–0.85 s`,
  which gives `Kp ≤ 0.64` at `K = 1.0` and `0.43` at `K = 1.5` for 60° of phase
  margin. [M] `kp_xy = 0.6` flew stable on EKF feedback; since the plant gain on
  this path has not been re-measured, that is a demonstration rather than a
  justification.

```
setpoint ─┐
          ▼
   e = p_ref − p̂ ──► rotate odom→body ──► Kp·e ──► saturate ──► slew ──► command/vel
          ▲                                                                    │
          │                                                          (×1.31 on yaw)
   feedback: localisation/pose (odom)          heading: attitude (DJI), sign-flipped
```

---

## 2. App-side changes (WildBridge)

The original app exposed only **normal** virtual stick: `[-1, 1]` stick
deflections passed through DJI's own nonlinear stick-to-speed curve. That curve
is undocumented, flight-mode dependent, and would have needed full
characterisation. The GPS waypoint routine already present in the app used the
**advanced** path (`sendVirtualStickAdvancedParam`) in SI units, so the change
was to expose that path to ROS.

**What was added (additive; the normal-stick path is untouched):**

- A velocity command path on the **existing UDP port 8082**, selected by a
  `mode` field in the JSON payload (`"stick"` or `"vel"`). A single port means
  one sequence counter and one watchdog; two ports would have meant two
  watchdogs able to fight each other.
- `DroneController.setVelocity(vx, vy, vz, yawRate)`, building a
  `VirtualStickFlightControlParam` with `RollPitchControlMode.VELOCITY`,
  `YawControlMode.ANGULAR_VELOCITY`, `VerticalControlMode.VELOCITY`,
  `FlightCoordinateSystem.BODY`. The pitch/roll assignment (`roll = vx`,
  `pitch = vy`) is the swap inherited from the working GPS routine; confirmed
  correct in flight (`controller_plant_model.md` §3).
- SDK-range clamps inside `setVelocity` (`VirtualStickRange`), a second line of
  defence below the ROS-side operating caps.
- A **mode-aware watchdog.** The existing 200 ms watchdog called
  `setStick(0,0,0,0)`, which is a no-op once advanced mode is engaged. It now
  calls `hover()`, which dispatches to the correct API for whichever mode is
  live. This was the one genuinely dangerous interaction and the reason the
  change had to touch the watchdog rather than being purely additive.
- Advanced mode latches on the first velocity packet and releases in
  `onDestroyView`, so the operator's manual arming workflow is unchanged.

**Confirmed facts about the interface** (`controller_plant_model.md`):
signs `+1/−1/+1/−1`, no dead zone above 0.01 m/s, drift floor 6 mm/s,
command-to-motion latency 0.28 s, per-axis gains near unity except yaw at
0.763 and a direction-asymmetric `vx` in [1.0, 1.5].

---

## 3. ROS side

`dji_node` gained a `command/vel` subscription (`TwistStamped`, body frame,
m/s and rad/s) selected by `command_mode`. It converts rad/s → deg/s, applies
the sign parameters, clamps to the arena caps, and sends the `mode`-tagged JSON
on 8082. It publishes only when a message arrives; there is no repeater, so a
dead controller lets the phone watchdog hover the aircraft rather than a stale
command persisting.

`TwistStamped` rather than `Twist`: the header is ignored by `dji_node` but the
controller stamps it from its own clock, which separates control-loop jitter
from transport jitter in the evaluation bags. A staleness check on that header
was deliberately **not** added — it would be a second dead-man switch competing
with the phone's.

The controller node (`position_heading_controller_node`) runs a 20 Hz timer,
gates on pose age, setpoint age, and `localisation/status.degraded`, and
publishes zero when any gate fails or when disabled. `yaw_source` selects the
heading source (`ekf_pose` or `dji_attitude`) with a per-session
`yaw_offset_deg`; `start_enabled` and a `controller/enable` topic arm it.

---

## 4. Validated behaviour on ground-truth feedback

Flown across four sessions with mocap relayed as `localisation/pose`, so the
controller saw a perfect estimate and any fault was its own.

**[M] Results:**

| property | result |
|---|---|
| Steady-state hold error | **±0.01 m** on all axes once settled |
| Stability range | stable and well-damped, `Kp` 0.25 → 0.6 |
| Overshoot, 0.3 m step, `Kp` 0.4 | under 3 cm (matches simulation) |
| Overshoot, 1 m step | comparable — set by approach speed × delay, not distance |
| Disturbance rejection | recovered smoothly from a manual stick nudge |
| Heading | tracks, turns the short way, settles |

The saturation-plus-P structure was confirmed to behave as a constant-velocity
approach at range: a distant setpoint cruises at `v_max` and enters the
proportional region only in the last `v_max/Kp` metres, so **overshoot in metres
is nearly independent of setpoint distance** — a single far waypoint is handled
like a velocity plan, and closely spaced waypoints are not needed for stability.

**Gains that flew:** `kp_xy` 0.25 (first flight) then 0.4 and 0.6;
`kp_z` 0.4–0.5; `kp_yaw` 0.8. `v_max_xy` 0.3–0.5, `v_max_z` 0.2–0.3,
`yaw_rate_max` 15 deg/s.

**Conclusion: the controller is complete and validated as a component.** Every
remaining problem is upstream of it, in the feedback.

---

### 4.1 Validated behaviour on EKF feedback

Two flights on `yaw_source: ekf_pose`, `kp_xy 0.6`, with mocap recorded on
domain 0 as a passive observer and never entering the loop. Manual flight until
the filter initialised, then virtual stick and a sequence of setpoint steps.
Scored by `flight_analysis.py`.

**[M] Results** (C01, 14 steps / C03, 19 steps):

| property | C01 | C03 |
|---|---|---|
| True error at the setpoint, at rest, median / p90 | **0.042 / 0.099 m** | **0.046 / 0.079 m** |
| Step magnitude, achieved / commanded, median | 0.998 | 0.994 |
| Step direction error, median / p90 | 2.8° / 5.0° | 2.3° / 4.5° |
| Cross-track at the end of the step, median | 0.031 m | 0.023 m |
| Settling to 0.05 m in the estimator's frame, median | 2.9 s (13/14) | 5.6 s (18/19) |
| Steady-state error in the estimator's frame, median | 0.025 m | 0.039 m |
| Overshoot, median | 0.055 m | 0.052 m |
| Excursion while holding, median / max | 0.062 / 0.098 m | 0.071 / 0.188 m |

**The cost of flying on the estimator rather than on truth is 0.01 m → 0.04 m**,
median, with a p90 of 0.08–0.10 m. That difference is the coupling this project
set out to quantify, and it is now measured rather than argued.

**[M] The control law is confirmed against the bag.** Re-simulating the node
from its own recorded inputs reproduces `command/vel` to the printed precision,
and regressing the gains out of the bag returns `kp_xy` 0.600, `kp_z`
0.498/0.500, `kp_yaw` 0.800/0.801 — the values passed on the command line.

**[M] Heading closed on `ekf_pose` works.** Published EKF yaw against mocap has
sd 1.06° (C03) and 0.47° (C01), so a fixed per-session offset is enough. The
offsets measured −39.0° and −49.5°; they are per session and must not be
inherited.

---

## 5. Closing the loop on the EKF — what made it work

Each of these was resolved before the flights of §4.1, and together they are why
those flights held a single VO epoch where earlier ones rebuilt every 24–90 s:

- **Heading source decided.** [M] `EKF yaw − mocap` is constant to sd 0.3–2.0°
  on clean flights, so `yaw_source: ekf_pose` needs no offset;
  `yaw_source: dji_attitude` needs a per-session `yaw_offset_deg` near the
  measured `EKF yaw − DJI attitude` (≈ −65 to −90° depending on session).
- **Attitude sign bug found and fixed.** The `sign_yaw` error propagated into
  `ekf_node`'s attitude update, which was rejecting ~99 % of attitude updates
  (2554 in one flight). After the fix, rejection fell to 1–16 per flight and the
  published-orientation yaw defect of `ekf_operating_conditions.md` §4.4
  collapsed from sd 28° to sd < 2°.
- **Health gate made usable.** `SIGMA_S_MAX` made relative to `s`, and
  `vo_discont` made a transient flag rather than a latching counter. `degraded`
  went from 100 % of samples to 20–30 %, so `gate_on_degraded` is now a
  meaningful gate.
- **Camera pipeline rebuilt.** The decoder was doing a colour conversion and a
  quality-95 JPEG encode single-threaded; it now decodes to grayscale, threads,
  and encodes at q70. Bench rate went from 11 Hz decaying with multi-second
  stalls to a steady 20–40 Hz.
- **Init made reliable.** Commanding above `V_LOW` first (`--amps 0.6`) makes
  the filter initialise in 40–100 s instead of never; earlier flights spent the
  clean part of the flight below the scale-observability threshold.

**[M] The EKF, on a clean segment, can be good:** step11b epoch 6 gave raw VO
increments matching mocap to a 4 % residual over 386 windows. So the geometry,
the texture, and VO itself are not fundamentally the problem.

---

## 6. The former blocker, and what it turned out to be

**Superseded.** This section previously recorded that closed-loop control on the
EKF had not been achieved, that scale never settled (`s` swinging 0.68→19.5,
9.2→217, 1.9→65, 0.70→146) and that RPE against mocap was 78–115 %. Both flights
of §4.1 contradict it.

### 6.1 Scale settles, and the epoch holds

[M] C01 and C03 each held **one VO epoch for the whole flight** — 695 s and
829 s of continuous track, no rebuild. With no rebuild there is nothing to
invalidate `s`, and the scale state stayed inside 4.554–4.763 (C01) and
3.406–4.562 (C03), with `σ_s/s` at 0.031–0.035. Against mocap the residual scale
factor is 1.044 and 1.008.

The mechanism recorded previously — a rebuild every ~25 s driving a noisy
re-measurement that overshoots — was real, but it was downstream of the
rebuilds, not a defect in the estimator. The re-measurement path was not
exercised on these flights, so it remains untested rather than fixed.

### 6.2 The video freeze did not occur

[M] C03 recorded the camera topic: 16.9 Hz steady, p99 frame interval 80 ms, max
0.45 s, **no gap over 0.5 s in 870 s**. C01 did not record the camera, but its
`vo/pose` gaps stayed under 0.29 s and `pose_valid` was 100 % of frames.

The mitigations of §7 item 1 were in use (letting the link settle, a warm-up
before the test sequence). **Which of them mattered, or whether the freeze is
simply intermittent, is not established** — the failure did not recur, which is
not the same as being fixed.

### 6.3 What still limits accuracy

Not scale, and not the controller. The residual error is:

- **`p_x`, `p_y` are observed by nothing** (`filter_design.md` §9), so the
  estimate drifts and the controller faithfully follows it. [M] Excursion while
  holding reached 0.19 m on C03.
- **`VO_DELAY` under-subtracts by 40–100 ms** (`filter_design.md` §4.1), a
  constant bias worth 1–3 cm at inspection speed.
- **[?] Direction-error scatter** across steps that no single frame rotation
  absorbs (§4.1).

## 7. Outcome

The fallback position recorded in the previous revision — controller validated
on ground truth, plant and estimator characterised, coupling quantified — is no
longer a fallback. The full result is available:

- **Controller:** validated on ground truth (§4) and on the onboard estimator
  (§4.1). Done.
- **Plant:** characterised by open-loop pulses (`controller_plant_model.md`).
  Done, with the gaps listed there.
- **Estimator:** characterised against ground truth on the current build
  (`ekf_operating_conditions.md` §1.1), with failure modes from the earlier
  build still mapped in §4 there.
- **Coupling, measured:** control accuracy on ground truth **±0.01 m**; on the
  onboard estimate **0.042–0.046 m median, 0.079–0.099 m p90**, with 2.3–2.8°
  of direction error and up to 0.19 m of excursion during a hold. The estimator
  remains the limiting term, and the reasons are measured rather than asserted.

That is the answer to how much localisation reliability costs the inspection
task, on this platform, in this arena.

## 8. Parameters as flown

**On EKF feedback (C01, C03).** [M] Confirmed from the bags by re-simulation and
regression, not from the command line:

```
kp_xy 0.6        kp_z 0.5        kp_yaw 0.8
v_max_xy 0.5     v_max_z 0.3     yaw_rate_max_deg 15
deadband_xy 0.02 deadband_z 0.02 deadband_yaw_deg 2.0
slew_xy 1.0      slew_z 1.0      slew_yaw_deg 60
yaw_scale 1.31   yaw_source ekf_pose      yaw_offset_deg 0.0
pose_timeout 0.5 setpoint_timeout 5.0     gate_on_degraded true
```

`kp_xy 0.6` is above the 0.4 recommended from the loop-shaping of
`controller_plant_model.md` §9, and flew stable and well damped on both flights.
The margin calculation rests on a plant gain that has not been re-measured on
this command path, so treat 0.6 as demonstrated rather than as justified.

Note `gate_on_degraded: true` was in force: [M] 4.6 % (C03) and 2.9 % (C01) of
status messages were degraded, and those samples produced zero commands.

**On mocap feedback (§4).**

```
kp_xy 0.4        kp_z 0.5        kp_yaw 0.8
v_max_xy 0.5     v_max_z 0.3     yaw_rate_max_deg 15
deadband_xy 0.02 deadband_z 0.02 deadband_yaw_deg 2.0
slew_xy 1.0      slew_z 1.0      slew_yaw_deg 60
yaw_scale 1.31   yaw_source ekf_pose (= mocap in these tests)
pose_timeout 0.5 setpoint_timeout 5.0
```

`dji_node`: `command_mode vel`, `sign_vx +1`, `sign_vy −1`, `sign_vz +1`,
`sign_yaw −1`, `v_max_horizontal 1.0`, `v_max_vertical 0.5`, `yaw_rate_max 30`.
