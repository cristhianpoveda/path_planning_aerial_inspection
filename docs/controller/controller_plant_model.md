# Command Interface — Plant Characterisation

Open-loop characterisation of the DJI advanced virtual stick velocity interface
on the Mini 4 Pro, measured against OptiTrack.

---

## 1. Configuration under test

| | |
|---|---|
| Command path | `command/vel` (`TwistStamped`) → `dji_node` → UDP/8082 → WildBridge → `sendVirtualStickAdvancedParam` |
| Control modes | `RollPitchControlMode.VELOCITY`, `YawControlMode.ANGULAR_VELOCITY`, `VerticalControlMode.VELOCITY` |
| Coordinate system | `FlightCoordinateSystem.BODY` |
| Pitch/roll assignment | `param.roll = vx`, `param.pitch = vy` (the swap inherited from `navigateToWaypointWithPID`; [M] confirmed correct, §3) |
| RC flight mode | Normal |
| Command rate | 20 Hz |
| Arena | 5 × 4 × 3 m, hover ~1.5 m |
| SDK limits | ±23 m/s roll/pitch, ±6 m/s vertical, ±100 deg/s yaw (`VirtualStickRange`) |

---

## 2. Excitation

Symmetric `+/−` velocity pulses on one axis at a time, separated by zero-command
dwells. Nose aligned to mocap +X before takeoff.

| bag | axis | amplitudes | pulse | dwell | purpose |
|---|---|---|---|---|---|
| `step3_signs` | vx, vy, vz, yaw | 0.2 m/s, 15 deg/s | 1.5 s | 3 s | signs, frame |
| `step4` | vx, vz, yaw | 0.05–0.5 m/s, 5–30 deg/s | 2.0 s | 3 s | gain curves |
| `step4` | vy | 0.05–0.5 m/s | 1.5 s | 3 s | gain curve |
| `step4b` | vx | 0.01–0.05 m/s | 4.0 s | 3 s | dead zone |
| `step4b` | vy | 0.4, 0.5 m/s | 2.5 s | 3 s | **invalid**, §7 |

---

## 3. Frame and signs

[M] Two independent bags:

| | `step3_signs` | `step4` |
|---|---|---|
| +vx direction, mocap body frame | +0.5° | −5.0° |
| +vy direction | −92.1° | −92.9° |
| angle +vx → +vy | −92.6° | −87.9° |

Perpendicular within measurement noise. **The pitch/roll assignment is correct
as flown; there is no swap problem on this path.**

| parameter | value | basis |
|---|---|---|
| `sign_vx` | **+1.0** | [M] +vx moves along the nose |
| `sign_vy` | **−1.0** | [M] +vy moves right; REP-103 is left-positive |
| `sign_vz` | **+1.0** | [M] +vz climbs |
| `sign_yaw` | **-1.0** | [M] DJI yaw is clockwise-positive; REP-103 is counter-clockwise. Confirmed in flight (step6) |

**Yaw convention.** [M] Later flights (step8, step10c, step11) established
this precisely: `mocap_yaw + dji_yaw` is constant to sd 0.3–0.6° across each
flight, so **DJI attitude yaw is the negative of mocap yaw plus a constant** —
DJI is clockwise-positive, mocap (REP-103) is counter-clockwise-positive. This
is why `sign_yaw = -1.0`, and it also required a sign fix in `ekf_node`'s
attitude update and in the controller's `dji_attitude` heading path. DJI
attitude reads 2–4 % below mocap in rate magnitude.

**Session constant.** [M] With the nose aligned to mocap +X at takeoff, the
mocap ↔ `base_link` yaw offset for this session is **+0.5°**. Per-session; do
not inherit (`filter_design.md`, Retired).

[M] Two later sessions confirm the convention and the need to re-fit it: from
`mocap_yaw = −1 × DJI_yaw + c`, `c` measured **+128.2° (C03, sd 0.31°)** and
**+139.3° (C01, sd 0.29°)** over 830 and 700 s. The sign is confirmed on both;
the constant is not transferable between sessions.

---

## 4. Steady-state gain

Measured / commanded, from least-squares position slopes over a
latency-shifted window. `step4` unless stated.

### 4.1 Horizontal, `vx` (fore/aft)

| \|cmd\| m/s | gain fwd | gain back | asym % | source |
|---|---|---|---|---|
| 0.010 | 1.05 | 1.56 | −39 | `step4b` |
| 0.020 | 1.61 | 2.64 | −48 | `step4b` |
| 0.030 | 0.82 | 1.15 | −34 | `step4b` |
| 0.050 | 1.21 | 1.77 | −38 | `step4` |
| 0.100 | 1.05 | 1.42 | −31 | `step4` |
| 0.200 | 1.06 | 1.24 | −16 | `step4` |
| 0.300 | 0.97 | 1.17 | −19 | `step4` |
| 0.500 | 1.02 | 1.10 | −8 | `step4` |

**[M] `vx` is direction-asymmetric.** Backward exceeds forward by 77 % at
0.05 m/s, falling to 10 % at 0.5 m/s. Forward is ~1.0 throughout.

Not environmental: a draught would have produced comparable drift in the `vy`
block, which showed −0.089 m against −0.865 m for `vx` over comparable blocks.

Scatter below 0.05 m/s is drift residual (§6), not plant nonlinearity.

**Design range: `K_vx ∈ [1.0, 1.5]`.**

### 4.2 Horizontal, `vy` (lateral)

| \|cmd\| m/s | gain + | gain − | asym % |
|---|---|---|---|
| 0.050 | 1.00 | 1.16 | −15 |
| 0.100 | 1.16 | 1.07 | +8 |
| 0.200 | 1.09 | 1.14 | −5 |
| 0.300 | 0.99 | 1.04 | −5 |

**`K_vy = 1.08`**, flat and near-symmetric to 0.3 m/s.
**[A] assumed flat to 0.5 m/s** — see §7.

### 4.3 Vertical, `vz`

| \|cmd\| m/s | gain up | gain down | asym % |
|---|---|---|---|
| 0.050 | 1.06 | 0.90 | +17 |
| 0.100 | 1.04 | 0.98 | +6 |
| 0.300 | 1.01 | 1.00 | +1 |
| 0.500 | 1.01 | 1.01 | +0 |

Fit: `measured = 1.008 · cmd + 0.002`.
**`K_vz = 1.01`.** Symmetric, linear, no compensation required.

### 4.4 Yaw rate

| \|cmd\| deg/s | gain + | gain − | asym % |
|---|---|---|---|
| 5 | 0.764 | 0.758 | +0.7 |
| 10 | 0.757 | 0.760 | −0.4 |
| 20 | 0.764 | 0.763 | +0.1 |
| 30 | 0.761 | 0.765 | −0.4 |

Fit: `measured = 0.763 · cmd − 0.011`.

**[M] `K_yaw = 0.763`, constant to ±1 % over a 6× amplitude range.** A fixed
scale factor, not a nonlinearity. **Apply `1/0.763 = 1.31` to commanded yaw
rate** to obtain true deg/s.

[?] Within each pulse the yaw rate decays ~15 % from first to second half of
the window, consistently across all amplitudes. The gain above is unaffected.
Not investigated.

---

## 5. Dead zone

**[M] None above 0.01 m/s.** Drift-corrected response at the smallest tested
command: 0.011 m/s forward, 0.016 m/s backward, against 0.010 m/s commanded.
No amplitude in any bag failed to produce motion.

The limit at small amplitude is the drift floor (§6), not a threshold.

---

## 6. Disturbance floor

Zero-command dwells, manual repositioning excluded.

| bag | dwells | total | x drift | y drift | z drift |
|---|---|---|---|---|---|
| `step4` | 35 | 154 s | −0.0009 ± 0.0058 | +0.0004 ± 0.0064 | +0.0006 ± 0.0024 |
| `step4b` | 11 | 169 s | −0.0035 ± 0.0055 | −0.0025 ± 0.0046 | +0.0006 ± 0.0042 |

m/s, mean ± sd.

| | value |
|---|---|
| Horizontal drift speed, median | **0.0056 m/s** |
| Horizontal drift speed, p90 | **0.013 m/s** |
| Max displacement in one dwell | **0.045 m** |

**[M] DJI's own hold is doing nearly all the work.** The position loop has
little disturbance to reject; terminal accuracy is set by this floor.

**[M] Lateral bias while translating.** During `vx` pulses the cross-axis
velocity is 0.010–0.017 m/s regardless of commanded speed (0.05–0.5 m/s), about
3× the stationary drift. Accounts for −0.68 m of lateral displacement across the
`step4b` `vx` block. Corrected by the `vy` channel in closed loop; relevant when
interpreting cross-track error.

---

## 7. Latency

| bag | cmd → `speed_vector` | minus 73 ms telemetry | cmd → mocap |
|---|---|---|---|
| `step3_signs` | 380 ms | 307 ms | 260 ms |
| `step4` | 440 ms | 367 ms | 290 ms |
| `step4b` | 370 ms | 297 ms | 270 ms |

Telemetry term from `filter_design.md` §4. The mocap route resolves at ~100 Hz
against 8.9 Hz for `speed_vector`.

**[M] Command-to-motion latency: 270–290 ms. Use 0.28 s.**

**[?] Velocity-loop time constant.** Estimated 0.0–0.4 s with wide scatter; the
estimator assumes one latency for all axes and is unreliable when the true onset
precedes it. Sufficient to state `tau` is well below the loop delay, so delay
dominates. Not pursued.

---

## 8. Loop delay budget

| term | value | source |
|---|---|---|
| Command to motion | 0.28 s | §8 [M] |
| Pose age at controller | 0.51–0.57 s | §9.1 [M] |
| **Total loop delay `L`** | **0.79–0.85 s** | |

### 8.1 The pose-age term, measured

[M] On C01 and C03 the age is read directly from the bags. Two quantities, and
they differ:

| | C01 | C03 |
|---|---|---|
| Arrival age — last pose in hand when the command is issued | 30 ms | 30 ms |
| Age against the pose STAMP — what the loop believes it has | 473 ms | 477 ms |
| Measured VO delay (`filter_design.md` §4.1) | 0.437 s | 0.44–0.51 s |
| **True age** — frame capture to command | **0.51 s** | **0.52–0.57 s** |

The believed age is the arrival age plus the `VO_DELAY` the node subtracted; the
true age replaces that with the measured delay. Queueing and publish rate
contribute only 30 ms — the rest is the OcuSync → phone → TCP → decode path.

**Supersedes** the previous note that the decoder rebuild had lowered the video
latency and that `VO_DELAY` might now over-subtract. It measures 0.44–0.51 s on
the current build, so 0.40 **under**-subtracts.

**Consequence for the gains.** `L` is now 0.79–0.85 s rather than 0.56–0.68.
[M] `kp_xy = 0.6` nevertheless flew stable and well damped on both flights
(`controller_design.md` §4.1). At `L = 0.82` s, 60° of phase margin gives
`kp ≤ 0.64` at `K = 1.0` and `0.43` at `K = 1.5`, so the flown gain is
consistent with the margin only if the effective gain on this path is near
unity. **The plant gain on the closed-loop path has not been re-measured** — the
attempt is in `controller_design.md` §4.1 and was unusable — so 0.6 is
demonstrated, not justified.

For plant `v = K·u` with position as the integrator, open loop
`Kp·K·e^{−sL}/s`, crossover `ω_c = Kp·K`, phase margin `90° − 57.3·ω_c·L`.

| target margin | `ω_c` | `Kp` at `K = 1.5` | `Kp` at `K = 1.0` |
|---|---|---|---|
| 60° | 0.78 rad/s | 0.52 | 0.78 |
| 45° | 1.17 rad/s | 0.78 | 1.17 |

**Recommended start: `Kp = 0.4` horizontal**, sized against the worst-case `vx`
gain of 1.5. [M] Validated in flight: closed-loop on mocap feedback was stable
and well-damped from `Kp = 0.25` to `Kp = 0.6`, and on EKF feedback at
`Kp = 0.6` (see `controller_design.md` §4 and §4.1).

Predicting the pose forward over its age would remove the 0.40 s term, leaving
`L ≈ 0.28 s` and roughly doubling the achievable bandwidth.

**Terminal accuracy.** At `Kp = 0.4`, the position error at which the commanded
velocity equals the drift floor is 1.4 cm (median drift) to 3.5 cm (p90).

---

## 9. Parameters for `dji_node`

```yaml
command_mode: "vel"
sign_vx:  1.0
sign_vy: -1.0
sign_vz:  1.0
sign_yaw: -1.0
v_max_horizontal: 1.0
v_max_vertical:   0.5
yaw_rate_max:    30.0
```

Yaw scale compensation (`×1.31`) is not applied here: it belongs to the
controller, so `command/vel` stays a physical-units interface and the
compensation is visible where the gain is chosen.

---

## 10. Method notes

- **Bag receive time is bursty.** [M] Mocap sample interval: median 8.3 ms, p1
  0.6–0.9 ms, p99 29 ms. Point-to-point differentiation is invalid and inflated
  velocities by up to 3×. All velocities here are least-squares slopes of
  position over ≥ 0.5 s windows, or derivatives of position resampled onto a
  uniform grid.
- **The measurement window is shifted by the command latency**, so it sits on
  the motion the command produced rather than straddling the acceleration ramp.
  Without the shift, gains read 20–25 % low.
- **A plateau check is reported per pulse** (`d%`, the disagreement between
  window halves). Above 20 % the pulse is flagged and excluded from
  conclusions.
- **Small-amplitude pulses are drift-corrected** using the dwells bracketing
  each pulse, rotated into that pulse's body frame.
- **Gains use the component along the commanded axis**, not the speed
  magnitude, which would rectify lateral drift into an apparent gain.
- **`analyse_pulses.py --selftest`** injects a known gain, latency and frame
  offset and recovers them. Per `filter_design.md` §12.4: validate the tool
  before trusting it on a bag.

