# Localisation Filter — Design Specification

Loosely-coupled error-state EKF. Monocular VO supplies unscaled motion as the
**propagation input**; DJI altitude, velocity and attitude are the **updates**
that recover metric scale and anchor the navigation frame.

VO appears only in prediction. It is never an update.

---

## 1. Frames and notation

| Symbol | Meaning |
|---|---|
| `n` | navigation frame — `odom`, origin at takeoff, ENU, gravity-aligned |
| `v` | VO world frame — ORB-SLAM3 first-keyframe frame, published as `vo_world` |
| `b` | `base_link` — rear of body, coincident with the OptiTrack rigid-body origin |
| `c` | `camera_optical_frame` |
| `R_n_v` | VO→nav rotation. Unknown, slowly drifting — estimated |
| `R_v_c(t)` | camera attitude in `v`, from ORB-SLAM3 |
| `R_b_c(t)` | body→camera, from the gimbal buffer |
| `r_l` | `base_link → camera_optical_frame`, metric, constant |
| `Δp_v` | unscaled VO translation increment, in `v` |

```
R_n_b(t) = R_n_v · R_v_c(t) · R_b_c(t)⁻¹
```

**Error-state convention.** `R_n_v = exp([δθ]ₓ) · R̄_n_v`. After every update:
`R̄_n_v ← exp([δθ]ₓ)·R̄_n_v`, then `δθ ← 0`. Because `δθ ≡ 0` at every
linearisation point, the `H` blocks in §5 are exact rather than first-order.

**[M] `r_l = [0.117, 0.0, −0.030] m`.**

---

## 2. State

```
x = [ p_x, p_y, p_z, s, b, δθ_x, δθ_y, δθ_z ]              (8 states)
```

| | |
|---|---|
| `p` | body position in `n` (m) |
| `s` | VO scale factor, `ṡ = 0` |
| `b` | altitude sensor bias (m) |
| `δθ` | nav-frame misalignment (rad), random walk |

No velocity state — velocity is a deterministic function of the VO input and `s`.
No attitude state — attitude comes from VO; only its frame offset is estimated.

**Why `δθ_x, δθ_y`.** ORB-SLAM3's world frame is not gravity-aligned; a tilt θ
leaks horizontal motion into `p_z`, which the altitude update can only absorb
by biasing `s` — self-consistently, so NEES will not reveal it.

**Why `δθ_z`.** [M] DJI velocity NED and DJI attitude yaw use **different yaw
datums**: the offset measured 51.84° ± 3.81 on F9_02, constant across the
flight (slope +0.013° per degree of heading). `δθ_z` is what absorbs it.

---

## 3. Sources

| Source | Topic | Rate | Role |
|---|---|---|---|
| VO pose | `vo/pose` | ~20–32 Hz [M] | propagation input |
| VO status | `vo/status` | every frame | tracking state, `vo_epoch` |
| Altitude | `relative_altitude` | 8.9 Hz [M] | observes `p_z`, `b` |
| Velocity | `speed_vector` | 8.9 Hz [M] | observes `s`, `δθ_z` |
| Attitude | `attitude` | 8.9 Hz [M] | observes `δθ` |
| Gimbal | `gimbal_joint_attitude` | 13.8–19 Hz [M] | supplies `R_b_c` |

Altitude, velocity and attitude share one telemetry packet and one timestamp.
`vo/pose` carries no covariance; increment noise comes from `Σ_base · g` (§4.3).

**Ground truth (evaluation only).** OptiTrack ~100–108 Hz. Three handling rules:
conjugate the published quaternion; **drop bit-identical consecutive poses**
([M] 7.6 % on F9_02 — interpolating through repeats makes velocity a comb);
and do not use the pre-takeoff window for frame constants — [M] the rigid body
is untracked on the ground (1 unique pose in 1915 samples on F9_02).

---

## 4. Timing

**[M] Every telemetry channel lags the physical event, and by different
amounts.** Measured by cross-correlation against mocap on three bags:

| channel | lag vs mocap | differential vs attitude |
|---|---|---|
| attitude (yaw rate) | 38.1 ± 3.0 ms | — (reference) |
| altitude (dz/dt) | 58.0 ± 5.4 ms | +20 ms |
| velocity (\|v\|) | 73.2 ± 3.8 ms | **+35 ms** |
| **VO pose** | **423 ± 17 ms** | **+385 ms** |

Only differentials matter: the common term is a clock offset plus shared
latency and cancels. Re-stamping is applied in the **filter frontend**, never
in the publisher — re-stamping at the source corrupts recorded bags and
desynchronises topics that share a packet stamp.

| constant | value | basis |
|---|---|---|
| `VO_DELAY` | **0.40 s** | [M] 0.345 (F9_02), 0.418 (F3_02), 0.430 (F6c) vs attitude |
| `VEL_DELAY` | **0.035 s** | [M] velocity − attitude differential |
| `ALT_DELAY` | not modelled | 20 ms, below altitude's dynamics |

**Why `VO_DELAY` is so large.** `camera_decoder_node` stamps every frame with
`get_clock().now()` at the moment it leaves the decoder. No latency is
subtracted. The 0.40 s is the whole OcuSync → phone → TCP → PyAV path. The
decoder is already tuned for low latency (`nobuffer`, `low_delay`,
`analyzeduration 0`, `thread_type NONE`), so this is a property of the link,
not of the decode.

> **[?] `VO_DELAY` is per-flight**, spread ±45 ms across three bags, constant
> to ~20 ms within each. A fixed 0.40 leaves ±45 ms rather than the 400 ms it
> replaces. Stamping at capture on the phone would remove it properly.

### 4.1 Re-measurement on the current build (C01, C03)

[M] The attitude lag is confirmed: **0.039, 0.052, 0.054, 0.060 s** across four
yaw-impulse windows on C03 (corr 0.77–0.99), against 38.1 ± 3.0 ms previously.
[M] The velocity−attitude differential reads **0.034 and 0.041 s** in the two
long windows, confirming `VEL_DELAY = 0.035`.

[M] **`VO_DELAY` is under-subtracted at 0.40.** Three routes, all pointing the
same way:

| route | C03 | C01 |
|---|---|---|
| VO \|ω\| vs DJI attitude \|ω\|, two 60–70 s yaw windows | **0.506, 0.506** (corr 0.61, 0.41) | not measured |
| same, whole flight | 0.495 (corr 0.18) | 0.437 (corr 0.25) |
| time shift minimising the moving-window RPE | −0.15 s, ≈ −0.10 after the anchor differential | −0.05 s |

So the true delay is **0.44–0.51 s**, still per-flight, and the applied 0.40
biases every published pose stamp early by 40–100 ms. At 0.3 m/s that is 1–3 cm
against a measured 4–5 cm position error, so it is not the limiting term.

> **Correlation needs excitation.** [M] Over a whole 870 s flight with p90 yaw
> rate of 1.8 deg/s the attitude channel correlates at 0.05 and the VO routes at
> 0.18. Restricted to 60–70 s windows containing yaw impulses, the same
> computation gives 0.77–0.99 and 0.41–0.61. Measure timing on windows with
> motion in them, not on whole flights.

---

## 5. Prediction

### 5.1 Increment conditioning

1. **Epoch guard, first.** If `vo_status.vo_epoch` changed, re-anchor, do not
   propagate, raise `VO_DISCONTINUITY`, apply the §8 reset policy.
2. **Pair.** `ΔT_c = T_v_c(t₀)⁻¹ · T_v_c(t₁)`; rotate into `v`.
3. **Guards.** Discard and re-anchor if the implied speed disagrees with
   `K_VEL·|v_dji|` beyond `GATE_V`, if VO rotation disagrees with the DJI
   attitude delta beyond `GATE_R`, or if `Δt ∉ [DT_MIN, DT_MAX]`.
4. **Gimbal lookup** by SLERP over the ring buffer. True interpolation, no ZOH.
5. **Lever arm** `Δℓ = (R_n_b(t₁) − R_n_b(t₀))·r_l`, metric. Never summed with
   `Δp_v`, which is not.

### 5.2 Propagation

With `u = R̄_n_v · Δp_v`:

```
p⁺ = p + s·u − Δℓ        s⁺ = s        b⁺ = b        δθ⁺ = δθ

       ⎡ I₃   u   0   −s·[u]ₓ ⎤
F  =   ⎢ 0    1   0      0    ⎥
       ⎢ 0    0   1      0    ⎥
       ⎣ 0    0   0     I₃    ⎦

Q = blkdiag( s²·(R̄_n_v Σ_v R̄_n_vᵀ), q_s·Δt, q_b·Δt, q_θ·Δt·I₃ )
```

**`q_s` must be non-zero.** [M] With `q_s = 0`, `P_ss` decreases monotonically;
combined with an understated `R_speed` it collapsed to 1 % relative on a 15 %
error and `s` froze. `q_s = 1e-4` keeps `s` able to move.

### 5.3 Process noise gain

`Σ_v = Σ_base · g`, `g = g_feat · g_slew`, `g_feat = clip(n_ref/n_map_points, 1, CAP)`.

**Sizing rule, not a free knob:** keep `s²·Σ_base` well below `q_b·Δt`, or
`p_z` absorbs the barometric drift instead of `b`.

---

## 6. Updates

### 6.0 Admission

**Dedupe altitude, not velocity.** [M] `T_HOLD = 0` for velocity: with the hold
applied, accepted velocity updates over 50 s fell from 279 to 50, and the
survivors were individually *worse* — residual rms 0.145 → 0.208 m/s and
`angle(z,h)` 3.93° → 7.80°. The surviving samples cluster at accelerations
where DJI and VO agree least. Altitude keeps the hold: repeated 0.1 m bins in
hover are genuinely uninformative and `P_zz` otherwise shrinks by a spurious √N.

> **Supersedes** the earlier reasoning that the dedupe was harmless. That was
> tested in simulation, where per-sample SNR is far better than on hardware.

**Ordering.** Strict effective-timestamp order, Joseph form, symmetrise after
every step.

### 6.1 Only velocity may move `s`

```python
K = P Hᵀ S⁻¹
if kind is not VELOCITY:
    K[IDX_S, :] = 0
```

**Why.** Altitude and attitude do not touch `s` in `H`, but propagation builds
a `p_z`–`s` cross-covariance via `F[IDX_P, IDX_S] = u`, and the gain moves `s`
through it — entangled with `b`. [M] Measured on F9_02: `cov_s_b` climbed
1e-5 → 2.9e-3 and `s` slid 3.61 → 3.19 over 75 s of slow flight while the
velocity update was correctly gated out. Blocking altitude alone left a 1.6 %
residual leak through `P_s_θ`; blocking both holds `s` exactly. Joseph form is
valid for any gain, so `P` stays symmetric and PSD.

### 6.2 Attitude — all three axes

```
z = Log( R_n_b^dji · R̄_n_b⁻¹ ),   H = [0₃ₓ₃ | 0 | 0 | I₃],   y = z
R_att = diag(σ_rp², σ_rp², σ_yaw²)
```

| | value | basis |
|---|---|---|
| `σ_rp0` | **0.012 rad (0.69°)** | [M] residual sd 0.66° roll, 0.80° pitch |
| `σ_yaw` | **0.011 rad (0.63°)** | [M] DJI attitude vs mocap: 0.19° sd over 229 s (F9_02), 0.28° (F3_02) |
| `accel_slope` | 1/g | physics, not tuned |

`σ_rp` is acceleration-scheduled: a quadrotor tilts to accelerate, so a
gravity-referenced tilt estimate under-reports by ~`atan(a_h/g)`. `a_h` must
come from differentiated DJI velocity, never from DJI's own tilt.

### 6.3 Altitude

```
h(x) = p_z + b,   H = [0, 0, 1, 0, 1, 0, 0, 0]
```

[M] Gain is unity; no direction-dependent correction. `R_alt` switches on
`|v_z| > VZ_INFL`. `b` cannot be datumed on the ground (the key returns exactly
0.000 when grounded), so its prior must be wide.

### 6.4 Velocity

```
w_v = Δp_v/Δt,  u_w = R̄_n_v·w_v
h(x) = s·u_w − Δℓ/Δt
H    = [ 0₃ₓ₃ | u_w·(speed ≥ V_LOW) | 0₃ₓ₁ | −s·[u_w]ₓ ]
z    = v_enu / K_VEL,   R = R_speed_eff
```

**`s` is refined on transits and held through passes.** The scale column is
zeroed unless **both** sides carry speed: `|v_dji| ≥ V_LOW` (measurement) and
`|u_w| ≥ V_VO_MIN` (prediction). [M] On F6 the VO increment was near zero
(87 % of `|h|` below 0.02 m/s) while DJI occasionally read 0.3–0.4 m/s, so the
innovation was the whole measurement and the scale gain was unbounded. [M] Below it the velocity error is a *bias*, not noise —
DJI's gain droops toward 0.77–0.82 against 0.93 at 1 m/s — so `V_INFL` slows
the damage without preventing it: an 87 s inspection segment dragged `s` from
3.53 toward 2.7 while `σ_s` stayed under 8 %.

**`R` must include the increment noise.** `h = s·u_w` is built from a
*measured* increment whose noise `Σ_v` otherwise enters only `Q`:

```
R_speed_eff = R_speed_base/K_VEL² + s_ref²·(R̄_n_v Σ_v R̄_n_vᵀ)/Δt²
```

[M] Scaled by `s ≈ 3.5` and divided by `Δt ≈ 0.03`, that term is *larger* than
`R_speed` itself (0.11 vs 0.057 m/s). Adding it dropped median accepted NIS
from 4.88 to 1.82 while `R_speed_h` was simultaneously reduced 3×. `s_ref`, not
the live `s`: `R` must not depend on the state being estimated.

| | value | basis |
|---|---|---|
| `K_VEL` | **0.87** | [M] see below |
| `R_speed_h` | 0.0011 m²/s² | [M] |
| `V_MIN` / `V_LOW` | 0.15 / 0.40 m/s | [M] dead-zone thresholds |
| `NIS_GATE_VEL` | 60.0 | loose: `H[:,s] = u_w` makes `S` asymmetric in the innovation sign, so a tight gate culls one sign |

**`K_VEL = 0.87`, by two independent routes.** Integrated path ratio against
mocap over windows ≥ 0.4 m/s gave 0.875 (F9_02) and 0.856 (F3_02); the method
reads 1–1.5 % low, putting it at 0.87–0.88. Independently, ground truth
requires init `s ≈ 3.78` on F9_02, and `s ∝ 1/K_VEL`, giving 0.873. Confirmed
on an untuned bag: F3_02's fitted scale factor is **0.9945 ± 0.036**.

## 7. Initialisation

- `p_x, p_y = 0`; **`p_z = z_alt − b_prior`**. [M] Init happens mid-flight, and
  altitude is takeoff-relative, so `p = 0` makes the first altitude residual
  the whole height: 240 of ~800 updates NIS-rejected and a permanent 1.5 m
  offset. `p_z` is observed by nothing else, so the rejection is self-sustaining.
- `R̄_n_v = R_n_b^dji(t₀) · R_b_c(t₀) · R_v_c(t₀)⁻¹`. This gravity-aligns `n`.
- `s` estimated from `K_VEL⁻¹·|v_dji| / (|Δp_v|/Δt)` over samples with
  `|v_dji| > V_LOW`.
- **Gate on PATH, not time.** Collection requires 60 samples spanning
  `INIT_PATH_M = 3.0 m`. [M] The previous 12 s excitation gate ran *in series*
  with collection and was unsatisfiable on inspection profiles: four of seven
  bags published nothing. `INIT_TRANSLATION_S` was sized for monocular
  parallax, which ORB-SLAM3 has already had by the time it reports `pose_valid`.

> **[?] The scale estimator is noisy.** [M] Per-sample scatter sd 3.54 on a
> median of 4.17 — 84 % relative — because the ratio divides by an
> instantaneous VO speed. Two draws from the same bag gave 3.776 (n=60) and
> 4.222 (n=131), a difference within its own standard error. No speed or time
> dependence was found. An integrated path ratio would remove the division;
> untested. Not critical while `estimate_scale = 1`, since the filter converges
> from any start — it matters only for the held configuration.

---

## 8. Failure handling

VO is the propagation input, so losing it is a **propagation gap**, not a
rejected measurement. Key on `pose_valid`, never on the string `LOST` — in pure
monocular, ORB-SLAM3 overwrites `LOST` within the same `Track()` call.

| Condition | Action |
|---|---|
| any state ≠ OK, once initialised | dead-reckon on `K_VEL⁻¹·v_dji`, `Q_p` inflated |
| no `vo/pose` > `VO_TIMEOUT` | dead-reckon |
| telemetry gap > 500 ms | degraded; hold state, inflate `Q` |

---

## 8. Observability

| Quantity | Observed by | Degenerate when |
|---|---|---|
| `s` | velocity, along `u_w`, above `V_LOW` | no transit |
| `b` | altitude over vertical range | small height range |
| `p_z` | altitude | — |
| `δθ_x, δθ_y` | attitude roll/pitch | — |
| `δθ_z` | attitude yaw; velocity ⊥ `u_w` | — |
| `p_x, p_y` | **nothing** — dead-reckoned | always |

Consequences: `P_xx`, `P_yy` grow monotonically, so **never gate degradation on
`tr(P_pp)`** — use `σ_s` or innovation consistency. Publish `odom→base_link`
only. **Evaluate with RPE over short windows, not ATE** — ATE on a
dead-reckoned `x, y` measures elapsed time, not quality.

---

## 9. Output contract

| Topic | Type | Consumer |
|---|---|---|
| `localisation/pose` | `PoseWithCovarianceStamped` | controller, planner, evaluation |
| `localisation/status` | `LocalisationStatus` | state machine |
| `/tf` | `TFMessage` | `odom→base_link` only |

Position from `p`. Orientation composed at publish time from
`R̄_n_v · R_v_c · R_b_c⁻¹`. **Stamp with the (VO_DELAY-corrected) VO stamp,
never `now()`** — confirmed by evaluation: the fitted mocap offset on a clean
segment is +0.015 to +0.025 s against an independently measured attitude lag of
+0.035 s.

Covariance is required, not optional: the planner needs localisation
uncertainty at viewpoint-selection time, and that coupling is the research
contribution.

---

## 10. Measured performance

F9_02, configuration of §13, against OptiTrack. RPE over 1 s windows per
`vo_epoch` segment. `scale` is the fitted residual factor: 1.00 = correct.

| segment | profile | scale | RPE 1 s (scale-corrected) | as the system runs |
|---|---|---|---|---|
| 0, 87 s | transit | 1.020 | **6.2 %** | 6.5 % |
| 2, 87 s | inspection after transit | **1.003** | 8.6 % | 8.4 % |
| 1, 22 s | [?] anomalous | 0.169 | 67 % | 125 % |

[M] `K_VEL = 0.87` is confirmed independently on F3_02, which was never tuned
against: fitted scale 0.985 ± 0.047.

The cross-bag envelope and the conditions it depends on are in
`operating_conditions.md` §1; they are not repeated here.

---

## 11. Configuration

```
estimate_scale 1.0    sigma_yaw 0.011    R_speed_h 0.0011
q_s 1e-4              T_HOLD 0.0         K_VEL 0.87
VO_DELAY 0.40         VEL_DELAY 0.035    INIT_PATH_M 3.0
V_VO_MIN 0.05         S_STEP_MAX 0.15
```

This is the configuration both C01 and C03 flew. Two constants are known to be
off and neither has been changed:

- **`VO_DELAY 0.40` under-subtracts by 40–100 ms** (§4.1). Worth 1–3 cm at
  inspection speed against a 4–5 cm measured error. A move to 0.45 is inside
  every live estimate and is the next thing to confirm in flight; the replay
  route cannot settle it (§12.7).
- **`K_VEL 0.87` reads high** for the flown speed range (§6.4), and the gain is
  speed-dependent to ~1 m/s.
