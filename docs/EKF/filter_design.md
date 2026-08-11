# Localisation Filter — Design Specification

Loosely-coupled error-state EKF. Monocular VO supplies unscaled motion as the
**propagation input**; DJI altitude, velocity and attitude are the **updates**
that recover metric scale and anchor the navigation frame.

VO appears only in prediction. It is never an update.

> Constants marked **[M]** are measured against OptiTrack (bags of 2026-08-06).
> Constants marked **[T]** await ORB-SLAM3 and NEES tuning.

---

## 1. Frames and notation

| Symbol | Meaning |
|---|---|
| `n` | navigation frame — `odom`, origin at takeoff, ENU, **gravity-aligned** |
| `v` | VO world frame — ORB-SLAM3's arbitrary first-keyframe frame |
| `b` | `base_link` — **rear of body, coincident with the OptiTrack rigid-body origin** |
| `c` | `camera_optical_frame` |
| `R_n_v` | VO-frame → nav-frame rotation. **Unknown, slowly drifting — estimated** |
| `R_v_c(t)` | camera attitude in `v`, from ORB-SLAM3 (~10 Hz) |
| `R_b_c(t)` | body→camera rotation, from tf (dynamic gimbal, 19 Hz) |
| `r_l` | `base_link → camera_optical_frame` translation in `b`, metric, **constant** |
| `r_imu` | `base_link → DJI IMU` translation in `b`, metric, constant |
| `Δp_v` | unscaled VO translation increment, in `v` |

Derived body attitude:

```
R_n_b(t) = R_n_v · R_v_c(t) · R_b_c(t)⁻¹
```

**`r_l` is constant.** [M] `r_l = [0.117, 0.0, −0.030] m`, `|r_l| = 0.121 m`
(`base_link → gimbal_base = [0.105, 0, −0.025]` plus
`gimbal_base → camera_link ≈ [0.012, 0, −0.005]`; `camera_link` and
`camera_optical_frame` share an origin). The gimbal-fixed component is
second-order: the gimbal holds camera roll/pitch near-fixed in `n`, so that
2 cm arm contributes ~1 mm to `Δℓ` versus ~9 mm from the body-fixed part.

**Error-state convention.** `R_n_v` is carried as a nominal `R̄_n_v` outside the
filter plus an error rotation `δθ` inside it:

```
R_n_v = exp([δθ]ₓ) · R̄_n_v
```

After every update: `R̄_n_v ← exp([δθ]ₓ)·R̄_n_v`, then `δθ ← 0`. `P` is retained.

---

## 2. State

```
x = [ p_x, p_y, p_z, s, b, δθ_x, δθ_y, δθ_z ]              (8 states)
```

| | |
|---|---|
| `p` | body position in `n` (m) |
| `s` | VO scale factor (unitless), `ṡ = 0` |
| `b` | altitude sensor bias (m), constant per flight |
| `δθ` | nav-frame misalignment (rad), random walk |

No velocity state — velocity is a deterministic function of the VO input and `s`.
No attitude state — attitude comes from VO at ~10 Hz; only its frame offset is
estimated.

> **Why `δθ_x, δθ_y` exist.** ORB-SLAM3's world frame is not gravity-aligned. A
> tilt of θ leaks horizontal motion into `p_z`: a 5 m traverse at θ = 5° gives
> 0.44 m of spurious altitude change, ~12σ_alt. The altitude update can only
> absorb that by biasing `s`, and it is self-consistent, so NEES will not
> reveal it.
>
> **Why `δθ_z` exists.** [M] DJI velocity is NED referenced to **magnetic
> north**, while `odom` is aligned to the room. The two differ by an angle
> measured at **~120°** in the flight arena — recovered independently from the
> velocity fit (117°, 119°) and from the attitude comparison (122°, 127°). That
> angle is unknown at runtime and drifts with the magnetometer. Without it the
> velocity update biases `s` by `cos ψ`.

---

## 3. Sources

| Source | Rate | Role |
|---|---|---|
| ORB-SLAM3 increments | ~10 Hz | propagation input (the quantity being scaled) |
| Altitude `RelativeAltitudeStamped` | **10 Hz** [M] | observes `p_z`, `b` |
| Velocity `Vector3Stamped` | **10 Hz** [M] | observes `s` directly and `δθ_z` |
| Attitude `AttitudeStamped` | **10 Hz** [M] | observes `δθ` |
| Gimbal `AttitudeStamped` (tf) | **19 Hz** [M] | supplies `R_b_c` |

Altitude, velocity and attitude **share one telemetry packet and one measurement
timestamp** [M] — verified by change-detection co-occurrence, so no
inter-source interpolation error.

Gimbal attitude arrives on a **separate 19 Hz packet with its own timestamp**
[M]. It is faster than the camera, so tf2 interpolation to any image time is
available within ≤53 ms.

**Timestamps.** All telemetry carries `header.stamp` in the ground-station ROS
clock, mapped from the phone monotonic clock by a min-over-window offset
estimator, with the phone-side stamp taken in a DJI key listener (true arrival
time). Residual velocity delay measured at ~75 ms — see §5.3.

Camera frames are stamped at decode time minus `VIDEO_LATENCY` **[T]**.

---

## 4. Prediction — driven by VO increments (~10 Hz)

### 4.1 Increment conditioning

1. **Pair.** `ΔT_c = T_v_c(t₀)⁻¹ · T_v_c(t₁)` → `Δp_c`, `Δt = t₁ − t₀`.
   Rotate into `v`: `Δp_v = R_v_c(t₀)·Δp_c`.

2. **Discontinuity guard.** Discard, re-anchor to `T_v_c(t₁)`, **do not
   propagate**, raise `VO_DISCONTINUITY` if any of:
   - map ID changed, or loop-closure / global-BA event signalled;
   - `ŝ·|Δp_v|/Δt` disagrees with `|v_dji|` beyond `GATE_V` [T] (enabled once
     `σ_s < SIGMA_S_OK`);
   - VO relative rotation disagrees with the DJI attitude delta over the same
     interval beyond `GATE_R` [T];
   - `Δt ∉ [0.5, 2.0] ×` nominal frame period.

3. **tf lookup.** `R_b_c` needed at both `t₀` and `t₁`. The gimbal runs at 19 Hz
   against a ~10 Hz camera, so tf2 interpolation covers `t₁` within ≤53 ms.
   Use true interpolation; no ZOH fallback and no significant buffering delay.

4. **Lever arm.** The gimbal auto-stabilises and cannot be locked, so the camera
   translates relative to `base_link` on any airframe attitude change. With
   `r_l` constant:

   ```
   Δℓ = ( R_n_b(t₁) − R_n_b(t₀) ) · r_l          [metres]
   ```

   `Δℓ` is metric, `Δp_v` is not — they cannot be summed here. Emit separately.
   Treated as state-independent (its `δθ` dependence scales with `|r_l|` and is
   second-order). Magnitude: ~9 mm per increment at 50 °/s, ~25% of `σ_alt`,
   systematic during manoeuvres — hence compensated rather than absorbed in `Q`.

5. **Emit** `{Δp_v, Δℓ, Δt, Σ_v, g, flags}`.

### 4.2 Propagation

With `u = R̄_n_v · Δp_v`:

```
p⁺  = p + s·u − Δℓ
s⁺  = s
b⁺  = b
δθ⁺ = δθ                                    (random walk)
```

```
       ⎡ I₃   u   0   −s·[u]ₓ ⎤
F  =   ⎢ 0    1   0      0    ⎥              (8×8)
       ⎢ 0    0   1      0    ⎥
       ⎣ 0    0   0     I₃    ⎦
```

```
Q = blkdiag( s²·(R̄_n_v · Σ_v · R̄_n_vᵀ),  q_s·Δt,  q_b·Δt,  q_θ·Δt·I₃ )
P⁺ = F·P·Fᵀ + Q
```

`q_s` [T] low but non-zero — holds `s` through hover.
`q_b` [T] ≈ 0 — `b` constant per flight; the state absorbs per-flight datum
variation, confirmed by the spread of fitted `b` across bags.
`q_θ` [T] low — tracks VO orientation drift and slow magnetic variation.

### 4.3 Process noise gain

```
Σ_v = Σ_base · g          g = g_track · g_feat · g_slew
```

| Factor | Condition | Value |
|---|---|---|
| `g_track` | `OK` | 1.0 |
| | `RECENTLY_LOST` | 10 |
| | `LOST` | no increment — see §6 |
| `g_feat` | always | `clip(n_ref / max(n_feat,1), 1, CAP)` [T] |
| `g_slew` | always | `1 + κ·‖ω_gimbal‖` [T] |

`κ` is expected to tune **low**: interpolation error over the 53 ms gimbal
period is much smaller than the 208 ms this term was originally sized for.

---

## 5. Updates

All three FC sources are 10 Hz and share one measurement timestamp. Process
strictly in timestamp order. Joseph-form covariance update throughout:

```
y = z − h(x);   S = H·P·Hᵀ + R;   K = P·Hᵀ·S⁻¹
x⁺ = x + K·y;   P⁺ = (I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ
```

### 5.1 Attitude

Applied **first** in each packet — it corrects the frame the other two are
compared in.

```
R̄_n_b = R̄_n_v · R_v_c(t) · R_b_c(t)⁻¹
y      = Log( R_n_b^dji · R̄_n_b⁻¹ )              (3-vector, nav frame)
H      = [ 0₃ₓ₃ | 0 | 0 | I₃ ]
R_att  = diag(σ_rp², σ_rp², σ_yaw²)
```

Because the residual is in the nav frame and nav z is up, the diagonal maps
onto the trust split with no special-casing in code.

| | | |
|---|---|---|
| `σ_rp` | **0.035 rad (2°)** | [M] in-flight, vibration included. Static floor is 0.1°; the in-flight residual is structured rather than white, so 2° is deliberately conservative |
| `σ_yaw` | **large — skip the yaw row** | **[T]** see §10 |

Static yaw drift measured at −1.2° over 116 s [M] — slow and bounded, which is
encouraging but not yet a substitute for a characterised `σ_yaw`.

### 5.2 Altitude — `RelativeAltitudeStamped`

Height above takeoff, terrain-blind. With `n` origin at takeoff it observes
`p_z` directly — no terrain term.

**[M] The gain is unity.** `KeyAltitude` fits OptiTrack with slope
`1 + k` where `k_up = −0.0020`, `k_down = +0.0002`, `k_hover = −0.0008` — all
within 0.2% of one, from 338 climb and 366 descend samples. The
direction-dependent correction inherited from the old `altitude_agl` key
(`k_up = −0.0699`, `k_down = −0.0468`) **does not apply to this key and is
removed**, along with `VZ_DEAD` and `K_DEFAULT`.

```
h(x) = p_z + b
H    = [ 0, 0, 1, 0, 1, 0, 0, 0 ]
R    = R_alt
```

| | | |
|---|---|---|
| `R_alt` (hover) | **0.00142 m²** (σ = 0.038 m) | [M] reproduces the earlier 0.00138 independently |
| `R_alt` (vertical motion) | **0.0104 m²** (σ = 0.102 m) | [M] noise grows ~7× during climbs |
| quantisation floor | 0.00083 m² (q²/12, q = 0.1 m) | ~60% of hover variance |
| `b` prior mean | **+0.042 m** | [M] from the vertical-sweep fit |
| `VZ_INFL` | 0.15 m/s | [T] threshold for switching to the inflated `R_alt` |

**`b` cannot be datumed on the ground.** `KeyAltitude` returns exactly 0.000
when grounded [M], so a grounded start carries no bias information. Take the
prior from the fitted value above and re-estimate per flight as a state.

No detectable ground effect: hover-bias-versus-height slopes were small and
inconsistent in sign across bags [M].

### 5.3 Velocity — `Vector3Stamped`

**[M] Frame is world (NED, magnetic-north-referenced).** Confirmed on three
bags with fit-residual ratios 3.53, 1.19, 1.76 against the body-frame
hypothesis. Gains: z = **−0.92 to −0.95** (down-positive, confirmed); the
horizontal 2×2 block is a **rotation of ~120°, not a signed permutation** —
so `axis_perm`/`signs` do not apply to x/y and the angle is estimated as
`δθ_z`.

**[M] The lag is a delay, not a filter.** Fitted one-pole `τ = 0.03–0.06 s`
(under half a sample period) with a pure delay of 50–100 ms. So the matched-
filter approach of applying `L{·}` to the prediction is unnecessary: **re-stamp
the measurement at `t − VEL_DELAY`** and treat it as white. This removes the
increment-buffer dependency from the velocity update.

```
w_v   = Δp_v / Δt                          (unscaled, in v, latest increment)
u_w   = R̄_n_v · w_v
h(x)  = s·u_w − Δℓ/Δt − ω_b × r_imu        (lever arms: DJI measures IMU-point
                                            body motion, VO measures camera)
H     = [ 0₃ₓ₃ | u_w | 0₃ₓ₁ | −s·[u_w]ₓ ]  (3×8)
R     = R_speed
```

One 3-vector, two observable quantities in orthogonal directions: **`s` along
`u_w`, `δθ_z` perpendicular to it.**

| | | |
|---|---|---|
| `VEL_DELAY` | **0.075 s** | [M] mean of 50/50/100/0 ms, quantised to the 50 ms analysis grid — do not over-fit |
| `R_speed` | **0.002 m²/s²** (σ ≈ 0.045 m/s) | [M] ~2.4× the quantisation floor; far cleaner than the "heavily pre-filtered, inflate generously" assumption |
| NED→ENU | `(x,y,z) → (y,x,−z)` | [M] z sign confirmed; residual yaw handled by `δθ_z` |
| `r_imu` | `[0.07, 0.0, 0.0] m` | `base_link` is at the body rear, so the DJI IMU sits ~7 cm forward |

**`r_imu` matters only during rotation.** `|ω × r_imu| ≈ 6 cm/s` at 50 °/s,
about 1.3σ of `R_speed`; at inspection rates (10 °/s) it is 1.2 cm/s and
negligible. Folded into the same lever-arm term as `Δℓ` rather than modelled
separately.

---

## 6. Failure handling

VO is the propagation input, so losing it is a **propagation gap**, not a
rejected measurement.

| Condition | Action |
|---|---|
| `RECENTLY_LOST` | propagate normally, `g_track = 10` |
| `LOST`, `VO_DISCONTINUITY`, or tf timeout | dead-reckon: `p⁺ = p + v_dji,n·Δt`, `Q_p` inflated; hold `s`, `b`, `δθ` |
| **telemetry transport gap** (no packet > 500 ms) | degraded; hold state, inflate `Q` |
| degraded > recovery timeout | state machine forces `Landing` |

Dead reckoning uses `R_n_b` **from DJI attitude directly**, not from the VO
chain — the VO chain is stale precisely when this path is active.

Degradation flag = `LOST` ∨ `VO_DISCONTINUITY` ∨ `TRANSPORT_GAP` ∨
`σ_s > SIGMA_S_MAX`. Do **not** gate on `tr(P_pp)` — see §7.

> Distinguish **transport gap** (no packets arriving — a real fault) from
> **value staleness** (packets arriving, values unchanged — normal when
> stationary, since DJI values are quantised and the stamp only advances on
> change).

---

## 7. Observability

| Quantity | Observed by | Degenerate when |
|---|---|---|
| `s` | velocity (direct, along `u_w`) | no motion |
| `s` | altitude (via `p_z` growth) | no vertical motion |
| `b` | altitude, over vertical range | small height range — correlates with `s` |
| `p_z` | altitude | — |
| `δθ_x, δθ_y` | attitude (roll/pitch, strong) | — |
| `δθ_z` | velocity (perpendicular to `u_w`); attitude yaw if trusted | no horizontal motion |
| `p_x, p_y` | **nothing** — dead-reckoned | always |

Consequences that change code:

- **`P_xx`, `P_yy` grow monotonically** and no update reduces them. Gating
  degradation on `tr(P_pp)` will fire on any sufficiently long healthy flight.
  Gate on `σ_s` or innovation consistency instead.
- **Publish `odom→base_link` only.** `map→odom` is identity, published
  separately (static publisher in `localisation.launch.py`), replaced later by
  an ArUco or other global corrector with no downstream change.
  `localisation/pose.header.frame_id = "odom"`.
- **Evaluate with RPE over short windows**, not ATE — ATE is dominated by drift
  this filter structurally cannot correct.
- Monitor the `s`–`b` off-diagonal of `P`, not just the marginals.
- With `σ_yaw` set large, **`δθ_z` depends entirely on the velocity update**, so
  it is unobservable in pure hover. Hold it with low `q_θ`.

---

## 8. Initialisation

- `p = 0`, `P_pp` small (takeoff datum).
- `b` ~ N(+0.042, σ_b²) [M] — re-estimate per flight; **do not datum on the
  ground**, `KeyAltitude` reads exactly zero there.
- **`R̄_n_v` from DJI attitude at `t₀`:**
  `R̄_n_v = R_n_b^dji(t₀) · R_b_c(t₀) · R_v_c(t₀)⁻¹`.
  This is what gravity-aligns the nav frame. Without it the entire chain
  inherits ORB-SLAM3's arbitrary initial orientation.
- `δθ = 0`; `P_θθ` per attitude uncertainty at init — **loose in yaw** (the
  room-to-magnetic-north offset is ~120° in this arena and is absorbed by
  `δθ_z`, so initialise `σ_δθz` wide).
- `s`: **not** a fixed constant. Estimate from `|v_dji| / (|Δp_v|/Δt)` over the
  first excited window; `P_ss` wide until `σ_s < SIGMA_S_OK`.
- §4.1 plausibility gate disabled until `σ_s < SIGMA_S_OK`.
- Gate init on the clock-offset tracker being warm and on the aircraft being
  airborne (altitude > 0.3 m and moving) — DJI keys return defaults on the
  ground.
- Monocular VO cannot initialise under pure rotation. Allow a deliberate
  translation after takeoff before declaring the filter ready.

---

## 9. Tuning and validation

- Tune `Σ_base`, `q_s`, `q_b`, `q_θ`, `κ`, `n_ref`, `CAP` by NEES / normalised
  innovation consistency vs OptiTrack (target ≈ 1).
- **Fix the OptiTrack rigid body before trusting any yaw comparison.** The
  current marker layout is near-symmetric, so the solver flips yaw by ~180°
  during rotation [M] — visible as ±170° plateaus in the attitude residual and
  as a 180° discrepancy in the fitted mocap↔`base_link` offset. Rebuild with an
  asymmetric layout.
- Identify `L`/`VEL_DELAY` from the **velocity residual against OptiTrack**,
  not from the autocorrelation of the velocity signal — the latter is dominated
  by smooth physical motion and over-estimated `τ` by ~5× in this dataset.
- When differentiating mocap position, **resample to the analysis grid first,
  then differentiate** — differentiating at native mocap rate and interpolating
  afterwards aliases noise and inflated the fitted gains by ~4.5×.
- **Check `δθ` against OptiTrack directly** — it is an estimated quantity with
  ground truth available, so it is independently verifiable rather than only
  visible through its effect on `s`.
- Vertical sweep with real height range to separate `s` from `b`.
- **Verify `F` and both `H` by finite differences before any flight data.** The
  `−s·[u]ₓ` blocks and the error-state reset make sign errors easy and quiet: a
  wrong sign yields a filter that converges plausibly and is wrong, and NEES
  tuning will not reveal it because the result is self-consistent.
- **Run offline on a rosbag, deterministically, from the first commit.** Every
  step above is an iteration over recorded data; a filter that only runs live
  cannot be tuned.

---

## 10. Constants

### Measured [M]

| Constant | Value |
|---|---|
| `r_l` | `[0.117, 0.0, −0.030] m` |
| `r_imu` | `[0.07, 0.0, 0.0] m` |
| `R_alt` (hover / vertical) | `0.00142` / `0.0104 m²` |
| `b` prior mean | `+0.042 m` |
| altitude gain | **unity** — `k` terms removed |
| velocity frame | world, NED, magnetic-north-referenced |
| velocity z sign | negative (down-positive) |
| `VEL_DELAY` | `0.075 s` |
| `R_speed` | `0.002 m²/s²` |
| `σ_rp` | `0.035 rad` |
| source rates | FC 10 Hz, gimbal 19 Hz |
| quantisation | altitude 0.1 m, velocity 0.1 m/s |

## 11. Decisions and superseded alternatives

Recorded because each replaced a plausible alternative that may otherwise
resurface.

- **VO is the propagation input, not a measurement.** Nützi/MSF treat VO as an
  update because an IMU propagates; no IMU is exposed here, so the roles invert
  and VO noise enters `Q`, never `R`. Listing VO under both would double-count
  the same data.
- **`s` and `b` are filter states, not preprocessing.** Rules out
  `robot_localization`, whose 15-state vector cannot be extended — unscaled VO
  fused against metric altitude there is averaged, not scaled.
- **Velocity is the primary scale cue, not altitude.** Only velocity's Jacobian
  touches `s` directly; altitude reaches it via the `p_z`–`s` cross-covariance
  and arrives entangled with `b`.
- **Altitude enters as a measurement model, not a pre-correction.** Dividing the
  measurement without dividing its covariance would misstate `R_alt`.
- **`odom→base_link` only.** `p_x, p_y` are unobserved, so this filter cannot
  claim global drift correction; `map→odom` stays identity until a corrector
  exists.
- **RPE over short windows, not ATE.** ATE is dominated by drift the structure
  cannot correct.
- **DJI velocity is world-frame NED, magnetic-north-referenced**, at ~120° to
  the room frame in this arena — hence `δθ_z` rather than an axis permutation.

### To determine [T]

| Constant | Source | Blocks |
|---|---|---|
| `σ_yaw` | asymmetric mocap rigid body + yaw survey | attitude yaw row (currently skipped) |
| `VIDEO_LATENCY` | camera-yaw-rate cross-correlation vs composed `R_n_b·R_b_c` | all VO timestamping |
| `Σ_base`, `n_ref`, `CAP`, `κ` | NEES tuning with VO | prediction noise |
| `q_s`, `q_b`, `q_θ` | NEES tuning | prediction noise |
| `GATE_V`, `GATE_R` | VO flight tests | discontinuity gating |
| `SIGMA_S_OK`, `SIGMA_S_MAX` | VO flight tests | init and degradation |
| `VZ_INFL` | altitude residual vs `|vz|` | `R_alt` switching |

Nothing in the **[T]** list blocks implementation — all have workable defaults.
