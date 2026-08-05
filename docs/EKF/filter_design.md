# Localisation Filter — Design Specification

Loosely-coupled error-state EKF. Monocular VO supplies unscaled motion as the
**propagation input**; DJI altitude, velocity and attitude are the **updates**
that recover metric scale and anchor the navigation frame.

VO appears only in prediction. It is never an update.

---

## 1. Frames and notation

| Symbol | Meaning |
|---|---|
| `n` | navigation frame — `odom`, origin at takeoff, ENU, **gravity-aligned** |
| `v` | VO world frame — ORB-SLAM3's arbitrary first-keyframe frame |
| `b` | `base_link` |
| `c` | `camera_optical_frame` |
| `R_n_v` | VO-frame → nav-frame rotation. **Unknown, slowly drifting — estimated** |
| `R_v_c(t)` | camera attitude in `v`, from ORB-SLAM3 (10 Hz) |
| `R_b_c(t)` | body→camera rotation, from tf (dynamic gimbal, ~17.5 Hz) |
| `r_l(t)` | `base_link → camera_optical_frame` translation in `b`, metric, time-varying |
| `Δp_v` | unscaled VO translation increment, in `v` |

Derived body attitude:

```
R_n_b(t) = R_n_v · R_v_c(t) · R_b_c(t)⁻¹
```

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
No attitude state — attitude comes from VO at 10 Hz; only its frame offset is
estimated.

> **Why `δθ` exists.** ORB-SLAM3's world frame is not gravity-aligned. A tilt of
> θ leaks horizontal motion into `p_z`: a 5 m traverse at θ = 5° produces 0.44 m
> of spurious altitude change, ~12σ_alt. The altitude update can only absorb that
> by biasing `s`. Tilt error is a first-order scale corruptor, and it is
> self-consistent, so NEES will not reveal it.

---

## 3. Sources

| Source | Rate | Role |
|---|---|---|
| ORB-SLAM3 increments | ~10 Hz | propagation input (the quantity being scaled) |
| Altitude `RelativeAltitudeStamped` | 10 Hz | observes `p_z`, `b` |
| Velocity `Vector3Stamped` | 10 Hz | observes `s` directly, and `δθ_z` if world-frame |
| **Attitude (telemetry)** | 10 Hz | observes `δθ`. Roll/pitch gravity-referenced; yaw magnetometer-referenced |

Altitude, velocity and attitude share one telemetry packet and one measurement
timestamp — no inter-source interpolation error.

Gimbal attitude arrives on a separate ~17.5 Hz packet with its own timestamp, so R_b_c is not synchronous with the FC trio. tf2 interpolation handles it; the point is that it's a distinct clock domain.

---

## 4. Prediction — driven by VO increments (~10 Hz)

### 4.1 Increment conditioning

1. **Pair.** `ΔT_c = T_v_c(t₀)⁻¹ · T_v_c(t₁)` → `Δp_c`, `Δt = t₁ − t₀`.
   Rotate into `v`: `Δp_v = R_v_c(t₀)·Δp_c`.

2. **Discontinuity guard.** Discard, re-anchor to `T_v_c(t₁)`, **do not
   propagate**, raise `VO_DISCONTINUITY` if any of:
   - map ID changed, or loop-closure / global-BA event signalled;
   - `ŝ·|Δp_v|/Δt` disagrees with `|v_dji|` beyond `GATE_V` (enabled once
     `σ_s < SIGMA_S_OK`);
   - VO relative rotation disagrees with the DJI attitude delta over the same
     interval beyond `GATE_R`;
   - `Δt ∉ [0.5, 2.0] ×` nominal frame period.

3. **No need to wait for tf** The gimbal updates faster than the ~10 Hz camera, so the buffering wait drops from ~208 ms to ≤57 ms. Keep true tf2 interpolation — drop the ZOH fallback entirely.

4. **Lever arm.** The gimbal auto-stabilises and cannot be locked, so the camera
   translates relative to `base_link` on any airframe attitude change:

   ```
   Δℓ = R_n_b(t₁)·r_l(t₁) − R_n_b(t₀)·r_l(t₀)          [metres]
   ```

   `Δℓ` is metric, `Δp_v` is not — they cannot be summed here. Emit separately.
   Treated as state-independent: its `δθ` dependence scales with `|r_l| ≈ 0.1 m`
   and is second-order.

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

`q_s` low but non-zero — holds `s` through hover.
`q_b ≈ 0` — `b` constant per flight; the state absorbs per-flight datum variation.
`q_θ` low — tracks VO orientation drift, not motion.

### 4.3 Process noise gain

```
Σ_v = Σ_base · g          g = g_track · g_feat · g_slew
```

| Factor | Condition | Value |
|---|---|---|
| `g_track` | `OK` | 1.0 |
| | `RECENTLY_LOST` | 10 |
| | `LOST` | no increment — see §6 |
| `g_feat` | always | `clip(n_ref / max(n_feat,1), 1, CAP)` |
| `g_slew` | always | `1 + κ·‖ω_gimbal‖` (tf interpolation error during slews) |

note κ is expected to tune low, since interpolation error over 57 ms is much smaller than the 208 ms the term was sized for.

---

## 5. Updates

All three sources are 10 Hz, stamped with measurement time. Process strictly in
timestamp order (IMPORTANT). Joseph-form covariance update throughout:

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

Because the residual is expressed in the nav frame and nav z is up, the diagonal
maps cleanly onto the trust split: **`σ_rp` small** (accelerometer-referenced,
absolute, drift-free), **`σ_yaw` large** (magnetometer-referenced; indoor
disturbance near steel is routinely tens of degrees). No special-casing in code —
the split lives entirely in `R_att`.

Set `σ_yaw → ∞` (skip the yaw row) if a magnetometer-health flag is available.

### 5.2 Altitude — `RelativeAltitudeStamped`

Height above takeoff, terrain-blind. With `n` origin at takeoff it observes `p_z`
directly — no terrain term. Applied as the measurement model, not as a
pre-correction, so `R_alt` is not distorted by division:

```
h(x) = (1+k)·p_z + b
H    = [ 0, 0, (1+k), 0, 1, 0, 0, 0 ]
R    = R_alt
```

```
k = k_up     if  v_z,up > +VZ_DEAD      (ascending)
    k_down   if  v_z,up < −VZ_DEAD      (descending)
    K_DEFAULT otherwise                 (hover — the two differ by 2.5%, below σ_alt)
```

> `v_z,up = −v_z,raw`. Confirmed: DJI reports negative when ascending, so the raw
> signal is down-positive.

| | |
|---|---|
| `k_up` | −0.0699 |
| `k_down` | −0.0468 |
| `K_DEFAULT` | `k_down` |
| `b` prior mean | −0.0466 m (grounded-start bag) — `b` is a state, not a constant |
| `R_alt` | 0.00138 m² (σ = 0.037 m; quantisation floor q²/12 = 0.00083) |
| quantisation | 0.1 m |

Effective altitude information rate stays below 10 Hz because of 0.1 m binning

### 5.3 Velocity — `Vector3Stamped`

DJI velocity is pre-filtered and lagged, so it is not a white measurement of
instantaneous velocity. **Apply the same lag to the prediction rather than
inflating R to cover the bias.** With `L{·}` the identified low-pass and `s`
constant, `s·L{w} = L{s·w}`, so:

```
w_v   = L{ Δp_v / Δt }                     (unscaled, in v, from the increment buffer)
u_w   = R̄_n_v · w_v
h(x)  = s·u_w − L{Δℓ/Δt}                   (lever-arm rate: DJI measures body
                                            motion, VO measures camera motion)
H     = [ 0₃ₓ₃ | u_w | 0₃ₓ₁ | −s·[u_w]ₓ ]  (3×8)
R     = R_speed_infl
```

One 3-vector, two observable quantities in orthogonal directions: **`s` along
`u_w`, `δθ_z` perpendicular to it.**

Frame handling, once characterisation resolves it:

| If `frame` = | Do | Scale sensitivity to yaw error |
|---|---|---|
| world (NED) | fixed permutation NED→ENU, no rotation | **biased by `cos ψ`** — 20° → 6%. `δθ_z` needed |
| body (FRD) | negate y,z → FLU, rotate by `R_n_b` | **cancels exactly** — same frame both sides |

Signs: z-component `−` (confirmed). Remaining placeholders resolve together from
the yaw-excited flight — `frame`, `axis_perm`, `signs`, `L`, `R_speed_infl` are
one question, not five.

---

## 6. Failure handling

VO is the propagation input, so losing it is a **propagation gap**, not a
rejected measurement.

| Condition | Action |
|---|---|
| `RECENTLY_LOST` | propagate normally, `g_track = 10` |
| `LOST`, `VO_DISCONTINUITY`, or tf timeout | dead-reckon: `p⁺ = p + v_dji,n·Δt`, `Q_p` inflated; hold `s`, `b`, `δθ` |
| degraded > recovery timeout | state machine forces `Landing` |

Dead reckoning uses `R_n_b` **from DJI attitude directly**, not from the VO
chain — the VO chain is stale precisely when this path is active. This is the
gap the attitude source closes.

Degradation flag = `LOST` ∨ `VO_DISCONTINUITY` ∨ `σ_s > SIGMA_S_MAX`.
Do **not** gate on `tr(P_pp)` — see §7.

---

## 7. Observability

| Quantity | Observed by | Degenerate when |
|---|---|---|
| `s` | velocity (direct, along `u_w`) | no motion |
| `s` | altitude (via `p_z` growth) | no vertical motion |
| `b` | altitude, over vertical range | small height range — correlates with `s` |
| `p_z` | altitude | — |
| `δθ_x, δθ_y` | attitude (roll/pitch, strong) | — |
| `δθ_z` | attitude (yaw, weak); velocity if world-frame | body-frame velocity + bad magnetometer |
| `p_x, p_y` | **nothing** — dead-reckoned | always |

Consequences that change code:

- **`P_xx`, `P_yy` grow monotonically** and no update reduces them. Gating
  degradation on `tr(P_pp)` will fire on any sufficiently long healthy flight.
  Gate on `σ_s` or innovation consistency instead.
- **Publish `odom→base_link` only.** `map→odom` is identity, published separately
  (static publisher in `localisation.launch.py`), replaced later by an ArUco or
  other global corrector with no downstream change.
  `localisation/pose.header.frame_id = "odom"`.
- **Evaluate with RPE over short windows**, not ATE — ATE is dominated by drift
  this filter structurally cannot correct.
- Monitor the `s`–`b` off-diagonal of `P`, not just the marginals.

---

## 8. Initialisation

- `p = 0`, `P_pp` small (takeoff datum).
- `b` ~ N(−0.0466, σ_b²) — re-datum each takeoff.
- **`R̄_n_v` from DJI attitude at `t₀`:**
  `R̄_n_v = R_n_b^dji(t₀) · R_b_c(t₀) · R_v_c(t₀)⁻¹`.
  This is what gravity-aligns the nav frame. Without it the entire chain inherits
  ORB-SLAM3's arbitrary initial orientation.
- `δθ = 0`, `P_θθ` per attitude uncertainty at init (loose in yaw).
- `s`: **not** a fixed constant. Estimate from `|v_dji| / (|Δp_v|/Δt)` over the
  first excited window; `P_ss` wide until `σ_s < SIGMA_S_OK`.
- §4.1 plausibility gate disabled until `σ_s < SIGMA_S_OK`.

---

## 9. Tuning and validation

- Tune `Σ_base`, `q_s`, `q_b`, `q_θ`, `R_speed_infl`, `σ_rp`, `σ_yaw` by NEES /
  normalised innovation consistency vs OptiTrack (target ≈ 1). Clean mocap
  glitches from GT first.
- Identify `L` from the **velocity residual against OptiTrack**, not from the
  autocorrelation of the velocity signal — the latter is dominated by smooth
  physical motion and over-estimates τ by roughly an order of magnitude.
- Vertical sweep with real height range to separate `s` from `b`.
- **Check `δθ` against OptiTrack directly** — it is now an estimated quantity
  with ground truth available, so it is independently verifiable rather than
  only visible through its effect on `s`.
- Characterise `σ_yaw` indoors at the flight location before trusting yaw at all.

---

## 10. Constants to determine

| Constant | Source | Blocks |
|---|---|---|
| `r_l` | tf chain at rest + CAD/tape | **propagation** — error biases `s`, not just variance |
| `frame`, `axis_perm`, `signs`, `L`, `R_speed_infl` | yaw-excited flight | velocity update (primary `s` cue) |
| `σ_rp`, `σ_yaw` | static test + indoor magnetometer check | attitude update |
| `Σ_base`, `n_ref`, `CAP`, `κ`, `q_s`, `q_b`, `q_θ` | NEES tuning | tuning only |
| `GATE_V`, `GATE_R`, `SIGMA_S_OK`, `SIGMA_S_MAX`, `VZ_DEAD` | flight tests | gating and state machine |
| `VIDEO_LATENCY`| Optitrack cross correlation | All VO timestamping |
