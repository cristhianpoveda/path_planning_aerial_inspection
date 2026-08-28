# Localisation Filter — Design Specification

Loosely-coupled error-state EKF. Monocular VO supplies unscaled motion as the
**propagation input**; DJI altitude, velocity and attitude are the **updates**
that recover metric scale and anchor the navigation frame.

VO appears only in prediction. It is never an update.

> **[M]** measured against OptiTrack. **[T]** still to determine.
> **[?]** open question — stated because it matters, not because it is settled.
>
> Revision 2026-08-28. Values here are measured on bags F3_01, F3_02, F6, F6c,
> F8, F9, F9_02 unless noted. Where a figure comes from one bag only, that is
> stated: several constants proved to be **per-flight**, not universal.

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

> **[?] The offset is per-flight.** Estimated at init on five bags: 53.5°
> (F9_02), 41.1° (F8), 26.5° (F3_01), 9.7° (F3_02), 152.6° (F6), 141.8° (F6c).
> Its source is not identified — see §12.

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

> **Supersedes** the previous claim that the attitude delay is under 20 ms and
> therefore not common-mode. Measured at 38 ms; the velocity differential is
> 35 ms, not the 75 ms previously recorded.

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

> **[?] The attitude yaw residual is not Gaussian and not small.** [M] On
> F9_02: roll mean +0.30° sd 0.66°, pitch +0.21° sd 0.80° — both correctly
> sized — but **yaw mean −22.8°, sd 28.0°**, giving 29 % acceptance at NIS 94.
> Since DJI attitude yaw tracks mocap to 0.19°, the error is in the VO chain
> (`R_LINK_OPTICAL`, the gimbal yaw negation in `camera_decoder`, or
> ORB-SLAM3 yaw drift), not in DJI. **Do not inflate `σ_yaw` to compensate** —
> that models a bias as noise. Unresolved; it costs published-orientation
> accuracy but demonstrably not position (see §11).

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

> **Supersedes** `K_VEL = 0.91`, which came from regression against
> differentiated mocap — differentiation at 100 Hz aliases noise and attenuates
> fitted gains. Raw arc length has the same failure: it measures the noise's
> own path, inflating low-speed path by 7–13 % and producing a fake dead-zone
> droop. Always resample → smooth → differentiate.

### 6.5 One update may not rewrite `s`

```python
if kind is VELOCITY and |dx[IDX_S]| > S_STEP_MAX · s:
    dx[IDX_S] = sign(dx[IDX_S]) · S_STEP_MAX · s
```

`s` is a per-flight constant estimated from many samples (§7). A single
innovation with authority to halve it means `P_ss` is wide and the sample is an
outlier, not that the scale changed.

[M] Without the cap, two velocity updates 100 ms after init moved `s`
2.778 → 1.032 → 0.247 on F6 and then to the 1e-3 clamp, where `p⁺ = p + 0.001·u`
froze position for 23 s of a 43 s flight (`|Δp_est|/|Δp_gt| = 0.003`). Both
updates passed every gate: `|u_w|` was 0.169 and 0.147, above `V_VO_MIN`.

`S_STEP_MAX = 0.15`. [M] Fires 12 times on F6, 8 on F6c, **2** on F9_02 over
200 s — inert on transit flights, where `s` converges in far smaller steps.

> **[?] The cap is applied to `dx`, not to `K`.** The Joseph form below uses
> the unlimited `K`, so `P_ss` contracts as though the full step had been
> taken. Measured `σ_s` stays at 0.47–0.50 on both F6 and F9_02, so the
> inconsistency is not currently biting. Scaling `K[IDX_S, :]` instead would be
> the consistent form; revisit if `σ_s` ever shrinks while `s_lim` climbs.

---

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

### 8.1 Reset policy on `vo_epoch` change

**Publishing never stops.** A rebuild invalidates `s` alone — `p` and `b` live
in the nav frame. [M] Dropping out of the initialised state cost 130 s of a
229 s flight, because the cold-start path then waited for excitation the
aircraft never supplied.

| Reason | Detection | Action |
|---|---|---|
| Sim(3) loop closure / merge | epoch changed, state stayed OK | re-anchor; inflate `P_ss`; keep `s` |
| Map rebuild / re-init | epoch changed via NOT_INITIALIZED | re-anchor; keep `s`; inflate `P_ss`; request re-measure |

**Do not reset `s` to 1.0.** It becomes *unknown*, not *one*, and the position
loop keeps running through the transient. On F9_02 the reset cost ~90 s of
recovery from a 3.8× scale error.

**Cap the inflation** at `min(P_ss·4, (0.15·s)²)`. [M] Uncapped, `P0_scale`
took `σ_s` to 28 % relative, which re-opened the `p_z`–`s` cross-covariance
before §6.1 was in place.

**The re-measure works.** [M] `_maybe_rescale` completed three times on F8
(`s` 2.951 → 3.799 and 3.754 → 4.231 from ~2 m of path) and once on F3_01
(4.040 → 5.002). It does not complete on a flight that never exceeds `V_LOW`.

> **[?] Whether a rebuilt map keeps the previous scale is bag-dependent.** On
> F9_02 two maps agreed to 1 % (implied truth 3.85 vs 3.79). On F8 the fitted
> factors ran 1.05–1.33 across 16 rebuilds. With a rebuild every ~12 s, `s` is
> re-measured before it has converged — a VO stability limit, not a gate
> problem. See `operating_conditions.md` §4.1.

---

## 9. Observability

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

## 10. Output contract

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

## 11. Measured performance

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

> **[?] Segment 1 is unexplained.** 2.2× excess motion with correct yaw
> (0.27° within-window) and a frame fit that reads 101°/127°/178° across
> otherwise identical runs — instability characteristic of a discontinuity
> *inside* the segment rather than a constant error.

---

## 12. Evaluation strategy

Every claim in §4, §6 and §11 came from one of the procedures below. They are
recorded so a change can be re-checked cheaply, and so a new bag can be brought
into the envelope without re-deriving the method.

### 12.1 Which bag answers which question

| bag | profile | what it is for |
|---|---|---|
| **F9_02** | transit + inspection, 5 epochs | the reference. Any change is regression-tested here first |
| **F3_02** | fast square | untuned cross-check. Its fitted scale (0.9945–0.985) is the independent confirmation of `K_VEL` |
| **F8** | vertical squares, 16 epochs in 191 s | behaviour under rapid map churn. Not a tuning case |
| **F6, F6c** | slow sweep at a wall | the failure boundary. VO under-translation, not a filter case |
| **F3_01** | slow square, 7 epochs in 106 s | as F8: no segment survives long enough to evaluate |
| **F9** | inspection, no transit | the correct-refusal case: never reaches `INIT_PATH_M` |

Never tune on F9_02 alone. [M] `psi_v`, epoch-straddling init pools and an
acceleration bias each looked well-founded on F9_02 and were each refuted —
twice by a cross-bag check, once by measuring the estimator's own scatter.

### 12.2 Regression, after any change (~10 min)

Replay **F9_02** and **F6** with `innovation_log` set and
`/drone_1/localisation/pose` recorded, then:

```bash
grep -E 'initialised:|rebuild|rescaled' $LOG
grep 'HEALTH' $LOG | tail -1
python3 scale_series.py $CSV 180
python3 evaluate_pose.py $POSE_BAG F9_02_mocap --flight F9_02
```

Pass conditions:

| check | expected | meaning if it moves |
|---|---|---|
| F9_02 seg 0 `RPE1s%` | 6.2 % | the change hurt the working case |
| F9_02 seg 2 `scale` | 1.00 ± 0.01 | scale estimation broken |
| accept rates | alt 99.9 %, vel ~62 %, att ~28 % | a channel started rejecting |
| `s_clamp` | 0 | `s` went non-physical |
| `s_lim` | ≤ 2 on F9_02 | the step cap is throttling real convergence |
| F6 `ratio1s` | 0.286 | position integration changed |

Current reference values: F9_02 seg 0 `RPE1s%` 6.21, seg 2 8.57, `s_lim` 2;
F6 `RPE1s%` 72.21, `ratio1s` 0.286. **If a change improves these, update the
table** — it is a reference, not a ratchet.

**Compare accept RATES, not HEALTH counts.** The counters are cumulative from
process start and scale with replay length; comparing counts across runs of
different length produces false alarms.

### 12.3 Full sweep, before a release (~40 min)

All seven bags, same configuration, `evaluate_pose.py` per bag. Report the §11
table plus the envelope in `operating_conditions.md` §1. A change that improves
F9_02 and degrades any other bag is not an improvement.

### 12.4 Characterisation, when a constant is in doubt

Each measures one quantity against mocap with no filter in the loop:

| script | measures | notes |
|---|---|---|
| `gt_align2.py` | flight↔mocap clock offset | three cues; requires all three to agree within ~50 ms |
| `channel_lags2.py` | per-channel latency incl. VO | differentials cancel the common clock term |
| `vo_delay.py` | `VO_DELAY`, mocap-free | VO \|ω\| vs DJI attitude \|ω\|; rejects segments with peak < 0.45 |
| `kvel_check.py` | `K_VEL` | integrated path ratio. Reads 1–1.5 % low |
| `yaw_datum.py` | which yaw datum is displaced | `A − B` is contaminated by the rigid-body offset — read `A` and `B` separately |

Two method rules, both learned by getting them wrong:

- **Resample → smooth → differentiate.** Raw arc length at 100 Hz sums the
  noise's own path: [M] it inflated low-speed path 7–13 % and produced a fake
  dead-zone droop in `K_VEL`.
- **Validate the tool on synthetic data with a known injected value before
  trusting it on a bag.** Every script above recovers a known input to better
  than 1 %; two of them did not until the validation exposed a sign error and a
  midpoint-stamping bias.

### 12.5 Analysing the innovation log

| script | question |
|---|---|
| `scale_series.py` | did `s` hold, and are the accept rates sane |
| `decompose_velocity.py` | is the velocity residual a direction error or a magnitude error |
| `residual_structure.py` | is it a scale error or a constant offset, and is `R` sized right |
| `gate_symmetry.py` | is the NIS gate culling one sign |

`evaluate_pose.py` reports two RPE columns. `RPE1s%` has scale removed by
Umeyama — the filter's achievable accuracy. `ro1s%` fixes the frame but not
scale — what the live system produces. **The gap between them is the cost of
the scale error**; when they converge, scale is no longer the limiting term.

A time offset pinned at `--scan` is not a fit: it means `s` is not constant
within that segment, and every number for that segment should be discarded.

---

## 13. Configuration

```
estimate_scale 1.0    sigma_yaw 0.011    R_speed_h 0.0011
q_s 1e-4              T_HOLD 0.0         K_VEL 0.87
VO_DELAY 0.40         VEL_DELAY 0.035    INIT_PATH_M 3.0
V_VO_MIN 0.05         S_STEP_MAX 0.15
```

### Decisions and superseded alternatives

- **VO is the propagation input, not a measurement.** No IMU is exposed, so VO
  noise enters `Q`, never `R`.
- **`s` and `b` are filter states**, which rules out `robot_localization`.
- **`estimate_scale = 0` does not hold `s` by itself.** It only zeroes the
  velocity Jacobian; the covariance must be held too (`P_ss` row and column
  zeroed). [M] Without it, an `s` seeded exactly at truth was dragged to 73 % of
  it in 80 s.
- **Only velocity moves `s`** (§6.1).
- **`K_VEL` is a constant, not a state** — co-linear with `s`.
- **Velocity re-stamping and `VO_DELAY` are applied in the frontend**, not in
  the publishers.
- **The attitude yaw row is enabled.** DJI attitude yaw is the best-measured
  quantity in the system (0.19° vs mocap).
- **A per-flight velocity yaw offset (`psi_v`) was tried and reverted.** [M]
  Applied in either sign it made things worse — attitude acceptance fell to
  61 % and 29 % respectively, from 98 %. The measured 53.5° is the disagreement
  between the attitude-seeded nav frame and the VO-derived heading, which is
  what `δθ_z` already absorbs; correcting the velocity signal removed the
  update's ability to correct the frame. It is still logged at init as a
  diagnostic.
- **`q_θz` is not sized from DJI yaw drift.** The −1.2° over 116 s figure is
  contradicted: DJI attitude yaw shows no drift against mocap over 229 s.

### Retired

The mocap↔nav tilt of 1.152° at bearing −164.9° and the mocap↔`base_link` yaw
of 128.5° **could not be re-verified** — the rigid body is untracked on the
ground, and the airborne fit gives −104.6° (F9_02) and −122.0° (F3_02), which
differ between sessions. Treat both as per-session constants requiring a fresh
fit, not as inherited values.

---

## 14. Open work, in order

1. **The attitude yaw residual** (§6.2). [M] Mean −22.8°, sd 28.0° on F9_02,
   against correctly sized roll and pitch. DJI attitude yaw tracks mocap to
   0.19°, so the error is in the VO chain. Isolate whether it enters via
   `R_b_c` or via VO by comparing each against mocap separately. This costs
   published-orientation accuracy, not position — but a viewpoint planner
   consumes orientation.
2. **`no_inc` losses** — [M] 478 of 1224 velocity events on F9_02, 401 of 1353
   on F8, lack a covering increment.
3. **Integrated-path scale estimator at init** (§7), if the held configuration
   is ever needed. [M] The present per-sample estimator has 84 % relative
   scatter.
4. **Transport-gap detection.** Dedupe happens at source, so a stationary
   aircraft and a dead link are identical on the wire. Needs a fixed-rate
   liveness heartbeat in `dji_node`.
5. **F9_02 segment 1** (§11), 22 s of 176, unexplained.

**Not filter work.** F6, F6c, F8 and F3_01 fail on VO input quality — see
`operating_conditions.md` §4.1. [M] On each, the filter's own scale is correct
or recovers; no update-side change can compensate for motion absent from the
propagation input or a frame re-anchored faster than it converges.