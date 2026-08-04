# Localisation Filter — Measurement Sources & Design

Loosely-coupled EKF that scales a no-scale monocular VO trajectory using altitude and
platform velocity. All measurements enter after per-sensor correction.

## Sources

### 1. Monocular VO/SLAM — ORB-SLAM3
- Provides: up-to-scale trajectory + orientation (gimbal camera). Consume **relative
  pose increments**, not absolute pose (see loop-closure note below).
- Role: the quantity being scaled (state carries scale factor `s`).
- Covariance: Based on the filter's state:
  - `R_vo = R_base * gain(state, n_features)`
  - `state = OK`          -> gain 1.0            (nominal `R_base`)
  - `state = RECENTLY_LOST` -> gain high (10x)  (coast on altitude/vel)
  - `state = LOST`        -> reject VO update       (do not fuse)
  - additionally scale by features: `gain *= clip(n_ref / max(n_feat,1), 1, cap)`
    (few tracked map points -> low-texture -> inflate R). `n_ref` ~ nominal count.
- This feature-count/tracking signal doubles as the planner's VO-degradation input.

The EKF receives relative increments between consecutive frames. On **Loop-closure / BA discontinuity:** the increment is resetto avoid propagating a global jump as motion.

### 2. Altitude (`RelativeAltitudeStamped`)
- Model: `z_dji = (1+k)·z_true + b + nu`
- `k_up   = -0.0699`   (ascending)
- `k_down = -0.0468`   (descending)
- `b      = -0.0466 m` (grounded-start bag)
- `R_alt  = 0.00138 m^2`  (sigma 0.037 m; floor q^2/12 = 0.00083)
- Rate 4.8 Hz, quantised 0.1 m.
- Correction before update:
  `k = k_up if vz<0 else k_down;  z_corr = (z_dji - b)/(1+k)`
- `b` treated as constant (no drift model). Re-init datum each takeoff.

### 3. Speed (`Vector3Stamped`) — CHARACTERISE!!
- **Filtered & lagged**: DJI velocity is pre-smoothed (lag-1 autocorr ~0.8–0.95) and
  trails truth — it is NOT a white measurement. Must be lag-corrected and R-inflated.
- Frame: body-vs-world unresolved (needs a yaw-excited flight to disambiguate).
- Rate 4.8 Hz, quantised 0.1 m/s. Vertical component sign: down-positive (negate).
- Placeholders (fill after the yaw-excited flight):
  - `frame`        = <body | world>
  - `axis_perm`    = <(i,j,k)>
  - `signs`        = <(±,±,±)>
  - `velocity_lag` = <___ ms>
  - `R_speed_raw`  = [<vx>, <vy>, <vz>] (m/s)^2
  - `R_speed_infl` = [<vx>, <vy>, <vz>] (m/s)^2   <- use this (AR(1)-inflated)
- Design note: because it is filtered/lagged/correlated, treat velocity as a
  **secondary** scale cue; altitude is the stronger anchor.

## Proposed filter

- **Type:** loosely-coupled EKF (Nützi-style: scale as explicit state; VO as input).
- **State:** `x = [p(3), v(3), s (scale), b_alt]`  (add speed-lag/bias states only if needed).
- **Prediction:** integrate scaled VO increments; `s_dot = 0` with low process noise
  (hold through hovers — scale is static when stationary).
- **Updates:**
  - Altitude: corrected `z_corr`, `R_alt` (per direction-dependent `k`).
  - Velocity: corrected (frame + sign + lag), inflated `R_speed_infl`; optionally
    downsample to reduce correlation. Negate vertical (down-positive).
  - VO: relative pose increments from ORB-SLAM3; `R_vo` scaled by tracking state +
    feature count (reject when LOST). Reset increment on loop-closure/BA jumps.
- **Degeneracies handled:** hover -> altitude/velocity carry no scale info, so `s`
  held via low process noise; smooth/straight motion -> weak scale excitation (known);
  low-texture -> VO R inflated via feature count (altitude carries scale there).
- **Validation:** run full EKF vs OptiTrack; tune all R via NEES / normalised-innovation
  consistency (target innovation variance ~1). Apply mocap glitch-cleaning to GT first.

## Open items
- Speed characterisation (yaw-excited flight) -> fill section 3 placeholders.
- Tune `R_base`, `gain` thresholds, and `n_ref` for the VO update via NEES.
- Optional: altitude `b` drift model (needs a ~10-min hover + Allan).
