# Localisation — Operating Conditions

What the filter needs to work, what it cannot do, and the routine that puts it
in a healthy state before an inspection.

Design and constants: `filter_design.md`. This document does not repeat them.

> **[M]** measured against OptiTrack on the seven `F*` bags.
> **[?]** uncertain — stated because it matters, not because it is settled.

---

## 1. Measured envelope

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

**Avoid:** slow translation parallel to a near, flat surface; pure hover; pure
vertical motion; rotation without translation; any profile that drives frequent
map rebuilds. **Log the `vo_epoch` rate** — more than about one rebuild per
30 s means the filter cannot converge between them.

### 4.2 Scale is unobservable without transit

`s` is observed only by the velocity update, only above `V_LOW`. [M] Below it
DJI's gain droops to 0.77–0.82 against 0.93 at 1 m/s — a bias, not noise, so
inflating `R` cannot remove it. During a pass the scale column is disabled and
`s` is held.

### 4.3 `p_x`, `p_y` are dead-reckoned

Nothing observes horizontal position, so `P_xx` and `P_yy` grow monotonically
and absolute error grows without bound. **Evaluate and gate on RPE over short
windows, never on ATE or `tr(P_pp)`.**

### 4.4 Published orientation yaw is not trustworthy

[M] The attitude yaw residual has mean −22.8° and sd 28.0°, against roll and
pitch residuals of 0.30°/0.66° and 0.21°/0.80° which are correctly sized. Since
DJI attitude yaw tracks mocap to 0.19°, the error is in the VO chain. Position
is demonstrably unaffected (§1). **A consumer needing pointing direction should
use DJI attitude yaw directly, not the published orientation.**

### 4.5 A dead link looks like a stationary aircraft

Telemetry is deduplicated at source, so `TRANSPORT_GAP` does not fire reliably.
Until `dji_node` emits a fixed-rate heartbeat, comms loss is detected only by
the phone-side watchdog.

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

> **[?] `S_CLAMPED` is not implemented.** `s` at its 1e-3 floor freezes
> position by construction (`filter_design.md` §6.5) and is not reported.
> `S_STEP_MAX` has made it rare — [M] `s_clamp = 0` on all seven bags — but a
> consumer cannot currently detect it. One line in `_publish_status`:
> `if x.s <= 1.01e-3: flags.append("S_CLAMPED")`.

---

## 6. What the consumer should read

| field | use |
|---|---|
| `pose.position` | valid within the envelope above |
| `pose.covariance` [0:3,0:3] | grows without bound in x, y — see §4.3 |
| `sigma_scale / scale` | scale confidence; gate readiness on this |
| `state`, `degraded`, `flags` | `state` is exactly `"OK"` or `"DEGRADED"` and carries no more information than `degraded`. The **flags** say why |
| `pose.orientation` yaw | **do not trust** — see §4.4 |

---

## 7. Open questions that bear on operation

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
