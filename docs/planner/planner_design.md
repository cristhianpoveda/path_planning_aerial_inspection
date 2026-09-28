# Viewpoint Planner — Design and Status

Offline planner that selects inspection viewpoints, visit order and edge speed
so that expected defect measurability is maximised **under the pose and scale
distribution the trajectory itself induces**.

---

## 1. What this is for

`filter_design.md` §9: `p_x, p_y` are observed by nothing. §6.4: `s` is observed
only by the velocity update, and DJI's velocity gain droops below ~1 m/s. So the
metric scale of the pose estimate — and therefore **achieved standoff, and
therefore ground sample distance on the defect** — is a function of how the
aircraft has been flying, not only of where it is.

Every planner in `second_lit_review.md` either computes imaging quality from a
nominal pose and buys the pose from infrastructure, or plans motion to reduce
estimator uncertainty with no imaging term. This one closes the loop:

```
plan  ->  sigma_s, Sigma_pp  ->  achieved standoff  ->  achieved GSD  ->  plan
```

**One line.** `Q` is nonlinear in depth, so `E[Q(d)] != Q(E[d])`. A planner that
evaluates quality at the nominal pose computes the right-hand side. This one
computes the left.

---

## 2. Scope

| decision | choice | why not the alternative |
|---|---|---|
| planning time | **offline** | [M] C01 and C03 held one VO epoch over 1565 s with no rebuild — nothing to replan against. Online also inherits `L ~ 0.79-0.85 s` and the §12.7 replay-vs-live divergence |
| online behaviour | **monitor and abort** | the state machine already owns `Recovering`; a second decision-maker is a hazard |
| environment | **known boxes + arena box** | exploration is a different contribution |
| decision variables | **viewpoint position, visit order, speed per edge** | orientation is fronto-parallel by construction (§4.2) |
| optimiser | **set-cover -> TSP, fixed-point loop on `sigma_s`** | same architecture as the baselines, so only `sigma_d` differs (§7.1) |
| initialisation | **manual, outside the planner** | the planner's problem starts at a known pose with a known `sigma_s` |
| capture | **continuous recording, frames chosen post hoc by pose** | no trigger logic, no timing race |

**Out of scope:** online replanning, exploration, occlusion-aware NBV,
continuous time allocation, orientation as a free variable, transit-leg
insertion (§11), large patches needing multiple lateral views.

---

## 3. Inputs, outputs, engagement

**In**
- `boxes` — obstacles and targets, axis-aligned boxes in the **arena frame**
- `arena` — the flyable volume
- `targets` — [T] placeholder: 2 patches on walls, 3 on interior boxes, each
  with a required defect width `w_t`
- `EkfParams` — the same object the flight configuration uses
- intrinsics — [M] `f_x = 1431.85`, 1920x1080, `camera_calibration.yaml`

**Out** — a `trajectory_msgs/MultiDOFJointTrajectory`: pose, velocity and
`time_from_start` per point. Chosen because it carries the speed profile in one
standard message; no new definition, no parallel arrays, no parameter races.

Emitted per viewpoint for evaluation: predicted `sigma_s/s`, predicted
`sigma_d`, `E[Q_def]`, nominal `Q_def`.

### 3.1 Engagement sequence

1. **Manual flight** until the filter initialises: >= `INIT_PATH_M` (3.0 m) at
   >= `V_LOW` (0.40 m/s), textured space, translation across the field of view.
2. **Hover at the checkerboard.** Compute `T(arena->odom)` (§9).
3. **Gate.** Accept the run only if `sigma_s/s < 0.10` at engagement. Consistent
   by construction: the forecast's initial condition is the `(0.10*s)^2` init
   floor, so predicted and actual start equal. Record the achieved value.
4. **Re-solve at the reported `s`.** [M] The forecast is **not** invariant to
   the planning scale (§5.1), so the offline plan is re-run through the
   fixed-point loop using the `s` that `localisation/status` reports at
   engagement. `s` is unknowable before flight — it is the arbitrary scale of
   whatever VO map was built — but it is known here, about a minute before the
   tour starts, and the loop takes seconds. The offline plan is the initial
   guess; the on-site solve is what flies.
5. **Verify the transform.** Command a hover at a known arena point and check
   against mocap before starting. Cheap insurance against a sign error.
6. **Engage the controller**, execute the tour.
7. **Close on the checkerboard.** The tour is a Hamiltonian cycle, so the board
   is imaged again at the end — giving accumulated `p_x, p_y` drift and a
   `K_VEL`-independent scale reading, per run.

The tour is planned in the arena frame and converted to `odom` by §9. **That
conversion is the only thing between a plan and a flight.**

---

## 4. The quality term

### 4.1 Composition

For a viewpoint at depth `d` from a patch with normal `n_hat` and incidence
`beta`, traversed at speed `v`:

```
g(d, beta) = (d / f_x) / cos(beta)              m per pixel on the surface
b_m        = f_x * t_exp * v_perp / d           px, motion smear
c          = k_dof * |d - d_f| / d              px, defocus
b_tot      = sqrt(b_0^2 + b_m^2 + c^2)          px, effective PSF width
w_min      = k_r * b_tot * g(d, beta)           m, finest resolvable feature
Q_def      = clip(w_t / w_min, 0, Q_max)
```

Quadrature composition turns GSD, blur and defocus into one scalar without three
arbitrary weights. `w_min` is **exactly what the printed line-pair panel
measures**, so the model is falsifiable rather than assumed.

### 4.2 Orientation

`phi_i` points at the patch centre along `-n_hat`. Not a decision variable.
Heading slews **during** the edge — the controller drives all four DOF to a
compound pose — so no rotation-without-translation occurs at nodes, which
`ekf_operating_conditions.md` §4.1 lists under **Avoid**.

Constraint: the required yaw change on an edge must be achievable at
`yaw_rate_max_deg` (15) within the edge duration. Rotational blur `b_omega` is
dropped on that basis. [A] Not measured.

### 4.3 Why defocus is not optional

Without it `Q_def ∝ 1/d`, which is **convex**, so `E[Q] > Q(E[d])` and pose
uncertainty would appear to *help*. Physically wrong, and it inverts the result.

Defocus makes `b_tot` grow either side of `d_f`, giving `Q_def` a peak and
concavity near it, hence `E[Q] < Q(E[d])`. [M] Verified numerically: with
placeholder constants the peak is at 1.29 m and `Q'' = -0.98` there.

**The DoF measurement (§10) is on the critical path, not a refinement.**

### 4.4 The expectation

Achieved depth is not planned depth:

```
sigma_d^2 = n_hat' * Sigma_pp * n_hat  +  ( (n_hat' * p) * sigma_s/s )^2
E[Q_def]  = sum_k omega_k * Q_def(d + xi_k * sigma_d, beta)     5-node Gauss-Hermite
```

- **position term** — [M] excursion while holding: median 0.062-0.071 m, max
  0.188 m (`controller_design.md` §4.1)
- **scale term** — proportional to **distance from the nav origin along the
  viewing normal**, not to standoff

**[M] The scale term dominates and varies strongly across the arena.** At
`sigma_s/s = 0.10`, `sigma_d` runs from ~0.07 m near the origin to ~0.40 m at
4 m out. That five-fold differential across candidates is what drives selection
— more than the absolute size of the effect at any one viewpoint.

---

## 5. The localisation model

`scale_forecast.py` drives the real `EkfCore` over a candidate trajectory. The
covariance recursion needs no measurement values, only `F, H, Q, R`, so the
prediction is consistent with the filter by construction rather than by
agreement between two models.

Mirrors `EkfCore.propagate` / `update_velocity` / `update_altitude` /
`update_attitude`, `IncrementBuilder.build`, `Scheduler._on_epoch`,
`ekf_node._maybe_rescale`, and the `(0.10*s)^2` floor from `_try_init`.

**Initial condition** `sigma_s/s = 0.10` — the init floor, the post-rescale
reset, and the engagement gate of §3.1, all the same number.

**Simplifications:**

| dropped | why |
|---|---|
| rebuild hazard | [M] zero rebuilds in 1565 s on C01 and C03; seven bags too thin to fit a rate |
| `n_map_points` from FOV geometry | unfitted, and never logged. See §6.5 |
| NIS rejection | forecast innovations are zero, so every update passes. [M] Real velocity acceptance ~62 %, so the forecast is **optimistic**; modelled as a constant `accept_rate` thinning |

### 5.1 [M] Speed is a hard threshold, not a graded trade-off

**Supersedes** the earlier claim that the `V_LOW` gate is replaced by a graded
droop. The scale column is gated on `|v_dji| >= V_LOW`, and
`v_dji = K_VEL * v_true`, so the gate is a **step** in true speed at
`V_LOW / K_VEL`. The measured droop only moves where that step sits:

| assumed gain | 0.87 (configured) | 0.845 (measured 0.3-0.5 m/s) | 0.80 |
|---|---|---|---|
| true speed needed | **0.46 m/s** | **0.47 m/s** | **0.50 m/s** |

[M] Confirmed by `check_forecast.py --tour` over a 5-viewpoint cycle:

| commanded | `v_dji` | `sigma_s/s` at the end | change |
|---|---|---|---|
| 0.3 | 0.26 | 0.115 | **+14 %** |
| 0.4 | 0.35 | 0.112 | **+11 %** |
| 0.6 | 0.52 | 0.016 | **-84 %** |
| 0.8 | 0.70 | 0.014 | **-86 %** |

Below the threshold `s` is unobservable and `sigma_s` grows on `q_s` alone;
above it, `sigma_s` collapses within one leg. **Speed levels must straddle
0.47 m/s** or the DP cannot express the trade-off (§6.4).

### 5.2 [M] The forecast is optimistic by about 2.2x

Forecast `sigma_s/s` settles at 0.014-0.016 above the threshold, against
[M] 0.031-0.035 measured on C01 and C03. The ratio is 2.2, against 1/0.62 = 1.6
from the velocity accept rate alone, with `no_inc` losses making up the rest.
Apply `accept_rate = 0.45` effective, or a 2.2x correction, and state which.

### 5.3 [M] The forecast is NOT scale-invariant

`sigma_s/s` at the end of a fixed tour, by planning scale:

| `s_ref` | 0.5 | 1.0 | 2.0 | 3.5 | 5.0 |
|---|---|---|---|---|---|
| `sigma_s/s` | 0.0231 | 0.0179 | 0.0156 | 0.0179 | 0.0230 |

A U-shape, minimum near `s = 2`, 48 % spread. Two effects: `q_s` is absolute,
so its relative contribution `q_s*dt/s^2` shrinks with `s`; while
`R_speed_eff` carries `s_ref^2` and `H[:, s] = u_w ∝ 1/s`, so information about
`s` falls off as `s` grows. They cross near 2.

**`q_s` is not made relative.** That would have the planner predicting a filter
that is not the one flying, which destroys the "consistent by construction"
argument for driving `EkfCore` at all. The plan is re-solved at the reported `s`
instead (§3.1 step 4).

---

## 6. The optimisation

### 6.1 Objective

```
maximise   J(plan) = w * Qbar  -  (1 - w) * That
```

- `Qbar` — mean `E[Q_def]` over covered targets, normalised to [0, 1]
- `That` — tour time divided by `T_ref`, the shortest feasible tour over the
  minimum covering set at maximum allowed speed. Computed once by the
  DECOUPLED planner and **held fixed across all variants and all `w`**
- `w` — swept 0 to 1 to trace the Pareto front

`T_ref`, not the battery budget: [M] 5-10 viewpoints in a 5x4x3 m arena gives a
2-3 minute tour against 10 minutes of endurance, so a battery normaliser never
binds and every `w` would return the same plan. Battery remains a hard
constraint on top.

**Localisation reliability is not a separate weighted term.** It enters through
`E[Q_def]`. A separate `sigma_s` term would double-count it and make the front
uninterpretable. The sweep is quality against **efficiency** — the trade-off
Maboudi §IV-B asks for by name.

### 6.2 Architecture — set-cover, TSP, fixed point

```
sigma_s <- 0.10 at every candidate           (the engagement value)
repeat up to 3 times:
    score      E[Q_def] for every candidate, given current sigma_s
    set-cover  minimum candidate subset covering every target
    TSP        Hamiltonian cycle from the board pose; edge cost = traversal
               time at the speed chosen in the previous iteration
    speed      forward DP over sigma_s at each edge boundary, 4 levels
    forecast   scale_forecast over the tour -> sigma_s per viewpoint
until the selected set and the order are both unchanged
```

Converged when set and order are stable. If it oscillates, keep the iteration
with the best `J` and **report that it did**. Report the iteration count per
plan.

Why this and not greedy: the baselines are select-then-route (Wu, Benshaaban),
so running COUPLED the same way leaves **one variable different** — whether
`sigma_d` is zero. Greedy would change objective and architecture at once and a
reviewer could attribute the result to either.

TSP: exact to 10-11 viewpoints, heuristic above. A cycle rather than a path, so
the board is imaged at both ends (§3.1).

**What the fixed point recovers, and what it does not.** Selection and ordering
respond to `sigma_s` across iterations, and speed feeds back through the edge
costs. What no select-then-route formulation expresses is a *prefix-dependent*
cost matrix; the iteration approximates it. No optimality guarantee — greedy had
none either.

### 6.3 Candidates and coverage

- **3 standoffs per target patch**, spread across the DoF band `[d_min, d_max]`.
  No lateral offsets: large patches are out of scope.
- **Set-cover** — every target imaged at least once. Standard inspection
  practice (Wu guarantees complete target coverage; Bircher covers every mesh
  face) and, more importantly, it makes the comparison valid: all variants cover
  the same set, so achieved quality is commensurable. Under a fixed-budget
  formulation they could cover different subsets and the numbers would not
  compare.

### 6.4 Feasibility constraints

Hard filters, applied before scoring:

| constraint | value | source |
|---|---|---|
| inside the arena, with clearance | 0.5 m | safety |
| predicted `sigma_s/s` on arrival | `< SIGMA_S_MAX` (0.20) | `EkfParams` |
| standoff | `[d_min, d_max]` [T] | §10 |
| edge speed | levels **0.35, 0.50, 0.70, 0.90** m/s | two either side of the 0.47 m/s observability threshold (§5.1); `v_max_horizontal` above |
| yaw change on an edge | achievable at 15 deg/s within the edge | §4.2 |
| line of sight to the patch | not blocked by another box | geometry |

### 6.5 VO degeneracy — diagnostic only in v1

[M] On F6 and F6c, slow translation parallel to a near flat wall gave 72 % RPE
with VO covering 29-32 % of true motion, while scale stayed correct (1.023,
1.016). The failure is **geometric degeneracy, not feature scarcity** — the
arena is textured to ~6 m in every direction.

Two proxies are computed and **logged, not costed**:

- fraction of the FOV subtended by near-coplanar geometry
- distance along the optical axis to the nearest surface

> **[?] This arena may not be able to produce the degeneracy.** The boundary is
> a see-through mesh with texture beyond it; F6 was a solid wall. Mitigation: at
> least one target on a **solid** surface, so the condition stays reachable.
> Until measured, a cost term would be a fitted constant with no data behind it.

> **[M] Every published metric would pass F6.** E2LOG is deterministic and sees
> adequate observability; FIF scores geometric visibility and features are
> plentiful; Maboudi states outright that reconstructability cannot predict
> failures on low-texture surfaces. That is why this would have to be built
> rather than adopted — and why it is out of v1.

---

## 7. Registration — arena frame to `odom`

The `odom` origin is wherever the filter initialised, which is arbitrary in the
arena. The plan is expressed in the arena frame.

```
T(opti->odom) = T(opti->board) * [ T(odom->base) * T(base->cam) * T(cam->board) ]^-1
```

- `T(opti->board)` — static, from the mocap rigid body and the measured
  `T_mocap_to_tag`. [M] Supplied; rotation orthonormal to 1.4e-4, a 3.4 deg tilt
  about [0.914, 0.380, -0.142]. Normalise before use.
- `T(cam->board)` — AprilTag PnP on a board-hover frame. **Undistort first**:
  [M] `k1 = 0.072`, `k2 = -0.084` are not negligible over a board filling half
  the frame.
- `T(base->cam)` — `/tf_static` plus gimbal attitude at that stamp.
- `T(odom->base)` — `localisation/pose` at the same stamp.

Tag detection and VO share the same image, so `VO_DELAY` cancels and no mocap
clock alignment is needed. The board is static, so `T(opti->board)` is
time-invariant.

**Two observations, opening and closing** (§3.1), give a `K_VEL`-independent
scale reading and the accumulated `p_x, p_y` drift that `filter_design.md` §9
says grows without bound but which has never been measured directly.

[?] Motive places the rigid-body origin at the marker centroid by default. If
the pivot was not set to the grid origin, every transform carries a constant
offset.

---

## 8. Calibration protocol

One short lab session, all static, drone stationary.

**Depth of field.** Panel at 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5 m. MTF50 from
the slanted edge at each. Fit `k_dof`; read `b_0` at the peak; set
`[d_min, d_max]` where MTF50 stays within a chosen fraction of peak.

**`k_r`.** At one known standoff, find the finest resolved line-pair group by
Michelson contrast and solve `k_r = w_min / (b_tot * g)`.

**Panel.** A4, 600 dpi, matte. Bars at 0.75, 1.0, 1.5, 2.0, 3.0, 4.0 mm; six
bars per group; bar width equals gap. Four slanted edges at 5 deg. A 40 mm ArUco
in one corner for registration and identity.

[M] At 1.5 m, GSD is 1.048 mm/px, so 2-3 px spans 2.1-3.1 mm. The 0.75 mm group
is unresolvable everywhere in the band and 4.0 mm resolvable everywhere — floor
and ceiling checks on the detection code.

---

## 9. Plan execution

The planner runs **offline** and stores a `MultiDOFJointTrajectory`. A small
publisher loads it and `waypoint_follower` advances along it. **No `goal`
message type is needed**, which settles the open item in `node_interfaces.md`.

**Carrot with lead compensation.** The follower advances a setpoint along the
path at the planned edge speed and publishes at 20 Hz. Speed is set by how fast
the carrot moves, not by saturation, so no parameter has to change mid-flight.

A P controller tracking a moving setpoint lags by `v / kp_xy` in steady state.
[M] At `kp_xy = 0.6` that is 0.58 m at 0.35 m/s and 1.50 m at 0.90 m/s — most
of a leg in a 5x4 m arena. **The carrot is therefore placed `v / kp_xy` ahead
of the intended point along the path**, so the aircraft sits on the plan rather
than behind it. Exact on straight segments; on corners the aircraft cuts
slightly, which is acceptable because capture happens at the nodes.

**Hard requirement: the publisher runs continuously for the whole tour.**
`setpoint_timeout` is 5 s, and [M] on C03 the setpoint was stale for 51 % of
hold time, which left the aircraft on DJI's own position hold rather than this
controller's (`ekf_operating_conditions.md` §5).

Capture is not triggered. The camera records throughout and frames are chosen
post hoc by pose (`evaluation_design.md` §3).

---

## 10. Plan execution

The planner is offline. Nothing about it runs in flight. What flies is a small
executor that walks the stored plan.

### 10.1 Carrot setpoints, led by the tracking lag

`waypoint_follower` is extended to subscribe to
`trajectory_msgs/MultiDOFJointTrajectory` and publish `setpoint` at 20 Hz — the
controller's own timer rate. Each tick advances a point along the planned path
by `v_edge * dt`.

**The carrot must be led ahead of the plan.** The controller is P plus
saturation (`controller_design.md` §1), so tracking a target moving at `v`
leaves a standing error `e = v / kp_xy`. [M] At `v = 0.5` and `kp_xy = 0.6` that
is **0.83 m** — most of a standoff distance. So the published setpoint is the
planned point advanced by

```
d_lead = v_edge / kp_xy      along the path tangent
```

which makes the aircraft's true path track the planned path rather than lagging
it. Plain feedforward, but without it every viewpoint is missed by `d_lead` and
corners are cut by the same amount.

[?] `d_lead` assumes the plant gain on this path is unity.
`controller_plant_model.md` §12 item 7 records that it could not be re-measured
in closed loop. Verify on the confirmation flight (§8 item 7) by comparing the
mocap path against the planned path; if it lags or leads consistently, scale
`d_lead`.

### 10.2 Requirements

- **The publisher runs continuously for the whole tour.** [M] `setpoint_timeout`
  is 5 s, and on C03 the setpoint was stale for 51 % of hold time, which put the
  aircraft on DJI's own position hold rather than this controller's. A tour with
  gaps is not a measurement of the plan.
- **Capture is continuous**; no trigger. Frames are chosen post hoc (§3 and
  `evaluation_design.md`).
- **`gate_on_degraded` stays true.** Zero commands while the estimate is
  untrusted is the correct behaviour; the resulting stalls are logged and
  reported, not suppressed.
- **Abort conditions**, monitored but not replanned against: `sigma_s/s >
  SIGMA_S_MAX`, `vo_epoch` change, or `degraded` for longer than the recovery
  timeout. All three already exist in the state machine.

### 10.3 What this settles

`node_interfaces.md` lists `goal` as an undecided message type and
`trajectory_gen_node` as a stub. With the planner offline and the plan stored,
**neither is needed**: a plan-publisher node loads the file and hands
`MultiDOFJointTrajectory` to `waypoint_follower`. The `goal` message can be
struck from the open list.
