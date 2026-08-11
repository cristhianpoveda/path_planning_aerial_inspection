# Data Flow / Sequence

Traces frames, pose, velocity and commands through the nodes over time.

---

## Localisation — runs at camera rate

```mermaid
sequenceDiagram
    autonumber
    participant DEC as camera_decoder
    participant SLAM as V-SLAM
    participant DJI as dji_node
    participant EKF as EKF

    Note over DEC,EKF: rate approx 10 Hz
    DEC->>SLAM: Camera frame
    SLAM->>SLAM: Track features, no-scale motion estimation
    SLAM->>EKF: Odometry (NO scale)

    par telemetry 10 Hz (same packet, one timestamp)
        DJI->>EKF: Velocity
        DJI->>EKF: Altitude_agl
        DJI->>EFK: Attitude
        DJI->>EFK: Gimbal_joint_attitude (19 Hz, separate packe)
    end

    EKF->>EKF: Fuse, resolve scale, correct drift
    EKF->>EKF: Broadcast /tf 
    Note right of EKF: Publishes estimated pose
```

**Timing note:** telemetry carries measurement-time stamps (phone key listener + clock-offset mapping), so its latency is resolved. Camera frames are stamped at decode time minus VIDEO_LATENCY, still to be measured. Fuse in timestamp order.

---

## 6b. Control loop - ~high rate

```mermaid
sequenceDiagram
    autonumber
    participant EKF as EKF
    participant WPF as waypoint_follower
    participant PC as position_controller
    participant DJI as dji_node
    participant PHONE as WildBridge (phone)

    Note over EKF,PHONE: rate approx CONTROL_HZ Hz
    EKF-->>PC: Current pose
    WPF-->>PC: Setpoint
    PC->>PC: Compute error then velocity cmd
    PC->>DJI: Stick cmd
    DJI->>PHONE: stick UDP/8082
    Note right of PHONE: watchdog reset<br/>timeout: 200ms -> hover
```

**Timing note:** pose age at the controller is ~310 ms (video ~200 + SLAM ~40 + gimbal tf ≤53 + filter ~10). This is not compared against the watchdog — the watchdog checks command freshness, not pose age. position_controller runs on a fixed timer and publishes regardless of pose arrival, so the watchdog is satisfied by construction. The 310 ms instead caps position-loop bandwidth at ~0.15 Hz, which is why waypoint_follower needs a conservative max_velocity.

---

## 6c. Discrete commands (takeoff / land) — event-driven, reliable

```mermaid
sequenceDiagram
    autonumber
    participant SRC as operator
    participant DJI as dji_node
    participant PHONE as WildBridge (phone)

    SRC->>DJI: discrete command
    DJI->>PHONE: HTTP/8080 POST
    PHONE->>PHONE: hand to DJI SDK
    PHONE-->>DJI: HTTP 200 (accepted, not completed)
    Note over DJI,PHONE: Cmd completion to be inferred from telemetry
```

---

## 6d. Evaluation flow (dev-only) — low rate

```mermaid
sequenceDiagram
    autonumber
    participant OPTI as OptiTrack (domain_id 0)
    participant DB as ros_domains_bridge
    participant EKF as EKF
    participant CMP as comparison_node

    OPTI->>DB: GT pose (domain 0)
    DB->>CMP: GT pose (domain 12)
    EKF->>CMP: Estimated pose
    CMP->>CMP: align frames , compute metrics
    Note right of CMP: log / save metrics
```

---

## Notes
- 6b + 6c share the same drone command path, hoever 6b is continuous/loss-tolerant (UDP) and 6c is discrete/reliable (HTTP + ACK).
