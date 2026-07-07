# System State Machine (flight modes + safety)

Behavioural changes and the transitions
between them. Also, it inlcudes the system safety logic.

---

## 7a. Command authority (manual vs autonomous)

The RC arbitrates in hardware — manual override always wins. This machine records that authority, not the drone firmware's internal logic.

```mermaid
stateDiagram-v2
    [*] --> Manual
    Manual --> Autonomous: Virtual stick enabled AND stack healthy
    Autonomous --> Manual: Operator presses takeover (H / swap mode)
    Autonomous --> Manual: Stack or bridge fails

    note right of Manual
        Safety pilot has full control.
        Path bypasses software.
    end note
    note right of Autonomous
        UDP stick stream.
    end note
```

---

## 7b. Autonomous flight behaviour

Hover is the safe default that every failure drains into.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Localising: Start requested
    Localising --> Hover: Pose valid and scale resolved
    Localising --> Idle: Init failed or aborted

    Hover --> Navigating: Goal accepted
    Navigating --> Hover: Goal reached
    Navigating --> Hover: New goal pending

    Navigating --> Recovering: Localisation degraded or pose jump
    Hover --> Recovering: Localisation degraded
    Recovering --> Hover: Localisation recovered
    Recovering --> Landing: Recovery timeout exceeded

    Hover --> Landing: Land requested or low battery
    Navigating --> Landing: Land requested or low battery
    Landing --> Idle: Touchdown confirmed

    note right of Hover
        Safe default state.
        Watchdog hover (zero stick)
        lands on comms loss.
    end note
    note right of Recovering
        Localisation untrusted.
        Hold position,
        wait for reacquire, else land.
    end note
```


## Thresholds and conditions to define
- Stack healthy: all flight-critical nodes up and pose valid.
- localisation "degraded": covariance above COV_MAX, or VO tracking lost, or
  pose jump above JUMP_MAX between updates.
- recovery timeout: 10 s before forcing Landing.
- low battery: 15 %.

## Cross-references
- Watchdog hover (200 ms comms loss) enters Hover in 7b and, on the phone side, is what the dead-man switch enforces regardless of stack state.