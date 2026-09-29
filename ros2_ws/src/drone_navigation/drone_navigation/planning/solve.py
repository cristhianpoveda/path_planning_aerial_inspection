"""Set-cover, TSP, speed DP, fixed point. planner_design.md 6.

    sigma_s <- the engagement value at every candidate
    repeat up to max_iter:
        score      E[Q_def] for every candidate, given current sigma_s
        set-cover  minimum candidate subset covering every target
        TSP        Hamiltonian cycle from the board pose
        speed      forward DP over sigma_s at each edge boundary
        forecast   sigma_s per viewpoint
    until the selected set and the order are both unchanged

Selection and ordering respond to sigma_s cross iterations and speed feeds back through the edge costs. What no select-then-route formulation expresses is a prefix-dependent cost matrix;mthis approximates it. No optimality guarantee -- greedy had none either.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import model as M
from . import viewpoints as V
from .forecast import Forecast, ForecastConfig, v_low_true

MAX_COMBINATIONS = 300_000


@dataclass
class SolveConfig:
    w: float = 0.5                      # quality vs efficiency
    coupled: bool = True                # False = score at the nominal pose
    speeds: tuple = (0.30, 0.45, 0.60, 0.90)
    max_iter: int = 3
    beam: int = 24                      # Pareto frontier cap in the speed DP
    image_dwell_s: float = 0.0
    sigma_s_ratio_0: float = 0.10       # 3.1 step 3, the engagement gate
    verbose: bool = False


@dataclass
class Plan:
    order: list = field(default_factory=list)      # candidate names, a cycle
    selected: list = field(default_factory=list)   # Candidate objects
    speeds: list = field(default_factory=list)     # one per edge
    legs: list = field(default_factory=list)       # (name, path, speed)
    sigma_s: dict = field(default_factory=dict)    # name -> sigma_s, absolute
    sigma_ratio: dict = field(default_factory=dict)  # name -> sigma_s / s
    quality: dict = field(default_factory=dict)    # target name -> E[Q]
    by_name: dict = field(default_factory=dict)
    time_s: float = 0.0
    J: float = 0.0
    Qbar: float = 0.0
    That: float = 0.0
    iterations: int = 0
    converged: bool = False


# --------------------------------------------------------------- set cover
def min_cover(cands, targets, score_of):
    """Minimum-cardinality covering subset, ties broken by total score.
    """
    names = [t.name for t in targets]
    sets = [set(c.covers) for c in cands]
    if not set().union(*sets) >= set(names):
        raise ValueError("candidates do not cover every target")

    n = len(cands)
    for k in range(1, n + 1):
        total = 1
        for i in range(k):
            total = total * (n - i) // (i + 1)
        if total > MAX_COMBINATIONS:
            break
        best, best_score = None, -np.inf
        for combo in itertools.combinations(range(n), k):
            u = set()
            for i in combo:
                u |= sets[i]
            if len(u) < len(names):
                continue
            sc = sum(score_of(cands[i]) for i in combo)
            if sc > best_score:
                best, best_score = combo, sc
        if best is not None:
            return [cands[i] for i in best]

    remaining, chosen = set(names), []
    while remaining:
        i = max(range(n), key=lambda j: (len(sets[j] & remaining),
                                         score_of(cands[j])))
        if not sets[i] & remaining:
            raise ValueError("greedy cover stalled")
        chosen.append(cands[i])
        remaining -= sets[i]
    return chosen


# ---------------------------------------------------------------------- TSP
def solve_order(names, cost):
    """Hamiltonian cycle over `names`, starting at index 0.
    """
    from python_tsp.exact import solve_tsp_dynamic_programming
    from python_tsp.heuristics import solve_tsp_simulated_annealing
    C = np.asarray(cost, float).copy()
    np.fill_diagonal(C, 0.0)
    if len(names) <= 11:
        perm, _ = solve_tsp_dynamic_programming(C)
    else:
        perm, _ = solve_tsp_simulated_annealing(C)
    k = perm.index(0)
    perm = perm[k:] + perm[:k]
    return [names[i] for i in perm]


# ------------------------------------------------------------------ speed DP
def speed_dp(order, paths, cfg, fcfg, params, gmap, q, nav_origin,
             sigma_s0, by_name):
    """Forward DP over sigma_s at each edge boundary.
    """
    Sigma_pp = np.diag([0.07 ** 2] * 3)     # [T] until 5 supplies it per node

    def node_quality(name, sig, v):
        """Mean E[Q] over the targets this node images, at this speed.
        """
        c = by_name.get(name)
        if c is None:
            return 0.0
        sc = V.score(c, gmap, q, v, sig / fcfg.s_true, Sigma_pp, nav_origin,
                     nominal=not cfg.coupled)
        return float(np.mean(list(sc.values()))) if sc else 0.0

    front = [(sigma_s0, 0.0, 0.0, [], [], None)]
    for i in range(len(order)):
        a = order[i - 1] if i > 0 else order[-1]
        b = order[i]
        path = paths[(a, b)]
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        nxt = []
        for sig, t, qa, sp, sgs, state in front:
            for v in cfg.speeds:
                f = Forecast(params, fcfg)
                if state is None:
                    f.reset(sigma_s0=sig)
                else:
                    f.core.x.P = state[0].copy()
                    f.core.x.s = state[1]
                f.fly(path, v, dwell=cfg.image_dwell_s)
                nxt.append((f.sigma_s,
                            t + length / v + cfg.image_dwell_s,
                            qa + node_quality(b, f.sigma_s, v),
                            sp + [v], sgs + [f.sigma_s],
                            (f.core.x.P, f.core.x.s)))
        nxt.sort(key=lambda r: (r[1], r[0], -r[2]))
        keep = []
        for r in nxt:
            if not any(o[0] <= r[0] and o[1] <= r[1] and o[2] >= r[2]
                       for o in keep):
                keep.append(r)
        if len(keep) > cfg.beam:
            ts = np.array([r[1] for r in keep])
            qs = np.array([r[2] for r in keep])
            tn = (ts - ts.min()) / max(ts.ptp(), 1e-12)
            qn = (qs - qs.min()) / max(qs.ptp(), 1e-12)
            rank = cfg.w * qn - (1.0 - cfg.w) * tn
            keep = [keep[i] for i in np.argsort(-rank)]
        front = keep[:cfg.beam]
    return front


# ------------------------------------------------------------------ objective
def evaluate(order, speeds, sigma_s, gmap, q, cfg, nav_origin, by_name,
             time_s, T_ref, Q_ref=None, force_coupled=False):
    """J = w * Qbar - (1 - w) * That.

    Q_ref normalises the quality axis the way T_ref normalises the time axis.
    
    force_coupled scores a plan under the true objective regardless of which
    planner produced it. That is the only fair comparison: a decoupled plan's
    own Qbar is its belief, computed with sigma_d set to zero, not what it
    achieves when flown in a world where sigma_d is not zero.
    """
    nominal = (not cfg.coupled) and not force_coupled
    Sigma_pp = np.diag([0.07 ** 2] * 3)     # [T] until 5 supplies it per node
    best = {}
    for i, name in enumerate(order):
        c = by_name[name]
        if c is None:
            continue
        v = speeds[i] if i < len(speeds) else speeds[-1]
        sc = V.score(c, gmap, q, v, sigma_s.get(name, cfg.sigma_s_ratio_0),
                     Sigma_pp, nav_origin, nominal=nominal)
        for t, val in sc.items():
            best[t] = max(best.get(t, 0.0), val)
    denom = Q_ref if Q_ref else q.Q_max
    Qbar = float(np.mean(list(best.values()))) / denom if best else 0.0
    That = time_s / T_ref if T_ref > 0 else 0.0
    return cfg.w * Qbar - (1.0 - cfg.w) * That, Qbar, That, best


# --------------------------------------------------------------------- solve
def plan(gmap, roadmap, cands, q, params, cfg=None, fcfg=None,
         board=None, nav_origin=None, T_ref=None, Q_ref=None):
    """Run the fixed point. Returns a Plan."""
    cfg = cfg or SolveConfig()
    fcfg = fcfg or ForecastConfig()
    nav_origin = gmap.arena.centre if nav_origin is None else np.asarray(
        nav_origin, float)
    sigma_s0 = cfg.sigma_s_ratio_0 * fcfg.s_true

    Sigma_pp = np.diag([0.07 ** 2] * 3)
    sigma_ratio = {c.name: cfg.sigma_s_ratio_0 for c in cands}

    prev_set = prev_order = None
    result = Plan()
    best_result = None
    for it in range(1, cfg.max_iter + 1):
        def score_of(c, _sr=sigma_ratio):
            sc = V.score(c, gmap, q, cfg.speeds[-1], _sr[c.name], Sigma_pp,
                         nav_origin, nominal=not cfg.coupled)
            return float(np.mean(list(sc.values()))) if sc else 0.0

        selected = min_cover(cands, gmap.targets, score_of)
        names = [c.name for c in selected]
        by_name = {c.name: c for c in selected}

        roadmap.clear_terminals()
        if board is not None:
            roadmap.add_terminal("BOARD", board)
            by_name["BOARD"] = None
        for c in selected:
            roadmap.add_terminal(c.name, c.position)
        nodes = (["BOARD"] if board is not None else []) + names

        L, paths, missing = roadmap.all_paths(nodes)
        if missing:
            raise ValueError(f"unreachable viewpoint pairs: {missing[:3]}")
        order = solve_order(nodes, L / cfg.speeds[-1])

        front = speed_dp(order, paths, cfg, fcfg, params, gmap, q,
                         nav_origin, sigma_s0, by_name)

        best = None
        for sig, t, _qa, sp, sgs, _ in front:
            sr = {n: g / fcfg.s_true for n, g in zip(order, sgs)}
            J, Qbar, That, qual = evaluate(order, sp, sr, gmap, q, cfg,
                                           nav_origin, by_name, t,
                                           T_ref or t, Q_ref)
            if best is None or J > best[0]:
                best = (J, Qbar, That, qual, sig, t, sp, sr, sgs)
        J, Qbar, That, qual, sig, t, sp, sr, sgs = best

        this = Plan(order=order, selected=selected, speeds=sp,
                      by_name=by_name,
                      legs=[(order[i], paths[(order[i - 1], order[i])], sp[i])
                            for i in range(len(order))],
                      sigma_s=dict(zip(order, sgs)),
                      sigma_ratio=dict(sr), quality=qual,
                      time_s=t, J=J, Qbar=Qbar, That=That, iterations=it)
        
        if best_result is None or J > best_result.J:
            best_result = this
        result = best_result
        
        realised = float(np.mean(list(sr.values()))) if sr else \
            cfg.sigma_s_ratio_0
        sigma_ratio = {c.name: sr.get(c.name, realised) for c in cands}

        if cfg.verbose:
            print(f"  iter {it}: {len(selected)} viewpoints, "
                  f"order {order}, T {t:.1f} s, J {J:.4f}")
        if prev_set == set(names) and prev_order == order:
            result.converged = True
            break
        prev_set, prev_order = set(names), order
    result.iterations = it
    return result


def reference_time(gmap, roadmap, cands, q, params, cfg=None, fcfg=None,
                   board=None):
    """T_ref: shortest tour over the minimum covering set at maximum speed.
    """
    cfg = cfg or SolveConfig()
    fcfg = fcfg or ForecastConfig()
    d = SolveConfig(**{**cfg.__dict__, "coupled": False, "max_iter": 1,
                       "w": 0.0, "verbose": False})
    p = plan(gmap, roadmap, cands, q, params, d, fcfg, board, T_ref=1.0)
    return p.time_s


def score_true(p: Plan, gmap, q, cfg, nav_origin, T_ref, Q_ref=None):
    """Re-score any plan under the COUPLED objective.
    """
    sr = {n: p.sigma_ratio.get(n, cfg.sigma_s_ratio_0) for n in p.order}
    return evaluate(p.order, p.speeds, sr, gmap, q, cfg, nav_origin,
                    p.by_name, p.time_s, T_ref, Q_ref, force_coupled=True)


def reference_quality(gmap, roadmap, cands, q, params, cfg=None, fcfg=None,
                      board=None, nav_origin=None, T_ref=None):
    """Q_ref: the best achievable Qbar, DECOUPLED, w = 1. Held fixed."""
    cfg = cfg or SolveConfig()
    d = SolveConfig(**{**cfg.__dict__, "coupled": False, "w": 1.0,
                       "verbose": False})
    p = plan(gmap, roadmap, cands, q, params, d, fcfg, board, nav_origin,
             T_ref or 1.0)
    return p.Qbar * q.Q_max
