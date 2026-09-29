#!/usr/bin/env python3
"""Acceptance checks for geometry.py and roadmap.py.

    python3 -m drone_navigation.planning.check_geometry arena.yaml

No ROS. Every check prints PASS or FAIL and the number behind it; the script
exits non-zero if any fail, so it can go in CI.
"""

import sys

import numpy as np

from .geometry import Box, GeometricMap, Target
from .roadmap import Roadmap, path_length

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def check_slab(n=4000, seed=0):
    """Slab method against dense sampling of the segment."""
    print("slab method vs brute force")
    rng = np.random.default_rng(seed)
    b = Box([0, 0, 0], [1, 2, 3])
    t = np.linspace(0, 1, 20000)[:, None]
    bad = 0
    for _ in range(n):
        p1, p2 = rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3)
        brute = bool(b.contains_many(p1 + (p2 - p1) * t).any())
        if b.intersects_segment(p1, p2) != brute:
            bad += 1
    check("segment test agrees with sampling", bad == 0, f"{bad}/{n} disagree")


def check_los():
    """A patch centre lies ON its box, so the ray must stop short of it."""
    print("line of sight")
    arena = Box([0, 0, 1.5], [6, 4, 3])
    blocker = Box([0.0, 0.0, 1.4], [0.4, 1.0, 1.0])   # spans x -0.2 .. 0.2
    t = Target("far_wall", [2.9, 0.0, 1.4], [-1, 0, 0], 0.002)
    m = GeometricMap(arena, [blocker], [t], 0.35, 0.5)
    near = m.has_line_of_sight(t.centre + 1.0 * t.normal, t.centre, t.normal)
    far = m.has_line_of_sight(t.centre + 3.9 * t.normal, t.centre, t.normal)
    off = m.has_line_of_sight([-1.0, 1.2, 1.4], t.centre, t.normal)
    check("target on a box is visible from in front", near)
    check("blocker between viewpoint and target occludes", not far)
    check("ray passing beside the blocker is clear", off)


def check_map(path):
    print(f"map: {path}")
    m = GeometricMap.from_yaml(path)
    print(f"  {m}")
    print(f"  flyable box after {m.arena_clearance} m clearance: "
          f"{m.lo.round(2).tolist()} .. {m.hi.round(2).tolist()}")
    check("arena is larger than twice the clearance", np.all(m.hi > m.lo))
    for b in m.obstacles:
        check(f"obstacle {b.centre.round(2).tolist()} centre is blocked",
              not m.is_point_valid(b.centre))
    for t in m.targets:
        vp = t.centre + 1.0 * t.normal
        check(f"target {t.name}: 1 m along its normal is free space",
              m.is_point_valid(vp),
              f"vp={vp.round(2).tolist()}")
        check(f"target {t.name}: visible from there",
              m.has_line_of_sight(vp, t.centre, t.normal))
    return m


def check_roadmap(m, n_samples=1500, radius=1.2, seed=1):
    print(f"roadmap: {n_samples} samples, radius {radius} m")
    r = Roadmap(m, n_samples, radius, seed).build()
    import networkx as nx
    comps = list(nx.connected_components(r.graph))
    big = max(len(c) for c in comps)
    check("roadmap is a single connected component", len(comps) == 1,
          f"{len(comps)} components, largest {big}/{n_samples}")

    names = []
    for t in m.targets:
        try:
            r.add_terminal(t.name, t.centre + 1.0 * t.normal)
            names.append(t.name)
        except ValueError as e:
            check(f"terminal {t.name} attaches", False, str(e))
    check("every target got a terminal", len(names) == len(m.targets),
          f"{len(names)}/{len(m.targets)}")

    L, paths, missing = r.all_paths(names)
    check("every viewpoint pair is reachable", not missing,
          f"{len(missing)} unreachable")

    worst = 0.0
    n_bad = 0
    for (a, b), p in paths.items():
        for i in range(len(p) - 1):
            if not m.is_segment_valid(p[i], p[i + 1]):
                n_bad += 1
        straight = float(np.linalg.norm(np.asarray(p[0]) - np.asarray(p[-1])))
        if straight > 1e-6:
            worst = max(worst, path_length(p) / straight)
    check("every path leg is collision-free", n_bad == 0, f"{n_bad} bad legs")
    check("pruned paths are not absurdly long", worst < 3.0,
          f"worst detour ratio {worst:.2f}x straight line")

    print("\n  pairwise path length (m):")
    print("   " + "".join(f"{n[:7]:>9s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:7]:>7s}" + "".join(f"{L[i, j]:9.2f}" for j in range(len(names))))
    return r


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "arena_example.yaml"
    check_slab()
    check_los()
    m = check_map(path)
    check_roadmap(m)
    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
