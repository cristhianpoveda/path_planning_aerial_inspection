"""Boxes, arena, collision checks.
"""

from dataclasses import dataclass, field

import numpy as np
import yaml

AXES = ("x", "y", "z")


class Box:
    """Box with optional yaw about z.
    """

    def __init__(self, centre, extents, margin=0.0, yaw=0.0, name=None):
        self.centre = np.asarray(centre, float)
        self.extents = np.asarray(extents, float) + 2.0 * margin
        self.yaw = float(yaw)
        self.name = name
        self.half = self.extents / 2.0
        c, s_ = np.cos(self.yaw), np.sin(self.yaw)
        self.R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        if self.yaw == 0.0:
            self.lo = self.centre - self.half
            self.hi = self.centre + self.half
        else:
            # AABB of the oriented box, for broad-phase and reporting only
            ext = np.abs(self.R) @ self.half
            self.lo = self.centre - ext
            self.hi = self.centre + ext

    def to_local(self, p):
        return (np.asarray(p, float) - self.centre) @ self.R

    def contains(self, p):
        return bool(np.all(np.abs(self.to_local(p)) <= self.half))

    def contains_many(self, pts):
        q = (np.asarray(pts, float) - self.centre) @ self.R
        return np.all(np.abs(q) <= self.half, axis=1)

    def intersects_segment(self, p1, p2, eps=1e-9):
        """Slab method in the box frame. True if the segment enters the box."""
        a = self.to_local(p1)
        b = self.to_local(p2)
        d = b - a
        d = np.where(np.abs(d) < eps, eps, d)
        t1 = (-self.half - a) / d
        t2 = (self.half - a) / d
        # per axis the near hit is min(t1, t2); t1 > t2 whenever d < 0
        t_enter = np.minimum(t1, t2).max()
        t_exit = np.maximum(t1, t2).min()
        return bool(t_enter <= t_exit and t_enter <= 1.0 and t_exit >= 0.0)

    FACES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0),
             "-y": (1, -1.0), "+z": (2, 1.0), "-z": (2, -1.0)}

    def face(self, name, offset=(0.0, 0.0)):
        """Centre and outward normal of a face, in world coordinates.
        """
        axis, sign = self.FACES[name]
        local = np.zeros(3)
        local[axis] = sign * self.half[axis]
        others = [i for i in range(3) if i != axis]
        for k, i in enumerate(others[:len(offset)]):
            local[i] = offset[k]
        if np.any(np.abs(local) > self.half + 1e-9):
            raise ValueError(f"offset {offset} falls outside face {name}")
        n = np.zeros(3)
        n[axis] = sign
        return self.centre + self.R @ local, self.R @ n

    def __repr__(self):
        y = f", yaw={np.degrees(self.yaw):.1f}deg" if self.yaw else ""
        nm = f"{self.name}: " if self.name else ""
        return (f"Box({nm}centre={np.round(self.centre, 3).tolist()}, "
                f"extents={np.round(self.extents, 3).tolist()}{y})")


@dataclass
class Target:
    """A patch to be inspected.
    """
    name: str
    centre: np.ndarray
    normal: np.ndarray
    w_t: float
    extents: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        self.centre = np.asarray(self.centre, float)
        n = np.asarray(self.normal, float)
        self.normal = n / np.linalg.norm(n)
        self.extents = np.asarray(self.extents, float)


def _box(d, name=None):
    """Accept either centre/extents or min/max.
    """
    yaw = float(np.radians(d["yaw_deg"])) if "yaw_deg" in d else float(d.get("yaw", 0.0))
    if "min" in d and "max" in d:
        lo, hi = np.asarray(d["min"], float), np.asarray(d["max"], float)
        if np.any(hi <= lo):
            raise ValueError(f"{name}: max must exceed min on every axis")
        return Box((lo + hi) / 2.0, hi - lo, yaw=yaw, name=d.get("name", name))
    return Box(d["centre"], d["extents"], yaw=yaw, name=d.get("name", name))


class GeometricMap:

    def __init__(self, arena, obstacles=(), targets=(), safety_margin=0.35,
                 arena_clearance=0.5):
        """arena: Box (uninflated).

        safety_margin inflates obstacles by the aircraft radius.
        arena_clearance shrinks the arena.
        """
        self.arena = arena
        self.safety_margin = float(safety_margin)
        self.arena_clearance = float(arena_clearance)
        self.obstacles = [Box(b.centre, b.extents, self.safety_margin,
                              b.yaw, b.name) for b in obstacles]
        self.obstacles_raw = list(obstacles)
        self.targets = list(targets)
        self.lo = arena.lo + self.arena_clearance
        self.hi = arena.hi - self.arena_clearance
        if np.any(self.lo >= self.hi):
            raise ValueError(
                f"arena {arena} is smaller than twice the clearance "
                f"{self.arena_clearance} m in some axis")

    # ------------------------------------------------------------ factories
    @classmethod
    def from_yaml(cls, path, safety_margin=0.35, arena_clearance=0.5):
        with open(path) as f:
            d = yaml.safe_load(f)

        arena = _box(d["arena"], name="arena")
        if arena.yaw:
            raise ValueError("the arena must be axis-aligned")

        obstacles = []
        for i, o in enumerate(d.get("obstacles", [])):
            obstacles.append(_box(o, name=o.get("name", f"obs{i}")))
        by_name = {b.name: b for b in obstacles}

        targets = []
        for i, t in enumerate(d.get("targets", [])):
            name = t.get("name", f"T{i}")
            if "box" in t:
                # on a face of a named obstacle: the trigonometry for a rotated box is done here, not by hand in the yaml
                b = by_name[t["box"]]
                centre, normal = b.face(t["face"], t.get("offset", (0.0, 0.0)))
            else:
                centre, normal = t["centre"], t["normal"]
            targets.append(Target(name=name, centre=centre, normal=normal,
                                  w_t=float(t["w_t"]),
                                  extents=t.get("extents", [0.0, 0.0])))
        return cls(arena, obstacles, targets, safety_margin, arena_clearance)

    # --------------------------------------------------------------- checks
    def in_arena(self, p):
        p = np.asarray(p, float)
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))

    def is_point_valid(self, p):
        if not self.in_arena(p):
            return False
        return not any(b.contains(p) for b in self.obstacles)

    def is_segment_valid(self, p1, p2):
        """Free of obstacles. Does not test the arena: both endpoints are
        already inside a convex box, so the segment is too."""
        return not any(b.intersects_segment(p1, p2) for b in self.obstacles)

    def has_line_of_sight(self, p, target_centre, normal=None, back_off=1e-3):
        """Camera ray test, against the uninflated obstacles.
        """
        p = np.asarray(p, float)
        c = np.asarray(target_centre, float)
        if normal is not None:
            end = c + back_off * np.asarray(normal, float)
        else:
            d = c - p
            n = np.linalg.norm(d)
            end = c - (back_off / n) * d if n > back_off else c
        return not any(b.intersects_segment(p, end)
                       for b in self.obstacles_raw)

    def filter_valid(self, pts):
        """Vectorised is_point_valid over an (N, 3) array."""
        pts = np.asarray(pts, float)
        ok = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
        for b in self.obstacles:
            ok &= ~b.contains_many(pts)
        return pts[ok]

    def sample_free(self, n, rng=None, oversample=3):
        """n valid points, uniform over the clearance-reduced arena."""
        rng = np.random.default_rng() if rng is None else rng
        out = np.zeros((0, 3))
        for _ in range(200):
            raw = rng.uniform(self.lo, self.hi, size=(n * oversample, 3))
            out = np.vstack([out, self.filter_valid(raw)])
            if len(out) >= n:
                return out[:n]
        raise RuntimeError(
            f"only {len(out)} of {n} samples were free; obstacles may fill "
            "the arena or the safety margin may be too large")

    def __repr__(self):
        return (f"GeometricMap(arena={self.arena}, "
                f"{len(self.obstacles)} obstacles, "
                f"{len(self.targets)} targets, margin={self.safety_margin})")
