"""Probabilistic roadmap, A* queries, line-of-sight pruning.

"""

import math

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


class Roadmap:

    def __init__(self, gmap, n_samples=1500, connection_radius=1.2, seed=0):
        """connection_radius default is metres and sized for a 5x4x3 m arena,
        not the coursework's 30x30x10 m one."""
        self.map = gmap
        self.n_samples = int(n_samples)
        self.connection_radius = float(connection_radius)
        self.rng = np.random.default_rng(seed)
        self.graph = nx.Graph()
        self.nodes = np.zeros((0, 3))
        self.kdtree = None
        self._terminals = {}
        self._path_cache = {}

    # ----------------------------------------------------------------- build
    def build(self, verbose=True):
        pts = self.map.sample_free(self.n_samples, self.rng)
        self.nodes = pts
        self.graph.add_nodes_from(range(len(pts)))
        self.kdtree = KDTree(pts)
        pairs = self.kdtree.query_pairs(self.connection_radius)
        n_edges = 0
        for i, j in pairs:
            if self.map.is_segment_valid(pts[i], pts[j]):
                self.graph.add_edge(i, j,
                                    weight=float(np.linalg.norm(pts[i] - pts[j])))
                n_edges += 1
        if verbose:
            comps = list(nx.connected_components(self.graph))
            big = max((len(c) for c in comps), default=0)
            print(f"roadmap: {len(pts)} nodes, {len(pairs)} candidate pairs, "
                  f"{n_edges} valid edges, {len(comps)} components "
                  f"(largest {big}, {100.0*big/max(len(pts),1):.1f}%)")
        return self

    # ------------------------------------------------------------- terminals
    def clear_terminals(self):
        for idx in self._terminals.values():
            if self.graph.has_node(idx):
                self.graph.remove_node(idx)
        self._terminals = {}
        self._path_cache = {}

    def add_terminal(self, name, position):
        """Attach a viewpoint (or the board pose) to the roadmap.
        """
        p = np.asarray(position, float)
        if not self.map.is_point_valid(p):
            raise ValueError(f"terminal {name} at {p.round(3)} is outside the "
                             "arena clearance or inside an obstacle")
        idx = len(self.nodes) + len(self._terminals)
        while self.graph.has_node(idx):
            idx += 1
        self.graph.add_node(idx)
        self._terminals[name] = idx
        self._node_pos = None

        near = self.kdtree.query_ball_point(p, self.connection_radius)
        n_edges = 0
        for j in near:
            if self.map.is_segment_valid(p, self.nodes[j]):
                self.graph.add_edge(idx, j,
                                    weight=float(np.linalg.norm(p - self.nodes[j])))
                n_edges += 1
        self._extra = getattr(self, "_extra", {})
        self._extra[idx] = p
        if n_edges == 0:
            self.graph.remove_node(idx)
            del self._terminals[name]
            raise ValueError(
                f"terminal {name} at {p.round(3)} connects to no roadmap node "
                f"within {self.connection_radius} m; increase the radius or "
                "the sample count")
        return idx

    def position(self, idx):
        if idx < len(self.nodes):
            return self.nodes[idx]
        return self._extra[idx]

    # --------------------------------------------------------------- queries
    def path(self, a, b, prune=True):
        """Collision-free polyline between two terminal names."""
        key = (a, b, prune)
        if key in self._path_cache:
            return self._path_cache[key]
        rev = (b, a, prune)
        if rev in self._path_cache:
            out = list(reversed(self._path_cache[rev]))
            self._path_cache[key] = out
            return out
        ia, ib = self._terminals[a], self._terminals[b]
        try:
            idx = nx.astar_path(
                self.graph, ia, ib,
                heuristic=lambda u, v: float(np.linalg.norm(
                    self.position(u) - self.position(v))),
                weight="weight")
        except nx.NetworkXNoPath:
            self._path_cache[key] = None
            return None
        pts = [self.position(i) for i in idx]
        if prune:
            pts = self.prune(pts)
        self._path_cache[key] = pts
        return pts

    def all_paths(self, names, prune=True, verbose=False):
        """Pairwise paths and lengths. Upper triangle only."""
        n = len(names)
        L = np.full((n, n), np.inf)
        np.fill_diagonal(L, 0.0)
        paths, missing = {}, []
        for i in range(n):
            for j in range(i + 1, n):
                p = self.path(names[i], names[j], prune)
                if p is None:
                    missing.append((names[i], names[j]))
                    continue
                d = path_length(p)
                L[i, j] = L[j, i] = d
                paths[(names[i], names[j])] = p
                paths[(names[j], names[i])] = list(reversed(p))
        if verbose and missing:
            print(f"WARNING: no path for {len(missing)} pairs, e.g. "
                  f"{missing[:3]}")
        return L, paths, missing

    # --------------------------------------------------------------- pruning
    def prune(self, pts):
        """Line-of-sight shortcutting. Always advances, so it terminates."""
        if len(pts) <= 2:
            return list(pts)
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            nxt = i + 1
            for k in range(len(pts) - 1, i, -1):
                if self.map.is_segment_valid(pts[i], pts[k]):
                    nxt = k
                    break
            out.append(pts[nxt])
            i = nxt
        return out


def path_length(pts):
    return float(sum(math.dist(pts[i], pts[i + 1])
                     for i in range(len(pts) - 1)))
