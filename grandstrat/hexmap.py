"""Axial hex map: index <-> (q, r), neighbors, distance, rings.

Coordinates are axial (q, r).  A map of radius R holds every hex with
hex_distance from origin <= R: count = 3R(R+1)+1 (R=12 -> 469).

Direction order is part of the pinned action encoding (MOVE d, d in 0..5):
    0:(+1,0) E   1:(+1,-1) NE   2:(0,-1) NW
    3:(-1,0) W   4:(-1,+1) SW   5:(0,+1) SE

Everything hot is precomputed at construction: an index<->coord pair of arrays
and an L x 6 neighbor-index table (-1 = off map).  Distance is closed-form O(1)
from axial deltas, so no distance matrix is needed.
"""

from __future__ import annotations

import numpy as np

DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
NUM_DIRS = 6


def hex_count(radius: int) -> int:
    return 3 * radius * (radius + 1) + 1


def hex_distance(dq: int, dr: int) -> int:
    """Distance between two hexes given their axial delta (dq, dr)."""
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2


class HexMap:
    """Immutable geometry for one map size.  Shared between state clones."""

    __slots__ = ("radius", "n", "qs", "rs", "idx", "neighbors", "_ring_cache")

    def __init__(self, radius: int):
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.radius = int(radius)
        coords = []
        R = self.radius
        for q in range(-R, R + 1):
            for r in range(max(-R, -q - R), min(R, -q + R) + 1):
                coords.append((q, r))
        self.n = len(coords)
        assert self.n == hex_count(R)
        self.qs = np.array([c[0] for c in coords], dtype=np.int16)
        self.rs = np.array([c[1] for c in coords], dtype=np.int16)
        self.idx = {(q, r): i for i, (q, r) in enumerate(coords)}
        neigh = np.full((self.n, NUM_DIRS), -1, dtype=np.int32)
        for i, (q, r) in enumerate(coords):
            for d, (dq, dr) in enumerate(DIRECTIONS):
                j = self.idx.get((q + dq, r + dr), -1)
                neigh[i, d] = j
        self.neighbors = neigh
        self._ring_cache: dict[int, list[tuple[int, int]]] = {}

    def index(self, q: int, r: int) -> int:
        return self.idx[(q, r)]

    def contains(self, q: int, r: int) -> bool:
        return (q, r) in self.idx

    def distance(self, i: int, j: int) -> int:
        return hex_distance(
            int(self.qs[i]) - int(self.qs[j]), int(self.rs[i]) - int(self.rs[j])
        )

    def ring_offsets(self, radius: int) -> list[tuple[int, int]]:
        """All axial offsets with hex_distance == radius (cached)."""
        off = self._ring_cache.get(radius)
        if off is None:
            off = []
            R = radius
            for dq in range(-R, R + 1):
                for dr in range(max(-R, -dq - R), min(R, -dq + R) + 1):
                    if hex_distance(dq, dr) == R:
                        off.append((dq, dr))
            self._ring_cache[radius] = off
        return off

    def within(self, i: int, radius: int) -> list[int]:
        """Indices of all on-map hexes within `radius` of hex i (inclusive)."""
        qi, ri = int(self.qs[i]), int(self.rs[i])
        out = []
        for dq in range(-radius, radius + 1):
            for dr in range(
                max(-radius, -dq - radius), min(radius, -dq + radius) + 1
            ):
                j = self.idx.get((qi + dq, ri + dr))
                if j is not None:
                    out.append(j)
        return out
