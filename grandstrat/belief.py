"""Belief construction over the hidden state (design §2).

Two belief mechanisms over the SAME hypothesis space, both built ONLY from a
civ's fog observation (memory layer + visibility history + own/visible
units + public terrain) and the public content tables -- never from the
hidden GameState:

* EXACT SUPPORT ENUMERATION for small maps: every hidden configuration consistent with the observation,
  where "consistent" is exact -- each tracked enemy unit is on a hex it can
  LEGALLY have reached from where it was last seen (time-expanded BFS under
  the unit's movement rules, through terrain, never crossing a hex the
  observer could see at that turn -- it would have been re-seen), not on a
  hex currently visible or known-occupied, and not colliding with another
  tracked unit.  A configuration where a tracked unit no longer exists (it
  may have died unseen) is the GONE hypothesis: the civ cannot rule it out,
  so it is part of the support.  Unreachable configurations are rejected by
  construction (design: "a configuration is in the support only if some
  legal history produces it").

* A PARTICLE FILTER with a fixed particle count for large maps: uniform
  sampling of the same candidate space with seeded draws from the named
  `belief` stream, weighted by consistency with the current observation and
  resampled systematically (deterministic given the seed).  The effective
  sample size (ESS) is reported so degeneracy is measurable.

Non-degeneracy contract:
the support must contain more than one configuration, must contain the true
configuration when one exists, and must not be ONLY the true configuration;
the particle ensemble must keep ESS > 1.  tests/test_belief.py asserts all of
these on real fog games.

Belief structures are built on request only (env.belief(civ) or the fog
module's small-map purchase accounting); nothing in the step path builds
them, and belief draws never touch the run's draw counters (the seed derives
purely from the master seed via rng.stream_seed).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .hexmap import HexMap
from .rng import Stream, stream_seed

GONE = None  # the "this unit no longer exists" hypothesis


class Track:
    """One hidden entity whose position the observer is uncertain about.

    `key` distinguishes the record within one belief ((owner, type_name,
    last_hex, last_turn) of the sighting that created it); candidates are
    hex indices for the ALIVE hypotheses; GONE is available for tracks that
    rest on memory only (the unit may have died unseen) and never for pinned
    (currently visible) tracks."""

    __slots__ = ("key", "owner", "type_name", "last_hex", "last_turn",
                 "candidates", "fixed")

    def __init__(self, key, owner: int, type_name: str, last_hex: int,
                 last_turn: int, candidates: np.ndarray, fixed: bool = False):
        self.key = key
        self.owner = owner
        self.type_name = type_name
        self.last_hex = int(last_hex)
        self.last_turn = int(last_turn)
        self.candidates = candidates
        self.fixed = fixed  # currently visible: exists, position known

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Track({self.key}, last={self.last_hex}@{self.last_turn}, "
                f"|cand|={len(self.candidates)})")


def _geometry(obs: dict) -> HexMap:
    return HexMap(int(obs["meta"]["radius"]))


def _blocked_now(obs: dict) -> set[int]:
    """Hexes a hidden unit cannot currently occupy: visible hexes, hexes of
    the observer's own units, and hexes of currently visible foreign units
    (those are separate pinned tracks)."""
    vis = obs["visible"]
    blocked = set(int(i) for i in np.flatnonzero(vis))
    hm = _geometry(obs)
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        j = hm.idx.get((int(q), int(r)))
        if j is not None:
            blocked.add(j)
    return blocked


def tracks_and_blocked(obs: dict, tables) -> tuple[list[Track], set[int]]:
    """Pinned tracks (visible foreign units, singleton candidates), mobile
    tracks (memory records unresolved by a sighting), and the blocked set."""
    hm = _geometry(obs)
    you = obs["you"]
    vis = obs["visible"]
    mem = obs["memory"]
    unit_names = obs["unit_names"]

    visible_foreign = []
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        if owner != you:
            # unit_list rows carry the type NAME (unlike the memory arrays,
            # which store table indices)
            j = hm.idx.get((int(q), int(r)))
            visible_foreign.append((j, int(owner), utype))

    blocked = set(int(i) for i in np.flatnonzero(vis))
    for j, owner, tname in visible_foreign:
        blocked.add(j)
    # own city hexes: an enemy unit cannot end on them (capture would be a
    # visible, memory-recorded event)
    for cid, owner, q, r, pop, hp, queue, bld, border, cap in obs["city_list"]:
        if owner == you:
            j = hm.idx.get((int(q), int(r)))
            if j is not None:
                blocked.add(j)

    now = int(obs["meta"]["turn"])
    vis_hist = obs.get("vis_history", {})

    pinned: list[Track] = []
    for j, owner, tname in visible_foreign:
        pinned.append(Track(("vis", owner, tname, j, now), owner, tname, j,
                            now, np.array([j], dtype=np.int64), fixed=True))

    move_costs = _move_cost_lookup(tables)
    mobile: list[Track] = []
    for i in np.flatnonzero(mem["unit_hp"] >= 0):
        owner = int(mem["unit_owner"][i])
        if owner == you:
            continue
        tname = unit_names[int(mem["unit_type"][i])]
        last_turn = int(mem["last_seen"][i])
        cands = _candidate_positions(i, last_turn, now, tname, owner,
                                     obs, tables, hm, vis_hist, move_costs,
                                     visible_foreign)
        if len(cands) == 0:
            continue  # fully constrained away: the record is stale/resolved
        track = Track(("mem", owner, tname, int(i), last_turn), owner, tname,
                      i, last_turn, cands)
        # resolution: if exactly one visible foreign unit of the same
        # (owner, type) is among the candidates, the record is subsumed by
        # that pinned sighting (it is the same unit, still in view).
        matches = [v for v in visible_foreign
                   if v[1] == owner and v[2] == tname and v[0] in set(cands.tolist())]
        if len(matches) == 1:
            continue
        mobile.append(track)
    return pinned + mobile, blocked


def _move_cost_lookup(tables) -> dict[str, int]:
    return {name: int(spec["move_cost"])
            for name, spec in tables.terrain.items()}


def _candidate_positions(last_hex: int, last_turn: int, now: int,
                         type_name: str, owner: int, obs: dict, tables,
                         hm: HexMap, vis_hist: dict, move_costs: dict,
                         visible_foreign) -> np.ndarray:
    """Exact time-expanded reach for one tracked unit (design §2 "consistent
    with a civ's observations"): turn-by-turn BFS from the last sighting.
    Staying put is always legal, so the reachable set grows monotonically;
    at every turn the unit must not stand on a hex the observer could see
    (it would have been re-seen and the memory refreshed), moves obey
    terrain passability and per-turn move budgets, and the final position
    cannot be currently visible or on a visible foreign unit's hex."""
    if now <= last_turn:
        return np.array([last_hex], dtype=np.int64)
    move = int(tables.units[type_name]["move"])
    terrain = obs["terrain"]
    terrain_names = obs["terrain_names"]
    foreign_hexes = {v[0] for v in visible_foreign}
    reach = {last_hex}
    for t in range(last_turn + 1, now + 1):
        blocked_t = vis_hist.get(t)
        new = set()
        for i in reach:
            for d in range(6):
                j = int(hm.neighbors[i, d])
                if j < 0 or j in reach or j in new or j in foreign_hexes:
                    continue
                spec = tables.terrain[terrain_names[int(terrain[j])]]
                if not spec["passable"] or move_costs[terrain_names[int(terrain[j])]] > move:
                    continue
                if blocked_t is not None and blocked_t[j]:
                    continue
                new.add(j)
        reach |= new
        if blocked_t is not None:
            reach = {i for i in reach if not blocked_t[i]}
        if not reach:
            break
    return np.array(sorted(reach), dtype=np.int64)


# ---------------------------------------------------------------------------
# exact support
# ---------------------------------------------------------------------------

def constraint_order(tracks: list[Track]) -> list[Track]:
    """Tracks sorted most-constrained first (deterministic tie: track key).
    `enumerate_support` yields configurations in THIS track order, and
    SupportResult stores its tracks in the same order so consumers (e.g.
    determinize) can align a configuration tuple with its tracks."""
    return [tracks[k] for k in
            sorted(range(len(tracks)), key=lambda k: (len(tracks[k].candidates),
                                                      str(tracks[k].key)))]


def enumerate_support(obs: dict, tables, cap: int | None = None
                      ) -> Iterator[tuple]:
    """Yield every hidden configuration consistent with the observation:
    one position (or GONE) per track, mutually non-colliding, none on a
    known-occupied hex.  Configurations are tuples over `constraint_order`
    tracks.  `cap`, when given, stops after that many configs.
    Exact on small maps; on large maps the count can be astronomical -- use
    count_support_size's cap or the particle filter instead."""
    tracks, blocked = tracks_and_blocked(obs, tables)
    tracks = constraint_order(tracks)
    n = len(tracks)
    chosen: list = [GONE] * n

    def rec(k: int, used: set[int]) -> Iterator[tuple]:
        if cap is not None and _count[0] >= cap:
            return
        if k == n:
            _count[0] += 1
            yield tuple(chosen)
            return
        tr = tracks[k]
        for pos in tr.candidates.tolist():
            if pos in used or pos in blocked:
                continue
            chosen[k] = pos
            yield from rec(k + 1, used | {pos})
        if not tr.fixed:
            chosen[k] = GONE
            yield from rec(k + 1, used)

    _count = [0]
    yield from rec(0, set())


def count_support_size(obs: dict, tables, cap: int = 200_000) -> int:
    """|support| exactly, up to `cap` (then return the negative count so a
    caller can tell a truncation from a true size)."""
    n = 0
    for _ in enumerate_support(obs, tables, cap=cap + 1):
        n += 1
        if n > cap:
            return -n
    return n


# ---------------------------------------------------------------------------
# particle filter (fixed particle count, deterministic)
# ---------------------------------------------------------------------------

class ParticleBelief:
    """Fixed-N particle ensemble over the same support space.

    Particles are sampled uniformly from per-track candidates with the named
    `belief` stream, weighted by consistency with the current observation
    (distinct positions, none blocked), then resampled systematically.  The
    true configuration has consistency weight 1 whenever it exists (it never
    sits on a visible hex), so systematic resampling keeps it in the
    ensemble.  `ess` is the pre-resample effective sample size."""

    def __init__(self, obs: dict, tables, seed: int, n_particles: int = 64):
        self.n = int(n_particles)
        self.tracks, self.blocked = tracks_and_blocked(obs, tables)
        self.turn = int(obs["meta"]["turn"])
        self.seed = int(seed)
        rng = Stream("belief", stream_seed(seed, "belief"))
        t0 = self.turn
        parts = []
        for p in range(self.n):
            cfg = []
            for t_idx, tr in enumerate(self.tracks):
                opts = tr.candidates.tolist()
                if not tr.fixed:
                    opts = opts + [GONE]
                cfg.append(opts[rng.randint(0, len(opts) - 1,
                                            "init", t0, p, t_idx)])
            parts.append(tuple(cfg))
        weights = np.array([self._weight(c) for c in parts], dtype=np.float64)
        total = weights.sum()
        self.ess = float(weights.sum() ** 2 / (weights ** 2).sum()) if total > 0 else 0.0
        self.particles = self._resample(parts, weights, rng, t0)
        self.raw_particles = parts
        self.raw_weights = weights

    def _weight(self, cfg: tuple) -> float:
        used = set()
        for pos, tr in zip(cfg, self.tracks):
            if pos is GONE:
                continue
            if pos in used:
                return 0.0
            # a pinned (visible) unit's position IS on a visible hex -- the
            # blocked set only rules out positions of HYPOTHESISED units
            if not tr.fixed and pos in self.blocked:
                return 0.0
            used.add(pos)
        return 1.0

    def _resample(self, parts, weights, rng: Stream, turn: int) -> list[tuple]:
        total = weights.sum()
        if total <= 0:
            # nothing consistent: keep the raw ensemble and flag it via ess=0
            return list(parts)
        w = weights / total
        cum = np.cumsum(w)
        u0 = rng.uniform("resample", turn)
        out = []
        j = 0
        for i in range(self.n):
            u = (i + u0) / self.n  # systematic: one pointer, monotone u
            while cum[j] < u and j < len(w) - 1:
                j += 1
            out.append(parts[j])
        return out

    def truth_index(self, truth: tuple) -> int | None:
        """Index of the true configuration in the ensemble, or None."""
        try:
            return self.particles.index(truth)
        except ValueError:
            return None

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ParticleBelief(n={self.n}, tracks={len(self.tracks)}, "
                f"ess={self.ess:.1f})")


def build_belief(obs: dict, tables, seed: int, particles: int | None = None,
                 kind: str = "auto"):
    """Build the belief for one observation: exact enumeration support on
    small maps, particle filter on large ones (`kind` forces a choice)."""
    R = int(obs["meta"]["radius"])
    small = R <= 4 and len(obs["terrain"]) <= 200
    if kind == "particles" or (kind == "auto" and not small):
        n = particles or int(tables.information["belief_particles"]["value"])
        return ParticleBelief(obs, tables, seed, n)
    support = enumerate_support(obs, tables)
    return SupportResult(list(support), tracks_and_blocked(obs, tables)[0])


class SupportResult:
    """Exact support: the list of consistent configurations plus the tracks
    they range over, in `constraint_order` (configuration tuple position k
    belongs to tracks[k])."""

    def __init__(self, configs: list[tuple], tracks: list[Track]):
        self.configs = configs
        self.tracks = constraint_order(tracks)

    def __len__(self) -> int:
        return len(self.configs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SupportResult(|support|={len(self.configs)}, tracks={len(self.tracks)})"
