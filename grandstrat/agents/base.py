"""Scripted baseline agents (design §3): expander, warmonger, turtle,
scout-heavy.  Deterministic given the seed: all tie-breaks are by id /
stable order, and the only draws come from the civ's own `agent_i` stream,
which is used ONLY when the agent is constructed with noise > 0 (documented
in docs/DESIGN_NOTES.md: at the defaults the agent_* streams draw
zero times).

Agents consume OBSERVATIONS ONLY (dict-of-arrays from grandstrat.observe plus
the public content tables) -- never the hidden GameState.  Every submitted
action is filtered against the env's legal-action list.
"""

from __future__ import annotations

import numpy as np

from ..hexmap import DIRECTIONS, hex_distance
from ..tables import load_tables


def legal_set(legal) -> set:
    return {tuple(a) for a in legal}


_GEO_CACHE: dict[tuple, dict] = {}


def geo_for(obs: dict) -> dict:
    """Axial lattice for the observed map shape (cached; pure geometry)."""
    key = (obs["meta"]["radius"], len(obs["terrain"]))
    geo = _GEO_CACHE.get(key)
    if geo is None:
        R = key[0]
        idx = {}
        coords = []
        for q in range(-R, R + 1):
            for r in range(max(-R, -q - R), min(R, -q + R) + 1):
                idx[(q, r)] = len(coords)
                coords.append((q, r))
        assert len(coords) == key[1]
        arr = np.array(coords, dtype=np.int16)
        geo = {
            "idx": idx,
            "coords": coords,
            "qs": arr[:, 0].astype(np.int32),
            "rs": arr[:, 1].astype(np.int32),
            "_dist": {},
        }
        _GEO_CACHE[key] = geo
    return geo


def dist_field(obs: dict, q: int, r: int) -> np.ndarray:
    """Vectorized hex-distance from (q, r) to every hex of the map.

    Pure geometry: the answer depends only on the map shape and (q, r), so it
    is memoized per shape (this was ~7% of
    engine time recomputed thousands of times per game).  The cached array is
    handed out READ-ONLY so a caller cannot poison the cache; every existing
    caller only reads it."""
    geo = geo_for(obs)
    cache = geo["_dist"]
    key = (q, r)
    d = cache.get(key)
    if d is None:
        qs, rs = geo["qs"], geo["rs"]
        dq = qs - q
        dr = rs - r
        d = (np.abs(dq) + np.abs(dq + dr) + np.abs(dr)) // 2
        d.flags.writeable = False
        cache[key] = d
    return d


def hex_index(obs: dict, q: int, r: int) -> int:
    return geo_for(obs)["idx"][(q, r)]


class Agent:
    """Base class: noise plumbing + obs accessors.

    `noise` (default 0) is the probability of replacing each chosen action
    with a uniform random legal action, drawn world-keyed (turn, slot) from
    the civ's own agent stream.
    """

    name = "base"

    def __init__(self, civ_id: int, params: dict | None = None):
        self.civ_id = civ_id
        self.params = dict(params or {})
        self.noise = float(self.params.get("noise", 0.0))
        self.stream_name = f"agent_{civ_id}"
        self._tables = load_tables()

    def reset(self) -> None:
        pass

    # Diplomacy layer trigger: action kinds that route into
    # `_diplomacy_layer`.  The default scripted policy only reacts to live
    # offers addressed to its civ.
    _diplomacy_needs = ("ACCEPT",)

    # -- to override -------------------------------------------------------
    def plan(self, obs: dict, legal: set) -> list[tuple]:
        raise NotImplementedError

    # -- public --------------------------------------------------------------
    @staticmethod
    def _actor(a):
        """(kind, id) of the actor an action binds, or None for SKIP.  Mirrors
        engine._UNIT_ACTIONS / _CITY_ACTIONS."""
        if a[0] in ("MOVE", "ATTACK_U", "ATTACK_C", "FOUND", "IMPROVE"):
            return ("u", a[1])
        if a[0] == "PRODUCE":
            return ("c", a[1])
        return None

    def _one_per_actor(self, actions):
        """Keep the FIRST action for each actor, drop later ones.

        Batch rule: the engine rejects a batch that binds
        one unit twice (it used to execute both).  A planner -- and in
        particular the noise substitution below, which draws from the whole
        legal pool -- can propose two actions for the same unit; dropping the
        later one here is deterministic and keeps every agent submittable."""
        out, bound = [], set()
        for a in actions:
            k = self._actor(a)
            if k is not None:
                if k in bound:
                    continue
                bound.add(k)
            out.append(a)
        return out

    def act(self, obs: dict, legal, rng) -> list[tuple]:
        ls = legal_set(legal)
        chosen = self._one_per_actor([a for a in self.plan(obs, ls) if a in ls])
        # cheap pre-filter on the plain list: the common case (diplomacy
        # module inactive) must not touch the set or allocate anything.
        # Subclasses that also OFFER widen `_diplomacy_needs`.
        if any(a[0] in self._diplomacy_needs for a in legal):
            chosen = self._diplomacy_layer(obs, ls, legal, chosen)
        if self.noise > 0.0 and chosen:
            out = []
            turn = obs["meta"]["turn"]
            pool = sorted(ls)
            for i, a in enumerate(chosen):
                if rng.chance(self.noise, "noise", turn, i):
                    out.append(pool[rng.randint(0, len(pool) - 1, "pick", turn, i)])
                else:
                    out.append(a)
            return self._one_per_actor(out)
        return chosen

    # -- diplomacy policy (no-op unless the diplomacy module is on)
    def _diplomacy_layer(self, obs: dict, ls: set, legal, chosen: list) -> list:
        """Default scripted diplomacy: ACCEPT live offers that meet the data
        reservation plus a PUBLIC pseudo-jitter derived from the offer id
        (offer*7 + civ*13 mod jitter -- deterministic, no rng draws, and the
        jitter is computable from the observation since offers are public
        news).  Scripted civs never OFFER and never REJECT (offers expire);
        agents that trade override this hook (see merchant).  Acceptance
        scans offers in pid order: set iteration order must never decide
        which offer is taken (PYTHONHASHSEED independence)."""
        accepts = sorted((a for a in ls if a[0] == "ACCEPT"), key=lambda a: a[1])
        if not accepts:
            return chosen
        K = self._tables.constants
        base = K["diplomacy_reservation"]["value"]
        jitter = int(K["diplomacy_threshold_jitter"]["value"])
        offers = {o["pid"]: o
                  for o in obs["meta"].get("diplomacy_offers", [])}
        for a in accepts:
            o = offers.get(a[1])
            if o is None:
                continue
            need = int(base[o["kind"]]) + (a[1] * 7 + self.civ_id * 13) % jitter
            if int(o["offer"]) >= need:
                chosen.append(a)
                break  # one diplomacy order per turn
        return chosen

    # -- obs accessors ---------------------------------------------------------
    def own_units(self, obs):
        return [u for u in obs["unit_list"] if u[1] == self.civ_id]

    def own_cities(self, obs):
        return [c for c in obs["city_list"] if c[1] == self.civ_id]

    def enemy_cities(self, obs):
        return [c for c in obs["city_list"] if c[1] != self.civ_id and c[1] >= 0]

    def passable(self, obs, q, r) -> bool:
        tname = obs["terrain_names"][int(obs["terrain"][hex_index(obs, q, r)])]
        return bool(self._tables.terrain[tname]["passable"])

    def hex_dist(self, q0, r0, q1, r1) -> int:
        return hex_distance(q1 - q0, r1 - r0)

    def step_toward(self, legal, uid, cur_q, cur_r, tq, tr):
        """Best legal single step toward (tq, tr); prefers moves that reduce
        the hex distance, falls back to a distance-neutral sidestep so units
        flow around each other instead of deadlocking; None if fully stuck."""
        d0 = self.hex_dist(cur_q, cur_r, tq, tr)
        best, best_d = None, d0
        best_side, side_d = None, None
        for d, (dq, dr) in enumerate(DIRECTIONS):
            a = ("MOVE", uid, d)
            if a not in legal:
                continue
            dist = self.hex_dist(cur_q + dq, cur_r + dr, tq, tr)
            if dist < best_d:
                best, best_d = a, dist
            elif dist == d0 and best_side is None:
                best_side, side_d = a, dist
        return best or best_side
