"""Fog of war + purchasable information (design §2).

`FogInformation` is the `information: "fog"` implementation of the
`observe.InformationModule` interface.  Full observability stays the default
module (design §0b); selecting this one via kernel config
`{"information": "fog"}` turns the fog dial W ∈ {0..K} into real structure:

* visible set per civ = hexes within the sight radius of that civ's units
  (per `units.json` `sight`) and cities (`city_sight`, data/information.json);
* a `last_seen` memory layer: the turn each hex was last seen plus the
  contents AS THEY WERE WHEN SEEN, never the current truth;
* the four gold purchases SPY / TRADE / DIPLO / ORACLE (SCOUT is priced in
  kind -- a unit's production + upkeep + exposure -- and adds no action),
  enabled and priced per dial rung from data/information.json and charged
  through the gold ledger so per-turn conservation still balances;
* the belief-support hook: purchase record entries carry the buyer's belief
  support size before/after on small maps (design §5).

ISOLATION RULE: the hidden GameState never aliases into an observation.  Every fog
array handed out is a fresh read-only copy; units and cities of other civs
appear only on currently visible hexes; per-civ tables (gold/score/techs/
research) are visible for self only (alive flags are public); no count of a
hidden entity leaves the module.  Every unit/city id in a
fog observation (unit rows, city rows, memory, and the agent-facing action
space) is an OBSERVER-LOCAL HANDLE -- a stable, per-civ negative integer.
The engine's global creation counters leak how many entities the whole world
has spawned (a first city with id 7 reveals seven cities exist), so they
never leave the module; `present_actions` re-keys the legal-action list into
handles and `resolve_action` maps them back before resolution.  At dial K
and after an ORACLE purchase the shared full-observation base is handed out
with true ids -- that rung IS the oracle view.

PERF NOTES (design §5 has ~0 headroom): visibility is recomputed per turn
with cached sight-reach matrices (one vectorised OR per civ); memory is
updated only on visible (dirty) hexes; at dial K -- and after an ORACLE
purchase -- the shared full-observation base is handed out unmasked, so the
top rung costs nothing over the default module.

Buffer contract: as in the full module, handed-out arrays are valid until the next
step/reset; the module serves per-(turn, civ) cached observation dicts.
"""

from __future__ import annotations

import numpy as np

from .engine import InvalidAction
from .hexmap import HexMap
from .observe import InformationModule, _ro
from .state import GameState

# owner-layer sentinel for "never seen": owner -1 means unowned, so the
# unknown marker must be distinct.
OWNER_UNKNOWN = -2
FOG_K = 4  # top dial rung = full observability (scenarios.FOG_K)


def hex_line(hm: HexMap, i0: int, i1: int) -> list[int]:
    """Hex indices along the hex line from i0 to i1 (inclusive), by axial
    lerp + cube rounding.  Pure geometry, deterministic."""
    q0, r0 = int(hm.qs[i0]), int(hm.rs[i0])
    q1, r1 = int(hm.qs[i1]), int(hm.rs[i1])
    n = hm.distance(i0, i1)
    out: list[int] = []
    for k in range(n + 1):
        t = k / n if n else 0.0
        x = q0 + (q1 - q0) * t
        z = r0 + (r1 - r0) * t
        y = -x - z
        rx, ry, rz = round(x), round(y), round(z)
        dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
        if dx > dy and dx > dz:
            rx = -ry - rz
        elif dy > dz:
            ry = -rx - rz
        # axial coordinates are (q, r) = (cube x, cube z)
        j = hm.idx.get((rx, rz))
        if j is not None and j not in out:
            out.append(j)
    return out


class _CivFog:
    """Per-civ visibility + memory bookkeeping (one per civ per game).

    It also carries the OBSERVER-LOCAL HANDLE maps: every unit/city id
    ever visible to this civ is re-keyed into a private, negative handle so
    observations never expose the engine's global creation counters (they
    leak how many units/cities the whole world has spawned -- e.g. a first
    city with id 7 tells the observer seven cities exist).  Handles are
    stable for the whole game, so actions and memory stay trackable; the
    engine's true ids remain the internal indexing handles everywhere."""

    __slots__ = (
        "vis", "vis_hist", "last_seen", "resource", "improve", "owner",
        "u_owner", "u_type", "u_hp",
        "c_id", "c_owner", "c_pop", "c_hp", "c_queue", "c_border",
        "c_capital", "c_bld",
        "spies", "shared", "pending_trade", "pending_diplo",
        "u_map", "u_rev", "u_next", "c_map", "c_rev", "c_next",
    )

    def __init__(self, L: int):
        self.vis = np.zeros(L, dtype=bool)
        self.vis_hist: dict[int, np.ndarray] = {}
        self.last_seen = np.full(L, -1, dtype=np.int32)
        self.resource = np.full(L, -1, dtype=np.int8)
        self.improve = np.full(L, -1, dtype=np.int8)
        self.owner = np.full(L, OWNER_UNKNOWN, dtype=np.int8)
        self.u_owner = np.full(L, OWNER_UNKNOWN, dtype=np.int8)
        self.u_type = np.full(L, -1, dtype=np.int8)
        self.u_hp = np.full(L, -1, dtype=np.int16)
        self.c_id = np.full(L, -1, dtype=np.int32)
        self.c_owner = np.full(L, OWNER_UNKNOWN, dtype=np.int8)
        self.c_pop = np.full(L, -1, dtype=np.int8)
        self.c_hp = np.full(L, -1, dtype=np.int16)
        self.c_queue = np.full(L, -1, dtype=np.int16)
        self.c_border = np.full(L, -1, dtype=np.int8)
        self.c_capital = np.full(L, -1, dtype=np.int8)
        self.c_bld = np.zeros(L, dtype=np.int32)
        self.spies: list[tuple[int, int]] = []  # (city_id, until_turn incl.)
        self.shared: set[int] = set()  # DIPLO partners (live map-layer exchange)
        self.pending_trade: list[np.ndarray] = []  # corridor hex indices
        self.pending_diplo: list[int] = []
        # observer-local handle maps: true id <-> handle (negative ints)
        self.u_map: dict[int, int] = {}
        self.u_rev: dict[int, int] = {}
        self.u_next = 0
        self.c_map: dict[int, int] = {}
        self.c_rev: dict[int, int] = {}
        self.c_next = 0


class FogInformation(InformationModule):
    name = "fog"

    def __init__(self, tables):
        self.tables = tables
        self._hexmaps: dict[int, HexMap] = {}
        self._reach: dict[tuple, np.ndarray] = {}
        # per-game state; rebuilt lazily whenever the GameState object changes
        self._game: list[_CivFog] = []
        self._oracle: set[int] = set()
        self._pending_support: list[dict] = []
        self._state_ref: GameState | None = None
        self._turn = -1
        self._obs_cache: dict[tuple, dict] = {}

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        """Drop all per-game bookkeeping (env.reset calls this)."""
        self._game = []
        self._oracle = set()
        self._pending_support = []
        self._state_ref = None
        self._turn = -1
        self._obs_cache = {}

    def _hm(self, R: int) -> HexMap:
        hm = self._hexmaps.get(R)
        if hm is None:
            hm = HexMap(R)
            self._hexmaps[R] = hm
        return hm

    def _reach_for(self, R: int, s: int) -> np.ndarray:
        """L x L uint8 matrix, 1 where hex distance <= s (cached per shape)."""
        key = (R, s)
        m = self._reach.get(key)
        if m is None:
            hm = self._hm(R)
            qs = hm.qs.astype(np.int32)
            rs = hm.rs.astype(np.int32)
            dq = qs[None, :] - qs[:, None]
            dr = rs[None, :] - rs[:, None]
            m = ((np.abs(dq) + np.abs(dq + dr) + np.abs(dr)) // 2 <= s).astype(np.uint8)
            self._reach[key] = m
        return m

    # -- observer-local handles -----------------------------------------------
    def _u_handle(self, fog: "_CivFog", uid: int) -> int:
        h = fog.u_map.get(uid)
        if h is None:
            h = -1 - fog.u_next
            fog.u_next += 1
            fog.u_map[uid] = h
            fog.u_rev[h] = uid
        return h

    def _c_handle(self, fog: "_CivFog", cid: int) -> int:
        h = fog.c_map.get(cid)
        if h is None:
            h = -1 - fog.c_next
            fog.c_next += 1
            fog.c_map[cid] = h
            fog.c_rev[h] = cid
        return h

    def _fog_for(self, state: GameState, civ: int) -> "_CivFog | None":
        if not self._game or civ >= len(self._game):
            return None
        return self._game[civ]

    def _city_sight(self, state: GameState, civ: int) -> int:
        """City sight radius for `civ`: the data base plus the civ's
        city_sight_bonus tech abilities."""
        from .economy import ability_sum

        return (int(self.tables.information["city_sight"]["value"])
                + ability_sum(state, state.civs[civ].techs, "city_sight_bonus"))

    # -- per-turn update -----------------------------------------------------
    def _ensure(self, state: GameState) -> None:
        if (self._state_ref is state and self._game
                and self._turn == state.turn):
            return
        self._update(state)

    def _update(self, state: GameState) -> None:
        R = int(state.params["R"])
        hm = self._hm(R)
        L = hm.n
        n_civs = len(state.civs)
        new_game = (not self._game or len(self._game) != n_civs
                    or self._state_ref is not state)
        if new_game:
            self._game = [_CivFog(L) for _ in range(n_civs)]
            self._oracle = set()
            self._pending_support = []
        self._state_ref = state
        self._turn = state.turn
        self._obs_cache = {}
        W = int(state.dials["fog"])
        full_dial = W >= FOG_K

        # -- base visibility: own units + own cities -------------------------
        by_civ_sight: dict[tuple, list[int]] = {}
        for u in state.units.values():
            if u.owner >= 0:
                s = int(self.tables.units[u.type]["sight"])
                by_civ_sight.setdefault((u.owner, s), []).append(u.i)
        own_city_idx: dict[int, list[int]] = {}
        for c in state.cities.values():
            if c.owner >= 0:
                own_city_idx.setdefault(c.owner, []).append(hm.index(c.q, c.r))
        for civ in range(n_civs):
            fog = self._game[civ]
            vis = np.zeros(L, dtype=bool)
            if full_dial or civ in self._oracle:
                vis[:] = True
            else:
                for (owner, s), idxs in by_civ_sight.items():
                    if owner == civ:
                        vis |= self._reach_for(R, s)[idxs].any(axis=0).astype(bool)
                if own_city_idx.get(civ):
                    cs = self._city_sight(state, civ)
                    vis |= self._reach_for(R, cs)[own_city_idx[civ]].any(axis=0).astype(bool)
            fog.vis = vis

        # -- observer-local handles for everything newly seen ------------------
        # state.units / state.cities iterate in id order (GameState contract),
        # so per-turn assignment order is deterministic.  Skipped entirely at
        # the full-view rungs: the base observation there carries true ids.
        if not full_dial:
            for civ in range(n_civs):
                if civ in self._oracle:
                    continue
                fog = self._game[civ]
                vis = fog.vis
                for u in state.units.values():
                    if u.owner == civ or vis[u.i]:
                        self._u_handle(fog, u.id)
                for c in state.cities.values():
                    if c.owner == civ or vis[hm.index(c.q, c.r)]:
                        self._c_handle(fog, c.id)

        # -- DIPLO sharing: live union of partners' visible sets -------------
        for civ in range(n_civs):
            fog = self._game[civ]
            if fog.shared and not (full_dial or civ in self._oracle):
                frontier = list(fog.shared)
                seen: set[int] = set()
                while frontier:
                    p = frontier.pop()
                    if p in seen:
                        continue
                    seen.add(p)
                    fog.vis |= self._game[p].vis
                    frontier.extend(self._game[p].shared)

        # -- spies: live visibility of the target city hex -------------------
        for civ in range(n_civs):
            fog = self._game[civ]
            if full_dial or civ in self._oracle or not fog.spies:
                continue
            keep = []
            for cid, until in fog.spies:
                if state.turn <= until:
                    city = state.cities.get(cid)
                    if city is not None:
                        fog.vis[hm.index(city.q, city.r)] = True
                    keep.append((cid, until))
            fog.spies = keep

        # -- memory: dirty (visible) hexes only -------------------------------
        bnames = sorted(self.tables.buildings)
        item_names = list(self.tables.units) + bnames
        for civ in range(n_civs):
            fog = self._game[civ]
            fog.vis_hist[state.turn] = fog.vis
            idxs = np.flatnonzero(fog.vis)
            fog.last_seen[idxs] = state.turn
            fog.resource[idxs] = state.resource[idxs]
            fog.improve[idxs] = state.improve[idxs]
            fog.owner[idxs] = state.owner[idxs]
            uids = state.occ_unit[idxs]
            fog.u_hp[idxs] = -1
            fog.u_owner[idxs] = OWNER_UNKNOWN
            fog.u_type[idxs] = -1
            for k in np.flatnonzero(uids >= 0):
                u = state.units[int(uids[k])]
                i = idxs[k]
                fog.u_owner[i] = u.owner
                fog.u_type[i] = self.tables.unit_ids[u.type]
                fog.u_hp[i] = u.hp
            cids = state.occ_city[idxs]
            fog.c_id[idxs] = -1
            fog.c_owner[idxs] = OWNER_UNKNOWN
            fog.c_pop[idxs] = -1
            fog.c_hp[idxs] = -1
            fog.c_queue[idxs] = -1
            fog.c_border[idxs] = -1
            fog.c_capital[idxs] = -1
            fog.c_bld[idxs] = 0
            for k in np.flatnonzero(cids >= 0):
                self._write_city_memory(fog, int(cids[k]), int(idxs[k]),
                                        state, item_names, bnames)

        # -- deferred reveals from purchases resolved earlier this step ------
        for civ in range(n_civs):
            fog = self._game[civ]
            while fog.pending_trade:
                corr = fog.pending_trade.pop()
                fog.last_seen[corr] = state.turn
                fog.resource[corr] = state.resource[corr]
                fog.improve[corr] = state.improve[corr]
                fog.owner[corr] = state.owner[corr]
                uids = state.occ_unit[corr]
                fog.u_hp[corr] = -1
                fog.u_owner[corr] = OWNER_UNKNOWN
                fog.u_type[corr] = -1
                for k in range(len(corr)):
                    i = int(corr[k])
                    if uids[k] >= 0:
                        u = state.units[int(uids[k])]
                        fog.u_owner[i] = u.owner
                        fog.u_type[i] = self.tables.unit_ids[u.type]
                        fog.u_hp[i] = u.hp
                    cid = int(state.occ_city[i])
                    if cid >= 0:
                        fog.c_id[i] = cid
                        self._write_city_memory(fog, cid, i, state,
                                                item_names, bnames)
            while fog.pending_diplo:
                p = fog.pending_diplo.pop()
                src = self._game[p]
                dst = fog
                merge = (dst.last_seen < 0) & (src.last_seen >= 0)
                if merge.any():
                    dst.last_seen[merge] = src.last_seen[merge]
                    dst.resource[merge] = src.resource[merge]
                    dst.improve[merge] = src.improve[merge]
                    dst.owner[merge] = src.owner[merge]
                    for arr in ("u_owner", "u_type", "u_hp", "c_id", "c_owner",
                                "c_pop", "c_hp", "c_queue", "c_border",
                                "c_capital", "c_bld"):
                        getattr(dst, arr)[merge] = getattr(src, arr)[merge]

        # -- fill in purchase-record support sizes on small maps -------------
        for ev in self._pending_support:
            ev["support_after"] = self._support_size(state, ev["_buyer"])
            del ev["_buyer"]
        self._pending_support = []

    def _write_city_memory(self, fog: _CivFog, cid: int, i: int,
                           state: GameState, item_names, bnames) -> None:
        c = state.cities[cid]
        fog.c_id[i] = cid
        fog.c_owner[i] = c.owner
        fog.c_pop[i] = c.pop
        fog.c_hp[i] = c.hp
        fog.c_queue[i] = -1 if c.queue is None else item_names.index(c.queue)
        fog.c_border[i] = c.border
        fog.c_capital[i] = int(c.capital)
        mask = 0
        for b in c.buildings:
            mask |= 1 << bnames.index(b)
        fog.c_bld[i] = mask

    # -- observation -----------------------------------------------------
    def build_observation(self, state: GameState, civ: int, base: dict) -> dict:
        self._ensure(state)
        W = int(state.dials["fog"])
        key = (state.turn, civ)
        cached = self._obs_cache.get(key)
        if cached is not None:
            return cached
        if W >= FOG_K or civ in self._oracle:
            obs = dict(base)
            obs["you"] = civ
            obs["meta"] = {**base["meta"], "information": "fog", "fog_dial": W,
                           "info_state": self._info_state(civ)}
            obs["memory"] = self._full_memory_payload(state)
            self._obs_cache[key] = obs
            return obs
        fog = self._game[civ]
        hm = self._hm(int(state.params["R"]))
        bnames = sorted(self.tables.buildings)
        item_names = list(self.tables.units) + bnames
        vis = fog.vis

        # map layers: terrain is public static geography (docs/DESIGN_NOTES.md §14);
        # resource/improve/owner show current truth only where visible now.
        resource = state.resource.copy()
        resource[~vis] = -1
        improve = state.improve.copy()
        improve[~vis] = -1
        owner = state.owner.copy()
        owner[~vis] = OWNER_UNKNOWN

        # units: own everywhere, foreign only on visible hexes.  Ids are
        # re-keyed into this civ's observer-local handles: the
        # engine's creation counters never leave the module.
        units_sorted = sorted(state.units.values(), key=lambda u: u.id)
        rows = [u for u in units_sorted if u.owner == civ or vis[u.i]]
        unit_cap = int(self.tables.constants["unit_cap"]["value"])
        u_id = np.full(unit_cap, -1, dtype=np.int32)
        u_owner = np.full(unit_cap, -1, dtype=np.int8)
        u_type = np.full(unit_cap, -1, dtype=np.int8)
        u_q = np.full(unit_cap, -1, dtype=np.int16)
        u_r = np.full(unit_cap, -1, dtype=np.int16)
        u_hp = np.zeros(unit_cap, dtype=np.int16)
        u_mp = np.zeros(unit_cap, dtype=np.int8)
        unit_list = []
        for k, u in enumerate(rows):
            h = self._u_handle(fog, u.id)
            u_id[k] = h
            u_owner[k] = u.owner
            u_type[k] = self.tables.unit_ids[u.type]
            u_q[k] = u.q
            u_r[k] = u.r
            u_hp[k] = u.hp
            u_mp[k] = u.mp
            unit_list.append((h, u.owner, u.type, u.q, u.r, u.hp, u.mp))

        # cities: own everywhere, foreign only on visible hexes
        cities_sorted = sorted(state.cities.values(), key=lambda c: c.id)
        crows = [c for c in cities_sorted
                 if c.owner == civ or vis[hm.index(c.q, c.r)]]
        city_cap = int(self.tables.constants["city_cap"]["value"])
        c_id = np.full(city_cap, -1, dtype=np.int32)
        c_owner = np.full(city_cap, -1, dtype=np.int8)
        c_q = np.full(city_cap, -1, dtype=np.int16)
        c_r = np.full(city_cap, -1, dtype=np.int16)
        c_pop = np.zeros(city_cap, dtype=np.int8)
        c_hp = np.zeros(city_cap, dtype=np.int16)
        c_food = np.zeros(city_cap, dtype=np.int32)
        c_prod = np.zeros(city_cap, dtype=np.int32)
        c_queue = np.full(city_cap, -1, dtype=np.int16)
        c_bld = np.zeros(city_cap, dtype=np.int32)
        c_border = np.ones(city_cap, dtype=np.int8)
        c_cap = np.zeros(city_cap, dtype=np.int8)
        city_list = []
        for k, c in enumerate(crows):
            h = self._c_handle(fog, c.id)
            c_id[k] = h
            c_owner[k] = c.owner
            c_q[k] = c.q
            c_r[k] = c.r
            c_pop[k] = c.pop
            c_hp[k] = c.hp
            c_food[k] = c.food_bucket
            c_prod[k] = c.prod_bucket
            if c.queue is not None:
                c_queue[k] = item_names.index(c.queue)
            mask = 0
            for b in c.buildings:
                mask |= 1 << bnames.index(b)
            c_bld[k] = mask
            c_border[k] = c.border
            c_cap[k] = int(c.capital)
            city_list.append((h, c.owner, c.q, c.r, c.pop, c.hp, c.queue,
                              c.buildings, c.border, bool(c.capital)))

        # per-civ tables: self always; others only through a reveal
        n_civs = len(state.civs)
        gold = base["gold"].copy()
        gold[[i for i in range(n_civs) if i != civ]] = -1
        score = base["score"].copy()
        score[[i for i in range(n_civs) if i != civ]] = -1
        techs = np.zeros_like(base["techs"])
        techs[civ] = base["techs"][civ]
        researching = [None] * n_civs
        researching[civ] = base["researching"][civ]
        rprog = np.zeros_like(base["research_progress"])
        rprog[civ] = base["research_progress"][civ]

        obs = {
            "meta": {**base["meta"], "information": "fog", "fog_dial": W,
                     "info_state": self._info_state(civ)},
            "you": civ,
            "terrain": base["terrain"],
            "terrain_names": base["terrain_names"],
            "resource": _ro(resource),
            "resource_names": base["resource_names"],
            "improve": _ro(improve),
            "improve_names": base["improve_names"],
            "owner": _ro(owner),
            "visible": _ro(vis.copy()),
            "memory": self._memory_payload(fog),
            "vis_history": dict(fog.vis_hist),
            "units": {
                "id": _ro(u_id), "owner": _ro(u_owner), "type": _ro(u_type),
                "q": _ro(u_q), "r": _ro(u_r), "hp": _ro(u_hp), "mp": _ro(u_mp),
                "count": len(rows),
            },
            "unit_names": base["unit_names"],
            "unit_list": tuple(unit_list),
            "cities": {
                "id": _ro(c_id), "owner": _ro(c_owner), "q": _ro(c_q), "r": _ro(c_r),
                "pop": _ro(c_pop), "hp": _ro(c_hp), "food": _ro(c_food),
                "prod": _ro(c_prod), "queue": _ro(c_queue), "buildings": _ro(c_bld),
                "border": _ro(c_border), "capital": _ro(c_cap),
                "count": len(crows),
            },
            "building_names": bnames,
            "city_list": tuple(city_list),
            "gold": _ro(gold),
            "score": _ro(score),
            "alive": base["alive"],
            "techs": _ro(techs),
            "tech_names": base["tech_names"],
            "researching": researching,
            "research_progress": _ro(rprog),
            "diplomacy": base["diplomacy"],
            "production_items": item_names,
        }
        self._obs_cache[key] = obs
        return obs

    def _memory_payload(self, fog: "_CivFog", map_ids: bool = True) -> dict:
        """Read-only COPIES of one civ's `last seen` layer.  Copies, not
        views: the module keeps mutating its internal arrays as hexes are
        re-seen, and handed-out observations must never change underneath a
        holder within a turn.  Memory city ids are handed out as this civ's
        observer-local handles (entries remembered through a DIPLO merge that
        this civ never saw directly get handles lazily here); `map_ids=False`
        (the full-view rungs) hands out true ids, matching the base view."""
        def c(a):
            return _ro(a.copy())

        if map_ids:
            cid_h = fog.c_id.copy()
            for i in np.flatnonzero(cid_h >= 0):
                cid_h[i] = self._c_handle(fog, int(cid_h[i]))
        else:
            cid_h = fog.c_id
        return {
            "last_seen": c(fog.last_seen),
            "resource": c(fog.resource),
            "improve": c(fog.improve),
            "owner": c(fog.owner),
            "unit_owner": c(fog.u_owner),
            "unit_type": c(fog.u_type),
            "unit_hp": c(fog.u_hp),
            "city_id": c(cid_h),
            "city_owner": c(fog.c_owner),
            "city_pop": c(fog.c_pop),
            "city_hp": c(fog.c_hp),
            "city_queue": c(fog.c_queue),
            "city_border": c(fog.c_border),
            "city_capital": c(fog.c_capital),
            "city_buildings": c(fog.c_bld),
        }

    def _full_memory_payload(self, state: GameState) -> dict:
        """Memory payload for full observability: everything seen this turn.
        Vectorised (this runs per civ per turn at the top rung)."""
        hm = self._hm(int(state.params["R"]))
        L = hm.n
        fog = _CivFog(L)
        fog.vis[:] = True
        fog.last_seen[:] = state.turn
        fog.resource = state.resource.copy()
        fog.improve = state.improve.copy()
        fog.owner = state.owner.copy()
        # per-unit-id lookup tables, then a single gather over the grid
        n_ids = max(state.next_unit_id, 1)
        owner_by_id = np.full(n_ids, OWNER_UNKNOWN, dtype=np.int8)
        type_by_id = np.full(n_ids, -1, dtype=np.int8)
        hp_by_id = np.full(n_ids, -1, dtype=np.int16)
        for u in state.units.values():
            owner_by_id[u.id] = u.owner
            type_by_id[u.id] = self.tables.unit_ids[u.type]
            hp_by_id[u.id] = u.hp
        occ = state.occ_unit
        live = occ >= 0
        idx = np.where(live, occ, 0)
        fog.u_owner = np.where(live, owner_by_id[idx], OWNER_UNKNOWN).astype(np.int8)
        fog.u_type = np.where(live, type_by_id[idx], -1).astype(np.int8)
        fog.u_hp = np.where(live, hp_by_id[idx], -1).astype(np.int16)
        bnames = sorted(self.tables.buildings)
        item_names = list(self.tables.units) + bnames
        for i in np.flatnonzero(state.occ_city >= 0):
            self._write_city_memory(fog, int(state.occ_city[i]), int(i),
                                    state, item_names, bnames)
        # full view: true ids, matching the base observation handed out here
        return self._memory_payload(fog, map_ids=False)

    def _info_state(self, civ: int) -> dict:
        """The civ's own information bookkeeping (its purchases to date), so
        purchase legality is computable from the observation alone."""
        fog = self._game[civ] if self._game and civ < len(self._game) else None
        return {
            "shared": sorted(fog.shared) if fog else [],
            "oracle": civ in self._oracle,
        }

    # -- purchases -----------------------------------------------------------
    def _rung(self, state: GameState):
        W = int(state.dials["fog"])
        enabled = self.tables.information["rung_purchases"]["value"].get(str(W), [])
        prices = self.tables.information["prices"]["value"]
        return W, set(enabled), prices

    def _price(self, prices, kind: str, W: int):
        p = prices.get(kind, {}).get(str(W))
        return None if p is None else int(p)

    def _known_enemy_cities(self, state: GameState, civ: int) -> list[int]:
        """Enemy cities known from sight or memory, without consulting hidden
        current ownership or existence. A stale target is checked at execution."""
        fog = self._game[civ]
        hm = self._hm(int(state.params["R"]))
        known = {int(fog.c_id[i]) for i in np.flatnonzero(fog.c_hp >= 0)
                 if int(fog.c_owner[i]) >= 0
                 and int(fog.c_owner[i]) != civ
                 and state.civs[int(fog.c_owner[i])].alive}
        for c in state.cities.values():
            if c.owner >= 0 and c.owner != civ and fog.vis[hm.index(c.q, c.r)]:
                known.add(c.id)
        return sorted(known)

    def purchases_legal(self, state: GameState, civ: int) -> list[tuple]:
        if not self._game or not state.civs[civ].alive:
            return []
        self._ensure(state)
        W, enabled, prices = self._rung(state)
        out: list[tuple] = []
        gold = state.civs[civ].gold
        if "spy" in enabled:
            price = self._price(prices, "spy", W)
            tech = self.tables.information["spy_tech"]["value"]
            if price is not None and gold >= price and tech in state.civs[civ].techs:
                # true ids here: present_actions re-keys the WHOLE legal list
                # (purchases included) into handles exactly once
                for cid in self._known_enemy_cities(state, civ):
                    out.append(("SPY", cid))
        if "trade" in enabled:
            price = self._price(prices, "trade", W)
            if price is not None and gold >= price:
                for p in range(len(state.civs)):
                    if p != civ and state.civs[p].alive:
                        out.append(("TRADE", p))
        if "diplo" in enabled:
            price = self._price(prices, "diplo", W)
            if price is not None and gold >= price:
                for p in range(len(state.civs)):
                    if p != civ and state.civs[p].alive \
                            and p not in self._game[civ].shared:
                        out.append(("DIPLO", p))
        if "oracle" in enabled and civ not in self._oracle:
            price = self._price(prices, "oracle", W)
            if price is not None and gold >= price:
                out.append(("ORACLE",))
        return out

    def resolve_purchase(self, state: GameState, report, civ: int, action) -> bool:
        """Resolve one validated purchase: charge gold, apply the reveal,
        write the record entry. Return False for a stale SPY target.
        Runs in the orders_information phase."""
        if not self._game:
            self._ensure(state)
        kind = action[0].lower()
        W, enabled, prices = self._rung(state)
        if kind not in enabled:
            raise InvalidAction(
                f"civ {civ}: purchase {action!r} not enabled at dial {W}")
        price = self._price(prices, kind, W)
        if price is None or state.civs[civ].gold < price:
            raise InvalidAction(f"civ {civ}: purchase {action!r} not affordable")
        fog = self._game[civ]
        ev = {
            "type": "info_purchase", "purchase": kind, "buyer": civ,
            "price": price, "turn": state.turn,
        }
        if kind == "spy":
            tech = self.tables.information["spy_tech"]["value"]
            if tech not in state.civs[civ].techs:
                raise InvalidAction(f"civ {civ}: SPY requires {tech}")
            if len(action) != 2:
                raise InvalidAction(f"civ {civ}: SPY takes a target city id")
            cid = int(action[1])
            if cid not in self._known_enemy_cities(state, civ):
                raise InvalidAction(
                    f"civ {civ}: SPY target city {cid} unknown to the buyer")
            city = state.cities.get(cid)
            if city is None or city.owner < 0 or city.owner == civ:
                return False  # stale remembered target; no purchase is made
            k = int(self.tables.information["spy_duration"]["value"])
            until = state.turn + k
            fog.spies.append((cid, until))
            ev["target_city"] = cid
            ev["target_owner"] = city.owner
            ev["revealed"] = {"city": cid, "hex": [city.q, city.r],
                              "until_turn": until}
        elif kind == "trade":
            partner = int(action[1])
            if partner == civ or not state.civs[partner].alive:
                raise InvalidAction(f"civ {civ}: TRADE partner {partner} invalid")
            hm = self._hm(int(state.params["R"]))
            # Standard worldgen founds capitals in civ-id order. Their city
            # ids and locations persist after capture, when the founder may
            # survive on secondary cities and no longer own a capital.
            mine = state.cities[civ]
            theirs = state.cities[partner]
            line = hex_line(hm, hm.index(mine.q, mine.r),
                            hm.index(theirs.q, theirs.r))
            r = int(self.tables.information["trade_corridor_radius"]["value"])
            corr: set[int] = set(line)
            for i in line:
                corr.update(hm.within(i, r))
            fog.pending_trade.append(np.array(sorted(corr), dtype=np.int64))
            ev["partner"] = partner
            ev["revealed"] = {"corridor_hexes": len(corr),
                              "endpoints": [[mine.q, mine.r], [theirs.q, theirs.r]]}
        elif kind == "diplo":
            partner = int(action[1])
            if partner == civ or not state.civs[partner].alive:
                raise InvalidAction(f"civ {civ}: DIPLO partner {partner} invalid")
            if partner in fog.shared:
                raise InvalidAction(
                    f"civ {civ}: maps already shared with civ {partner}")
            fog.shared.add(partner)
            fog.pending_diplo.append(partner)
            self._game[partner].shared.add(civ)
            self._game[partner].pending_diplo.append(civ)
            ev["partner"] = partner
            ev["revealed"] = {"shared_with": partner}
        elif kind == "oracle":
            self._oracle.add(civ)
            ev["revealed"] = {"full_state": True}
        else:
            raise InvalidAction(f"civ {civ}: unknown purchase {action!r}")

        # charge through the ledger (income_phase writes the `purchase` column)
        state.civs[civ].gold -= price
        report.purchases[civ] = report.purchases.get(civ, 0) + price

        # belief-support size before/after on small maps (design §5)
        if int(state.params["R"]) <= 4:
            ev["support_before"] = self._support_size(state, civ)
            ev["_buyer"] = civ  # "after" filled at the next observation build
            self._pending_support.append(ev)
        report.events.append(ev)
        return True

    def _support_size(self, state: GameState, civ: int):
        """Exact belief-support size for `civ` from its current memory; small maps only, capped.  A diagnostic
        column: it must never break purchase resolution."""
        from .belief import count_support_size

        try:
            return count_support_size(self._belief_view(state, civ), self.tables)
        except Exception:
            return -1

    # -- handle / full-view / diplomacy hooks ---------------------------------
    def _full_view(self, state: GameState, civ: int) -> bool:
        """True when this civ currently receives the shared full-observation
        base (dial-K rung or ORACLE purchase): the oracle view carries true
        ids, so the action mapping must be the identity to match."""
        return (int(state.dials["fog"]) >= FOG_K or civ in self._oracle)

    def owner_known(self, state: GameState, civ: int, idx: int) -> int:
        """The territory owner of `idx` as `civ` knows it: current truth on
        visible hexes, the last-seen owner from memory elsewhere, the
        OWNER_UNKNOWN sentinel for never-seen hexes.  This is what makes
        closed-borders legality computable from the observation alone."""
        if self._full_view(state, civ):
            return int(state.owner[idx])
        fog = self._game[civ] if self._game and civ < len(self._game) else None
        if fog is None:
            return OWNER_UNKNOWN
        if fog.vis[idx]:
            return int(state.owner[idx])
        return int(fog.owner[idx])

    def present_actions(self, state: GameState, civ: int, actions) -> list:
        """Re-key engine true ids into this civ's observer-local handles so
        the agent-facing action space matches the observation (the engine's
        creation counters are never exposed).  Identity at the full-view
        rungs, which hand out the base observation with true ids."""
        if self._full_view(state, civ):
            return list(actions)
        fog = self._game[civ] if self._game and civ < len(self._game) else None
        if fog is None:
            return list(actions)
        out = []
        for a in actions:
            k = a[0]
            if k == "MOVE":
                out.append((k, self._u_handle(fog, a[1]), a[2]))
            elif k == "ATTACK_U":
                out.append((k, self._u_handle(fog, a[1]),
                            self._u_handle(fog, a[2])))
            elif k == "ATTACK_C":
                out.append((k, self._u_handle(fog, a[1]),
                            self._c_handle(fog, a[2])))
            elif k in ("FOUND", "IMPROVE"):
                out.append((k, self._u_handle(fog, a[1])))
            elif k == "PRODUCE":
                out.append((k, self._c_handle(fog, a[1]), a[2]))
            elif k == "SPY":
                out.append((k, self._c_handle(fog, a[1])))
            else:
                out.append(tuple(a))
        return out

    def resolve_action(self, civ: int, action) -> tuple:
        """Map handles back to true engine ids before resolution (the inverse
        of present_actions).  An unresolvable handle raises: legality was
        checked against the presented list, so this would be a broken
        module, not a player error.  Identity at the full-view rungs (the
        module's `_state_ref` is current: validate/`purchases_legal` ran
        `_ensure` for this very turn)."""
        from .engine import InvalidAction

        if not self._game or civ >= len(self._game):
            return tuple(action)
        if self._state_ref is not None and self._full_view(self._state_ref, civ):
            return tuple(action)
        fog = self._game[civ]

        def uh(uid):
            t = fog.u_rev.get(int(uid))
            if t is None:
                raise InvalidAction(f"civ {civ}: unknown unit handle {uid!r}")
            return t

        def ch(cid):
            t = fog.c_rev.get(int(cid))
            if t is None:
                raise InvalidAction(f"civ {civ}: unknown city handle {cid!r}")
            return t

        k = action[0]
        if k == "MOVE":
            return (k, uh(action[1]), action[2])
        if k == "ATTACK_U":
            return (k, uh(action[1]), uh(action[2]))
        if k == "ATTACK_C":
            return (k, uh(action[1]), ch(action[2]))
        if k in ("FOUND", "IMPROVE"):
            return (k, uh(action[1]))
        if k == "PRODUCE":
            return (k, ch(action[1]), action[2])
        if k == "SPY":
            return (k, ch(action[1]))
        return tuple(action)

    def apply_share_maps(self, state: GameState, a: int, b: int) -> None:
        """Activate the same live map exchange the DIPLO purchase uses, on
        behalf of an accepted diplomacy share_maps offer."""
        self._ensure(state)
        if not self._game or a >= len(self._game) or b >= len(self._game):
            return
        fa, fb = self._game[a], self._game[b]
        if b in fa.shared and a in fb.shared:
            return
        fa.shared.add(b)
        fa.pending_diplo.append(b)
        fb.shared.add(a)
        fb.pending_diplo.append(a)

    def _belief_view(self, state: GameState, civ: int) -> dict:
        """Observation-shaped dict for belief construction at resolution
        time: the civ's real observation, built over the true base view."""
        from .observe import build_base_observation

        self._ensure(state)
        return self.build_observation(state, civ, build_base_observation(state))
