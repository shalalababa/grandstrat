"""Expander: founds cities early, develops them, keeps a modest escort.

Priorities
  settlers : found a city the moment the hex is legal; otherwise walk to the
             best-scoring valid foundation spot within a bounded radius.
  cities   : settlers while below `target_cities` (and settlers on the map <
             2), then one worker per two cities, then the strongest military
             the techs allow, then buildings (cheap-first), else any unit.
  military : attack adjacent/in-range enemies near own cities; otherwise stay
             within `leash` hexes of the nearest own city.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import Agent, hex_index

_IMPROVABLE_CACHE: dict = {}
_SPOT_CACHE: dict = {}
_CACHE_CAP = 256  # bounded: content digests vary with fog visibility


def _terrain_digest(obs) -> str:
    """Stable digest of the world's PUBLIC terrain layer (terrain never
    changes and is public under every information module, so this identifies
    the world's static geography within one process).  Caches keyed only by
    map SHAPE used to leak one world's content-derived fields into a later
    different-seeded world of the same shape (found by the test suite)."""
    return hashlib.sha1(np.asarray(obs["terrain"]).tobytes()).hexdigest()


def _content_digest(obs) -> str:
    """Digest of terrain + resource layers.  The resource layer is
    visibility-dependent under fog, so entries are bounded by _CACHE_CAP."""
    h = hashlib.sha1()
    h.update(np.asarray(obs["terrain"]).tobytes())
    h.update(np.asarray(obs["resource"]).tobytes())
    return h.hexdigest()


class ExpanderAgent(Agent):
    name = "expander"

    def __init__(self, civ_id, params=None):
        super().__init__(civ_id, params)
        self.target_cities = int(self.params.get("target_cities", 6))
        self.leash = int(self.params.get("leash", 3))
        self._dmin_key = None
        self._dmin_cache = None
        self._atk_key = None
        self._atk_map = {}

    def reset(self) -> None:
        # per-instance memos must not leak across games when an agent
        # instance is reused for several games (record.play_batch pattern)
        self._dmin_key = None
        self._dmin_cache = None
        self._atk_key = None
        self._atk_map = {}

    # -- foundation-spot scoring ------------------------------------------
    def best_spot(self, obs, u, radius=4):
        """Best foundation spot near the settler (vectorized; cached static
        neighbourhood-score field per map shape)."""
        import numpy as np

        from .base import dist_field, geo_for

        key = (obs["meta"]["radius"], len(obs["terrain"]), _content_digest(obs))
        cached = _SPOT_CACHE.get(key)
        if cached is None:
            if len(_SPOT_CACHE) >= _CACHE_CAP:
                _SPOT_CACHE.clear()
            geo = geo_for(obs)
            tables = self._tables
            L = len(obs["terrain"])
            nb_score = np.zeros(L, dtype=np.float64)
            for i in range(L):
                qi, ri = geo["coords"][i]
                s = 0.0
                for dq, dr in ((0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)):
                    j = geo["idx"].get((qi + dq, ri + dr))
                    if j is None:
                        continue
                    tname = obs["terrain_names"][int(obs["terrain"][j])]
                    t = tables.terrain[tname]
                    s += 2.0 * t["food"] + 1.5 * t["prod"] + 1.0 * t["gold"]
                    if int(obs["resource"][j]) > 0:  # -1 = unseen (fog)
                        s += 1.5
                nb_score[i] = s
            passable = np.zeros(L, dtype=bool)
            for i in range(L):
                tname = obs["terrain_names"][int(obs["terrain"][i])]
                passable[i] = bool(tables.terrain[tname]["passable"])
            cached = (geo, nb_score, passable)
            _SPOT_CACHE[key] = cached
        geo, nb_score, passable = cached
        tables = self._tables
        uq, ur = u[3], u[4]
        settler_d = dist_field(obs, uq, ur)
        # distance to the nearest city: shared across all settlers of this
        # civ within one turn (the map and cities do not change mid-plan)
        L = len(obs["terrain"])
        ck = (obs["meta"]["turn"], self.civ_id, L)
        if self._dmin_key == ck:
            dmin = self._dmin_cache
        else:
            cities = obs["city_list"]
            if cities:
                dmin = np.full(L, 1 << 30, dtype=np.int64)
                for c in cities:
                    np.minimum(dmin, dist_field(obs, c[2], c[3]), out=dmin)
            else:
                dmin = np.full(L, 99, dtype=np.int64)
            self._dmin_key, self._dmin_cache = ck, dmin
        occupied = np.zeros(len(obs["terrain"]), dtype=bool)
        for uu in obs["unit_list"]:
            if uu[0] != u[0]:
                j = geo["idx"].get((uu[3], uu[4]))
                if j is not None:
                    occupied[j] = True
        owner_ok = (np.asarray(obs["owner"]) == -1) | (np.asarray(obs["owner"]) == self.civ_id)
        mask = (passable & owner_ok & ~occupied
                & (dmin >= int(tables.constants["city_min_dist"]["value"]))
                & (settler_d <= radius))
        if not mask.any():
            return None, None
        score = nb_score - 0.3 * settler_d
        cur = geo["idx"].get((uq, ur))
        if cur is not None and mask[cur]:
            score[cur] += 2.0  # stickiness: settling in place beats wandering
        score = np.where(mask, score, -1e18)
        best = int(np.argmax(score))
        if score[best] <= -1e17:
            return None, None
        return geo["coords"][best]

    def _idx(self, obs):
        from .base import geo_for

        return geo_for(obs)["idx"]

    # -- city production ----------------------------------------------------
    def _city_action(self, obs, city, legal, n_settlers, n_workers, n_mil):
        tables = self._tables
        civ_techs = {
            t for i, t in enumerate(obs["tech_names"]) if obs["techs"][self.civ_id][i]
        }
        n_cities = len(self.own_cities(obs))
        want = None
        if n_cities < self.target_cities and n_settlers < 2:
            want = "settler"
        if want is None and n_workers < max(1, n_cities // 2):
            want = "worker"
        if want is None:
            mil = sorted(
                (
                    (int(s["strength"]), -int(s["cost"]), name)
                    for name, s in tables.units.items()
                    if not s.get("civilian")
                    and (s.get("tech") is None or s["tech"] in civ_techs)
                ),
                reverse=True,
            )
            want = mil[0][2] if mil else "warrior"
        a = ("PRODUCE", city[0], want)
        if a in legal:
            return [a]
        # fallback: any producible building, else any legal produce
        for name in sorted(tables.buildings):
            spec = tables.buildings[name]
            if (spec.get("tech") is None or spec["tech"] in civ_techs) \
                    and ("PRODUCE", city[0], name) in legal:
                return [("PRODUCE", city[0], name)]
        for a in sorted(legal):
            if a[0] == "PRODUCE" and a[1] == city[0]:
                return [a]
        return []

    # -- plan ----------------------------------------------------------------
    def plan(self, obs, legal):
        actions = []
        units = self.own_units(obs)
        cities = self.own_cities(obs)
        n_settlers = sum(1 for u in units if u[2] == "settler")
        n_workers = sum(1 for u in units if u[2] == "worker")
        n_mil = sum(
            1 for u in units
            if not self._tables.units.get(u[2], {}).get("civilian", False)
        )
        for city in cities:
            actions += self._city_action(obs, city, legal, n_settlers, n_workers, n_mil)
        for u in units:
            actions += self._unit_plan(obs, legal, u)
        return actions

    def _unit_plan(self, obs, legal, u) -> list[tuple]:
        actions = []
        uid, _, utype, q, r, hp, mp = u
        if utype == "settler":
            spot = self.best_spot(obs, u)
            if spot[0] is None or (spot[0], spot[1]) == (q, r):
                if ("FOUND", uid) in legal:
                    actions.append(("FOUND", uid))
                    return actions
            else:
                step = self.step_toward(legal, uid, q, r, spot[0], spot[1])
                if step:
                    actions.append(step)
                    return actions
        elif utype == "worker":
            if ("IMPROVE", uid) in legal:
                actions.append(("IMPROVE", uid))
                return actions
            target = self._nearest_improvable(obs, q, r)
            if target is not None:
                step = self.step_toward(legal, uid, q, r, target[0], target[1])
                if step:
                    actions.append(step)
                    return actions
        else:
            atk = self._best_attack(legal, uid, obs["meta"]["turn"])
            if atk:
                actions.append(atk)
                return actions
            home = self._nearest_own_city(obs, q, r)
            if home is not None:
                d = self.hex_dist(q, r, home[2], home[3])
                if d > self.leash:
                    step = self.step_toward(legal, uid, q, r, home[2], home[3])
                    if step:
                        actions.append(step)
                        return actions
        actions.append(("SKIP",))
        return actions

    # -- helpers ------------------------------------------------------------
    def _best_attack(self, legal, uid, turn=None):
        """Memoized per (turn, legal-set): one scan builds [uid -> best
        attack].  Ties break by smallest target id: the legal container is a
        SET, so first-seen order is PYTHONHASHSEED-dependent."""
        key = (turn, id(legal))
        if self._atk_key != key:
            m = {}
            for a in legal:
                if len(a) < 3:
                    continue
                if a[0] == "ATTACK_C":
                    e = m.setdefault(a[1], [None, None, None])
                    if e[0] is None or a[2] < e[0][2]:
                        e[0] = a
                elif a[0] == "ATTACK_U":
                    e = m.setdefault(a[1], [None, None, None])
                    if e[1] is None or a[2] < e[2]:
                        e[1], e[2] = a, a[2]
            self._atk_key, self._atk_map = key, m
        e = self._atk_map.get(uid)
        if not e:
            return None
        return e[0] or e[1]

    def _nearest_own_city(self, obs, q, r):
        cities = self.own_cities(obs)
        if not cities:
            return None
        return min(cities, key=lambda c: (self.hex_dist(q, r, c[2], c[3]), c[0]))

    def _nearest_improvable(self, obs, q, r):
        """Nearest own-territory hex that still lacks its improvement
        (vectorized over the map; terrain is static so the buildable mask is
        cached per map shape)."""
        import numpy as np

        from .base import dist_field, geo_for

        key = (obs["meta"]["radius"], len(obs["terrain"]),
               _terrain_digest(obs))
        cached = _IMPROVABLE_CACHE.get(key)
        if cached is None:
            if len(_IMPROVABLE_CACHE) >= _CACHE_CAP:
                _IMPROVABLE_CACHE.clear()
            geo = geo_for(obs)
            tables = self._tables
            buildable = np.zeros(len(obs["terrain"]), dtype=bool)
            passable = np.zeros(len(obs["terrain"]), dtype=bool)
            for i, (qq, rr) in enumerate(geo["coords"]):
                tname = obs["terrain_names"][int(obs["terrain"][i])]
                passable[i] = bool(tables.terrain[tname]["passable"])
                buildable[i] = any(
                    tname in spec["terrain"] for spec in tables.improvements.values()
                )
            cached = (buildable, passable, geo)
            _IMPROVABLE_CACHE[key] = cached
        buildable, passable, geo = cached
        terr = np.asarray(obs["terrain"], dtype=np.int64)
        own = np.asarray(obs["owner"]) == self.civ_id
        no_impr = np.asarray(obs["improve"]) == 0
        city_hexes = np.zeros(len(terr), dtype=bool)
        for c in obs["city_list"]:
            i = geo["idx"].get((c[2], c[3]))
            if i is not None:
                city_hexes[i] = True
        mask = own & no_impr & buildable & passable & ~city_hexes
        if not mask.any():
            return None
        d = dist_field(obs, q, r)
        d = np.where(mask, d, 1 << 30)
        i = int(np.argmin(d))
        if d[i] >= (1 << 30):
            return None
        return geo["coords"][i]
