"""Scout-heavy: opens with scouts that sweep the map, then expands.

Exploration is agent-side knowledge: the agent marks hexes within the sight
radius of its units as visited (exactly the routine a fog module would feed
it) and marches scouts toward the nearest unvisited hex.  Once the
map is mostly swept it falls back to expander-style development.
"""

from __future__ import annotations

import numpy as np

import numpy as np

from .base import Agent, hex_index
from .expander import ExpanderAgent, _terrain_digest

_PASSABLE_CACHE: dict = {}


class ScoutHeavyAgent(ExpanderAgent):
    name = "scout"

    def __init__(self, civ_id, params=None):
        super().__init__(civ_id, params)
        self.target_cities = int(self.params.get("target_cities", 4))
        self.n_scouts = int(self.params.get("scouts", 2))
        self.visited = None

    def reset(self) -> None:
        self.visited = None

    def _mark_visible(self, obs, u):
        from .base import dist_field

        sight = int(self._tables.units.get(u[2], {}).get("sight", 2))
        self.visited |= dist_field(obs, u[3], u[4]) <= sight

    def plan(self, obs, legal):
        if self.visited is None or len(self.visited) != len(obs["terrain"]):
            self.visited = np.zeros(len(obs["terrain"]), dtype=bool)
        for u in self.own_units(obs):
            self._mark_visible(obs, u)

        actions = []
        units = self.own_units(obs)
        cities = self.own_cities(obs)
        n_scouts = sum(1 for u in units if u[2] == "scout")
        n_settlers = sum(1 for u in units if u[2] == "settler")
        n_workers = sum(1 for u in units if u[2] == "worker")
        explored = float(self.visited.mean())

        for city in cities:
            actions += self._city_action(obs, city, legal, n_settlers, n_workers, n_scouts)

        for u in units:
            uid, _, utype, q, r, hp, mp = u
            if utype == "scout" and explored < 0.85:
                target = self._nearest_unvisited(obs, q, r)
                if target is not None:
                    step = self.step_toward(legal, uid, q, r, target[0], target[1])
                    actions.append(step or ("SKIP",))
                    continue
            if utype == "worker" and ("IMPROVE", uid) in legal:
                actions.append(("IMPROVE", uid))
                continue
            # otherwise: expander behaviour (settling, escort, city development)
            actions += super()._unit_plan(obs, legal, u)
        return actions

    def _city_action(self, obs, city, legal, n_settlers, n_workers, n_scouts):
        if n_scouts < self.n_scouts and ("PRODUCE", city[0], "scout") in legal:
            return [("PRODUCE", city[0], "scout")]
        return super()._city_action(obs, city, legal, n_settlers, n_workers, n_scouts)

    def _nearest_unvisited(self, obs, q, r):
        import numpy as np

        from .base import dist_field, geo_for

        geo = geo_for(obs)
        # terrain CONTENT key: passability is terrain-dependent, and two
        # same-shape worlds in one process must not share the field
        key = (obs["meta"]["radius"], len(obs["terrain"]),
               _terrain_digest(obs))
        passable = _PASSABLE_CACHE.get(key)
        if passable is None:
            tables = self._tables
            passable = np.array([
                bool(tables.terrain[obs["terrain_names"][int(t)]]["passable"])
                for t in obs["terrain"]
            ], dtype=bool)
            _PASSABLE_CACHE[key] = passable
        mask = ~self.visited & passable
        if not mask.any():
            return None
        d = np.where(mask, dist_field(obs, q, r), 1 << 30)
        i = int(np.argmin(d))
        if d[i] >= (1 << 30):
            return None
        return geo["coords"][i]
