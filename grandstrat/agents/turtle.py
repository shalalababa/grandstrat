"""Turtle: fortify, build infrastructure, expand slowly, fight only at home.

Priorities
  cities   : walls first, then granary/library/market, one worker per city,
             minimal garrison, rare expansion (`target_cities`, default 2).
  military : hold at the city (no action) unless an enemy is in attack range.
  workers  : improve; else idle near home.
"""

from __future__ import annotations

from .expander import ExpanderAgent


class TurtleAgent(ExpanderAgent):
    name = "turtle"

    def __init__(self, civ_id, params=None):
        super().__init__(civ_id, params)
        self.target_cities = int(self.params.get("target_cities", 2))

    def _city_action(self, obs, city, legal):
        civ_techs = {
            t for i, t in enumerate(obs["tech_names"]) if obs["techs"][self.civ_id][i]
        }
        buildings = city[7]
        for name in ("walls", "granary", "library", "market", "barracks"):
            if name in buildings:
                continue
            spec = self._tables.buildings.get(name)
            if spec is None:
                continue
            if spec.get("tech") is not None and spec["tech"] not in civ_techs:
                continue
            if ("PRODUCE", city[0], name) in legal:
                return [("PRODUCE", city[0], name)]
        n_cities = len(self.own_cities(obs))
        n_workers = sum(1 for u in self.own_units(obs) if u[2] == "worker")
        n_mil = sum(
            1 for u in self.own_units(obs)
            if not self._tables.units.get(u[2], {}).get("civilian", False)
        )
        want = None
        if n_cities < self.target_cities and sum(
            1 for u in self.own_units(obs) if u[2] == "settler"
        ) == 0:
            want = "settler"
        elif n_workers < n_cities:
            want = "worker"
        elif n_mil < n_cities:
            want = "archer"
        if want and ("PRODUCE", city[0], want) in legal:
            return [("PRODUCE", city[0], want)]
        for a in sorted(legal):
            if a[0] == "PRODUCE" and a[1] == city[0]:
                return [a]
        return []

    def plan(self, obs, legal):
        actions = []
        cities = self.own_cities(obs)
        for city in cities:
            actions += self._city_action(obs, city, legal)
        for u in self.own_units(obs):
            uid, _, utype, q, r, hp, mp = u
            if utype == "worker":
                if ("IMPROVE", uid) in legal:
                    actions.append(("IMPROVE", uid))
                    continue
                actions.append(("SKIP",))
                continue
            if utype == "settler":
                actions += super()._unit_plan(obs, legal, u)
                continue
            # military: attack only what is already in range (defence)
            atk = None
            for a in sorted(legal):
                if a[0] in ("ATTACK_U", "ATTACK_C") and a[1] == uid:
                    atk = a
                    break
            actions.append(atk or ("SKIP",))
        return actions
