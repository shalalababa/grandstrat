"""Warmonger: mass military, march on the nearest enemy city, siege it.

Priorities
  cities   : one settler if below `min_cities`, then the strongest available
             military, forever.
  military : the first `len(cities)` units (closest to a city) defend; the
             rest march on the nearest enemy city and attack anything in
             range, cities first.
  civilians: fall back to the nearest own city.
"""

from __future__ import annotations

from .expander import ExpanderAgent


class WarmongerAgent(ExpanderAgent):
    name = "warmonger"

    def __init__(self, civ_id, params=None):
        super().__init__(civ_id, params)
        self.min_cities = int(self.params.get("min_cities", 2))

    def _city_action(self, obs, city, legal):
        civ_techs = {
            t for i, t in enumerate(obs["tech_names"]) if obs["techs"][self.civ_id][i]
        }
        n_cities = len(self.own_cities(obs))
        n_settlers = sum(1 for u in self.own_units(obs) if u[2] == "settler")
        want = None
        if n_cities < self.min_cities and n_settlers == 0:
            want = "settler"
        else:
            mil = sorted(
                (
                    (-int(s["strength"]), int(s["cost"]), name)
                    for name, s in self._tables.units.items()
                    if not s.get("civilian")
                    and (s.get("tech") is None or s["tech"] in civ_techs)
                ),
            )
            want = mil[0][2] if mil else "warrior"
        if ("PRODUCE", city[0], want) in legal:
            return [("PRODUCE", city[0], want)]
        for a in sorted(legal):
            if a[0] == "PRODUCE" and a[1] == city[0]:
                return [a]
        return []

    def plan(self, obs, legal):
        actions = []
        units = self.own_units(obs)
        cities = self.own_cities(obs)
        for city in cities:
            actions += self._city_action(obs, city, legal)

        # defenders: one military unit per city, the closest one
        defenders = set()
        for city in cities:
            cand = [
                u for u in units
                if u[0] not in defenders
                and not self._tables.units.get(u[2], {}).get("civilian", False)
            ]
            if cand:
                defenders.add(min(cand, key=lambda u: (self.hex_dist(u[3], u[4], city[2], city[3]), u[0]))[0])

        targets = self.enemy_cities(obs)
        for u in units:
            uid, _, utype, q, r, hp, mp = u
            spec = self._tables.units.get(utype, {})
            civilian = spec.get("civilian", False)
            if not civilian and uid not in defenders:
                atk = self._attack_in_range(legal, uid, obs)
                if atk:
                    actions.append(atk)
                    continue
                if targets:
                    tq, tr = min(
                        ((c[2], c[3]) for c in targets),
                        key=lambda p: (self.hex_dist(q, r, p[0], p[1]), p[0], p[1]),
                    )
                    step = self.step_toward(legal, uid, q, r, tq, tr)
                    if step:
                        actions.append(step)
                        continue
            elif civilian:
                actions += super()._unit_plan(obs, legal, u)
                continue
            actions.append(("SKIP",))
        return actions

    def _attack_in_range(self, legal, uid, obs):
        """City attacks first (siege), then the weakest adjacent unit."""
        city_atk = None
        unit_atk = None
        best = None
        for a in legal:
            if len(a) < 2 or a[1] != uid:
                continue
            if a[0] == "ATTACK_C":
                # smallest target id: set iteration order is hash-dependent
                if city_atk is None or a[2] < city_atk[2]:
                    city_atk = a
            elif a[0] == "ATTACK_U":
                hp = self._unit_hp(obs, a[2])
                if best is None or (hp, a[2]) < best:
                    unit_atk, best = a, (hp, a[2])
        return city_atk or unit_atk

    def _unit_hp(self, obs, uid):
        for u in obs["unit_list"]:
            if u[0] == uid:
                return u[5]
        return 1 << 30

    def _nearest_own_city(self, obs, q, r):
        cities = self.own_cities(obs)
        if not cities:
            return None
        return min(cities, key=lambda c: (self.hex_dist(q, r, c[2], c[3]), c[0]))
