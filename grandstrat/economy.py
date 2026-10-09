"""Economy subsystem (design §1): city yields, food growth, production,
research, gold income/upkeep, bankruptcy, scoring.

Conservation is exact: every mutation writes explicit arithmetic into the
TurnReport ledger -- {"before", +income, -spent, =after} rows for gold,
research, and every city's food/production buckets -- and the tests re-derive
each row's balance from the ledger alone (test_conservation.py).

Turn-income ordering (see docs/DESIGN_NOTES.md): culture/borders -> per-city
food/production -> gold income + science -> research -> upkeep -> bankruptcy
disbands -> healing -> scores.
"""

from __future__ import annotations

from .engine import (TurnReport, city_wid, remove_unit, spawn_unit,
                     terrain_passable)
from .state import GameState
from .worldgen import recompute_territory

# Improvement id -> name (observation encoding; 0 = none).  Pinned order.
IMPROVE_NAMES = ("", "farm", "mine", "lumberyard", "trading_post")


def hex_yield(state: GameState, idx: int) -> tuple:
    """(food, prod, gold) yielded by working hex idx (terrain + resource +
    improvement), before buildings.  Kept current by worldgen fill and
    engine._improve; O(1) array reads."""
    return int(state.yf[idx]), int(state.yp[idx]), int(state.yg[idx])


def work_score(state: GameState, idx: int) -> float:
    w = state.tables.constants["work_score_weights"]["value"]
    return (w["food"] * state.yf[idx] + w["prod"] * state.yp[idx]
            + w["gold"] * state.yg[idx])


def _rework_hexes(state: GameState, city) -> None:
    """Choose worked ring hexes: best `pop` hexes by work_score within the
    border (center always worked for free).  Deterministic: (-score, index)."""
    import numpy as np

    hm = state.hexmap
    cands = []
    for rad in range(1, city.border + 1):
        for dq, dr in hm.ring_offsets(rad):
            j = hm.idx.get((city.q + dq, city.r + dr))
            if j is None or int(state.occ_city[j]) >= 0:
                continue
            if int(state.owner[j]) != city.owner:
                continue
            cands.append(j)
    if not cands:
        city.worked = ()
        city.worked_stale = False
        return
    idx = np.array(cands, dtype=np.int64)
    w = state.tables.constants["work_score_weights"]["value"]
    scores = (w["food"] * state.yf[idx] + w["prod"] * state.yp[idx]
              + w["gold"] * state.yg[idx])
    # deterministic: by (-score, hex index)
    order = sorted(range(len(cands)), key=lambda k: (-scores[k], cands[k]))
    city.worked = tuple(cands[k] for k in order[: city.pop])
    city.worked_stale = False


def city_yields(state: GameState, city) -> dict:
    if city.worked_stale:
        _rework_hexes(state, city)
    y = dict(state.tables.constants["city_center_yield"]["value"])
    for j in city.worked:
        y["food"] += int(state.yf[j])
        y["prod"] += int(state.yp[j])
        y["gold"] += int(state.yg[j])
    for b in city.buildings:
        for k, v in state.tables.buildings[b]["yield"].items():
            y[k] = y.get(k, 0) + int(v)
    # population brings commerce (keeps the economy near balance with upkeep)
    y["gold"] = y.get("gold", 0) + city.pop // 2
    return y


def city_max_hp(state: GameState, city) -> int:
    hp = int(state.tables.constants["city_hp_max"]["value"])
    for b in city.buildings:
        hp += int(state.tables.buildings[b]["hp"])
    return hp


def city_defense(state: GameState, city) -> float:
    d = float(state.tables.constants["city_base_def"]["value"])
    for b in city.buildings:
        d += float(state.tables.buildings[b]["def"])
    return d


def item_cost(state: GameState, item: str) -> int:
    if item in state.tables.units:
        return int(state.tables.units[item]["cost"])
    return int(state.tables.buildings[item]["cost"])


def ability_sum(state: GameState, techs, key: str) -> int:
    """Sum of one data-driven tech ability over a civ's researched techs
    (techs.json entries may carry an `abilities` object; currently
    `gold_per_turn` and `city_sight_bonus` exist).  Every tech unlocks
    something -- units, buildings, improvements, other techs, or these."""
    total = 0
    for t in techs:
        ab = state.tables.techs[t].get("abilities")
        if ab:
            total += int(ab.get(key, 0))
    return total


def _culture_and_borders(state: GameState, report: TurnReport) -> bool:
    K = state.tables.constants
    thresholds = K["border_culture"]["value"]
    changed = False
    for city in state.cities.values():
        city.culture += int(K["base_culture_per_turn"]["value"])
        new_border = city.border
        for rad in range(1, len(thresholds) + 1):
            if city.culture >= thresholds[rad - 1]:
                new_border = rad
        if new_border > city.border:
            city.border = new_border
            city.worked_stale = True
            changed = True
            report.events.append({
                "type": "border_grown", "city": city.id, "owner": city.owner,
                "border": new_border,
            })
    return changed


def _food_and_production(state: GameState, report: TurnReport) -> None:
    K = state.tables.constants
    cap = int(K["unit_cap"]["value"])
    for city in state.cities.values():
        y = city_yields(state, city)
        # -- food ----------------------------------------------------------
        row = {
            "city": city.id, "owner": city.owner,
            "pop_before": city.pop, "food_before": city.food_bucket,
            "food_income": y["food"], "food_eaten": K["eat_per_pop"]["value"] * city.pop,
            "food_spent": 0, "food_reset": 0,
        }
        city.food_bucket += row["food_income"] - row["food_eaten"]
        need = K["growth_base"]["value"] + K["growth_per_pop"]["value"] * city.pop
        if city.food_bucket >= need:
            city.pop += 1
            city.worked_stale = True
            row["food_spent"] = need
            city.food_bucket -= need
            report.events.append({
                "type": "pop_growth", "city": city.id, "owner": city.owner,
                "pop": city.pop,
            })
        elif city.food_bucket < 0:
            if city.pop > 1:
                city.pop -= 1
                city.worked_stale = True
                report.events.append({
                    "type": "starvation", "city": city.id, "owner": city.owner,
                    "pop": city.pop,
                })
            row["food_reset"] = city.food_bucket  # negative: clamped to 0
            city.food_bucket = 0
        row["pop_after"] = city.pop
        row["food_after"] = city.food_bucket
        report.ledger.setdefault("city_food", []).append(row)
        # -- production ------------------------------------------------------
        prow = {
            "city": city.id, "owner": city.owner, "queue": city.queue,
            "prod_before": city.prod_bucket, "prod_income": y["prod"],
            "prod_spent": 0, "prod_after": None, "completed": None,
        }
        city.prod_bucket += y["prod"]
        if city.queue is not None:
            cost = item_cost(state, city.queue)
            if city.prod_bucket >= cost and _complete_item(state, city, report, cap):
                prow["prod_spent"] = cost
                prow["completed"] = city.queue
                city.prod_bucket -= cost
                city.queue = None
        prow["prod_after"] = city.prod_bucket
        report.ledger.setdefault("city_prod", []).append(prow)


def _complete_item(state: GameState, city, report: TurnReport, cap: int) -> bool:
    """Complete the queued item.  Return False (staying queued, bucket held at
    or above cost to retry next turn) when the spawn is blocked."""
    item = city.queue
    if item in state.tables.units:
        if len(state.units) >= cap:
            return False
        # spawn on the city hex if free, else the lowest-index adjacent free
        # passable, cityless hex (a garrison must not block production forever)
        hm = state.hexmap
        i0 = hm.index(city.q, city.r)
        spot = i0 if int(state.occ_unit[i0]) < 0 else -1
        if spot < 0:
            for d in range(6):
                j = int(hm.neighbors[i0, d])
                if (j >= 0 and int(state.occ_unit[j]) < 0
                        and int(state.occ_city[j]) < 0
                        and terrain_passable(state, j)):
                    spot = j
                    break
            if spot < 0:
                return False  # fully surrounded this turn; retry next turn
        if item == "settler":
            if city.pop < int(state.tables.constants["settler_min_pop"]["value"]):
                return False  # starved below the founding threshold
            city.pop -= 1
            city.worked_stale = True
        u = spawn_unit(state, city.owner, item, int(hm.qs[spot]), int(hm.rs[spot]),
                       birth_key=city_wid(state, city))
        report.events.append({
            "type": "unit_built", "unit": u.id, "kind": item,
            "owner": city.owner, "city": city.id, "wid": list(u.wid),
        })
        return True
    city.buildings = tuple(sorted(city.buildings + (item,)))
    report.events.append({
        "type": "building_built", "building": item, "owner": city.owner,
        "city": city.id,
    })
    return True


def _research(state: GameState, report: TurnReport, science: list[int]) -> None:
    for civ_id, civ in enumerate(state.civs):
        if not civ.alive:
            continue
        row = {
            "owner": civ_id, "before": civ.research_bucket,
            "income": science[civ_id], "spent": 0, "after": None,
            "completed": [],
        }
        civ.research_bucket += science[civ_id]
        while True:
            if civ.researching is None or civ.researching in civ.techs:
                avail = sorted(
                    (state.tables.techs[t]["cost"], t)
                    for t in state.tables.techs_available(civ.techs)
                )
                civ.researching = avail[0][1] if avail else None
            if civ.researching is None:
                civ.research_bucket = 0
                break
            cost = int(state.tables.techs[civ.researching]["cost"])
            if civ.research_bucket < cost:
                break
            civ.research_bucket -= cost
            row["spent"] += cost
            civ.techs = tuple(sorted(civ.techs + (civ.researching,)))
            row["completed"].append(civ.researching)
            report.events.append({
                "type": "tech_researched", "owner": civ_id, "tech": civ.researching,
            })
        row["after"] = civ.research_bucket
        if row["spent"] or row["income"]:
            report.ledger.setdefault("research", []).append(row)


def income_phase(state: GameState, report: TurnReport, diplomacy=None) -> None:
    K = state.tables.constants
    borders_changed = _culture_and_borders(state, report)
    if borders_changed or state.territory_stale:
        recompute_territory(state)
        for city in state.cities.values():
            city.worked_stale = True

    _food_and_production(state, report)

    # gold + science per civ (city iteration is city-id order)
    income = [0] * len(state.civs)
    science = [0] * len(state.civs)
    for city in state.cities.values():
        y = city_yields(state, city)
        income[city.owner] += y["gold"]
        science[city.owner] += y["science"]
    # tech abilities: e.g. guilds' gold_per_turn
    for civ_id, civ in enumerate(state.civs):
        if civ.alive:
            income[civ_id] += ability_sum(state, civ.techs, "gold_per_turn")

    gold_before = [c.gold for c in state.civs]
    _research(state, report, science)
    for civ, inc in zip(state.civs, income):
        civ.gold += inc

    upkeep = [0] * len(state.civs)
    for u in state.units.values():
        if u.owner >= 0:
            upkeep[u.owner] += int(state.tables.units[u.type]["upkeep"])
    for civ, up in zip(state.civs, upkeep):
        civ.gold -= up

    # bankruptcy: disband highest-upkeep first (tie: oldest id) until >= 0.
    #
    # Bankruptcy fix: disbanding used to leave `civ.gold`
    # untouched, so the loop could only ever terminate by exhausting the civ's
    # ENTIRE army and clamping the treasury at 0.  Measured on the default
    # scenario that fired 37 times in 6 games, annihilating 320 units (up to 31
    # at once) -- a single deficit turn wiped a civ's whole military.  A
    # disbanded unit is disbanded INSTEAD OF being paid for, so its upkeep is
    # returned to the treasury and removed from this turn's upkeep charge; the
    # loop now stops at the first solvent moment and the ledger still balances
    # (the row's `upkeep` column reports what was actually paid).
    for civ_id, civ in enumerate(state.civs):
        if not civ.alive:
            continue
        disbanded = []
        clamped = False
        while civ.gold < 0:
            own = [u for u in state.units.values() if u.owner == civ_id]
            if not own:
                clamped = True
                civ.gold = 0  # nothing left to disband; treasury clamps
                break
            victim = sorted(
                own, key=lambda u: (-int(state.tables.units[u.type]["upkeep"]), u.id)
            )[0]
            refund = int(state.tables.units[victim.type]["upkeep"])
            if refund <= 0:
                # every remaining unit is free to maintain: disbanding cannot
                # restore solvency, so clamp instead of looping forever
                clamped = True
                civ.gold = 0
                break
            civ.gold += refund
            upkeep[civ_id] -= refund
            remove_unit(state, victim)
            disbanded.append(victim.id)
        if disbanded or clamped:
            report.events.append({
                "type": "bankruptcy_disband", "owner": civ_id,
                "units": disbanded, "clamped": clamped,
            })

    # healing: units that did not act and stand in own territory (or, with an
    # open-borders agreement, in the partner's -- diplomacy module);
    # unattacked cities heal
    max_hp = int(K["unit_hp_max"]["value"])
    for u in state.units.values():
        if not u.acted and u.owner >= 0 and u.hp < max_hp:
            i = state.hexmap.index(u.q, u.r)
            owner_i = int(state.owner[i])
            if owner_i == u.owner or (diplomacy is not None
                                      and owner_i >= 0
                                      and diplomacy.borders_open(u.owner, owner_i)):
                u.hp = min(max_hp, u.hp + int(K["heal_per_turn"]["value"]))
    for city in state.cities.values():
        if not city.attacked:
            mhp = city_max_hp(state, city)
            if city.hp < mhp:
                city.hp = min(mhp, city.hp + int(K["heal_per_turn"]["value"]))
        city.attacked = False

    report.ledger["gold"] = []
    for i in range(len(state.civs)):
        # information purchases and diplomacy transfers were settled BEFORE
        # this phase, so gold_before is already net of them; both are shown
        # explicitly (purchase / transfer columns) so the ledger alone
        # re-derives every balance:
        #   before - purchase + income - upkeep + bounty + transfer == after
        spent = int(report.purchases.get(i, 0))
        transfer = int(report.transfers.get(i, 0))
        # `before` is the true pre-turn gold (purchases and transfers were
        # settled earlier in the turn, so gold_before + spent - transfer
        # reconstructs it); raw is the pre-clamp expectation of `after`.
        raw = gold_before[i] + income[i] - upkeep[i]
        clamped = state.civs[i].gold == 0 and raw < 0
        report.ledger["gold"].append({
            "owner": i, "before": gold_before[i] + spent - transfer,
            "income": income[i], "upkeep": upkeep[i], "bounty": 0,
            "purchase": spent, "transfer": transfer,
            "clamped": clamped,  # treasury floored at 0 (nothing left to disband)
            "after": state.civs[i].gold,
        })


def recompute_scores(state: GameState) -> None:
    w = state.tables.constants["score_weights"]["value"]
    for civ_id, civ in enumerate(state.civs):
        if not civ.alive:
            civ.score = 0
            continue
        n_cities = sum(1 for c in state.cities.values() if c.owner == civ_id)
        civ.score = (
            int(w["city"]) * n_cities
            + int(w["tech"]) * len(civ.techs)
            + int(w["territory"]) * civ.territory_count
            + civ.gold // int(w["gold_per"])
        )
