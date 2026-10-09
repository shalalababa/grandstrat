"""Combat subsystem: strength contest with terrain/flanking modifiers and a
damage formula, resolved with COMMON RANDOM NUMBERS keyed on WORLD identity.

The world-key rule (design §1): every roll key is
    (purpose, turn, attacker_id, defender_id, roll_index)
drawn from the `combat` stream.  No path counters -- not a fight index, not an
action ordinal, not a call counter.  Consequences (tested in
test_crn.py):

* Two different fights in the same turn draw different values.
* The same (attacker, defender) pair fighting at the same turn from two
  different game lineages (different histories, same world position) draws
  byte-identical values.
* Two arms that differ only in the information dial share rolls for identical
  fights.

This module is a swappable kernel module (design §0b): the env holds a
CombatModule instance selected by config; only the "crn" implementation
exists.
"""

from __future__ import annotations

from .economy import city_defense, city_max_hp
from .engine import move_unit, remove_unit, terrain_def_mod
from .hexmap import NUM_DIRS
from .state import GameState


class CombatModule:
    """Interface (design §0b): combat is behind a module boundary."""

    name = "combat"

    def resolve_attack_unit(self, state: GameState, report: TurnReport,
                            attacker: "Unit-like", defender: "Unit-like") -> None:
        raise NotImplementedError

    def resolve_attack_city(self, state: GameState, report: TurnReport,
                            attacker, city) -> None:
        raise NotImplementedError


def _hp_factor(state: GameState, hp: int) -> float:
    K = state.tables.constants
    frac = hp / float(K["unit_hp_max"]["value"])
    floor = float(K["hp_strength_floor"]["value"])
    return floor + (1.0 - floor) * frac


def _flankers(state: GameState, defender) -> int:
    """Adjacent military units friendly to the defender."""
    hm = state.hexmap
    i0 = defender.i
    n = 0
    for d in range(NUM_DIRS):
        j = int(hm.neighbors[i0, d])
        if j < 0:
            continue
        uid = int(state.occ_unit[j])
        if uid >= 0:
            u = state.units[uid]
            if u.owner == defender.owner and not state.tables.units[u.type].get("civilian", False):
                n += 1
    return n


def _roll_value(state: GameState, report: TurnReport, purpose: str,
                atk_wid, def_wid, roll_index: int) -> float:
    """One CRN draw keyed on WORLD identity (docs/DESIGN_NOTES.md §13):
    (purpose, turn, attacker_wid, defender_wid, roll_index) where unit wids
    are (owner, birth_turn, birth_key, ordinal) birth facts and a city's wid
    is ("city", founding_hex_index).  The creation-counter integer ids are
    path-counted -- arms that diverge in production order would stop pairing
    if the key used them -- so the key carries wids, never `unit.id`."""
    turn = state.turn
    key = (purpose, turn, atk_wid, def_wid, roll_index)
    u = state.rng.uniform("combat", *key)
    report.rolls.append({"key": list(key), "u": u, "stream": "combat"})
    return u


def city_wid(state: GameState, city) -> tuple:
    return ("city", state.hexmap.index(city.q, city.r))


def _damage(state: GameState, A: float, D: float, u: float) -> int:
    K = state.tables.constants
    ratio = (A / D) ** float(K["damage_exponent"]["value"])
    return int(round(float(K["damage_base"]["value"]) * ratio * u))


def _strength(state: GameState, unit) -> float:
    return float(state.tables.units[unit.type]["strength"])


def _ranged_strength(state: GameState, unit) -> float:
    return float(state.tables.units[unit.type]["ranged_strength"])


def _kill_unit(state: GameState, report: TurnReport, u) -> None:
    report.events.append({"type": "unit_died", "unit": u.id, "owner": u.owner,
                          "kind": u.type, "q": u.q, "r": u.r,
                          "wid": list(u.wid)})
    remove_unit(state, u)


class CrnCombat(CombatModule):
    """The default implementation: world-keyed CRN rolls."""

    name = "crn"

    def resolve_attack_unit(self, state, report, attacker, defender) -> None:
        if defender.owner == attacker.owner:
            return  # never friendly fire (kernel re-checks, this is belt+braces)
        K = state.tables.constants
        spec = state.tables.units[attacker.type]
        turn = state.turn
        ranged = int(spec["range"]) >= 1 and int(spec["ranged_strength"]) > 0 \
            and state.hexmap.distance(attacker.i, defender.i) > 1
        attacker.acted = True
        attacker.mp = 0
        if ranged:
            A = _ranged_strength(state, attacker) * _hp_factor(state, attacker.hp)
            D = _strength(state, defender) * _hp_factor(state, defender.hp) \
                * terrain_def_mod(state, state.hexmap.index(defender.q, defender.r))
            u = _roll_value(state, report, "ranged", attacker.wid, defender.wid, 0)
            dmg = _damage(state, A, D, u)
            report.rolls[-1]["dmg"] = dmg
            defender.hp -= dmg
        else:
            A = _strength(state, attacker) * _hp_factor(state, attacker.hp)
            D = _strength(state, defender) * _hp_factor(state, defender.hp) \
                * terrain_def_mod(state, state.hexmap.index(defender.q, defender.r)) \
                * (1.0 + float(K["flank_bonus"]["value"])
                   * min(_flankers(state, defender), int(K["flank_max"]["value"])))
            u = _roll_value(state, report, "melee", attacker.wid, defender.wid, 0)
            dmg = _damage(state, A, D, u)
            report.rolls[-1]["dmg"] = dmg
            defender.hp -= dmg
            if defender.hp > 0 and attacker.hp > 0:
                Ar = D  # retaliation: roles swap
                Dr = A
                u2 = _roll_value(state, report, "retal", attacker.wid, defender.wid, 0)
                dmg2 = _damage(state, Ar, Dr, u2)
                report.rolls[-1]["dmg"] = dmg2
                attacker.hp -= dmg2
        if defender.hp <= 0:
            _kill_unit(state, report, defender)
            if not ranged and attacker.hp > 0 \
                    and int(state.occ_unit[state.hexmap.index(defender.q, defender.r)]) < 0:
                # melee kill: attacker advances into the vacated hex.  The
                # occupancy guard keeps the advance out of a hex another
                # resolution in the same simultaneous turn already took
                # (a kill-advance that stacked two units on one hex fired
                # once in an earlier version).
                move_unit(state, attacker, defender.q, defender.r)
        if attacker.hp <= 0:
            _kill_unit(state, report, attacker)

    def resolve_attack_city(self, state, report, attacker, city) -> None:
        K = state.tables.constants
        spec = state.tables.units[attacker.type]
        ranged = int(spec["range"]) >= 1 and int(spec["ranged_strength"]) > 0 \
            and state.hexmap.distance(attacker.i, state.hexmap.index(city.q, city.r)) > 1
        attacker.acted = True
        attacker.mp = 0
        city.attacked = True
        D = city_defense(state, city)
        if ranged:
            A = _ranged_strength(state, attacker) * _hp_factor(state, attacker.hp)
            u = _roll_value(state, report, "city_ranged", attacker.wid, city_wid(state, city), 0)
            dmg = _damage(state, A, D, u)
            dmg = int(round(dmg * float(K["city_ranged_damage_factor"]["value"])))
            report.rolls[-1]["dmg"] = dmg
            city.hp -= dmg
        else:
            A = _strength(state, attacker) * _hp_factor(state, attacker.hp)
            u = _roll_value(state, report, "city_melee", attacker.wid, city_wid(state, city), 0)
            dmg = _damage(state, A, D, u)
            report.rolls[-1]["dmg"] = dmg
            city.hp -= dmg
            if attacker.hp > 0:
                u2 = _roll_value(state, report, "city_retal", attacker.wid, city_wid(state, city), 0)
                dmg2 = _damage(state, D, A, u2)
                report.rolls[-1]["dmg"] = dmg2
                attacker.hp -= dmg2
        if attacker.hp <= 0:
            _kill_unit(state, report, attacker)
            if city.hp <= 0:
                city.hp = 1  # attacker died; city holds at 1 hp (no capture)
            return
        if city.hp <= 0:
            if attacker.owner < 0 or city.owner < 0:
                # barbarians (and anomalies) never capture: city holds at 1 hp
                city.hp = 1
                return
            self._capture(state, report, attacker, city)
            if not ranged \
                    and int(state.occ_unit[state.hexmap.index(city.q, city.r)]) < 0:
                move_unit(state, attacker, city.q, city.r)

    def _capture(self, state, report, attacker, city) -> None:
        K = state.tables.constants
        old_owner = city.owner
        # any garrison of the old owner on the city hex is destroyed -- except
        # the attacker itself (it can be the garrison if the city was already
        # friendly-owned at this point in the resolution order)
        uid = int(state.occ_unit[state.hexmap.index(city.q, city.r)])
        if uid >= 0:
            g = state.units.get(uid)
            if g is not None and g.owner == old_owner and g.id != attacker.id:
                _kill_unit(state, report, g)
        report.events.append({
            "type": "city_captured", "city": city.id, "from": old_owner,
            "to": attacker.owner, "turn": state.turn,
        })
        city.owner = attacker.owner
        city.hp = max(1, int(round(city_max_hp(state, city)
                                   * float(K["city_capture_hp_fraction"]["value"]))))
        city.pop = max(1, city.pop // 2)
        city.worked_stale = True
        state.territory_stale = True
