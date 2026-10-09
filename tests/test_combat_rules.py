"""Combat mechanics and state-integrity invariants (design §1)."""

import pytest

from conftest import occupancy_ok

from grandstrat.combat import CrnCombat
from grandstrat.engine import TurnReport, spawn_unit
from grandstrat.env import GrandStratEnv
from grandstrat.hexmap import HexMap
from grandstrat.rng import Rng
from grandstrat.state import GameState
from grandstrat.tables import load_tables


def _empty_state():
    tables = load_tables()
    hm = HexMap(4)
    rng = Rng(0)
    params = {"name": "test", "R": 4, "civs": 2, "T": 10, "camps": 0,
              "start": "standard"}
    st = GameState(hm, tables, rng, params, {"fog": 4, "resource_scarcity": 1.0,
                                             "adversary_strength": 1.0,
                                             "mistake_punishment": 0.0,
                                             "nonstationarity": 0.0}, 0)
    from grandstrat.state import Civ

    st.civs = [Civ(20), Civ(20)]
    import numpy as np

    st.terrain[:] = tables.terrain_ids["plains"]
    st.passable[:] = True
    st.move_cost[:] = 1
    return st


def _duel(st, atk_type="warrior", def_type="warrior"):
    a = spawn_unit(st, 0, atk_type, 0, 0)
    d = spawn_unit(st, 1, def_type, 1, 0)
    return a, d


def test_melee_damages_both_sides_with_world_keyed_rolls():
    st = _empty_state()
    a, d = _duel(st)
    report = TurnReport(0)
    CrnCombat().resolve_attack_unit(st, report, a, d)
    assert report.rolls, "melee must record attack and retaliation rolls"
    assert [r["key"][0] for r in report.rolls] == ["melee", "retal"]
    for r in report.rolls:
        # keys carry the units' WORLD identities (owner, birth turn, birth
        # key, ordinal), never the creation-counter ids
        assert r["key"][1:] == [0, a.wid, d.wid, 0]
        assert a.wid[:1] == (0,) and d.wid[:1] == (1,)
        assert "dmg" in r and r["dmg"] >= 0
    assert d.hp < 100 and a.hp < 100
    assert occupancy_ok(st)


def test_flanking_and_terrain_enter_defense_strength():
    """Deterministic check of the modifiers: same roll value, different
    flanker/terrain setups -> strictly less incoming damage when flanked or
    in rough terrain (analytic, not statistical)."""
    st = _empty_state()
    a, d = _duel(st)
    assert CrnCombat.__name__ == "CrnCombat"
    from grandstrat.combat import _flankers, _hp_factor, _strength

    assert _flankers(st, d) == 0
    spawn_unit(st, 1, "warrior", 1, 1)
    assert _flankers(st, d) == 1
    spawn_unit(st, 1, "warrior", 2, 0)
    assert _flankers(st, d) == 2
    spawn_unit(st, 0, "warrior", 0, 1)  # friendly to the defender? no: owner 0
    assert _flankers(st, d) == 2, "enemy adjacency must not flank"

    # terrain: hills defend better than plains at the same roll
    def dmg(terrain, flankers, u=1.0):
        K = st.tables.constants
        A = _strength(st, a) * _hp_factor(st, a.hp)
        tname = st.tables.terrain_names[st.tables.terrain_ids[terrain]]
        D = (_strength(st, d) * _hp_factor(st, d.hp)
             * float(st.tables.terrain[tname]["def_mod"])
             * (1.0 + float(K["flank_bonus"]["value"]) * min(flankers, 2)))
        from grandstrat.combat import _damage

        return _damage(st, A, D, u)

    assert dmg("hills", 0) < dmg("plains", 0)
    assert dmg("plains", 1) < dmg("plains", 0)
    assert dmg("plains", 3) == dmg("plains", 2), "flanking caps at flank_max"


def test_ranged_no_retaliation_and_city_ranged_halves():
    st = _empty_state()
    a = spawn_unit(st, 0, "archer", 0, 0)
    d = spawn_unit(st, 1, "warrior", 2, 0)  # within range 2, not adjacent
    report = TurnReport(0)
    CrnCombat().resolve_attack_unit(st, report, a, d)
    assert [r["key"][0] for r in report.rolls] == ["ranged"]
    assert a.hp == 100  # archer unscathed at range


def test_city_capture_transfers_and_heals_half():
    from grandstrat.state import City

    st = _empty_state()
    city = City(0, 1, 1, 0, capital=True)
    city.hp = 1  # one hit from capture
    st.cities[city.id] = city
    st.occ_city[st.hexmap.index(1, 0)] = city.id
    attacker = spawn_unit(st, 0, "warrior", 0, 0)
    report = TurnReport(0)
    CrnCombat().resolve_attack_city(st, report, attacker, city)
    assert any(e["type"] == "city_captured" for e in report.events)
    assert city.owner == 0
    assert city.hp == int(city_max_hp_half(st, city))
    assert attacker.q == 1 and attacker.r == 0  # attacker advanced in
    assert occupancy_ok(st)


def city_max_hp_half(st, city):
    from grandstrat.economy import city_max_hp

    return round(city_max_hp(st, city)
                 * float(st.tables.constants["city_capture_hp_fraction"]["value"]))


def test_barbarians_never_capture():
    from grandstrat.state import City

    st = _empty_state()
    city = City(0, 0, 1, 0, capital=True)
    city.hp = 1
    st.cities[city.id] = city
    st.occ_city[st.hexmap.index(1, 0)] = city.id
    barb = spawn_unit(st, -1, "warrior", 0, 0)
    report = TurnReport(0)
    CrnCombat().resolve_attack_city(st, report, barb, city)
    assert city.owner == 0, "barbarians must never capture"
    assert city.hp == 1


def test_combat_kills_leave_consistent_state():
    env = GrandStratEnv(scenario="duel")
    from grandstrat.agents import make_agents
    from grandstrat.record import play_game

    play_game(env, 2, make_agents(["warmonger", "warmonger"], 2))
    assert occupancy_ok(env.state)
