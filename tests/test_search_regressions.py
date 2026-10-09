"""Executable engine/observation boundaries for determinization and search."""
import copy

import numpy as np

from conftest import make_env
from grandstrat.belief import SupportResult, Track
from grandstrat.engine import can_move_to
from grandstrat.search import determinize, model_id, observation_action
from grandstrat.tables import load_tables


TABLES = load_tables()


def fog_observation():
    env = make_env("tiny", dials={"fog": 0}, config={"information": "fog"})
    env.reset(4)
    return env.observation(0)


def test_fog_handles_are_not_empty_engine_occupancy():
    obs = fog_observation()
    assert any(row[0] < 0 for row in obs["unit_list"])
    state = determinize(obs, None, TABLES)
    for unit in state.units.values():
        assert unit.id >= 0
        assert state.occ_unit[unit.i] == unit.id
    for city in state.cities.values():
        assert city.id >= 0
        assert state.occ_city[state.hexmap.index(city.q, city.r)] == city.id
    # Real engine movement must see an adjacent occupied hex as occupied.
    unit = next(u for u in state.units.values() if u.owner == 0)
    idx = next(int(i) for i in state.hexmap.neighbors[unit.i]
               if i >= 0 and state.passable[i] and state.occ_city[i] < 0)
    from grandstrat.engine import spawn_unit
    if state.occ_unit[idx] < 0:
        spawn_unit(state, 0, "warrior", int(state.hexmap.qs[idx]), int(state.hexmap.rs[idx]))
    assert not can_move_to(state, unit, idx)


def test_observation_actions_round_trip_both_id_spaces():
    for obs in [fog_observation(), {"unit_list": [(3,), (9,)], "city_list": [(4,)]}]:
        uid = obs["unit_list"][0][0]
        cid = obs["city_list"][0][0]
        assert observation_action(("MOVE", model_id(uid), 2), obs) == ("MOVE", uid, 2)
        assert observation_action(("ATTACK_C", model_id(uid), model_id(cid)), obs) == ("ATTACK_C", uid, cid)
        assert observation_action(("PRODUCE", model_id(cid), "warrior"), obs) == ("PRODUCE", cid, "warrior")


def test_determinization_key_selects_different_supported_worlds():
    obs = fog_observation()
    base = determinize(obs, None, TABLES)
    empty = np.flatnonzero(base.passable & (base.occ_unit < 0) & (base.occ_city < 0))[:8]
    assert len(empty) > 1
    track = Track((1, "warrior", int(empty[0]), 0), 1, "warrior", int(empty[0]), 0, empty)
    support = SupportResult([(int(i),) for i in empty], [track])
    configurations = set()
    for key in range(32):
        state = determinize(obs, support, TABLES, key=key)
        assert state.state_hash() == determinize(obs, support, TABLES, key=key).state_hash()
        sampled = [u.i for uid, u in state.units.items() if uid not in base.units]
        assert len(sampled) == 1 and sampled[0] in empty
        configurations.add(sampled[0])
    assert len(configurations) > 1


def test_remembered_negative_city_handle_is_restored():
    obs = copy.deepcopy(fog_observation())
    base = determinize(obs, None, TABLES)
    idx = int(np.flatnonzero(base.passable & (base.occ_city < 0))[0])
    memory = {k: np.array(v, copy=True) for k, v in obs["memory"].items()}
    for field, value in {"city_id": -101, "city_hp": 200, "city_owner": 1,
                         "city_pop": 2, "city_queue": -1, "city_buildings": 0,
                         "city_border": 1, "city_capital": 0}.items():
        memory[field][idx] = value
    obs["memory"] = memory
    state = determinize(obs, None, TABLES)
    city = state.cities[100]
    assert (city.owner, city.pop) == (1, 2)
    assert state.occ_city[idx] == city.id
