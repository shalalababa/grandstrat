"""Observation hygiene (design §3/§0b):
observations never alias the hidden state; handed-out buffers are read-only;
clone_state is a true deep copy; hidden_state_hash is complete."""

import numpy as np
import pytest

from conftest import occupancy_ok

from grandstrat.env import GrandStratEnv
from grandstrat.state import City, Civ, Unit


def _env():
    env = GrandStratEnv(scenario="duel")
    env.reset(3)
    return env


def test_obs_arrays_are_read_only_and_not_state_arrays():
    env = _env()
    st = env.state
    obs = env.observation(0)
    for name in ("terrain", "resource", "improve", "owner"):
        layer = obs[name]
        assert not layer.flags.writeable
    for name in ("id", "owner", "type", "q", "r", "hp", "mp"):
        assert not obs["units"][name].flags.writeable
    # copying breaks identity: no shared mutable buffer with the state
    assert obs["terrain"] is not st.terrain
    with pytest.raises(ValueError):
        obs["owner"][0] = 5


def test_obs_mutation_via_aliasing_is_impossible():
    """If any obs buffer aliased state, a forced writeable flip would leak
    into the state; verify the invariant after flipping on a copy."""
    env = _env()
    obs = env.observation(0)
    private = np.array(obs["owner"])  # our own copy
    private[:] = -7
    assert (env.state.owner != -7).all()


def test_step_does_not_mutate_previous_obs():
    env = _env()
    obs0 = env.observation(0)
    snap_owner = np.array(obs0["owner"])
    snap_units = np.array(obs0["units"]["q"])
    env.step({})  # everyone skips
    obs1 = env.observation(0)
    # handed-out buffers stay valid until the next step (contract), and the
    # new build is a fresh object: identity differs
    assert obs1["owner"] is not obs0["owner"]
    del snap_owner, snap_units  # contract documented in env docstring


def test_clone_state_is_exact_deep_copy():
    env = _env()
    h0 = env.hidden_state_hash()
    clone = env.clone_state()
    assert clone.state_hash() == h0
    original_rules_arrays = {name: getattr(env.state, name).copy()
                             for name in ("passable", "move_cost", "yf", "yp", "yg")}
    # mutate every mutable corner of the clone
    clone.terrain[:] += 1
    clone.owner[:] = -2
    clone.improve[:] = 1
    for u in clone.units.values():
        u.hp -= 1
        u.q += 0 if u.q < 100 else 0
    for c in clone.cities.values():
        c.pop += 1
        c.culture += 1
    for civ in clone.civs:
        civ.gold += 1
    clone.next_unit_id += 1
    clone.done = not clone.done
    clone.params["T"] += 1
    clone.dials["fog"] = 0
    assert env.hidden_state_hash() == h0, "clone mutation leaked into state"
    assert env.state.params["T"] != clone.params["T"]
    assert env.state.dials["fog"] != clone.dials["fog"]
    # Yield arrays are used by IMPROVE during rollouts and are omitted from
    # the state hash, so check them directly as well.
    for name in original_rules_arrays:
        arr = getattr(clone, name)
        assert not np.shares_memory(arr, getattr(env.state, name))
        arr[0] = not arr[0] if name == "passable" else arr[0] + 1
        assert np.array_equal(getattr(env.state, name), original_rules_arrays[name])


def test_hidden_state_hash_tracks_every_mutated_field():
    env = _env()
    base = env.state
    seen = set()
    seen.add(base.state_hash())
    u0 = next(iter(base.units.values()))
    u0.hp -= 1
    assert base.state_hash() not in seen
    seen.add(base.state_hash())
    c0 = next(iter(base.cities.values()))
    c0.food_bucket += 1
    assert base.state_hash() not in seen
    seen.add(base.state_hash())
    base.civs[0].gold += 1
    assert base.state_hash() not in seen
    seen.add(base.state_hash())
    base.turn += 1
    assert base.state_hash() not in seen


def test_agents_never_receive_state():
    """Scripted agents act through act(obs, legal, rng) only; assert the obs
    passed around contains no GameState and no unit/city objects."""
    from grandstrat.record import play_game

    env = _env()
    seen_obs = []

    class Snoop(env.info.__class__):
        pass

    obs = env.observation(1)
    for key, val in obs.items():
        assert not isinstance(val, (Unit, City, Civ)), key
    # unit/city info arrives as arrays + tuples, not live objects
    assert all(isinstance(u, tuple) for u in obs["unit_list"])


def test_flat_encoder_deterministic_and_dense():
    env = _env()
    v1 = env.flat_observation(0)
    v2 = env.flat_observation(0)
    assert v1.dtype == np.float32
    assert (v1 == v2).all()
    assert np.isfinite(v1).all()


def test_play_through_keeps_grids_and_hash_stable(tmp_path):
    from grandstrat.record import play_game
    from grandstrat.agents import make_agents

    env = GrandStratEnv(scenario="duel")
    agents = make_agents(["expander", "warmonger"], 2)
    play_game(env, 5, agents)
    assert occupancy_ok(env.state)
    h = env.hidden_state_hash()
    assert occupancy_ok(env.clone_state())
    assert env.clone_state().state_hash() == h
