"""Env API surface (design §3, §5, §0b) and the text renderer."""

import json

import numpy as np
import pytest

from conftest import make_env

from grandstrat.observe import flat_layout
from grandstrat.scenarios import DIALS, SCENARIOS


def test_api_shapes_parallel_pettingzoo():
    env = make_env("tiny")
    obs = env.reset(1)
    assert set(obs) == set(env.possible_agents)
    actions = {}
    for civ in env.possible_agents:
        legal = env.legal_actions(civ)
        assert legal
        actions[civ] = [legal[0]]
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert set(rewards) == set(env.possible_agents)
    assert set(terms) == set(env.possible_agents)
    assert set(truncs) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    assert isinstance(env.hidden_state_hash(), str)
    clone = env.clone_state()
    assert clone.state_hash() == env.hidden_state_hash()


def test_terminal_reward_lands_with_done():
    """The winner's terminal reward must not be dropped on the final step
    (so a learner never misses it)."""
    from grandstrat.record import play_game
    from grandstrat.agents import make_agents

    env = make_env("duel")
    agents = make_agents(["warmonger", "expander"], 2)
    env.reset(0)
    last = None
    while not env.state.done:
        actions = {}
        for civ, ag in enumerate(agents):
            if not env.state.civs[civ].alive:
                continue
            actions[civ] = ag.act(env.observation(civ), env.legal_actions(civ),
                                  env.state.rng.stream(ag.stream_name))
        obs, rewards, terms, truncs, infos = env.step(actions)
        last = (rewards, terms)
    rewards, terms = last
    assert all(terms[c] for c in terms)
    winner = env.state.winner
    term = float(env.tables.constants["terminal_reward"]["value"])
    if winner is not None and winner >= 0:
        assert rewards[winner] >= term  # includes the terminal bonus


def test_scenario_and_dial_registry():
    assert "default" in SCENARIOS and SCENARIOS["default"]["R"] == 12
    for name in ("fog", "resource_scarcity", "adversary_strength",
                 "mistake_punishment", "nonstationarity"):
        assert name in DIALS
    with pytest.raises(ValueError):
        make_env("nope")
    with pytest.raises(ValueError):
        make_env(dials={"fog": 99})
    with pytest.raises(ValueError):
        make_env(dials={"no_such_dial": 1})


def test_stream_seed_override_plumbed_through_env():
    env = make_env("tiny", stream_seeds={"combat": 777})
    obs = env.reset(1)
    assert env.state.rng._effective_seed("combat") == 777
    assert env.state.rng._effective_seed("barbarians") != 777


def test_flat_encoder_layout_stable():
    env = make_env("tiny")
    env.reset(0)
    layout = flat_layout(env.state)
    assert layout["total"] > 0
    # layout offsets must not collide
    spans = [(off, off + size) for name, span in layout.items()
             if name != "total" for off, size in [span]]
    spans.sort()
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0


def test_text_renderer_deterministic_and_wellformed():
    env = make_env("tiny")
    env.reset(2)
    txt = env.render_text(0)
    assert set(txt) == {"json", "summary"}
    payload = json.loads(txt["json"])
    for field in ("turn", "you", "your_civ", "your_units", "your_cities",
                  "legal_actions", "gold", "scores"):
        assert field in payload
    legal = env.legal_actions(0)
    assert [list(a) for a in legal] == payload["legal_actions"]
    # determinism: identical within a turn, JSON keys sorted
    txt2 = env.render_text(0)
    assert txt == txt2
    assert "Legal actions" in txt["summary"]
    # after stepping, the renderer reflects the new turn
    env.step({})
    assert env.render_text(0)["json"] != txt["json"]


def test_observation_interface_fog_ready():
    """The information module is swappable: a custom module returning a
    masked view plugs in without env/state changes."""
    from grandstrat.observe import InformationModule

    class HalfObservable(InformationModule):
        name = "half"

        def build_observation(self, state, civ, base):
            obs = dict(base)
            hidden = np.array(base["owner"])
            hidden[hidden != civ] = -1
            obs["owner"] = hidden
            obs["you"] = civ
            return obs

    env = make_env("duel")
    env.info = HalfObservable()
    env.reset(0)
    o0 = env.observation(0)
    o1 = env.observation(1)
    assert (o0["owner"][o0["owner"] == 1] == -1).all()
    assert (o1["owner"][o1["owner"] == 0] == -1).all()
    # the state itself is untouched
    assert (env.state.owner != -1).any()
