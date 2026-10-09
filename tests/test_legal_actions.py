"""Legal-action completeness (design §6): every listed action executes;
every non-listed probe action is rejected; non-terminal states always offer
an action."""

import random

import pytest

from conftest import occupancy_ok

from grandstrat.engine import InvalidAction, legal_actions
from grandstrat.env import GrandStratEnv


def _collect_states(scenario, seeds, steps):
    """(state, kernel) snapshots taken mid-game from scripted play."""
    from grandstrat.record import play_game

    snapshots = []
    for seed in seeds:
        env = GrandStratEnv(scenario=scenario)
        agents = _agents_for(env)
        env.reset(seed)
        for t in range(steps):
            if env.state.done:
                break
            actions = _agent_actions(env, agents)
            if t in (2, 5, 9):
                snapshots.append((env.clone_state(), seed))
            env.step(actions)
    return snapshots


def _agents_for(env):
    from grandstrat.agents import make_agents

    return make_agents(["expander", "warmonger", "turtle", "scout"][:env.resolved_params()["civs"]],
                       env.resolved_params()["civs"])


def _agent_actions(env, agents):
    actions = {}
    for civ, agent in enumerate(agents):
        if not env.state.civs[civ].alive:
            continue
        actions[civ] = agent.act(env.observation(civ), env.legal_actions(civ),
                                 env.state.rng.stream(agent.stream_name))
    return actions


SCENARIOS = [("tiny", (0, 1, 2)), ("duel", (0, 1)), ("dense", (0,))]


@pytest.mark.parametrize("scenario,seeds", SCENARIOS)
def test_every_legal_action_executes(scenario, seeds):
    """Apply each listed action as a single-civ batch on a clone: must step
    cleanly, never crash, and leave a consistent occupancy grid."""
    from grandstrat.observe import build_base_observation

    executed = 0
    for state, seed in _collect_states(scenario, seeds, 12):
        for civ in range(len(state.civs)):
            if not state.civs[civ].alive:
                continue
            legal = legal_actions(state, civ)
            assert legal, "non-terminal state must offer actions"
            assert ("SKIP",) in legal
            for action in legal:
                clone = state.clone()
                e2 = GrandStratEnv(scenario=scenario)
                e2.state = clone
                e2._prev_scores = [c.score for c in clone.civs]
                e2._obs_base = build_base_observation(clone)
                obs, rewards, terms, truncs, infos = e2.step({civ: [action]})
                executed += 1
                report = infos[civ]["report"]
                # a solo batch cannot create staleness for its own action:
                # any rejection of the listed action is an
                # enumeration/execution mismatch
                for r in report.events:
                    if r["type"] == "rejected":
                        assert tuple(r["action"]) != action, (
                            f"listed action {action} did not execute: "
                            f"rejected ({r['reason']})"
                        )
                assert occupancy_ok(clone), "occupancy grid corrupted"
    assert executed > 120, f"expected broad coverage, got {executed}"


@pytest.mark.parametrize("scenario,seeds", SCENARIOS)
def test_every_illegal_action_is_rejected(scenario, seeds):
    """Probe actions that are NOT in the legal set must raise InvalidAction
    (and only that) when submitted."""
    rng = random.Random(7)
    probed = 0
    for state, seed in _collect_states(scenario, seeds, 10):
        for civ in range(len(state.civs)):
            if not state.civs[civ].alive:
                continue
            legal = set(legal_actions(state, civ))
            live_units = [u.id for u in state.units.values() if u.owner == civ]
            live_cities = [c.id for c in state.cities.values() if c.owner == civ]
            probes = set()
            for _ in range(40):
                kind = rng.choice(["MOVE", "ATTACK_U", "ATTACK_C", "FOUND",
                                   "IMPROVE", "PRODUCE", "JUNK", "SKIP"])
                if kind == "JUNK":
                    probes.add(("TELEPORT", rng.choice(live_units or [0])))
                elif kind == "MOVE":
                    probes.add(("MOVE", rng.choice(live_units or [0]), rng.randrange(6)))
                elif kind == "ATTACK_U":
                    probes.add(("ATTACK_U", rng.choice(live_units or [0]),
                                rng.randrange(10_000)))
                elif kind == "ATTACK_C":
                    probes.add(("ATTACK_C", rng.choice(live_units or [0]),
                                rng.randrange(10_000)))
                elif kind == "FOUND":
                    probes.add(("FOUND", rng.choice(live_units or [0])))
                elif kind == "IMPROVE":
                    probes.add(("IMPROVE", rng.choice(live_units or [0])))
                elif kind == "PRODUCE":
                    probes.add(("PRODUCE", rng.choice(live_cities or [0]),
                                rng.choice(["battleship", "giant_robot"])))
                # SKIP alone is always legal: not a probe
            for probe in probes:
                if probe in legal:
                    continue
                probed += 1
                e2 = GrandStratEnv(scenario=scenario)
                e2.state = state.clone()
                with pytest.raises(InvalidAction):
                    e2.step({civ: [probe]})
    assert probed > 120, f"expected broad rejection coverage, got {probed}"


def test_wrong_types_rejected():
    env = GrandStratEnv(scenario="tiny")
    env.reset(0)
    with pytest.raises(InvalidAction):
        env.step({0: "MOVE somewhere"})
    with pytest.raises(InvalidAction):
        env.step({99: [("SKIP",)]})
    with pytest.raises(InvalidAction):
        env.step({0: [("MOVE", 0, 9)]})  # direction 9 does not exist
