"""Regression tests for engine invariants.

Each test pins a defect that was found and fixed in an earlier version:

B1  a batch that bound the SAME unit twice passed `validate` and executed both
    actions (a unit moved twice; two attacks on one defender drew the SAME
    world-keyed combat roll, because roll_index is always 0).
B2  bankruptcy disbanding never returned gold to the treasury, so the disband
    loop could only terminate by exhausting the civ's ENTIRE army and clamping
    at 0 -- measured 37 times in 6 default games, 320 units destroyed.
B3  the shipped conservation test only re-derives the ledger's own arithmetic.
    This one recomputes the upkeep charge from the unit multiset the engine
    actually held at income time, i.e. from outside the ledger.
"""

import pytest
from conftest import LINEUP, make_env

from grandstrat.agents import AGENTS, make_agents
from grandstrat.engine import InvalidAction, legal_actions
from grandstrat.hexmap import DIRECTIONS


# --------------------------------------------------------------------------
# B1 -- one action per unit per turn is a BATCH invariant
# --------------------------------------------------------------------------

def _find_double_move_candidate():
    """A (seed, action) whose unit has enough move points to move twice."""
    for seed in range(20):
        env = make_env("default")
        env.reset(seed=seed)
        st = env.state
        for a in legal_actions(st, 0):
            if a[0] != "MOVE":
                continue
            u = st.units[a[1]]
            if u.mp < 2:
                continue
            dq, dr = DIRECTIONS[a[2]]
            twice = (u.q + 2 * dq, u.r + 2 * dr)
            if twice in st.hexmap.idx:
                j = st.hexmap.idx[twice]
                if st.passable[j] and int(st.occ_unit[j]) < 0 \
                        and int(st.move_cost[j]) <= u.mp - 1:
                    return seed, a
    pytest.skip("no double-move candidate in the first 20 seeds")


def test_duplicate_action_for_one_unit_is_rejected():
    seed, a = _find_double_move_candidate()
    env = make_env("default")
    env.reset(seed=seed)
    with pytest.raises(InvalidAction, match="one action per unit per turn"):
        env.step({0: [a, a], 1: [], 2: [], 3: []})


def test_two_different_actions_for_one_unit_are_rejected():
    env = make_env("default")
    env.reset(seed=11)
    by_unit = {}
    for a in legal_actions(env.state, 0):
        if a[0] == "MOVE":
            by_unit.setdefault(a[1], []).append(a)
    pair = next(v[:2] for v in by_unit.values() if len(v) >= 2)
    with pytest.raises(InvalidAction, match="one action per unit per turn"):
        env.step({0: list(pair), 1: [], 2: [], 3: []})


def test_two_produce_orders_for_one_city_are_rejected():
    env = make_env("default")
    env.reset(seed=4)
    prod = [a for a in legal_actions(env.state, 0) if a[0] == "PRODUCE"]
    same_city = {}
    for a in prod:
        same_city.setdefault(a[1], []).append(a)
    pair = next(v[:2] for v in same_city.values() if len(v) >= 2)
    with pytest.raises(InvalidAction, match="one PRODUCE per city per turn"):
        env.step({0: list(pair), 1: [], 2: [], 3: []})


def test_distinct_units_may_act_in_the_same_batch():
    """The rule must not over-reject: one action each for two units is legal."""
    env = make_env("default")
    env.reset(seed=11)
    by_unit = {}
    for a in legal_actions(env.state, 0):
        if a[0] == "MOVE":
            by_unit.setdefault(a[1], []).append(a)
    uids = sorted(by_unit)[:2]
    assert len(uids) == 2
    env.step({0: [by_unit[uids[0]][0], by_unit[uids[1]][0]], 1: [], 2: [], 3: []})


def test_noisy_agents_never_submit_two_actions_for_one_unit():
    """The noise substitution draws from the whole legal pool and used to
    produce duplicate-unit batches (it double-attacked one defender)."""
    env = make_env("tiny")
    agents = [AGENTS["expander"](0, {"noise": 0.5}),
              AGENTS["warmonger"](1, {"noise": 0.5})]
    from grandstrat.record import play_game

    play_game(env, 1, agents)  # would raise InvalidAction if it regressed


# --------------------------------------------------------------------------
# B2 -- bankruptcy disbands only as far as solvency
# --------------------------------------------------------------------------

def test_bankruptcy_stops_at_solvency_and_never_wipes_the_army():
    """Before the fix EVERY bankruptcy event was `clamped`: disbanding returned
    no gold, so the loop could only end by destroying the civ's entire army
    (37/37 events, 320 units, up to 31 in one turn over 6 default games)."""
    env = make_env("default")
    agents = make_agents(LINEUP, 4)
    events = 0
    clamped = 0
    emptied = 0
    biggest = 0
    for seed in range(6):
        env.reset(seed=seed)
        while not env.state.done:
            acts = {c: agents[c].act(env.observation(c), env.legal_actions(c), c)
                    for c in range(4)}
            _, _, _, _, infos = env.step(acts)
            for e in infos[0]["report"].events:
                if e["type"] != "bankruptcy_disband":
                    continue
                events += 1
                owner = e["owner"]
                left = [u for u in env.state.units.values() if u.owner == owner]
                biggest = max(biggest, len(e["units"]))
                # solvency is restored either way; never a negative treasury
                assert env.state.civs[owner].gold >= 0, (
                    f"seed {seed}: bankruptcy left civ {owner} in debt")
                if e["clamped"]:
                    clamped += 1
                    # the clamp is only reachable with no payable upkeep left
                    assert all(
                        int(env.state.tables.units[u.type]["upkeep"]) == 0
                        for u in left), (
                        f"seed {seed}: clamped while units with upkeep remain")
                if not left:
                    emptied += 1
    assert events > 0, "no bankruptcy fired; the test would be vacuous"
    # the defect's signature: bankruptcy as an army-wipe rather than a trim
    assert clamped == 0, f"{clamped}/{events} bankruptcies clamped the treasury"
    assert emptied <= events // 5, (
        f"{emptied}/{events} bankruptcies emptied the civ's army")
    assert biggest <= 10, f"one bankruptcy disbanded {biggest} units at once"


# --------------------------------------------------------------------------
# B3 -- adversarial conservation: upkeep recomputed OUTSIDE the ledger
# --------------------------------------------------------------------------

def test_upkeep_ledger_matches_an_independent_recount(monkeypatch):
    """Recompute each turn's upkeep charge from the unit multiset the kernel
    held when `income_phase` started, plus units built during the phase, minus
    units disbanded for bankruptcy -- then compare with the ledger column.

    This is deliberately not derived from the ledger: the shipped conservation
    test only checks that before + income - upkeep + bounty == after, which
    holds for any consistently wrong pair of income/upkeep numbers."""
    from grandstrat import economy, kernel

    snapshots = {}
    real_income_phase = economy.income_phase

    def spy(state, report, **kw):
        # id -> (type, owner) for every civ-owned unit at income time
        snapshots[state.turn] = {u.id: (u.type, u.owner)
                                 for u in state.units.values() if u.owner >= 0}
        return real_income_phase(state, report, **kw)

    monkeypatch.setattr(kernel.economy, "income_phase", spy)

    env = make_env("default")
    agents = make_agents(LINEUP, 4)
    checked = 0
    for seed in (0, 5):
        env.reset(seed=seed)
        while not env.state.done:
            acts = {c: agents[c].act(env.observation(c), env.legal_actions(c), c)
                    for c in range(4)}
            _, _, _, _, infos = env.step(acts)
            rep = infos[0]["report"]
            st = env.state
            at_income = snapshots[rep.turn]
            # units produced inside income_phase pay upkeep the same turn
            built = [(e["unit"], e["kind"], e["owner"]) for e in rep.events
                     if e["type"] == "unit_built"]
            # a disbanded unit is disbanded INSTEAD of being paid for
            disbanded = {uid for e in rep.events
                         if e["type"] == "bankruptcy_disband"
                         for uid in e["units"]}
            upk = st.tables.units
            per_civ = [0] * len(st.civs)
            for uid, (utype, owner) in at_income.items():
                if uid not in disbanded:
                    per_civ[owner] += int(upk[utype]["upkeep"])
            for uid, kind, owner in built:
                if uid not in disbanded:
                    per_civ[owner] += int(upk[kind]["upkeep"])
            for row in rep.ledger["gold"]:
                assert row["upkeep"] == per_civ[row["owner"]], (
                    f"seed {seed} turn {rep.turn} civ {row['owner']}: ledger "
                    f"upkeep {row['upkeep']} != independent recount "
                    f"{per_civ[row['owner']]}")
            checked += 1
    assert checked > 100, "probe covered too few turns to be adversarial"


# --------------------------------------------------------------------------
# docs/CONSTANTS.md documents every assumed constant
# --------------------------------------------------------------------------

def test_constants_documented():
    """Design §7: 'Data constants are ASSUMED and listed in CONSTANTS.md'.
    Every key of the data tables must be named in docs/CONSTANTS.md."""
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = open(os.path.join(root, "docs", "CONSTANTS.md")).read()
    for table in ("constants", "information"):
        consts = json.load(open(os.path.join(root, "grandstrat", "data",
                                             f"{table}.json")))
        missing = sorted(k for k in consts
                         if k != "_comment" and k not in doc)
        assert not missing, f"{table}.json keys absent from CONSTANTS.md: {missing}"


def test_every_constant_is_tagged_assumed():
    """Nothing may claim fidelity to a licensed ruleset (design §7)."""
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("constants", "units", "buildings", "techs", "terrain",
                 "improvements", "information"):
        table = json.load(open(os.path.join(root, "grandstrat", "data",
                                            f"{name}.json")))
        for key, spec in table.items():
            if key == "_comment" or not isinstance(spec, dict):
                continue
            if "source" in spec:
                assert spec["source"] == "ASSUMED", (
                    f"{name}.json:{key} source={spec['source']!r}")
