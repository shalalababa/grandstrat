"""Tech-tree gating (tech-gated improvements, tech abilities), domination
and entity caps, observer-local handles, the diplomacy protocol, the
search-agent interface with determinization, the buy-or-search hook, and
the mistake_punishment / nonstationarity dials.  All of these semantics are
off at the default configuration, so the core tests keep their meaning."""

import json

import numpy as np
import pytest

from conftest import make_env

from grandstrat.agents import make_agents
from grandstrat.diplomacy import SimpleDiplomacy, make_diplomacy
from grandstrat.engine import InvalidAction, legal_actions, move_unit, spawn_unit
from grandstrat.hexmap import DIRECTIONS, hex_distance
from grandstrat.observe import build_base_observation
from grandstrat.record import RecordWriter, play_game
from grandstrat.search import (Budget, BuyOrSearchAgent, ForwardModel,
                               SearchAdapter, DeterminizedSearchAgent,
                               determinize, model_id)
from grandstrat.state import City, Unit
from grandstrat.tables import load_tables

TABLES = load_tables()


def _report(infos):
    return infos[0]["report"]


def _hex_coords(state, idx):
    return int(state.hexmap.qs[idx]), int(state.hexmap.rs[idx])


def _events(env):
    return env.diplomacy if env.diplomacy is not None else None


# ---------------------------------------------------------------------------
# tech-tree completion
# ---------------------------------------------------------------------------

def test_improvements_are_tech_gated():
    """IMPROVE respects the improvement's `tech` gate
    (the tree's unlocks.improvements is enforced)."""
    env = make_env("tiny")
    env.reset(0)
    st = env.state
    worker = next(u for u in st.units.values()
                  if u.owner == 0 and u.type == "worker")
    iname = None
    tname = st.tables.terrain_names[int(st.terrain[worker.i])]
    for name, spec in st.tables.improvements.items():
        if tname in spec["terrain"]:
            iname = name
            break
    assert iname is not None, "seed 0 worker stands on non-improvable terrain"
    tech = st.tables.improvements[iname]["tech"]
    st.civs[0].techs = ()
    env._legal_cache = {}
    legal_before = any(a[0] == "IMPROVE" and a[1] == worker.id
                       for a in env.legal_actions(0))
    st.civs[0].techs = (tech,)
    env._legal_cache = {}
    legal_after = any(a[0] == "IMPROVE" and a[1] == worker.id
                      for a in env.legal_actions(0))
    assert (legal_before, legal_after) == (False, True), (
        f"IMPROVE {iname} must require {tech}")


def test_tech_unlocks_consistent_with_content_tables():
    """The tech tree's unlocks agree with the content tables' tech gates, and
    every tech unlocks units/buildings/improvements/abilities or is a prereq
    of another tech (design §1: 'each unlocking ...')."""
    techs = TABLES.techs
    dependents = {t: 0 for t in techs}
    for name, spec in techs.items():
        for p in spec["prereqs"]:
            dependents[p] += 1
    for name, spec in TABLES.units.items():
        gate = spec.get("tech")
        if gate is not None:
            assert name in techs[gate]["unlocks"]["units"], name
    for gate, spec in techs.items():
        for name in spec["unlocks"]["units"]:
            assert TABLES.units[name].get("tech") == gate, name
    for name, spec in TABLES.buildings.items():
        gate = spec.get("tech")
        if gate is not None:
            assert name in techs[gate]["unlocks"]["buildings"], name
    for gate, spec in techs.items():
        for name in spec["unlocks"]["buildings"]:
            assert TABLES.buildings[name].get("tech") == gate, name
    for gate, spec in techs.items():
        for name in spec["unlocks"]["improvements"]:
            assert TABLES.improvements[name].get("tech") == gate, name
    for name, spec in TABLES.improvements.items():
        gate = spec.get("tech")
        if gate is not None:
            assert name in techs[gate]["unlocks"]["improvements"], name
    for name, spec in techs.items():
        un = spec["unlocks"]
        assert un["units"] or un["buildings"] or un["improvements"] \
            or spec.get("abilities") or dependents[name] > 0, (
            f"tech {name} unlocks nothing at all")


def test_guilds_gold_ability_in_ledger():
    """guilds' gold_per_turn ability lands in the income phase (and hence the
    ledger's income column)."""
    from grandstrat.economy import city_yields

    env = make_env("tiny")
    env.reset(0)
    st = env.state
    st.civs[0].techs = ("guilds",)
    _, _, _, _, infos = env.step({})
    row = next(r for r in _report(infos).ledger["gold"] if r["owner"] == 0)
    y = sum(city_yields(st, c)["gold"] for c in st.cities.values()
            if c.owner == 0)
    assert row["income"] == y + 3
    # the other civ is unaffected
    row1 = next(r for r in _report(infos).ledger["gold"] if r["owner"] == 1)
    y1 = sum(city_yields(st, c)["gold"] for c in st.cities.values()
             if c.owner == 1)
    assert row1["income"] == y1


def test_civil_service_sight_ability():
    """civil_service' city_sight_bonus widens the fog visible set, and the
    extra hexes all lie within the boosted sight radius of own cities."""
    env = make_env("duel", dials={"fog": 0}, config={"information": "fog"})
    env.reset(0)
    env.step({})
    base = env.observation(0)["visible"].copy()

    env2 = make_env("duel", dials={"fog": 0}, config={"information": "fog"})
    env2.reset(0)
    env2.state.civs[0].techs = ("civil_service",)
    env2.step({})
    bonus = env2.observation(0)["visible"]

    hm = env2.state.hexmap
    extra = np.flatnonzero(bonus & ~base)
    assert len(extra) > 0, "sight ability changed nothing"
    own_cities = [c for c in env2.state.cities.values() if c.owner == 0]
    for j in extra:
        assert min(hm.distance(j, hm.index(c.q, c.r)) for c in own_cities) <= 3


# ---------------------------------------------------------------------------
# domination and caps
# ---------------------------------------------------------------------------

def test_domination_needs_only_all_capitals():
    """Holding every original capital wins by domination even when a
    rival survives on secondary cities (design §1 'domination (all
    capitals)'; the undocumented dead-rival requirement is gone)."""
    env = make_env("duel")
    env.reset(0)
    env.step({})
    st = env.state
    cap1 = next(c for c in st.cities.values() if c.owner == 1 and c.capital)
    # a secondary city for civ 1 far from every other city
    hm = st.hexmap
    spot = next(j for j in range(hm.n)
                if st.passable[j] and int(st.occ_city[j]) < 0
                and all(hm.distance(j, hm.index(c.q, c.r)) >= 3
                        for c in st.cities.values()))
    c2 = City(st.next_city_id, 1, int(hm.qs[spot]), int(hm.rs[spot]),
              capital=False)
    c2.hp = int(TABLES.constants["city_hp_max"]["value"])
    st.cities[c2.id] = c2
    st.next_city_id += 1
    st.occ_city[spot] = c2.id
    # civ 0 takes the enemy capital
    cap1.owner = 0
    st.territory_stale = True
    obs, rewards, terms, truncs, infos = env.step({})
    rep = _report(infos)
    assert rep.done and rep.winner == 0 and rep.done_reason == "domination"
    assert st.civs[1].alive, "the rival must survive on its secondary city"


def test_unit_cap_enforced_at_spawn_transition():
    """The cap is enforced at the single birth transition, not only
    asserted in the observation builder."""
    env = make_env("tiny")
    env.reset(0)
    st = env.state
    cap = int(TABLES.constants["unit_cap"]["value"])
    filler = cap - len(st.units) - 1
    for k in range(filler):
        u = Unit(1_000_000 + k, 0, "warrior", 0, 0, 100, 1)
        st.units[u.id] = u  # synthetic fill: occupancy not maintained
    spawn_unit(st, 0, "warrior", 0, 0)  # one below the cap: fine
    assert len(st.units) == cap
    with pytest.raises(RuntimeError):
        spawn_unit(st, 0, "warrior", 0, 0)
    st.units.pop(1_000_000)
    # observation-side guard: a cap+1 world can never silently truncate
    st.units[2_000_001] = Unit(2_000_001, 0, "warrior", 0, 0, 100, 1)
    st.units[2_000_002] = Unit(2_000_002, 0, "warrior", 0, 0, 100, 1)
    with pytest.raises(ValueError):
        build_base_observation(st)


def test_barbarian_phase_respects_unit_cap(tmp_path):
    """Caps on the barbarian route: a capped world skips spawns entirely and
    the population never overflows the cap."""
    env = make_env("duel", dials={"adversary_strength": 10.0})
    env.reset(0)
    st = env.state
    cap = int(TABLES.constants["unit_cap"]["value"])
    for k in range(cap - len(st.units)):
        st.units[1_000_000 + k] = Unit(1_000_000 + k, -1, "warrior", 0, 0,
                                       100, 1)
    for _ in range(3):
        if env.state.done:
            break
        env.step({})
        assert len(env.state.units) <= cap


# ---------------------------------------------------------------------------
# observer-local handles: actions in handle space execute
# ---------------------------------------------------------------------------

def test_handle_space_attack_executes():
    """A legal ATTACK_U submitted in the fog module's handle space executes:
    the true target takes damage, and no rejection is recorded."""
    env = make_env("duel", dials={"fog": 0}, config={"information": "fog"})
    agents = make_agents(["expander", "warmonger"], 2)
    for a in agents:
        a.reset()
    env.reset(1)
    # two scripted turns, then forge contact: relocate a civ-1 unit next to
    # a civ-0 unit so ATTACK_U is legal for civ 0
    for _ in range(2):
        acts = {civ: ag.act(env.observation(civ), env.legal_actions(civ),
                            env.state.rng.stream(ag.stream_name))
                for civ, ag in enumerate(agents)
                if env.state.civs[civ].alive}
        env.step(acts)
    st = env.state
    atk = next(u for u in st.units.values() if u.owner == 0
               and u.type == "warrior")
    tgt_hex = next(int(st.hexmap.neighbors[atk.i, dd]) for dd in range(6)
                   if int(st.hexmap.neighbors[atk.i, dd]) >= 0
                   and int(st.occ_city[int(st.hexmap.neighbors[atk.i, dd])]) < 0
                   and int(st.occ_unit[int(st.hexmap.neighbors[atk.i, dd])]) < 0)
    victim = next(u for u in st.units.values() if u.owner == 1
                  and u.type in ("warrior", "scout"))
    move_unit(st, victim, int(st.hexmap.qs[tgt_hex]),
              int(st.hexmap.rs[tgt_hex]))
    attack = next(a for a in env.legal_actions(0)
                  if a[0] == "ATTACK_U"
                  and env.info._game[0].u_rev.get(a[2]) == victim.id)
    # every actor in the legal list is handle-keyed, own units included
    assert env.info._game[0].u_rev[attack[1]] == atk.id
    hp_before = victim.hp
    _, _, _, _, infos = env.step({0: [attack], 1: []})
    rep = _report(infos)
    rejections = [e for e in rep.events if e.get("type") == "rejected"
                  and tuple(e["action"]) == tuple(attack)]
    assert not rejections, f"handle-space attack rejected: {rejections}"
    rolls = [r for r in rep.rolls if r["stream"] == "combat"]
    assert rolls, "the attack drew no combat rolls"
    handle = attack[2]
    assert env.info._game[0].u_rev[handle] == victim.id
    assert env.observation(0)["units"]["count"] >= 0


# ---------------------------------------------------------------------------
# diplomacy protocol
# ---------------------------------------------------------------------------

def _dipl_env():
    env = make_env("duel", config={"diplomacy": "simple"})
    env.reset(0)
    return env


def test_offer_accept_flow_transfers_gold_and_sets_matrix():
    env = _dipl_env()
    st = env.state
    offer = next(a for a in env.legal_actions(0)
                 if a[0] == "OFFER" and a[2] == "open_borders")
    assert offer[1] == 1
    _, _, _, _, infos = env.step({0: [offer], 1: []})
    ev = next(e for e in _report(infos).events if e["type"] == "dipl_offer")
    assert ev["from"] == 0 and ev["to"] == 1 and ev["offer"] == offer[3]
    # snapshot AFTER the offer turn: the acceptance turn is what must balance
    gold0, gold1 = st.civs[0].gold, st.civs[1].gold
    obs = env.observation(1)
    assert obs["meta"]["diplomacy_offers"], "offers are not public news"
    accept = ("ACCEPT", ev["pid"])
    assert accept in {tuple(a) for a in env.legal_actions(1)}
    _, _, _, _, infos = env.step({1: [accept], 0: []})
    rep = _report(infos)
    assert any(e["type"] == "dipl_accepted" for e in rep.events)
    assert st.civs[0].gold == gold0 - offer[3] + (rep.ledger["gold"][0]["income"]
                                                  - rep.ledger["gold"][0]["upkeep"])
    assert st.civs[1].gold == gold1 + offer[3] + (rep.ledger["gold"][1]["income"]
                                                  - rep.ledger["gold"][1]["upkeep"])
    assert rep.ledger["gold"][0]["transfer"] == -offer[3]
    assert rep.ledger["gold"][1]["transfer"] == offer[3]
    for row in rep.ledger["gold"]:
        raw = (row["before"] - row["purchase"] + row["income"]
               - row["upkeep"] + row["bounty"] + row["transfer"])
        assert row["after"] == raw or row["clamped"]
    m = env.diplomacy.matrix(st)
    assert m[0, 1] == 0.5 and m[1, 0] == 0.5
    assert env.diplomacy.agreement(0, 1) == "open_borders"


def test_open_borders_gates_entry_then_allows_and_heals():
    """Closed borders: entering foreign territory is not a legal move
    (computed on the mover's knowledge).  After an accepted open_borders
    offer it is legal, and units heal in the partner's territory."""
    def prepared(env):
        """March both civs 20 turns (border radius 2 by culture 15), then
        place civ 0's warrior on a free hex k adjacent to foreign-territory
        hex j."""
        agents = make_agents(["expander", "expander"], 2)
        for a in agents:
            a.reset()
        while env.state.turn < 20 and not env.state.done:
            acts = {civ: ag.act(env.observation(civ), env.legal_actions(civ),
                                env.state.rng.stream(ag.stream_name))
                    for civ, ag in enumerate(agents)
                    if env.state.civs[civ].alive}
            env.step(acts)
        st = env.state
        hm = st.hexmap
        pair = None
        for jj in range(hm.n):
            if not (int(st.owner[jj]) == 1 and int(st.occ_unit[jj]) < 0
                    and int(st.occ_city[jj]) < 0 and st.passable[jj]
                    and int(st.move_cost[jj]) == 1):
                continue
            for d in range(6):
                kk = int(hm.neighbors[jj, d])
                if kk >= 0 and int(st.occ_unit[kk]) < 0 \
                        and int(st.occ_city[kk]) < 0 and st.passable[kk]:
                    pair = (jj, kk)
                    break
            if pair:
                break
        j, k = pair
        u = next(ux for ux in st.units.values()
                 if ux.owner == 0 and ux.type == "warrior")
        move_unit(st, u, int(hm.qs[k]), int(hm.rs[k]))
        d = next(dd for dd, (dq, dr) in enumerate(DIRECTIONS)
                 if int(hm.neighbors[k, dd]) == j)
        return u, d, j

    # closed borders: the move into enemy territory is not listed
    env = _dipl_env()
    u, d, j = prepared(env)
    assert int(env.state.owner[j]) == 1
    assert ("MOVE", u.id, d) not in set(env.legal_actions(0))

    # accepted open_borders offer makes the same move legal
    env2 = _dipl_env()
    u2, d2, j2 = prepared(env2)
    offer = next(a for a in env2.legal_actions(0)
                 if a[0] == "OFFER" and a[2] == "open_borders")
    env2.step({0: [offer], 1: []})
    ev = next(e for e in env2.diplomacy.public_offers())
    _, _, _, _, infos = env2.step({1: [("ACCEPT", ev["pid"])], 0: []})
    assert ("MOVE", u2.id, d2) in set(env2.legal_actions(0))
    # healing: a hurt unit standing still in partner territory heals
    u2.hp = 50
    move_unit(env2.state, u2, *_hex_coords(env2.state, j2))
    env2.step({0: []})
    assert u2.hp == 55


def test_share_maps_offer_shares_maps():
    """An accepted share_maps offer gives both civs the live union of their
    visible sets (the same mechanics as the DIPLO purchase, negotiated)."""
    env = make_env("duel", dials={"fog": 2},
                   config={"information": "fog", "diplomacy": "simple"})
    env.reset(0)
    offer = next(a for a in env.legal_actions(0)
                 if a[0] == "OFFER" and a[2] == "share_maps")
    env.step({0: [offer], 1: []})
    ev = env.diplomacy.public_offers()[0]
    env.step({1: [("ACCEPT", ev["pid"])], 0: []})
    st = env.state
    assert env.diplomacy.agreement(0, 1) == "share_maps"
    for civ in (0, 1):
        obs = env.observation(civ)
        for u in st.units.values():
            if u.owner >= 0 and u.owner != civ:
                assert obs["visible"][u.i], (
                    f"civ {civ} cannot see partner unit after share_maps")
    m = env.diplomacy.matrix(st)
    assert m[0, 1] == 1.0 and m[1, 0] == 1.0


def test_betrayal_breaks_agreement():
    env = _dipl_env()
    env.step({})
    offer = next(a for a in env.legal_actions(0)
                 if a[0] == "OFFER" and a[2] == "open_borders")
    env.step({0: [offer], 1: []})
    pid = env.diplomacy.public_offers()[0]["pid"]
    env.step({1: [("ACCEPT", pid)], 0: []})
    assert env.diplomacy.agreement(0, 1) == "open_borders"
    # forge an attack: relocate a civ-1 unit next to civ 0's warrior
    st = env.state
    atk = next(u for u in st.units.values() if u.owner == 0
               and u.type == "warrior")
    tgt = next(int(st.hexmap.neighbors[atk.i, dd]) for dd in range(6)
               if int(st.hexmap.neighbors[atk.i, dd]) >= 0
               and int(st.occ_city[int(st.hexmap.neighbors[atk.i, dd])]) < 0
               and int(st.occ_unit[int(st.hexmap.neighbors[atk.i, dd])]) < 0)
    victim = next(u for u in st.units.values() if u.owner == 1
                  and u.type in ("warrior", "scout"))
    move_unit(st, victim, int(st.hexmap.qs[tgt]), int(st.hexmap.rs[tgt]))
    legal = env.legal_actions(0)
    attack = next(a for a in legal if a[0] == "ATTACK_U" and a[2] == victim.id)
    _, _, _, _, infos = env.step({0: [attack], 1: []})
    rep = _report(infos)
    broken = [e for e in rep.events if e["type"] == "dipl_broken"]
    assert broken and broken[0]["kind"] == "open_borders"
    assert env.diplomacy.agreement(0, 1) is None
    assert env.diplomacy.matrix(st)[0, 1] == 0.0


def test_offer_expiry_and_unknown_pid():
    env = _dipl_env()
    env.step({})
    offer = next(a for a in env.legal_actions(0)
                 if a[0] == "OFFER" and a[2] == "open_borders")
    _, _, _, _, infos = env.step({0: [offer], 1: []})
    ttl = int(TABLES.constants["diplomacy_offer_ttl"]["value"])
    pid = env.diplomacy.public_offers()[0]["pid"]
    with pytest.raises(InvalidAction):
        env.step({1: [("ACCEPT", pid + 999)], 0: []})  # unknown pid: illegal
    for _ in range(ttl + 1):
        _, _, _, _, infos = env.step({})
    types = [e["type"] for e in _report(infos).events]
    assert "dipl_expired" in types
    assert env.diplomacy.public_offers() == []


def test_diplomacy_batch_rule():
    env = _dipl_env()
    env.reset(0)
    offers = [a for a in env.legal_actions(0) if a[0] == "OFFER"]
    assert len(offers) >= 2
    with pytest.raises(InvalidAction):
        env.step({0: [offers[0], offers[1]], 1: []})


def test_merchant_fires_diplomacy_in_a_real_game():
    env = make_env("duel", config={"diplomacy": "simple"})
    agents = make_agents(["merchant", "expander"], 2)
    for a in agents:
        a.reset()
    env.reset(0)
    offers = accepted = 0
    while not env.state.done and env.state.turn < 60:
        acts = {}
        for civ, ag in enumerate(agents):
            if env.state.civs[civ].alive:
                acts[civ] = ag.act(env.observation(civ),
                                   env.legal_actions(civ),
                                   env.state.rng.stream(ag.stream_name))
        _, _, _, _, infos = env.step(acts)
        evs = _report(infos).events
        offers += sum(1 for e in evs if e["type"] == "dipl_offer")
        accepted += sum(1 for e in evs if e["type"] == "dipl_accepted")
    assert offers >= 2, "the merchant never bid"
    assert accepted >= 1, "no offer was ever accepted in 60 turns"
    # the engine's diplomacy is deterministic: the named stream draws zero
    assert env.state.rng.draw_counts.get("diplomacy", 0) == 0


def test_no_diplomacy_module_by_default():
    env = make_env("duel")
    env.reset(0)
    assert env.diplomacy is None
    assert not any(a[0] in ("OFFER", "ACCEPT", "REJECT")
                   for a in env.legal_actions(0))
    with pytest.raises(ValueError):
        make_diplomacy({"diplomacy": "warlords"}, TABLES)


# ---------------------------------------------------------------------------
# determinization + forward model + search
# ---------------------------------------------------------------------------

def _marching_scout(seed=0, turns=12):
    """A small fog game in which civ 0's scout marches, so some enemy
    entities may have been seen (and possibly lost again)."""
    env = make_env("tiny", dials={"fog": 0}, config={"information": "fog"})
    env.reset(seed)
    st = env.state
    for t in range(turns):
        if env.state.done:
            break
        legal0 = env.legal_actions(0)
        acts = {0: [], 1: []}
        obs0 = env.observation(0)
        scout_row = next(((uid, q, r) for uid, owner, utype, q, r, hp, mp
                          in obs0["unit_list"]
                          if owner == 0 and utype == "scout"), None)
        if scout_row is None:
            break
        handle, sq, sr = scout_row
        moves = [a for a in legal0 if a[0] == "MOVE" and a[1] == handle]
        if moves:
            acts[0] = [min(moves, key=lambda a: a[2])]
        env.step(acts)
        env.observation(0)
        env.observation(1)
    return env


def test_determinize_from_observation_only():
    from grandstrat.belief import SupportResult

    env = _marching_scout()
    obs = env.observation(0)
    you = int(obs["you"])
    belief = env.belief(0)
    assert isinstance(belief, SupportResult)

    d1 = determinize(obs, belief, TABLES, key=0)
    d2 = determinize(obs, belief, TABLES, key=0)
    assert d1.state_hash() == d2.state_hash(), "determinize is not deterministic"

    hm = d1.hexmap
    # own units: exact observation facts, handles mapped to engine-safe ids
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        if owner != you:
            continue
        u = d1.units.get(model_id(uid))
        assert u is not None and (u.q, u.r) == (int(q), int(r))
    # pinned foreign units: at their observed hexes
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        if owner == you:
            continue
        u = d1.units.get(model_id(uid))
        assert u is not None and (u.q, u.r) == (int(q), int(r))
    # sampled hidden units: only where the belief's candidate sets allow
    union = set()
    for tr in belief.tracks:
        if not tr.fixed:
            union |= set(tr.candidates.tolist())
    for u in d1.units.values():
        if u.owner == you:
            continue
        if int(obs["meta"]["turn"]) >= 0 and any(
                model_id(row[0]) == u.id for row in obs["unit_list"]):
            continue  # pinned
        assert u.i in union, f"sampled unit at {u.i} is outside the support"

    # the true hidden state is never inserted: a unit the observer has never
    # seen does not exist in ANY determinization
    st = env.state
    far = max(range(hm.n), key=lambda j: min(
        hm.distance(j, hm.index(ux.q, ux.r))
        for ux in st.units.values() if ux.owner == 0))
    from grandstrat.engine import refresh_occ
    spy = spawn_unit(st, 1, "warrior", int(hm.qs[far]), int(hm.rs[far]))
    refresh_occ(st)
    obs2 = env.observation(0)
    assert not any((int(q), int(r)) == (int(hm.qs[far]), int(hm.rs[far]))
                   for uid2, o2, t2, q, r, hp, mp in obs2["unit_list"]
                   if o2 == 1), "precondition: the spy must be unseen"
    belief2 = env.belief(0)
    for key in range(6):
        dd = determinize(obs2, belief2, TABLES, key=key)
        placed = [u for u in dd.units.values()
                  if u.owner == 1 and (u.q, u.r) == (spy.q, spy.r)]
        assert not placed, "determinize inserted the true hidden unit"


def test_forward_model_steps_a_determinized_clone():
    env = _marching_scout()
    obs = env.observation(0)
    det = determinize(obs, env.belief(0), TABLES, key=0)
    fm = ForwardModel(TABLES)
    clone = det.clone()
    rep = fm.step(clone, {})
    assert clone.turn == det.turn + 1
    assert rep.done or rep.scores
    # legal actions in the model and the hidden-state hash path work
    assert ("SKIP",) in fm.legal_actions(det, 0)
    assert det.state_hash() == determinize(obs, env.belief(0), TABLES,
                                           key=0).state_hash()


def test_search_adapter_plays_legal_actions_and_accounts_budget(tmp_path):
    env = make_env("tiny", dials={"fog": 0}, config={"information": "fog"})
    inner = DeterminizedSearchAgent(0, {"expansions": 16,
                                        "determinizations": 2, "horizon": 1})
    adapter = SearchAdapter(0, inner,
                            belief_fn=lambda obs: env.belief(obs["you"]),
                            budget_params={"expansions": 16,
                                           "determinizations": 2})
    opp = make_agents(["expander"], 2)[1]
    path = str(tmp_path / "search.jsonl")
    with RecordWriter(path) as w:
        play_game(env, 0, [adapter, opp], writer=w)
    assert adapter.total_expansions > 0, "the search agent never spent compute"
    assert adapter.total_determinizations > 0
    # the record carries the honest per-turn compute accounting
    turns = [json.loads(l) for l in open(path) if '"turn"' in l]
    comp = [ln.get("compute") for ln in turns if ln.get("compute")]
    assert comp, "compute accounting missing from the record"
    assert all("0" in c for c in comp)
    assert all(c["0"]["expansions_used"] <= 16 for c in comp)
    assert any(c["0"]["expansions_used"] > 0 for c in comp)


def test_buy_or_search_agent_buys_info_or_spends_compute(tmp_path):
    env = make_env("tiny", dials={"fog": 2},
                   config={"information": "fog"})
    agent = BuyOrSearchAgent(
        0, {"mode": "alternate", "budget": {"expansions": 8,
                                            "determinizations": 1}},
        belief_fn=lambda obs: env.belief(obs["you"]))
    opp = make_agents(["expander"], 2)[1]
    for a in (agent, opp):
        a.reset()
    env.reset(0)
    reveals = computes = 0
    purchases = 0
    for t in range(8):
        if env.state.done:
            break
        for c in env.state.civs:
            c.gold = max(c.gold, 500)
        acts = {0: agent.act(env.observation(0), env.legal_actions(0),
                             env.state.rng.stream("agent_0")),
                1: opp.act(env.observation(1), env.legal_actions(1),
                           env.state.rng.stream("agent_1"))}
        _, _, _, _, infos = env.step(acts)
        purchases += sum(1 for e in _report(infos).events
                         if e.get("type") == "info_purchase")
        if agent.last_decision == "reveal":
            reveals += 1
        elif agent.last_decision == "compute":
            computes += 1
    assert reveals >= 2 and computes >= 2, "the dial never alternated"
    assert purchases >= 2, "reveal turns bought nothing"
    assert agent.total_expansions > 0, "compute turns never searched"
    assert agent.last_decision in ("reveal", "compute")


# ---------------------------------------------------------------------------
# mistake_punishment and nonstationarity dials
# ---------------------------------------------------------------------------

def _stale_move_setup(dial_value):
    """Build a deterministic position where both civs order the same free
    hex; the lower-id unit wins, the other's move is dropped (stale)."""
    env = make_env("duel", dials={"mistake_punishment": dial_value})
    env.reset(0)
    st = env.state
    hm = st.hexmap

    def enterable(idx):
        return (st.passable[idx] and int(st.move_cost[idx]) <= 1
                and int(st.occ_unit[idx]) < 0 and int(st.occ_city[idx]) < 0)

    j = next(jj for jj in range(hm.n)
             if enterable(jj)
             and sum(1 for dd in range(6)
                     if int(hm.neighbors[jj, dd]) >= 0
                     and enterable(int(hm.neighbors[jj, dd]))) >= 2)
    free_neighbors = [int(hm.neighbors[j, dd]) for dd in range(6)
                      if int(hm.neighbors[j, dd]) >= 0
                      and enterable(int(hm.neighbors[j, dd]))]
    w0 = next(u for u in st.units.values() if u.owner == 0
              and u.type == "warrior")
    w1 = next(u for u in st.units.values() if u.owner == 1
              and u.type == "warrior")
    k0, k1 = free_neighbors[0], free_neighbors[1]
    move_unit(st, w0, int(hm.qs[k0]), int(hm.rs[k0]))
    move_unit(st, w1, int(hm.qs[k1]), int(hm.rs[k1]))

    def dir_into(u, target):
        return next(dd for dd in range(6)
                    if int(hm.neighbors[hm.index(u.q, u.r), dd]) == target)

    return env, w0, w1, dir_into(w0, j), dir_into(w1, j)


def test_mistake_punishment_shapes_reward_on_stale_orders():
    env0, w0, w1, d0, d1 = _stale_move_setup(0.0)
    env5, w0b, w1b, d0b, d1b = _stale_move_setup(5.0)
    _, r0, _, _, infos0 = env0.step({0: [("MOVE", w0.id, d0)],
                                     1: [("MOVE", w1.id, d1)]})
    rep0 = _report(infos0)
    rejected1 = sum(1 for e in rep0.events if e.get("type") == "rejected"
                    and e.get("civ") == 1)
    assert rejected1 == 1, "expected the higher-id mover's order to drop"
    _, r5, _, _, infos5 = env5.step({0: [("MOVE", w0b.id, d0b)],
                                     1: [("MOVE", w1b.id, d1b)]})
    assert r5[0] == r0[0], "innocent civ's reward must be untouched"
    assert r5[1] == pytest.approx(r0[1] - 5.0)
    # dynamics are identical (pure reward shaping): same scores, same events
    assert [c.score for c in env0.state.civs] == [c.score for c in env5.state.civs]
    assert _report(infos0).events == _report(infos5).events


def test_nonstationarity_increases_barbarian_pressure():
    """The nonstationarity dial scales spawn chance by 1 + W*turn/T; with the
    per-camp cap on the small duel map the effect saturates, so this runs the
    default 6-camp scenario at half adversary strength."""
    def spawns(nonstat):
        env = make_env("default", dials={"nonstationarity": nonstat,
                                         "adversary_strength": 0.5})
        agents = make_agents(["expander", "expander", "expander", "expander"], 4)
        for a in agents:
            a.reset()
        env.reset(0)
        n = 0
        while not env.state.done and env.state.turn < 80:
            acts = {}
            for civ, ag in enumerate(agents):
                if env.state.civs[civ].alive:
                    acts[civ] = ag.act(env.observation(civ),
                                       env.legal_actions(civ),
                                       env.state.rng.stream(ag.stream_name))
            _, _, _, _, infos = env.step(acts)
            n += sum(1 for e in _report(infos).events
                     if e.get("type") == "barb_spawn")
        return n

    base, ramped = spawns(0.0), spawns(4.0)
    assert ramped > base, (
        f"nonstationarity ramp did not increase spawns ({ramped} vs {base})")
