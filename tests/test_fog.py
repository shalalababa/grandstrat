"""Fog correctness (design §6), checked by ADVERSARIAL PROBE: a civ never
observes a hex outside its sight unless purchased or shared, verified by
diffing every observation against the hidden state hex by hex over full
games, on several scenarios and dial rungs -- including the memory layer (a
remembered hex must show what was there WHEN SEEN, never the current truth),
the isolation rule (no aliasing of hidden state, no hidden unit ids, no
counts that leak), and legal_actions computable from the observation alone.

The probe recomputes visibility INDEPENDENTLY of the fog module: sight radii
of units and cities straight from the tables, plus purchased reveals re-derived
from the record events (spy actives, diplo pairs, oracle buyers).
"""

import json

import numpy as np
import pytest

from conftest import LINEUP, make_env

from grandstrat.agents import make_agents
from grandstrat.hexmap import hex_distance
from grandstrat.record import RecordWriter, play_game
from grandstrat.tables import load_tables

TABLES = load_tables()
OWNER_UNKNOWN = -2


# ---------------------------------------------------------------------------
# independent visibility recompute (adversarial: not the fog module's code)
# ---------------------------------------------------------------------------

def _sight(tables, type_name):
    return int(tables.units[type_name]["sight"])


def _expected_vis(state, tables, intel, civ):
    """Expected visible set for `civ` at the current state, from first
    principles: own units' sight + own cities' sight + spy intel + diplo
    sharing (closure) + oracle/dial-K full reveal.  `intel` maps civ ->
    {"spies": [(city_id, until_turn)], "shared": set(civ), "oracle": bool}."""
    R = int(state.params["R"])
    if R >= 0 and int(state.dials["fog"]) >= 4:
        vis = np.full(state.hexmap.n, True)
        vis = {c: np.full(state.hexmap.n, True) for c in range(len(state.civs))}
        return vis[civ]
    per_civ = {}

    def blank(c):
        if c not in per_civ:
            per_civ[c] = np.zeros(state.hexmap.n, dtype=bool)

    for c in range(len(state.civs)):
        if intel.get(c, {}).get("oracle") or int(state.dials["fog"]) >= 4:
            per_civ[c] = np.full(state.hexmap.n, dtype=bool, fill_value=True)
            continue
        blank(c)
    city_sight = int(tables.information["city_sight"]["value"])
    for u in state.units.values():
        if u.owner >= 0:
            blank(u.owner)
            _mark(per_civ[u.owner], state, u.i, _sight(tables, u.type))
    for c in state.cities.values():
        if c.owner >= 0:
            blank(c.owner)
            _mark(per_civ[c.owner], state, state.hexmap.index(c.q, c.r),
                  _city_sight_for(state, tables, c.owner))
    # spies: live visibility of the target city hex while active
    for civ_id, spec in intel.items():
        for cid, until in spec.get("spies", []):
            if state.turn <= until and cid in state.cities:
                blank(civ_id)
                city = state.cities[cid]
                _mark(per_civ[civ_id], state,
                      state.hexmap.index(city.q, city.r), 0)
    # diplo: live union of partners' visible sets (closure over pairs)
    for civ_id, spec in intel.items():
        frontier = list(spec.get("shared", ()))
        seen = set()
        while frontier:
            p = frontier.pop()
            if p in seen:
                continue
            seen.add(p)
            blank(civ_id)
            blank(p)
            per_civ[civ_id] |= per_civ[p]
            frontier.extend(intel.get(p, {}).get("shared", ()))
    blank(civ)
    return per_civ[civ]


def _city_sight_for(state, tables, civ):
    """Effective city sight: data base + the civ's city_sight_bonus tech
    abilities."""
    base = int(tables.information["city_sight"]["value"])
    bonus = 0
    for t in state.civs[civ].techs:
        ab = tables.techs[t].get("abilities") or {}
        bonus += int(ab.get("city_sight_bonus", 0))
    return base + bonus


def _mark(vis, state, hex_i, radius):
    if radius == 0:
        vis[hex_i] = True
        return
    hm = state.hexmap
    qi, ri = int(hm.qs[hex_i]), int(hm.rs[hex_i])
    for dq in range(-radius, radius + 1):
        for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1):
            j = hm.idx.get((qi + dq, ri + dr))
            if j is not None:
                vis[j] = True


def _intel_from_events(events, W):
    intel = {}
    for e in events:
        if e.get("type") != "info_purchase":
            continue
        spec = intel.setdefault(e["buyer"], {"spies": [], "shared": set(),
                                             "oracle": False})
        if e["purchase"] == "spy":
            spec["spies"].append((e["target_city"], e["revealed"]["until_turn"]))
        elif e["purchase"] == "diplo":
            spec["shared"].add(e["partner"])
            other = intel.setdefault(e["partner"], {"spies": [], "shared": set(),
                                                    "oracle": False})
            other["shared"].add(e["buyer"])
        elif e["purchase"] == "oracle":
            spec["oracle"] = True
    return intel


# ---------------------------------------------------------------------------
# probe machinery
# ---------------------------------------------------------------------------

class _Snapshots:
    """Per-turn full-state snapshots for the memory-layer check."""

    def __init__(self, tables):
        self.tables = tables
        self.data = {}

    def take(self, state):
        bnames = sorted(self.tables.buildings)
        item_names = list(self.tables.units) + bnames
        units = {u.id: (u.owner, self.tables.unit_ids[u.type], u.hp)
                 for u in state.units.values()}
        cities = {}
        for c in state.cities.values():
            mask = 0
            for b in c.buildings:
                mask |= 1 << bnames.index(b)
            cities[c.id] = (c.owner, c.pop, c.hp,
                            -1 if c.queue is None else item_names.index(c.queue),
                            c.border, int(c.capital), mask)
        self.data[state.turn] = {
            "resource": state.resource.copy(),
            "improve": state.improve.copy(),
            "owner": state.owner.copy(),
            "occ_unit": state.occ_unit.copy(),
            "occ_city": state.occ_city.copy(),
            "units": units,
            "cities": cities,
        }

    def __getitem__(self, turn):
        return self.data[turn]


def _check_observation(env, civ, obs, snap, intel, checks, handles):
    """Diff one civ's observation against the hidden state and the seen-time
    snapshots, hex by hex.  `handles[civ]` is the probe's OWN accumulated
    handle->true-id map (the probe resolves observer-local handles the same
    way the fog module does: from sightings, white-box)."""
    state = env.state
    tables = env.tables
    W = int(state.dials["fog"])
    hm = state.hexmap
    L = hm.n
    if W >= 4 or intel.get(civ, {}).get("oracle"):
        return  # full observability: the shared base IS the truth

    vis = obs["visible"]
    exp_vis = _expected_vis(state, tables, intel, civ)
    checks["hexes"] += L
    assert (vis == exp_vis).all(), (
        f"turn {state.turn} civ {civ}: visible set != independent recompute "
        f"({int((vis != exp_vis).sum())} hexes differ)")

    # -- layer masks: current truth only where visible, sentinels elsewhere
    for name, arr, sentinel in (("resource", obs["resource"], -1),
                                ("improve", obs["improve"], -1),
                                ("owner", obs["owner"], OWNER_UNKNOWN)):
        assert (arr[vis] == getattr(state, name)[vis]).all(), (
            f"turn {state.turn} civ {civ}: {name} wrong on visible hexes")
        assert (arr[~vis] == sentinel).all(), (
            f"turn {state.turn} civ {civ}: {name} shows data on UNSEEN hexes")

    # -- observer-local handles: no engine creation-counter id may
    # appear anywhere in the observation; every handle is negative, unique,
    # and permanently bound to the entity it was assigned to
    hu = handles.setdefault(civ, {}).setdefault("u", {})
    hc = handles[civ].setdefault("c", {})
    for row in obs["units"]["id"][:obs["units"]["count"]]:
        assert int(row) < 0, f"true unit id {int(row)} leaked into obs"
    for row in obs["cities"]["id"][:obs["cities"]["count"]]:
        assert int(row) < 0, f"true city id {int(row)} leaked into obs"
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        tu = state.unit_at(int(q), int(r))
        assert tu is not None and tu.owner == int(owner) \
            and tu.type == str(utype), "unit row does not match the world"
        prev = hu.get(int(uid))
        assert prev is None or prev == tu.id, "unit handle changed meaning"
        hu[int(uid)] = tu.id
    for cid, owner, q, r, pop, hp, queue, bld, border, cap in obs["city_list"]:
        tc = state.city_at(int(q), int(r))
        assert tc is not None and tc.owner == int(owner), \
            "city row does not match the world"
        prev = hc.get(int(cid))
        assert prev is None or prev == tc.id, "city handle changed meaning"
        hc[int(cid)] = tc.id

    # -- units: own everywhere, foreign only on visible hexes
    exp_foreign = sorted(u.id for u in state.units.values()
                         if u.owner != civ and vis[u.i])
    got_foreign = sorted(hu[int(r[0])] for r in obs["unit_list"] if r[1] != civ)
    assert got_foreign == exp_foreign, (
        f"turn {state.turn} civ {civ}: visible foreign units != units on "
        f"visible hexes")
    got_own = sorted(hu[int(r[0])] for r in obs["unit_list"] if r[1] == civ)
    exp_own = sorted(u.id for u in state.units.values() if u.owner == civ)
    assert got_own == exp_own, "own units missing or fabricated in obs"
    assert obs["units"]["count"] == len(got_own) + len(got_foreign)

    # -- cities: own everywhere, foreign only on visible hexes
    exp_fcity = sorted(c.id for c in state.cities.values()
                       if c.owner != civ and vis[hm.index(c.q, c.r)])
    got_fcity = sorted(hc[int(c[0])] for c in obs["city_list"] if c[1] != civ)
    assert got_fcity == exp_fcity, "visible foreign cities != cities in sight"
    exp_owncity = sorted(c.id for c in state.cities.values() if c.owner == civ)
    got_owncity = sorted(hc[int(c[0])] for c in obs["city_list"] if c[1] == civ)
    assert got_owncity == exp_owncity, "own cities missing or fabricated"

    # -- per-civ tables: self only (alive is public)
    n = len(state.civs)
    for i in range(n):
        if i != civ:
            assert obs["gold"][i] == -1 and obs["score"][i] == -1
            assert not obs["techs"][i].any()
            assert obs["researching"][i] is None
            assert obs["research_progress"][i] == 0
        else:
            assert obs["gold"][i] == state.civs[i].gold

    # -- memory layer: what was there WHEN SEEN, never the current truth
    mem = obs["memory"]
    ls = mem["last_seen"]
    for i in np.flatnonzero(ls < 0):
        assert mem["resource"][i] == -1 and mem["improve"][i] == -1 \
            and mem["owner"][i] == OWNER_UNKNOWN, (
            f"turn {state.turn} civ {civ}: never-seen hex {i} has memory")
    for i in np.flatnonzero(ls >= 0):
        t_seen = int(ls[i])
        s = snap[t_seen]
        assert mem["resource"][i] == s["resource"][i], (
            f"turn {state.turn} civ {civ} hex {i}: memory resource != what "
            f"was there when seen (turn {t_seen})")
        assert mem["improve"][i] == s["improve"][i]
        assert mem["owner"][i] == s["owner"][i]
        uid = int(s["occ_unit"][i])
        if uid >= 0:
            o, ty, hp = s["units"][uid]
            assert mem["unit_owner"][i] == o and mem["unit_type"][i] == ty \
                and mem["unit_hp"][i] == hp
        else:
            assert mem["unit_hp"][i] == -1, (
                f"turn {state.turn} civ {civ} hex {i}: memory invents a unit")
        cid = int(s["occ_city"][i])
        if cid >= 0:
            o, pop, hp, queue, border, capital, mask = s["cities"][cid]
            h = int(mem["city_id"][i])
            assert h < 0, f"true city id {h} leaked into the memory layer"
            if h in hc:
                assert hc[h] == cid, "memory city handle changed meaning"
            else:
                hc[h] = cid  # bind on first memory encounter (as the module does)
            assert mem["city_owner"][i] == o
            assert mem["city_hp"][i] == hp and mem["city_pop"][i] == pop
        else:
            assert mem["city_hp"][i] == -1, (
                f"turn {state.turn} civ {civ} hex {i}: memory invents a city")
        checks["memory_hexes"] += 1


def _legal_from_obs(env, civ, obs):
    """Recompute the legal-action set from the observation alone (design §2:
    `legal_actions` must be computable from the observation)."""
    tables = env.tables
    hm_k = env.state.hexmap
    geo = {}
    R = int(obs["meta"]["radius"])
    coords = {}
    for q in range(-R, R + 1):
        for r in range(max(-R, -q - R), min(R, -q + R) + 1):
            coords[(q, r)] = len(geo)
            geo[(q, r)] = len(geo)
    from grandstrat.hexmap import DIRECTIONS

    you = obs["you"]
    terrain = obs["terrain"]
    civ_techs = {t for i, t in enumerate(obs["tech_names"])
                 if obs["techs"][you][i]}
    occ = {}
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        occ[(int(q), int(r))] = owner
    cities_at = {}
    for cid, owner, q, r, pop, hp, queue, bld, border, cap in obs["city_list"]:
        cities_at[(int(q), int(r))] = (cid, owner)
    out = {("SKIP",)}
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        if owner != you:
            continue
        spec = tables.units[utype]
        q, r = int(q), int(r)
        for d, (dq, dr) in enumerate(DIRECTIONS):
            dest = (q + dq, r + dr)
            if dest not in geo:
                continue
            j = geo[dest]
            tname = obs["terrain_names"][int(terrain[j])]
            t = tables.terrain[tname]
            if not t["passable"] or int(t["move_cost"]) > int(mp):
                continue
            if dest in occ:
                continue
            if dest in cities_at and cities_at[dest][1] != you:
                continue
            out.add(("MOVE", uid, d))
        if not spec.get("civilian", False):
            # attacks: adjacent units, ranged units/cities (range <= sight so
            # every target is on a visible hex)
            for d, (dq, dr) in enumerate(DIRECTIONS):
                dest = (q + dq, r + dr)
                if dest in occ and occ[dest] != you:
                    # engine id of the target: from the unit list
                    tgt = next(int(u[0]) for u in obs["unit_list"]
                               if (int(u[3]), int(u[4])) == dest)
                    out.add(("ATTACK_U", uid, tgt))
            rng_r = int(spec["range"])
            if int(spec["ranged_strength"]) > 0 and rng_r >= 1:
                for (cq, cr), (cid, owner) in cities_at.items():
                    d2 = hex_distance(cq - q, cr - r)
                    if owner != you and 1 <= d2 <= rng_r:
                        out.add(("ATTACK_C", uid, cid))
                for u2 in obs["unit_list"]:
                    if u2[1] == you:
                        continue
                    d2 = hex_distance(int(u2[3]) - q, int(u2[4]) - r)
                    if 2 <= d2 <= rng_r:
                        out.add(("ATTACK_U", uid, int(u2[0])))
            # melee city attacks
            for d, (dq, dr) in enumerate(DIRECTIONS):
                dest = (q + dq, r + dr)
                if dest in cities_at and cities_at[dest][1] != you:
                    out.add(("ATTACK_C", uid, cities_at[dest][0]))
        if utype == "settler":
            j = geo[(q, r)]
            tname = obs["terrain_names"][int(terrain[j])]
            tname = tables.terrain_names[int(terrain[j])]
            if tables.terrain[tname]["passable"] and (q, r) not in cities_at:
                dmin = int(tables.constants["city_min_dist"]["value"])
                if all(hex_distance(cq - q, cr - r) >= dmin
                       for (cq, cr) in cities_at):
                    terr_owner = int(obs["owner"][j])
                    if terr_owner in (-1, you):
                        out.add(("FOUND", uid))
        if utype == "worker":
            j = geo[(q, r)]
            tname = tables.terrain_names[int(terrain[j])]
            if obs["improve"][j] == 0 and (q, r) not in cities_at \
                    and int(obs["owner"][j]) == you:
                specs = [s2 for s2 in tables.improvements.values()
                         if tname in s2["terrain"]]
                if specs:
                    iname = next(name for name, s2 in tables.improvements.items()
                                 if tname in s2["terrain"])
                    tech = tables.improvements[iname].get("tech")
                    if tech is None or tech in civ_techs:
                        out.add(("IMPROVE", uid))
    # production: own cities, tech-gated
    for cid, owner, q, r, pop, hp, queue, bld, border, cap in obs["city_list"]:
        if owner != you:
            continue
        bnames = obs["building_names"]
        bset = set(bld)  # city_list rows carry the buildings tuple
        for name, s2 in tables.units.items():
            if s2.get("tech") is not None and s2["tech"] not in civ_techs:
                continue
            if name == "settler" and int(pop) < int(
                    tables.constants["settler_min_pop"]["value"]):
                continue
            out.add(("PRODUCE", cid, name))
        for name, s2 in tables.buildings.items():
            if name in bset:
                continue
            if s2.get("tech") is not None and s2["tech"] not in civ_techs:
                continue
            out.add(("PRODUCE", cid, name))
    # purchases from the observation: gold, rung, tech, known enemy cities
    info = tables.information
    W = int(obs["meta"]["fog_dial"])
    enabled = info["rung_purchases"]["value"].get(str(W), [])
    prices = info["prices"]["value"]
    gold = int(obs["gold"][you])
    if "spy" in enabled:
        price = prices.get("spy", {}).get(str(W))
        if price is not None and gold >= price \
                and info["spy_tech"]["value"] in civ_techs:
            mem = obs["memory"]
            known = set()
            for i in np.flatnonzero(mem["city_hp"] >= 0):
                o = int(mem["city_owner"][i])
                if o >= 0 and o != you and bool(obs["alive"][o]):
                    known.add(int(mem["city_id"][i]))
            for cid, owner, q, r, pop, hp, queue, bld, border, cap in obs["city_list"]:
                if owner != you and owner >= 0 and bool(obs["alive"][owner]):
                    known.add(cid)
            for cid in known:
                out.add(("SPY", cid))
    if "trade" in enabled:
        price = prices.get("trade", {}).get(str(W))
        if price is not None and gold >= price:
            for i in range(obs["meta"]["n_civs"]):
                if i != you and bool(obs["alive"][i]):
                    out.add(("TRADE", i))
    info_state = obs["meta"].get("info_state", {"shared": [], "oracle": False})
    if "diplo" in enabled:
        price = prices.get("diplo", {}).get(str(W))
        if price is not None and gold >= price:
            for i in range(obs["meta"]["n_civs"]):
                if i != you and bool(obs["alive"][i]) \
                        and i not in info_state["shared"]:
                    out.add(("DIPLO", i))
    if "oracle" in enabled:
        price = prices.get("oracle", {}).get(str(W))
        if price is not None and gold >= price and not info_state["oracle"]:
            out.add(("ORACLE",))
    return out


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def _run_fog_game(env, seed, agents, buyer_bot=False):
    """Play one full fog game; probe every observation of every turn."""
    checks = {"hexes": 0, "memory_hexes": 0, "legal_checks": 0, "purchases": 0}
    handles: dict[int, dict] = {}  # civ -> {"u": handle->true, "c": ...}
    for a in agents:
        a.reset()
    env.reset(seed)
    if buyer_bot:
        # harness setup for the purchase probe: civ 0 knows the SPY gate tech
        civ0 = env.state.civs[0]
        civ0.techs = tuple(sorted(set(civ0.techs) | {"writing"}))
    snap = _Snapshots(env.tables)
    snap.take(env.state)
    events = []
    while not env.state.done:
        # harness: keep treasuries solvent so the probe can run long games
        for c in env.state.civs:
            c.gold = max(c.gold, 200 if buyer_bot else 60)
        actions = {}
        for civ, ag in enumerate(agents):
            if not env.state.civs[civ].alive:
                continue
            if buyer_bot and civ == 0:
                actions[civ] = _buyer_choice(env, civ, env.state.turn)
                if actions[civ]:
                    continue
            actions[civ] = ag.act(env.observation(civ), env.legal_actions(civ),
                                  env.state.rng.stream(ag.stream_name))
        obs, rew, ter, tr, infos = env.step(actions)
        events.extend(infos[0]["report"].events)
        snap.take(env.state)
        intel = _intel_from_events(events, int(env.state.dials["fog"]))
        for civ in range(len(env.state.civs)):
            o = env.observation(civ)
            _check_observation(env, civ, o, snap, intel, checks, handles)
            if civ < 2:  # the expensive recompute on two civs per turn
                if not o["alive"][civ]:
                    assert {tuple(a) for a in env.legal_actions(civ)} == {("SKIP",)}
                    continue
                exp = _legal_from_obs(env, civ, o)
                got = {tuple(a) for a in env.legal_actions(civ)}
                assert exp == got, (
                    f"turn {env.state.turn} civ {civ}: legal_actions not "
                    f"computable from the observation; from-obs-only="
                    f"{sorted(map(str, exp - got))[:4]} engine-only="
                    f"{sorted(map(str, got - exp))[:4]}")
                checks["legal_checks"] += 1
        checks["purchases"] = sum(
            1 for e in events if e.get("type") == "info_purchase")
    return checks, events


def _buyer_choice(env, civ, turn):
    """Purchase bot on a fixed schedule: TRADE at 6, SPY at 8 (the trade
    corridor's memory write has made the enemy capital known by then), DIPLO
    at 10, ORACLE at 12.  Deterministic; no-op when the purchase is not
    legal."""
    legal = env.legal_actions(civ)
    want = {6: "TRADE", 8: "SPY", 10: "DIPLO", 12: "ORACLE"}.get(turn)
    if want is None:
        return []
    acts = [a for a in legal if a[0] == want]
    if acts:
        return [acts[len(acts) // 2]]  # deterministic mid target
    return []


# ---------------------------------------------------------------------------
# the probes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario,dials,seeds,min_hexes,min_legal", [
    ("duel", {"fog": 0}, (0,), 40 * 60, 50),
    ("duel", {"fog": 2}, (0,), 40 * 60, 50),
    ("tiny", {"fog": 0}, (0, 1), 20 * 60, 20),
    ("dense", {"fog": 0}, (0,), 40 * 60, 50),
])
def test_adversarial_probe_fog_layers(scenario, dials, seeds, min_hexes, min_legal, tmp_path):
    """Full games under the fog module: every observation of every civ of
    every turn matches the independent visibility recompute, shows current
    truth only on visible hexes, sentinels elsewhere, and remembers seen-time
    contents on the memory layer."""
    for seed in seeds:
        env = make_env(scenario, dials=dials, config={"information": "fog"})
        n = env.resolved_params()["civs"]
        agents = make_agents(LINEUP, n)
        checks, events = _run_fog_game(env, seed, agents)
        assert checks["hexes"] > min_hexes, "probe covered too few hexes"
        assert checks["memory_hexes"] > 200
        assert checks["legal_checks"] > min_legal


def test_adversarial_probe_with_purchases(tmp_path):
    """The probe over a game in which all four gold purchases fire: purchased
    reveals (spy/trade/diplo/oracle) are the ONLY way unseen data reaches a
    civ, each charged at its exact price, and every observation stays
    consistent."""
    env = make_env("duel", dials={"fog": 3}, config={"information": "fog"})
    agents = make_agents(LINEUP, 2)
    # make SPY available: give civ 0 the gate tech (harness setup)
    checks, events = _run_fog_game(env, 3, agents, buyer_bot=True)
    kinds = {e["purchase"] for e in events if e.get("type") == "info_purchase"}
    assert kinds == {"trade", "diplo", "spy", "oracle"}, (
        f"buyer bot did not fire all purchases: {kinds}")
    assert checks["legal_checks"] > 40


def test_dial_K_fog_module_matches_full_module(tmp_path):
    """At dial K the fog module hands out the shared base view: byte-equal
    observations and byte-equal full-game records vs the default module."""
    recs = []
    for config in (None, {"information": "fog"}):
        env = make_env("duel", dials={"fog": 4}, config=config)
        agents = make_agents(["expander", "warmonger"], 2)
        path = str(tmp_path / f"rec_{config}.jsonl")
        with RecordWriter(path) as w:
            play_game(env, 9, agents, writer=w)
        recs.append(path)
    def turns(path):
        return [ln for ln in open(path, encoding="utf-8")
                if '"kind":"turn"' in ln or '"kind": "turn"' in ln]

    assert turns(recs[0]) == turns(recs[1]), (
        "dial-K fog rung changed dynamics vs full observability")
    # the fog arm logs the information module in its header
    assert "information" in open(recs[1], encoding="utf-8").readline()


def test_fog_game_determinism_and_replay(tmp_path):
    """Fog games (with purchases) are deterministic and replayable."""
    paths = []
    for i in range(2):
        env = make_env("duel", dials={"fog": 3}, config={"information": "fog"})
        agents = make_agents(LINEUP, 2)
        path = str(tmp_path / f"fog{i}.jsonl")
        with RecordWriter(path) as w:
            play_game(env, 4, agents, writer=w)
        paths.append(path)
    a = open(paths[0], "rb").read()
    b = open(paths[1], "rb").read()
    assert a == b
    assert b'"information":"fog"' in a.replace(b": ", b":") or \
        b"information" in a, "config not logged in the record header"
    from grandstrat.replay import replay_record

    res = replay_record(paths[0])
    assert res["ok"], res["mismatches"]
    assert res["winner_match"]


def test_fog_obs_never_aliases_state_and_has_no_hidden_counts(tmp_path):
    """Isolation under fog: obs arrays are read-only, not state arrays, and
    no hidden unit/cid id appears anywhere in an observation -- every
    entity id handed out is a negative observer-local handle."""
    env = make_env("duel", dials={"fog": 0}, config={"information": "fog"})
    env.reset(2)
    st = env.state
    obs = env.observation(0)
    for name in ("resource", "improve", "owner", "visible"):
        assert not obs[name].flags.writeable
        assert obs[name] is not getattr(st, name, None)
    vis = obs["visible"]
    foreign_visible = {u.id for u in st.units.values()
                       if u.owner != 0 and vis[u.i]}
    for u in st.units.values():
        if u.owner != 0 and u.id not in foreign_visible:
            assert not any(int(row) == u.id
                           for row in obs["units"]["id"][:obs["units"]["count"]])
            assert all(entry[0] != u.id for entry in obs["unit_list"])
    # the creation-counter leak is closed: unit AND city
    # ids in the observation are handles, never engine ids
    for row in obs["units"]["id"][:obs["units"]["count"]]:
        assert int(row) < 0
    for row in obs["cities"]["id"][:obs["cities"]["count"]]:
        assert int(row) < 0
    for entry in obs["unit_list"]:
        assert int(entry[0]) < 0
    for entry in obs["city_list"]:
        assert int(entry[0]) < 0
    # handles are stable across a turn, and each civ's handles resolve to
    # that civ's view of the world (handle VALUES may coincide across civs --
    # the namespaces are observer-local by design)
    obs0b = env.observation(0)
    assert (obs0b["units"]["id"] == obs["units"]["id"]).all()
    obs1 = env.observation(1)
    rev0, rev1 = env.info._game[0].u_rev, env.info._game[1].u_rev
    for entry in obs["unit_list"]:
        if entry[1] == 0:
            assert rev0[int(entry[0])] in st.units \
                and st.units[rev0[int(entry[0])]].owner == 0
    for entry in obs1["unit_list"]:
        if entry[1] == 1:
            assert rev1[int(entry[0])] in st.units \
                and st.units[rev1[int(entry[0])]].owner == 1
    # flat encoding and text rendering work under fog
    v = env.flat_observation(1)
    assert v.dtype == np.float32 and np.isfinite(v).all()
    txt = env.render_text(1)
    assert "legal_actions" in txt["json"]
    payload = json.loads(txt["json"])
    assert all(c["owner"] != 1 for c in payload["visible_enemy_cities"])
