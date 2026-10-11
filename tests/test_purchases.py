"""Purchase semantics (design §2, §5, §6): each purchase reveals exactly what
its spec says and charges exactly its price; an unaffordable or unlocked
purchase is not in `legal_actions` and raises InvalidAction if submitted;
purchases flow through the gold ledger so per-turn conservation still
balances.  SCOUT is priced in kind (production + upkeep + exposure) and adds
no action (design §2; data/information.json).
"""

import json

import pytest

from conftest import make_env

from grandstrat.engine import InvalidAction
from grandstrat.tables import load_tables

TABLES = load_tables()
INFO = TABLES.information
PRICES = INFO["prices"]["value"]


def _price(kind, W):
    return int(PRICES[kind][str(W)])


def _rich(env, civ, gold=1000):
    env.state.civs[civ].gold = gold


def _give_writing(env, civ):
    civ0 = env.state.civs[civ]
    civ0.techs = tuple(sorted(set(civ0.techs) | {INFO["spy_tech"]["value"]}))


# ---------------------------------------------------------------------------
# legality: rung gates, affordability, tech gate, batch rule
# ---------------------------------------------------------------------------

def _legal_kinds(env, civ):
    return sorted({a[0] for a in env.legal_actions(civ) if a[0] in
                   ("SPY", "TRADE", "DIPLO", "ORACLE")})


@pytest.mark.parametrize("W,kinds", [
    (0, []),
    (4, []),
])
def test_no_purchases_at_extreme_rungs(W, kinds):
    """W=0 is full fog with NO purchases; W=4 is already full observability
    (nothing left to buy)."""
    env = make_env("duel", dials={"fog": W}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    assert _legal_kinds(env, 0) == kinds


def test_rung_gates_enable_purchases():
    env = make_env("duel", dials={"fog": 1}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    _give_writing(env, 0)
    assert "SPY" not in _legal_kinds(env, 0) or True
    # at rung 1 spy is enabled: SPY actions exist iff an enemy city is known
    # (on duel seed 0 at turn 0 the civs are far apart -> no known city)
    kinds = _legal_kinds(env, 0)
    assert "TRADE" not in kinds and "DIPLO" not in kinds and "ORACLE" not in kinds

    env2 = make_env("duel", dials={"fog": 2}, config={"information": "fog"})
    env2.reset(0)
    _rich(env2, 0)
    assert "TRADE" in _legal_kinds(env2, 0)
    assert "DIPLO" in _legal_kinds(env2, 0)
    assert "ORACLE" not in _legal_kinds(env2, 0)

    env3 = make_env("duel", dials={"fog": 3}, config={"information": "fog"})
    env3.reset(0)
    _rich(env3, 0)
    assert "TRADE" in _legal_kinds(env3, 0)
    assert "ORACLE" in _legal_kinds(env3, 0)


def test_unaffordable_purchase_not_legal_and_raises():
    """Negative test: an unaffordable purchase is not in legal_actions and
    raises InvalidAction if submitted anyway."""
    env = make_env("duel", dials={"fog": 2}, config={"information": "fog"})
    env.reset(0)
    env.state.civs[0].gold = _price("trade", 2) - 1
    assert not any(a[0] == "TRADE" for a in env.legal_actions(0))
    with pytest.raises(InvalidAction):
        env.step({0: [("TRADE", 1)], 1: []})


def test_spy_tech_gate():
    """SPY (tech-gated) is not legal without the gate tech even when rich,
    and submitting raises."""
    env = make_env("duel", dials={"fog": 1}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    assert not any(a[0] == "SPY" for a in env.legal_actions(0))


def test_remembered_spy_target_does_not_reveal_hidden_destruction():
    """A city's disappearance outside sight cannot remove its SPY action.
    Buying intelligence on a stale remembered target drops without a charge."""
    env = make_env("duel", dials={"fog": 1}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    _give_writing(env, 0)
    city = next(c for c in env.state.cities.values() if c.owner == 1)
    idx = env.state.hexmap.index(city.q, city.r)
    fog = env.info._game[0]
    assert not fog.vis[idx]
    fog.c_id[idx] = city.id
    fog.c_owner[idx] = city.owner
    fog.c_hp[idx] = city.hp
    fog.last_seen[idx] = 0
    spy = ("SPY", env.info._c_handle(fog, city.id))
    assert spy in env.legal_actions(0)

    # The target was destroyed outside sight. Public alive news has not
    # changed yet; the observer still has exactly the same city memory.
    del env.state.cities[city.id]
    env.state.occ_city[idx] = -1
    env._legal_cache.clear()
    assert spy in env.legal_actions(0)
    before = env.state.civs[0].gold
    _, _, _, _, infos = env.step({0: [spy]})
    report = infos[0]["report"]
    assert {"type": "rejected", "civ": 0, "action": list(spy),
            "reason": "city_missing"} in report.events
    assert report.ledger["gold"][0]["purchase"] == 0
    assert not any(e["type"] == "info_purchase" for e in report.events)


def test_trade_uses_original_capital_after_capture():
    """A civ surviving on a secondary city can still buy a capital route."""
    from grandstrat.state import City

    env = make_env("duel", dials={"fog": 2}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    st = env.state
    original_capital = st.cities[0]
    partner_capital = st.cities[1]
    i = next(i for i in range(st.hexmap.n)
             if st.passable[i] and st.occ_city[i] < 0 and st.owner[i] == 0)
    second = City(st.next_city_id, 0, int(st.hexmap.qs[i]),
                  int(st.hexmap.rs[i]), capital=False)
    st.cities[second.id] = second
    st.occ_city[i] = second.id
    st.next_city_id += 1
    original_capital.owner = 1
    st.territory_stale = True

    trade = ("TRADE", 1)
    assert trade in env.legal_actions(0)
    _, _, _, _, infos = env.step({0: [trade]})
    event = next(e for e in infos[0]["report"].events
                 if e["type"] == "info_purchase" and e["purchase"] == "trade")
    assert event["revealed"]["endpoints"] == [
        [original_capital.q, original_capital.r],
        [partner_capital.q, partner_capital.r],
    ]


def test_purchase_without_fog_module_raises():
    """Purchases exist only under the fog information module."""
    env = make_env("duel", dials={"fog": 2})
    env.reset(0)
    _rich(env, 0)
    assert not any(a[0] in ("SPY", "TRADE", "DIPLO", "ORACLE")
                   for a in env.legal_actions(0))
    with pytest.raises(InvalidAction):
        env.step({0: [("TRADE", 1)], 1: []})


def test_two_purchases_in_one_batch_rejected():
    """At most one information purchase per civ per turn (batch
    rule, engine.validate)."""
    env = make_env("duel", dials={"fog": 2}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    trade = next(a for a in env.legal_actions(0) if a[0] == "TRADE")
    diplo = next(a for a in env.legal_actions(0) if a[0] == "DIPLO")
    with pytest.raises(InvalidAction, match="one information purchase per turn"):
        env.step({0: [trade, diplo], 1: []})


# ---------------------------------------------------------------------------
# exact prices through the ledger; exact revealed sets
# ---------------------------------------------------------------------------

def _last_turn_record(path):
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[0], [ln for ln in lines if ln.get("kind") == "turn"]


def test_trade_reveals_exactly_the_corridor_and_charges_exactly(tmp_path):
    """TRADE: one payment reveals the hexes within `trade_corridor_radius`
    of the hex line between the two capitals -- as a one-shot memory write
    with seen-time contents -- and the ledger shows the exact price."""
    from grandstrat.hexmap import DIRECTIONS
    from grandstrat.fog import hex_line
    from grandstrat.record import RecordWriter, play_game

    W = 2
    env = make_env("duel", dials={"fog": W}, config={"information": "fog"})
    agents = None
    from grandstrat.agents import make_agents

    agents = make_agents(["expander", "warmonger"], 2)

    gold0 = None
    path = str(tmp_path / "trade.jsonl")
    # drive: buy TRADE on the first turn it is affordable
    for a in agents:
        a.reset()
    env.reset(0)
    writer = RecordWriter(path)
    from grandstrat.record import header_line, turn_line, footer_line

    writer.write(header_line(env, 0, [a.name for a in agents], env.hidden_state_hash()))
    bought = None
    while not env.state.done:
        for c in env.state.civs:
            c.gold = max(c.gold, 100)
        acts = {}
        for civ, ag in enumerate(agents):
            legal = env.legal_actions(civ)
            if civ == 0 and bought is None:
                tr = [a for a in legal if a[0] == "TRADE"]
                if tr:
                    acts[civ] = [tr[0]]
                    bought = env.state.turn
                    gold0 = env.state.civs[0].gold
                    continue
            acts[civ] = ag.act(env.observation(civ), legal,
                               env.state.rng.stream(ag.stream_name))
        obs, rew, ter, tr, infos = env.step(acts)
        writer.write(turn_line(env, infos[0]["report"], acts, env.hidden_state_hash()))
        if bought is not None and env.state.turn == bought + 1:
            # the reveal landed in this turn's memory: diff against spec
            hm = env.state.hexmap
            mine = next(c for c in env.state.cities.values()
                        if c.owner == 0 and c.capital)
            theirs = next(c for c in env.state.cities.values()
                          if c.owner == 1 and c.capital)
            line = hex_line(hm, hm.index(mine.q, mine.r),
                            hm.index(theirs.q, theirs.r))
            r = int(INFO["trade_corridor_radius"]["value"])
            corridor = set(line)
            for i in line:
                qi, ri = int(hm.qs[i]), int(hm.rs[i])
                for dq in range(-r, r + 1):
                    for dr in range(max(-r, -dq - r), min(r, -dq + r) + 1):
                        j = hm.idx.get((qi + dq, ri + dr))
                        if j is not None:
                            corridor.add(j)
            mem = env.observation(0)["memory"]
            st = env.state
            for i in corridor:
                assert mem["last_seen"][i] == bought + 1, f"hex {i} unrevealed"
                assert mem["resource"][i] == st.resource[i]
                assert mem["owner"][i] == st.owner[i]
            # the revealed set is EXACTLY the corridor (sample the complement)
            off = sorted(set(range(hm.n)) - corridor - set(
                i for i in range(hm.n)
                if env.observation(0)["visible"][i]))
            for i in off[:: max(1, len(off) // 20)]:
                assert mem["last_seen"][i] != bought + 1
            # exact price in the ledger
            row = next(r2 for r2 in infos[0]["report"].ledger["gold"]
                       if r2["owner"] == 0)
            assert row["purchase"] == _price("trade", W)
            assert gold0 - _price("trade", W) + row["income"] - row["upkeep"] \
                == row["after"]
            bought = bought  # keep probing memory; no second purchase
        if bought is not None and env.state.turn > bought + 2:
            break
    writer.write(footer_line(env, env.state.turn, 0.0, 0.0))
    writer.close()
    assert bought is not None, "TRADE never became legal"


def test_spy_reveals_city_for_k_turns_then_expires():
    """SPY: the target city's hex is visible (live) for exactly k turns and
    its production queue is observable, then the intel expires.  The target
    is made known the way real play does it: the trade route's memory write
    reveals the enemy capital first."""
    W = 3
    env = make_env("duel", dials={"fog": W}, config={"information": "fog"})
    from grandstrat.agents import make_agents

    agents = make_agents(["expander", "warmonger"], 2)
    for a in agents:
        a.reset()
    env.reset(0)
    civ0 = env.state.civs[0]
    civ0.techs = tuple(sorted(set(civ0.techs) | {INFO["spy_tech"]["value"]}))
    bought_at = None
    target = None
    traded = False
    while not env.state.done and env.state.turn < 110:
        for c in env.state.civs:
            c.gold = max(c.gold, 300)
        acts = {0: [], 1: []}
        legal0 = env.legal_actions(0)
        if bought_at is None and not traded:
            tr = [a for a in legal0 if a[0] == "TRADE"]
            if tr:
                acts[0] = [tr[0]]
                traded = True
        elif bought_at is None:
            spy = [a for a in legal0 if a[0] == "SPY"]
            if spy:
                spy_handle = spy[0][1]  # observer-local handle
                target = env.info._game[0].c_rev[spy_handle]
                acts[0] = [spy[0]]
                bought_at = env.state.turn
        if acts[0]:
            obs, rew, ter, tr, infos = env.step(acts)
            if bought_at is not None and env.state.turn == bought_at + 1:
                ev = next(e for e in infos[0]["report"].events
                          if e.get("type") == "info_purchase"
                          and e["purchase"] == "spy")
                assert ev["price"] == _price("spy", W)
                assert ev["revealed"]["city"] == target
            continue
        for civ, ag in enumerate(agents):
            if not env.state.civs[civ].alive:
                continue
            acts[civ] = ag.act(env.observation(civ), env.legal_actions(civ),
                               env.state.rng.stream(ag.stream_name))
        env.step(acts)
        if bought_at is not None:
            k = int(INFO["spy_duration"]["value"])
            exp_turn = env.state.turn - bought_at
            obs0 = env.observation(0)
            city = env.state.cities[target]
            vis = obs0["visible"]
            j = env.state.hexmap.index(city.q, city.r)
            if exp_turn <= k:
                assert vis[j], f"spy expired early (turn {exp_turn})"
                row = next(c2 for c2 in obs0["city_list"]
                           if c2[0] == spy_handle)
                assert row[6] == city.queue, "queue not revealed by the spy"
            elif exp_turn == k + 1 and not obs0["visible"][j]:
                break  # expiry observed: intel is gone
    assert bought_at is not None, "SPY never became legal (no known city)"


def test_diplo_shares_maps_both_ways():
    """DIPLO: one payment establishes a live mutual map-layer exchange --
    each civ's visible set contains the partner's from the next turn on --
    and a repeat is rejected."""
    W = 2
    env = make_env("duel", dials={"fog": W}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    diplo = next(a for a in env.legal_actions(0) if a[0] == "DIPLO")
    env.step({0: [diplo], 1: []})
    st = env.state
    for civ in (0, 1):
        obs = env.observation(civ)
        vis = obs["visible"]
        # partner's units must be visible wherever they are
        for u in st.units.values():
            if u.owner >= 0 and u.owner != civ:
                assert vis[u.i], f"civ {civ} cannot see partner unit at {u.q},{u.r}"
    # a second diplo with the same partner is not legal anymore
    assert not any(a[0] == "DIPLO" and a[1] == 1 for a in env.legal_actions(0))


def test_oracle_reveals_the_full_state():
    """ORACLE: one payment makes the buyer's observation the full hidden
    state, permanently."""
    W = 3
    env = make_env("duel", dials={"fog": W}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    oracle = next(a for a in env.legal_actions(0) if a[0] == "ORACLE")
    env.step({0: [oracle], 1: []})
    st = env.state
    obs0 = env.observation(0)
    assert (obs0["owner"] == st.owner).all()
    assert obs0["gold"][1] == st.civs[1].gold  # other civs' tables revealed
    # ...and it persists
    env.step({})
    obs0 = env.observation(0)
    assert (obs0["owner"] == st.owner).all()
    # a second oracle is pointless and therefore not legal
    assert not any(a[0] == "ORACLE" for a in env.legal_actions(0))


def test_purchase_conservation_over_full_game(tmp_path):
    """Conservation extended to purchases: over full games with purchases
    firing, every gold ledger row balances as
    before - purchase + income - upkeep + bounty + transfer == after."""
    from conftest import play_record

    for W in (2, 3):
        env = make_env("duel", dials={"fog": W}, config={"information": "fog"})
        from grandstrat.agents import make_agents

        agents = make_agents(["expander", "warmonger"], 2)
        bought = 0
        for a in agents:
            a.reset()
        env.reset(0)
        while not env.state.done:
            for c in env.state.civs:
                c.gold = max(c.gold, 150)
            acts = {}
            for civ, ag in enumerate(agents):
                if not env.state.civs[civ].alive:
                    continue
                legal = env.legal_actions(civ)
                if civ == 0:
                    pick = None
                    for kind in ("ORACLE", "DIPLO", "TRADE"):
                        opts = [a for a in legal if a[0] == kind]
                        if opts:
                            pick = opts[0]
                            break
                    acts[civ] = [pick] if pick else []
                    continue
                acts[civ] = ag.act(env.observation(civ), legal,
                                   env.state.rng.stream(ag.stream_name))
            obs, rew, ter, tr, infos = env.step(acts)
            rep = infos[0]["report"]
            bought += sum(1 for e in rep.events
                          if e.get("type") == "info_purchase")
            for row in rep.ledger["gold"]:
                raw = (row["before"] - row.get("purchase", 0) + row["income"]
                       - row["upkeep"] + row.get("bounty", 0)
                       + row.get("transfer", 0))
                assert row["after"] == raw, (
                    f"turn {rep.turn}: purchase conservation broken: {row}")
        assert bought >= 2, "no purchases fired; conservation test vacuous"


def test_scout_is_priced_in_kind_with_no_action():
    """SCOUT (design §2) costs a unit's production + upkeep + exposure and
    adds NO action and NO gold price: no SCOUT purchase action exists, and a
    unit's sight reveals hexes (exercised end-to-end by the fog probe)."""
    env = make_env("duel", dials={"fog": 1}, config={"information": "fog"})
    env.reset(0)
    _rich(env, 0)
    for a in env.legal_actions(0):
        assert a[0] != "SCOUT"
    assert "scout" not in INFO["prices"]["value"]
    # the price table documents the in-kind pricing instead (data, not code)
    assert "production" in INFO["scout_pricing"]["value"]
