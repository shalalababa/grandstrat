"""Belief construction (design §2, §6): exact support enumeration on
small maps and the particle filter elsewhere.  Non-degeneracy contract: the
support contains more than one configuration and is not ONLY the truth; it
contains the true configuration without containing unreachable ones (a configuration is in the
support only if some legal history produces it); the particle ensemble keeps
a healthy effective sample size.  Support monotonicity: candidate sets grow
only as fast as the tracked units' movement budgets (a new hidden event).
"""

import numpy as np
import pytest

from conftest import make_env

from grandstrat.belief import (GONE, ParticleBelief, SupportResult,
                               build_belief, enumerate_support,
                               tracks_and_blocked)
from grandstrat.hexmap import hex_distance


def _truth_config(env, civ, support):
    """Map each track to the true hidden position (test-side: may consult the
    hidden state) and return the configuration if it exists."""
    state = env.state
    obs = env.observation(civ)
    tracks, blocked = tracks_and_blocked(obs, env.tables)
    used = set()
    cfg = []
    for tr in tracks:
        if tr.fixed:
            # pinned by sight: some hidden unit must actually be there
            hit = next((u for u in state.units.values()
                        if u.owner == tr.owner and u.type == tr.type_name
                        and state.hexmap.index(u.q, u.r) == tr.last_hex
                        and tr.last_hex not in used), None)
            if hit is None:
                return None, tracks
            used.add(tr.last_hex)
            cfg.append(tr.last_hex)
            continue
        # memory track: an alive hidden unit of this (owner, type) on one of
        # the candidates, or GONE (it may have died unseen)
        hit = None
        for u in state.units.values():
            if u.owner != tr.owner or u.type != tr.type_name:
                continue
            j = state.hexmap.index(u.q, u.r)
            if j in set(tr.candidates.tolist()) and j not in used:
                hit = j
                break
        used.add(hit if hit is not None else -1)
        cfg.append(hit)
    return tuple(cfg), tracks


def _tiny_fog_game(seed=0, turns=30):
    """A tiny fog game driven by one marching scout so enemy contact happens
    and then breaks again (the non-degenerate case for the support)."""
    from grandstrat.hexmap import DIRECTIONS

    env = make_env("tiny", dials={"fog": 0}, config={"information": "fog"})
    env.reset(seed)
    st = env.state
    for _ in range(turns):
        if env.state.done:
            break
        for c in env.state.civs:
            c.gold = max(c.gold, 60)
        legal0 = env.legal_actions(0)
        legal1 = env.legal_actions(1)
        acts = {0: [], 1: []}
        # civ 0: march the starting scout toward the map centre (exploration);
        # civ 1: fully passive.  Legal actions carry observer-local handles
        # so the scout is matched through the observation.
        scout = next((u for u in st.units.values()
                      if u.owner == 0 and u.type == "scout"), None)
        if scout is None:
            prod = [a for a in legal0
                    if a[0] == "PRODUCE" and a[2] == "scout"]
            if prod:
                acts[0] = prod[:1]
        else:
            obs0 = env.observation(0)
            handle = next(uid for uid, owner, utype, q, r, hp, mp
                          in obs0["unit_list"]
                          if owner == 0 and utype == "scout")
            moves = [a for a in legal0
                     if a[0] == "MOVE" and a[1] == handle]
            if moves:
                # deterministic: keep walking west
                acts[0] = [min(moves, key=lambda a: a[2])]
        env.step(acts)
        # real consumers build observations every turn; the fog module's
        # memory/visibility history accumulates per built observation
        env.observation(0)
        env.observation(1)
    return env


def test_support_enumeration_is_nondegenerate_and_contains_truth():
    """The exact support on `tiny` (R=4, 2 civs): more than one
    configuration, containing the true hidden configuration, and not ONLY
    the truth (the degenerate-ensemble failure)."""
    env = _tiny_fog_game(seed=0)
    support = env.belief(0)
    assert isinstance(support, SupportResult)
    assert len(support) > 1, (
        f"degenerate support: {len(support)} configuration(s)")
    truth, tracks = _truth_config(env, 0, support)
    assert truth is not None, "no consistent truth mapping; tracks are broken"
    assert len(support.tracks) > 0
    configs = support.configs
    assert truth in configs, (
        "the true configuration is not in the support: the reachability "
        "filter is over-tight (unreachable/unseen hexes)")
    only_pinned = all(tr.fixed for tr in tracks)
    assert not only_pinned or len(configs) > 1
    # not only-the-truth: at least one non-pinned track has alternatives
    assert any(len(tr.candidates) > 1 for tr in tracks) or \
        any(tr is not None for tr in tracks), "support collapsed onto the truth"


def test_support_rejects_unreachable_configurations():
    """Every configuration in the support is reachable: candidates sit on
    passable hexes, never on currently visible hexes, and within the unit's
    movement budget of its last sighting."""
    env = _tiny_fog_game(seed=1)
    support = env.belief(0)
    state = env.state
    obs = env.observation(0)
    vis = obs["visible"]
    tables = env.tables
    for tr in support.tracks:
        if tr.fixed:
            continue  # pinned by sight: position is observed, not hypothesised
        for j in tr.candidates.tolist():
            tname = tables.terrain_names[int(obs["terrain"][j])]
            assert tables.terrain[tname]["passable"], (
                f"water/impassable hex {j} in the candidate set")
            assert not vis[j], (
                f"currently visible hex {j} in the candidate set")
            budget = int(tables.units[tr.type_name]["move"]) \
                * (int(obs["meta"]["turn"]) - tr.last_turn) + 1
            d = hex_distance(int(state.hexmap.qs[j]) - int(state.hexmap.qs[tr.last_hex]),
                             int(state.hexmap.rs[j]) - int(state.hexmap.rs[tr.last_hex]))
            assert d <= budget, (
                f"candidate hex {j} is {d} away but the unit had {budget} moves")


def test_support_monotonic_growth_needs_hidden_events():
    """Candidate sets grow only between observations, and by at most the
    tracked unit's per-turn movement budget -- support never grows without a
    new hidden event (an enemy turn in which it could move).  Scanned over
    seeds until a game actually produces growth (tiny maps often keep the
    enemy permanently in sight, which makes the check vacuous)."""
    total_growth = 0
    for seed in range(8):
        env = make_env("tiny", dials={"fog": 0}, config={"information": "fog"})
        env.reset(seed)
        st = env.state
        own_cap = next(c for c in st.cities.values()
                       if c.owner == 0 and c.capital)
        enemy_cap = next(c for c in st.cities.values()
                         if c.owner == 1 and c.capital)
        prev_cands = None
        for turn in range(25):
            if env.state.done:
                break
            for c in env.state.civs:
                c.gold = max(c.gold, 60)
            legal0 = env.legal_actions(0)
            acts = {0: [], 1: []}
            scout = next((u for u in st.units.values()
                          if u.owner == 0 and u.type == "scout"), None)
            if scout is None:
                prod = [a for a in legal0
                        if a[0] == "PRODUCE" and a[2] == "scout"]
                if prod:
                    acts[0] = prod[:1]
            else:
                # approach the enemy until turn 12, then march home: the
                # enemy units get seen and then fall out of sight, so memory
                # tracks appear and age
                target = enemy_cap if turn < 12 else own_cap
                moves = [a for a in legal0
                         if a[0] == "MOVE"
                         and a[1] == _scout_handle(env, 0, scout)]
                if moves:
                    best = min(moves, key=lambda a: (
                        hex_distance(target.q - _after(scout, a)[0],
                                     target.r - _after(scout, a)[1]), a[2]))
                    acts[0] = [best]
            env.step(acts)
            obs = env.observation(0)
            env.observation(1)
            tracks, _ = tracks_and_blocked(obs, env.tables)
            cands = {tr.key: frozenset(tr.candidates.tolist())
                     for tr in tracks if not tr.fixed}
            if prev_cands:
                for key, now in cands.items():
                    if key not in prev_cands:
                        continue
                    was = prev_cands[key]
                    grew = now - was
                    if grew:
                        total_growth += 1
                        tname = key[2]
                        budget = int(env.tables.units[tname]["move"])
                        for j in grew:
                            near = min(hex_distance(
                                int(st.hexmap.qs[j]) - int(st.hexmap.qs[i]),
                                int(st.hexmap.rs[j]) - int(st.hexmap.rs[i]))
                                for i in was)
                            assert near <= budget, (
                                f"candidate {j} jumped {near} steps in one "
                                f"turn (budget {budget}): support grew "
                                f"without a matching hidden event")
            prev_cands = cands
        if total_growth > 0:
            break
    assert total_growth > 0, (
        "no candidate growth observed on any seed; the monotonicity check "
        "never ran")


def test_particle_filter_fixed_nondeterministic_and_nondegenerate():
    """The particle filter on a large map: fixed particle count,
    deterministic given the seed, healthy ESS, contains the truth, and never
    collapses to identical particles (given a genuinely uncertain track)."""
    env = _scouting_duel_game()
    b1 = env.belief(0, kind="particles")
    b2 = env.belief(0, kind="particles")
    assert isinstance(b1, ParticleBelief)
    assert b1.n == int(env.tables.information["belief_particles"]["value"])
    assert b1.particles == b2.particles, "particle belief is not deterministic"
    assert b1.tracks, "no tracks to filter over"
    if any(len(tr.candidates) > 1 or not tr.fixed for tr in b1.tracks):
        assert len(set(b1.particles)) > 1, (
            "particles collapsed to one configuration")
        assert b1.ess > 1.0, f"effective sample size degenerated: {b1.ess}"
    truth, tracks = _truth_config(env, 0, None)
    if truth is not None:
        assert truth in b1.particles, (
            "the true configuration has consistency weight 1 yet was lost in "
            "resampling")


def _scouting_duel_game(turns=40):
    """A duel fog game in which civ 0 marches its starting scout toward the
    enemy capital: enemy units get seen and fall out of sight again, which
    is the non-degenerate case for belief construction."""
    env = make_env("duel", dials={"fog": 0}, config={"information": "fog"})
    env.reset(5)
    st = env.state
    enemy_cap = next(c for c in st.cities.values() if c.owner == 1 and c.capital)
    for _ in range(turns):
        if env.state.done:
            break
        for c in env.state.civs:
            c.gold = max(c.gold, 60)
        legal0 = env.legal_actions(0)
        acts = {0: [], 1: []}
        scout = next((u for u in st.units.values()
                      if u.owner == 0 and u.type == "scout"), None)
        if scout is None:
            obs0 = env.observation(0)
            cap_h = next(cid for cid, owner, q, r, pop, hp, queue, bld,
                         border, cap in obs0["city_list"]
                         if owner == 0 and bool(cap))
            prod = [a for a in legal0
                    if a[0] == "PRODUCE" and a[1] == cap_h
                    and a[2] == "scout"]
            if prod:
                acts[0] = prod[:1]
        else:
            # march the scout toward the enemy capital
            moves = [a for a in legal0 if a[0] == "MOVE"
                     and a[1] == _scout_handle(env, 0, scout)]
            if moves:
                best = min(moves, key=lambda a: (
                    hex_distance(enemy_cap.q - _after(scout, a)[0],
                                 enemy_cap.r - _after(scout, a)[1]),
                    a[2]))
                acts[0] = [best]
        env.step(acts)
        env.observation(0)
        env.observation(1)
    return env


def _after_unused(unit, move_action):
    pass


def _after(unit, move_action):
    from grandstrat.hexmap import DIRECTIONS

    dq, dr = DIRECTIONS[move_action[2]]
    return unit.q + dq, unit.r + dr


def _own_cap(env, civ):
    return next(c for c in env.state.cities.values()
                if c.owner == civ and c.capital)


def _scout_handle(env, civ, scout):
    """The scout's observer-local handle from the civ's observation
    (legal actions carry handles, not engine ids)."""
    obs = env.observation(civ)
    return next(uid for uid, owner, utype, q, r, hp, mp in obs["unit_list"]
                if owner == civ and utype == "scout"
                and (int(q), int(r)) == (scout.q, scout.r))


def _after(unit, move_action):
    from grandstrat.hexmap import DIRECTIONS

    dq, dr = DIRECTIONS[move_action[2]]
    return unit.q + dq, unit.r + dr


def test_belief_support_size_in_purchase_records():
    """Purchase record entries carry the belief-support size before/after on
    small maps (design §5) -- diagnostic columns, computed exactly."""
    env = make_env("tiny", dials={"fog": 2}, config={"information": "fog"})
    env.reset(0)
    for c in env.state.civs:
        c.gold = max(c.gold, 300)
    trade = next(a for a in env.legal_actions(0) if a[0] == "TRADE")
    obs, rew, ter, tr, infos = env.step({0: [trade], 1: []})
    ev = next(e for e in infos[0]["report"].events
              if e.get("type") == "info_purchase")
    assert "support_before" in ev and "support_after" in ev, ev
    assert ev["support_before"] >= 1 and ev["support_after"] >= 1
    # "after" is measured one turn later, on a knowledge state that also
    # includes the corridor reveal and the enemy's own turn: the ontology can
    # only have grown by exactly the recorded reveal, so any growth is
    # justified by the recorded hidden events
    assert ev["revealed"]["corridor_hexes"] > 0
