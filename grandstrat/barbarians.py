"""Barbarian subsystem: camps as neutral threats (design §1).

Camps are placed by worldgen.  Each turn:
  spawn -- per alive camp, Bernoulli(spawn_chance * adversary_strength dial
           * nonstationarity ramp) keyed ("spawn", turn, camp_id) on
           the `barbarians` stream; spawn position and unit type keyed on the
           same world identity.  The global unit_cap is respected before the
           roll (a capped world skips the spawn entirely).
  act   -- each barbarian unit (id order): attack an adjacent civ unit/city,
           else step toward the nearest civ unit/city within the aggression
           radius (greedy, tie by direction order), else hold.

Barbarians are owner -1, never found cities, never capture (combat module
clamps city hp at 1 for them), and respect no borders (the diplomacy
module gates only civ-owned movers).  All draws are world-keyed:
(turn, camp_id / unit_id, purpose).
"""

from __future__ import annotations

from .combat import CrnCombat
from .engine import (TurnReport, can_move_to, move_unit, spawn_unit,
                     terrain_move_cost, terrain_passable)
from .hexmap import NUM_DIRS
from .state import GameState


def barbarian_phase(state: GameState, report: TurnReport, combat: CrnCombat) -> None:
    K = state.tables.constants
    _spawns(state, report, K)
    _acts(state, report, combat, K)


def _spawns(state: GameState, report: TurnReport, K) -> None:
    chance = float(K["barb_spawn_chance"]["value"]) * float(state.dials["adversary_strength"])
    # nonstationarity dial (simple documented semantics): the world
    # grows linearly more dangerous toward the horizon,
    # chance *= 1 + nonstationarity * turn / T.  0 (the default) changes
    # nothing; the dial is data-driven and logged in every record header.
    nonstat = float(state.dials.get("nonstationarity", 0.0) or 0.0)
    if nonstat > 0.0:
        T = max(1, int(state.params["T"]))
        chance *= 1.0 + nonstat * (state.turn / T)
    if chance <= 0.0:
        return
    max_per = int(K["barb_max_per_camp"]["value"])
    radius = int(K["barb_spawn_radius"]["value"])
    unit_cap = int(K["unit_cap"]["value"])
    for camp in state.camps.values():
        if not camp.alive:
            continue
        if len(state.units) >= unit_cap:
            break  # the cap is respected before the roll, never overflowed
        ci = state.hexmap.index(camp.q, camp.r)
        nearby = sum(
            1 for u in state.units.values()
            if u.owner < 0
            and state.hexmap.distance(ci, state.hexmap.index(u.q, u.r)) <= radius + 1
        )
        if nearby >= max_per:
            continue
        if not state.rng.chance("barbarians", chance, "spawn", state.turn, camp.id):
            continue
        # spawn position: free passable land hex within `radius`, seeded choice
        cands = []
        for dq, dr in state.hexmap.ring_offsets(radius) if radius else []:
            j = state.hexmap.idx.get((camp.q + dq, camp.r + dr))
            if j is not None and terrain_passable(state, j) and int(state.occ_unit[j]) < 0:
                cands.append(j)
        if not cands:
            continue
        j = cands[state.rng.randint("barbarians", 0, len(cands) - 1,
                                    "pos", state.turn, camp.id)]
        horse = state.rng.chance("barbarians",
                                 float(K["barb_horseman_chance"]["value"]),
                                 "type", state.turn, camp.id)
        utype = "horseman" if horse else "warrior"
        u = spawn_unit(state, -1, utype, int(state.hexmap.qs[j]), int(state.hexmap.rs[j]),
                       birth_key=camp.id)
        report.events.append({
            "type": "barb_spawn", "camp": camp.id, "unit": u.id,
            "kind": utype, "q": u.q, "r": u.r, "wid": list(u.wid),
        })


def _nearest_target(state: GameState, u, radius: int):
    """Nearest civ unit or city within `radius` of the barbarian (tie: id).
    Inline arithmetic; hot on barbarian-heavy turns."""
    hm = state.hexmap
    i0 = u.i
    qi, ri = int(hm.qs[i0]), int(hm.rs[i0])
    best = None
    best_key = (1 << 30, 1 << 30)
    for t in state.units.values():
        if t.owner >= 0:
            dq = int(hm.qs[t.i]) - qi
            dr = int(hm.rs[t.i]) - ri
            d = (abs(dq) + abs(dq + dr) + abs(dr)) // 2
            if d <= radius and (d, t.id) < best_key:
                best, best_key = t, (d, t.id)
    for c in state.cities.values():
        ci = hm.index(c.q, c.r)
        dq = int(hm.qs[ci]) - qi
        dr = int(hm.rs[ci]) - ri
        d = (abs(dq) + abs(dq + dr) + abs(dr)) // 2
        if d <= radius and (d, c.id) < best_key:
            best, best_key = c, (d, c.id)
    return best


def _acts(state: GameState, report: TurnReport, combat, K) -> None:
    radius = int(K["barb_aggression_radius"]["value"])
    for u in list(state.units.values()):
        if u.owner >= 0 or u.id not in state.units:
            continue
        target = _nearest_target(state, u, radius)
        if target is None:
            continue
        hm = state.hexmap
        i0 = u.i
        ti = hm.index(target.q, target.r)
        d = hm.distance(i0, ti)
        if d == 1:
            if hasattr(target, "pop"):  # city
                combat.resolve_attack_city(state, report, u, target)
            else:
                combat.resolve_attack_unit(state, report, u, target)
            continue
        # greedy step toward target: neighbor minimizing hex distance,
        # tie by direction order; never onto cities.
        best_d, best_j, best_dir = d, -1, -1
        for dd in range(NUM_DIRS):
            j = int(hm.neighbors[i0, dd])
            if j < 0 or not can_move_to(state, u, j):
                continue
            if int(state.occ_city[j]) >= 0:
                continue
            dj = hm.distance(j, ti)
            if dj < best_d:
                best_d, best_j, best_dir = dj, j, dd
        if best_j >= 0:
            u.mp -= max(1, terrain_move_cost(state, best_j))
            u.acted = True
            move_unit(state, u, int(hm.qs[best_j]), int(hm.rs[best_j]))


def clear_camps(state: GameState, report: TurnReport) -> None:
    """A civ unit ending the turn alone on a camp hex clears it for a bounty.
    The bounty is added to civ gold AND to the gold ledger row so per-turn
    conservation arithmetic stays exact."""
    K = state.tables.constants
    for camp in state.camps.values():
        if not camp.alive:
            continue
        i = state.hexmap.index(camp.q, camp.r)
        uid = int(state.occ_unit[i])
        if uid < 0:
            continue
        u = state.units[uid]
        if u.owner < 0:
            continue
        camp.alive = False
        bounty = int(K["barb_clear_bounty"]["value"])
        state.civs[u.owner].gold += bounty
        rows = report.ledger.get("gold") or []
        for row in rows:
            if row["owner"] == u.owner:
                row["bounty"] = row.get("bounty", 0) + bounty
                row["after"] += bounty  # row was snapshotted before cleanup
        report.events.append({
            "type": "camp_cleared", "camp": camp.id, "by": u.owner,
            "bounty": bounty,
        })
