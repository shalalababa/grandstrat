"""Turn engine: phase list, action spec, legality, order resolution.

ACTION SPEC (schema gs.record/v1; actions are plain tuples, serialised as
JSON arrays in records):
    ("SKIP",)
    ("MOVE",   unit_id, d)              d in 0..5, see hexmap.DIRECTIONS
    ("ATTACK_U", unit_id, target_unit_id)
    ("ATTACK_C", unit_id, target_city_id)
    ("FOUND",  unit_id)                 settler founds a city here
    ("IMPROVE", unit_id)                worker builds the terrain improvement
    ("PRODUCE", city_id, item_name)     set the (single-item) production queue
    ("SPY",    target_city_id)          buy intel on one known enemy city
    ("TRADE",  partner_civ)             buy a reveal of the partner's map
    ("DIPLO",  partner_civ)             buy a mutual map-layer exchange
    ("ORACLE",)                         buy a full reveal of the hidden state
    ("OFFER",  partner, kind, gold)     bid `gold` for an agreement
                                        (diplomacy module only; kind in
                                        {"open_borders", "share_maps"})
    ("ACCEPT", proposal_id)             take a live offer addressed to us
    ("REJECT", proposal_id)             decline a live offer addressed to us

The four information purchases exist only when the fog information module is
active and the fog dial's rung enables them (data/information.json); they are
appended to `legal_actions` by the information module, never enumerated here.
The three diplomacy orders exist only when the diplomacy module is active
(config `diplomacy: "simple"`); they are appended by that module.

Entity ids in SUBMITTED actions are observer-local under the fog module:
the information module re-keys unit/city ids into per-civ handles
(`present_actions`) and maps them back before resolution (`resolve_action`).
Under the default full module the mapping is the identity.

Contract (design §6 "legal-action completeness"): `legal_actions` lists
exactly the actions that execute; `validate` (called up front by
env.step) rejects anything else with InvalidAction.  A submitted action that
was legal when validated but stale at resolution time (its unit died to an
earlier-resolved order in the same simultaneous turn) is DROPPED and recorded
as a {"type": "rejected", ...} event -- it never crashes and never silently
vanishes.

Turn phase list (deterministic; see docs/DESIGN_NOTES.md for the ordering choice):
    orders_produce, orders_move, orders_attack, orders_civilian,
    income, barbarians, cleanup
"""

from __future__ import annotations

import math

from .hexmap import NUM_DIRS
from .tables import Tables
from .state import City, GameState, Unit

# Resources: id -> name (observation encoding; 0 = none). Pinned API.
RESOURCE_NAMES = ("", "wheat", "iron", "gold_ore")


class InvalidAction(Exception):
    """Raised by validate/step when a submitted action is not legal."""


class TurnReport:
    """Everything the record needs from one resolved turn."""

    __slots__ = ("turn", "events", "rolls", "ledger", "scores", "gold",
                 "purchases", "transfers", "done", "winner", "done_reason")

    def __init__(self, turn: int):
        self.turn = turn
        self.events: list[dict] = []
        self.rolls: list[dict] = []
        self.ledger: dict = {}
        self.scores: list[int] = []
        self.gold: list[int] = []
        # civ id -> gold spent on information purchases this turn; written
        # into the ledger's gold rows as an explicit `purchase` column so
        # per-turn conservation still balances (design §5).
        self.purchases: dict[int, int] = {}
        # civ id -> NET gold received via diplomacy transfers this turn
        # (positive received, negative paid); shown as the ledger's
        # `transfer` column.
        self.transfers: dict[int, int] = {}
        self.done = False
        self.winner = None
        self.done_reason = None


# ---------------------------------------------------------------------------
# occupancy / unit lifecycle helpers (single writers of occ grids)
# ---------------------------------------------------------------------------

def refresh_occ(state: GameState) -> None:
    state.occ_unit.fill(-1)
    state.occ_city.fill(-1)
    hm = state.hexmap
    for u in state.units.values():
        state.occ_unit[hm.index(u.q, u.r)] = u.id
    for c in state.cities.values():
        state.occ_city[hm.index(c.q, c.r)] = c.id


def city_wid(state: GameState, city: City) -> int:
    """A city's world identity: its founding hex (design §1 "cities already
    have a world identity (founding hex); use it").  Eternal: capture changes
    ownership, never the founding hex."""
    return state.hexmap.index(city.q, city.r)


def spawn_unit(state: GameState, owner: int, utype: str, q: int, r: int,
               birth_key: int = -1) -> Unit:
    """Spawn a unit and pin its WORLD IDENTITY (docs/DESIGN_NOTES.md §13).

    `birth_key` is the birth city's world identity (its founding hex index)
    for production births, the camp id for barbarian births, and -1 for
    ad-hoc spawns.  The ordinal counts births with the same
    (owner, turn, birth_key) this game, so the full wid
    (owner, birth_turn, birth_key, ordinal) is path-independent: two arms
    that diverge in production order still give corresponding units the same
    wid, which is what combat CRN keys on.  The integer id stays the
    indexing handle only.

    unit_cap (enforced at the transition): this is the single choke point
    every birth in the game goes through; the cap is enforced HERE rather
    than only asserted in the observation builder, and callers with a soft
    policy (production holds its bucket, barbarians skip the spawn) pre-check
    `len(state.units)` before rolling."""
    if len(state.units) >= int(state.tables.constants["unit_cap"]["value"]):
        raise RuntimeError(
            f"unit_cap {state.tables.constants['unit_cap']['value']} reached; "
            f"spawn of {utype!r} for owner {owner} refused at the transition")
    spec = state.tables.units[utype]
    u = Unit(state.next_unit_id, owner, utype, q, r,
             int(state.tables.constants["unit_hp_max"]["value"]), int(spec["move"]))
    key = (owner, state.turn, birth_key)
    n = state.birth_counts.get(key, 0)
    state.birth_counts[key] = n + 1
    u.wid = (owner, state.turn, birth_key, n)
    u.i = state.hexmap.index(q, r)
    state.units[u.id] = u
    state.next_unit_id += 1
    state.occ_unit[u.i] = u.id
    return u


def remove_unit(state: GameState, u: Unit) -> None:
    del state.units[u.id]
    if u.i >= 0 and int(state.occ_unit[u.i]) == u.id:
        state.occ_unit[u.i] = -1


def move_unit(state: GameState, u: Unit, q: int, r: int) -> None:
    if u.i >= 0 and int(state.occ_unit[u.i]) == u.id:
        state.occ_unit[u.i] = -1
    u.q, u.r = q, r
    u.i = state.hexmap.index(q, r)
    state.occ_unit[u.i] = u.id


def terrain_passable(state: GameState, idx: int) -> bool:
    return bool(state.passable[idx])


def terrain_move_cost(state: GameState, idx: int) -> int:
    mc = int(state.move_cost[idx])
    return mc if mc > 0 else -1


def terrain_def_mod(state: GameState, idx: int) -> float:
    name = state.tables.terrain_names[state.terrain[idx]]
    return float(state.tables.terrain[name]["def_mod"])


# ---------------------------------------------------------------------------
# legality (shared by legal_actions and resolution re-checks)
# ---------------------------------------------------------------------------

def can_move_to(state: GameState, u: Unit, idx: int) -> bool:
    """idx is the destination hex index.  Hot path: array/occ lookups only."""
    if not state.passable[idx] or u.mp <= 0:
        return False
    if int(state.occ_unit[idx]) >= 0:
        return False
    cid = int(state.occ_city[idx])
    if cid >= 0:
        c = state.cities.get(cid)
        if c is not None and c.owner != u.owner:
            return False  # enemy cities are entered by capture, not movement
    return int(state.move_cost[idx]) <= u.mp


def can_found_city(state: GameState, u: Unit, owner_at: int | None = None) -> bool:
    """`owner_at`, when given, is the TERRITORY OWNER AS THE ACTING CIV KNOWS
    IT (diplomacy: legality is computable from the observation, and
    resolution re-checks the true owner).  None = the true owner."""
    if u.type != "settler":
        return False
    hm = state.hexmap
    i = hm.index(u.q, u.r)
    if not terrain_passable(state, i):
        return False
    if int(state.occ_city[i]) >= 0:
        return False
    dmin = state.tables.constants["city_min_dist"]["value"]
    for c in state.cities.values():
        if hm.distance(i, hm.index(c.q, c.r)) < dmin:
            return False
    terr_owner = int(state.owner[i]) if owner_at is None else owner_at
    return terr_owner in (-1, u.owner)


def improvement_for(state: GameState, idx: int):
    """The improvement buildable on this hex's terrain, or None (tech gate is
    checked separately: see can_improve)."""
    tname = state.tables.terrain_names[state.terrain[idx]]
    for name, spec in state.tables.improvements.items():
        if tname in spec["terrain"]:
            return name
    return None


def can_improve(state: GameState, u: Unit, owner_at: int | None = None) -> bool:
    """Tech-gated (the tech tree's `unlocks.improvements` data is enforced
    through the `tech` field in improvements.json: a worker cannot build a
    mine without mining).  `owner_at` is the territory
    owner as the acting civ KNOWS it (diplomacy module legality); None = the
    true owner."""
    if u.type != "worker":
        return False
    i = state.hexmap.index(u.q, u.r)
    if state.improve[i] != 0:
        return False
    if int(state.occ_city[i]) >= 0:
        return False
    iname = improvement_for(state, i)
    if iname is None:
        return False
    tech = state.tables.improvements[iname].get("tech")
    if tech is not None and tech not in state.civs[u.owner].techs:
        return False
    terr_owner = int(state.owner[i]) if owner_at is None else owner_at
    return terr_owner == u.owner


def melee_targets(state: GameState, u: Unit) -> list[Unit]:
    out = []
    for d in range(NUM_DIRS):
        j = int(state.hexmap.neighbors[u.i, d])
        if j < 0:
            continue
        uid = int(state.occ_unit[j])
        if uid >= 0:
            t = state.units[uid]
            if t.owner != u.owner:
                out.append(t)
    return out


def ranged_targets(state: GameState, u: Unit) -> list[Unit]:
    spec = state.tables.units.get(u.type)
    if spec is None or int(spec["range"]) < 1 or int(spec["ranged_strength"]) <= 0:
        return []
    hm = state.hexmap
    i0 = u.i
    rng_r = int(spec["range"])
    out = []
    seen = {i0}
    frontier = [i0]
    for _ in range(rng_r):
        nxt = []
        for i in frontier:
            for d in range(NUM_DIRS):
                j = int(hm.neighbors[i, d])
                if j < 0 or j in seen:
                    continue
                seen.add(j)
                nxt.append(j)
                uid = int(state.occ_unit[j])
                if uid >= 0:
                    t = state.units[uid]
                    if t.owner != u.owner:
                        out.append(t)
        frontier = nxt
    return out


def melee_city_targets(state: GameState, u: Unit) -> list[City]:
    out = []
    hm = state.hexmap
    i0 = u.i
    for d in range(NUM_DIRS):
        j = int(hm.neighbors[i0, d])
        if j < 0:
            continue
        cid = int(state.occ_city[j])
        if cid >= 0:
            c = state.cities.get(cid)
            if c is not None and c.owner != u.owner:
                out.append(c)
    return out


def ranged_city_targets(state: GameState, u: Unit) -> list[City]:
    spec = state.tables.units.get(u.type)
    if spec is None or int(spec["range"]) < 1 or int(spec["ranged_strength"]) <= 0:
        return []
    hm = state.hexmap
    i0 = u.i
    rng_r = int(spec["range"])
    out = []
    seen = set()
    frontier = [i0]
    for _ in range(rng_r):
        nxt = []
        for i in frontier:
            for d in range(NUM_DIRS):
                j = int(hm.neighbors[i, d])
                if j < 0 or j in seen:
                    continue
                seen.add(j)
                nxt.append(j)
                cid = int(state.occ_city[j])
                if cid >= 0:
                    c = state.cities.get(cid)
                    if c is not None and c.owner != u.owner:
                        out.append(c)
        frontier = nxt
    return out


def production_options(state: GameState, city: City) -> list[str]:
    """Items this city could build (tech-gated; no duplicates; settler needs pop)."""
    civ = state.civs[city.owner]
    K = state.tables.constants
    techs = civ.techs
    out = []
    for name, spec in state.tables.units.items():
        tech = spec.get("tech")
        if tech is not None and tech not in techs:
            continue
        if name == "settler" and city.pop < K["settler_min_pop"]["value"]:
            continue
        out.append(name)
    for name, spec in state.tables.buildings.items():
        if name in city.buildings:
            continue
        tech = spec.get("tech")
        if tech is not None and tech not in techs:
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# legal action enumeration + validation
# ---------------------------------------------------------------------------

def _entry_allowed(state: GameState, information, civ: int, j: int,
                   diplomacy) -> bool:
    """Closed-borders filter (diplomacy module): entering another
    civ's territory needs an open-borders agreement.  The check runs on the
    TERRITORY OWNER AS THE ACTING CIV KNOWS IT (`information.owner_known`) so
    `legal_actions` stays computable from the observation alone; execution
    re-checks the true owner and drops a stale move (`closed_borders`).  With
    no diplomacy module (the default) nothing is ever blocked."""
    if not diplomacy.closed_borders:
        return True
    if information is not None:
        owner = information.owner_known(state, civ, j)
    else:
        owner = int(state.owner[j])
    return diplomacy.entry_ok(owner, civ)


def legal_actions(state: GameState, civ: int,
                  information=None, diplomacy=None) -> list[tuple]:
    """Enumerate the legal actions for `civ`.  `information` (an information
    module) may append purchase actions and re-key entity ids into
    civ-local handles (see fog.present_actions); `diplomacy` may
    append OFFER/ACCEPT/REJECT and gate foreign-territory entry.  Base
    enumeration stays pure so ordinary legality never depends on the
    optional subsystems."""
    civ_state = state.civs[civ]
    out: list[tuple] = [("SKIP",)]
    if not civ_state.alive:
        return out
    hm = state.hexmap
    knowledge = (diplomacy is not None and diplomacy.closed_borders
                 and information is not None)
    for u in state.units.values():
        if u.owner != civ:
            continue
        i0 = u.i
        for d in range(NUM_DIRS):
            j = int(hm.neighbors[i0, d])
            if j < 0 or not can_move_to(state, u, j):
                continue
            if diplomacy is not None and not _entry_allowed(state, information, civ, j, diplomacy):
                continue
            out.append(("MOVE", u.id, d))
        spec = state.tables.units.get(u.type)
        if spec is not None and not spec.get("civilian", False):
            for t in melee_targets(state, u):
                out.append(("ATTACK_U", u.id, t.id))
            if int(spec["ranged_strength"]) > 0 and int(spec["range"]) >= 1:
                for t in ranged_targets(state, u):
                    out.append(("ATTACK_U", u.id, t.id))
                for c in ranged_city_targets(state, u):
                    out.append(("ATTACK_C", u.id, c.id))
            for c in melee_city_targets(state, u):
                out.append(("ATTACK_C", u.id, c.id))
        if can_found_city(state, u, information.owner_known(state, civ, i0)
                          if knowledge else None):
            out.append(("FOUND", u.id))
        if can_improve(state, u, information.owner_known(state, civ, i0)
                       if knowledge else None):
            out.append(("IMPROVE", u.id))
    for c in state.cities.values():
        if c.owner != civ:
            continue
        for item in production_options(state, c):
            out.append(("PRODUCE", c.id, item))
    if information is not None:
        out.extend(information.purchases_legal(state, civ))
    if diplomacy is not None:
        out.extend(diplomacy.diplomacy_actions(state, civ))
    # de-dup, preserving order (ranged+melee can both reach a target)
    seen = set()
    dedup = []
    for a in out:
        if a not in seen:
            seen.add(a)
            dedup.append(a)
    if information is not None:
        present = getattr(information, "present_actions", None)
        if present is not None:
            dedup = present(state, civ, dedup)
    return dedup


# Which element of an action tuple names the actor whose ONE action per turn
# the batch rule below enforces (docs/DESIGN_NOTES.md §2).
_UNIT_ACTIONS = ("MOVE", "ATTACK_U", "ATTACK_C", "FOUND", "IMPROVE")
_CITY_ACTIONS = ("PRODUCE",)
# Purchases bind no unit or city (they act on the buyer civ); at most ONE per
# civ per turn (docs/DESIGN_NOTES.md §15).  Diplomacy orders
# share the same one-per-civ budget: OFFER/ACCEPT/REJECT are civ-level news.
_INFO_ACTIONS = ("SPY", "TRADE", "DIPLO", "ORACLE")
_DIPLO_ORDERS = ("OFFER", "ACCEPT", "REJECT")


def validate(state: GameState, civ: int, actions, legal_list=None,
             information=None, diplomacy=None) -> None:
    """Raise InvalidAction unless every submitted action is currently legal AND
    the batch binds each actor at most once.

    `legal_list` lets a caller reuse the legal list it already enumerated for
    this (turn, civ) -- pure redundancy avoidance, same result either way
    (replay has no cache, so it recomputes with BOTH modules).

    BATCH RULE: membership in `legal_actions` is
    a per-action property; a batch that lists two legal actions for the SAME
    unit used to pass and both used to execute, letting a unit move twice (or
    move and attack, or attack the same defender twice -- which draws the SAME
    world-keyed combat roll twice, since the key is
    (purpose, turn, attacker, defender, roll_index) and roll_index is always
    0).  "One action per unit per turn" is a batch-level invariant, so it is
    enforced here, at the only place that sees the whole batch.  One PRODUCE
    per city likewise (two would have silently let the last one win)."""
    civ_state = state.civs[civ]
    if not isinstance(actions, (list, tuple)):
        raise InvalidAction(f"civ {civ}: action batch must be a list, got {type(actions)!r}")
    if not civ_state.alive:
        allowed = {("SKIP",)}
        for a in actions:
            if tuple(a) not in allowed:
                raise InvalidAction(f"civ {civ} is eliminated; only [SKIP] is accepted")
        return
    if legal_list is not None:
        legal = set(legal_list)
    else:
        legal = set(legal_actions(state, civ, information, diplomacy))
    bound_units: set[int] = set()
    bound_cities: set[int] = set()
    n_purchases = 0
    n_diplo = 0
    for a in actions:
        at = tuple(a)
        if at not in legal:
            raise InvalidAction(f"civ {civ}: illegal action {at!r}")
        if at[0] in _UNIT_ACTIONS:
            if at[1] in bound_units:
                raise InvalidAction(
                    f"civ {civ}: unit {at[1]} already has an action this turn "
                    f"(second action {at!r}); one action per unit per turn")
            bound_units.add(at[1])
        elif at[0] in _CITY_ACTIONS:
            if at[1] in bound_cities:
                raise InvalidAction(
                    f"civ {civ}: city {at[1]} already has a PRODUCE this turn "
                    f"(second action {at!r}); one PRODUCE per city per turn")
            bound_cities.add(at[1])
        elif at[0] in _INFO_ACTIONS:
            n_purchases += 1
            if n_purchases > 1:
                raise InvalidAction(
                    f"civ {civ}: at most one information purchase per turn "
                    f"(second purchase {at!r})")
        elif at[0] in _DIPLO_ORDERS:
            n_diplo += 1
            if n_diplo > 1:
                raise InvalidAction(
                    f"civ {civ}: at most one diplomacy order per turn "
                    f"(second order {at!r})")
