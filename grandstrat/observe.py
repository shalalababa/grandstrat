"""Observation construction (design §5) and the flat RL encoder (§0b).

SEPARATION RULE: observations are DERIVED, read-only views built by an
Information module; the full GameState is never handed out.  Under the
default `full` observability module every civ receives the same shared
view -- arrays are marked non-writeable, which makes sharing safe.  The fog
module (grandstrat.fog) plugs in here: it produces per-civ masked
copies from the same builder without any env/state change.

Observation layout (schema documented, pinned):
  meta      turn, horizon, n_civs, radius, scenario name, dials, done/winner
  you       the observing civ id (per-civ shallow wrapper)
  layers    terrain, resource, improve, owner          int8[L]
  units     id/owner/type/q/r/hp/mp arrays + count + python unit_list
  cities    id/owner/q/r/pop/hp/queue/buildings-mask/border/capital + list
  civs      gold, score, alive, techs[n_civs x n_techs], researching, research bucket
  diplomacy float32[n_civs, n_civs] -- all 1.0 (peace) unless the diplomacy module is on
"""

from __future__ import annotations

import numpy as np

from .engine import RESOURCE_NAMES
from .economy import IMPROVE_NAMES
from .state import GameState


def _ro(a: np.ndarray) -> np.ndarray:
    """Mark array read-only (shared observations stay safe)."""
    a.flags.writeable = False
    return a


class InformationModule:
    """Interface (design §0b): the information subsystem is optional; the
    kernel defaults to full observability.  Two implementations exist: this
    `full` module and the fog module (grandstrat.fog), selected by
    KernelConfig `information: "full" | "fog"` via make_information()."""

    name = "full"

    def build_observation(self, state: GameState, civ: int, base: dict) -> dict:
        """Return the observation dict for `civ`.  With full observability the
        shared base view is returned with a per-civ 'you' field."""
        obs = dict(base)
        obs["you"] = civ
        return obs

    # -- information-purchase hooks (the full module has none) -----------
    def reset(self) -> None:
        """Drop per-game bookkeeping (env.reset calls this)."""

    def purchases_legal(self, state: GameState, civ: int) -> list:
        """Purchase actions currently available to `civ` (empty: no module)."""
        return []

    def resolve_purchase(self, state: GameState, report, civ: int, action) -> None:
        from .engine import InvalidAction

        raise InvalidAction(f"civ {civ}: no information module; {action!r} "
                            f"cannot be resolved")

    # -- handle/diplomacy hooks (identity behaviour under the full module) --
    def owner_known(self, state: GameState, civ: int, idx: int) -> int:
        """The territory owner of hex idx AS CIV KNOWS IT (engine legality
        under the diplomacy module uses this so legal actions stay computable
        from the observation).  Full observability: the true owner."""
        return int(state.owner[idx])

    def present_actions(self, state: GameState, civ: int, actions) -> list:
        """Re-key entity ids into civ-local handles for the agent-facing
        action space.  Identity under the full module (its observations are
        the oracle view; see fog.py for the information-safe mapping)."""
        return list(actions)

    def resolve_action(self, civ: int, action) -> tuple:
        """Inverse of present_actions, applied before resolution."""
        return tuple(action)

    def apply_share_maps(self, state: GameState, a: int, b: int) -> None:
        """Diplomacy-driven map sharing.  No-op under full
        observability: there is nothing left to share."""


def make_information(config: dict | None, tables):
    """Build the information module named by KernelConfig (design §0b:
    the module is optional and defaults to full observability)."""
    name = (config or {}).get("information", None) or "full"
    if name == "full":
        return InformationModule()
    if name == "fog":
        from .fog import FogInformation

        return FogInformation(tables)
    raise ValueError(f"unknown information module {name!r}")


def build_base_observation(state: GameState, diplomacy_matrix=None) -> dict:
    """Build the shared full-information view.  One call per env step.
    `diplomacy_matrix`, when given (diplomacy module), replaces the
    default all-peace matrix; agreements are public news."""
    hm = state.hexmap
    tables = state.tables
    n_civs = len(state.civs)
    unit_cap = int(tables.constants["unit_cap"]["value"])
    city_cap = int(tables.constants["city_cap"]["value"])
    L = hm.n

    units_sorted = sorted(state.units.values(), key=lambda u: u.id)
    cities_sorted = sorted(state.cities.values(), key=lambda c: c.id)
    # The caps are enforced at the spawn/founding transitions; this is the
    # observation-side guard so array truncation can never happen silently.
    if len(units_sorted) > unit_cap:
        raise ValueError(f"{len(units_sorted)} units exceed unit_cap {unit_cap}")
    if len(cities_sorted) > city_cap:
        raise ValueError(f"{len(cities_sorted)} cities exceed city_cap {city_cap}")

    u_id = np.full(unit_cap, -1, dtype=np.int32)
    u_owner = np.full(unit_cap, -1, dtype=np.int8)
    u_type = np.full(unit_cap, -1, dtype=np.int8)
    u_q = np.full(unit_cap, -1, dtype=np.int16)
    u_r = np.full(unit_cap, -1, dtype=np.int16)
    u_hp = np.zeros(unit_cap, dtype=np.int16)
    u_mp = np.zeros(unit_cap, dtype=np.int8)
    unit_list = []
    unit_ids = tables.unit_ids
    for k, u in enumerate(units_sorted):
        u_id[k] = u.id
        u_owner[k] = u.owner
        u_type[k] = unit_ids[u.type]
        u_q[k] = u.q
        u_r[k] = u.r
        u_hp[k] = u.hp
        u_mp[k] = u.mp
        unit_list.append((u.id, u.owner, u.type, u.q, u.r, u.hp, u.mp))

    c_id = np.full(city_cap, -1, dtype=np.int32)
    c_owner = np.full(city_cap, -1, dtype=np.int8)
    c_q = np.full(city_cap, -1, dtype=np.int16)
    c_r = np.full(city_cap, -1, dtype=np.int16)
    c_pop = np.zeros(city_cap, dtype=np.int8)
    c_hp = np.zeros(city_cap, dtype=np.int16)
    c_food = np.zeros(city_cap, dtype=np.int32)
    c_prod = np.zeros(city_cap, dtype=np.int32)
    c_queue = np.full(city_cap, -1, dtype=np.int16)
    c_bld = np.zeros(city_cap, dtype=np.int32)  # bitmask over building names
    c_border = np.ones(city_cap, dtype=np.int8)
    c_cap = np.zeros(city_cap, dtype=np.int8)
    city_list = []
    bnames = sorted(tables.buildings)
    # production item ids share one namespace: units (table order) then buildings
    item_names = list(tables.units) + bnames
    for k, c in enumerate(cities_sorted):
        c_id[k] = c.id
        c_owner[k] = c.owner
        c_q[k] = c.q
        c_r[k] = c.r
        c_pop[k] = c.pop
        c_hp[k] = c.hp
        c_food[k] = c.food_bucket
        c_prod[k] = c.prod_bucket
        if c.queue is not None:
            c_queue[k] = item_names.index(c.queue)
        mask = 0
        for b in c.buildings:
            mask |= 1 << bnames.index(b)
        c_bld[k] = mask
        c_border[k] = c.border
        c_cap[k] = int(c.capital)
        city_list.append((c.id, c.owner, c.q, c.r, c.pop, c.hp, c.queue,
                          c.buildings, c.border, bool(c.capital)))

    gold = np.array([c.gold for c in state.civs], dtype=np.int32)
    score = np.array([c.score for c in state.civs], dtype=np.int32)
    alive = np.array([c.alive for c in state.civs], dtype=bool)
    techs = np.zeros((n_civs, len(tables.techs)), dtype=bool)
    tnames = tables.tech_names
    researching = []
    for i, c in enumerate(state.civs):
        for t in c.techs:
            techs[i, tnames.index(t)] = True
        researching.append(c.researching)
    rprog = np.array([c.research_bucket for c in state.civs], dtype=np.int32)
    if diplomacy_matrix is None:
        diplomacy = np.ones((n_civs, n_civs), dtype=np.float32)  # peace everywhere
    else:
        diplomacy = np.array(diplomacy_matrix, dtype=np.float32).copy()
        assert diplomacy.shape == (n_civs, n_civs), (
            f"diplomacy matrix shape {diplomacy.shape} != {(n_civs, n_civs)}")

    obs = {
        "meta": {
            "turn": state.turn,
            "horizon": int(state.params["T"]),
            "n_civs": n_civs,
            "radius": int(state.params["R"]),
            "scenario": state.params.get("name", "custom"),
            "dials": dict(state.dials),
            "done": bool(state.done),
            "winner": state.winner,
            "done_reason": state.done_reason,
        },
        "terrain": _ro(state.terrain.copy()),
        "terrain_names": list(tables.terrain),
        "resource": _ro(state.resource.copy()),
        "resource_names": list(RESOURCE_NAMES),
        "improve": _ro(state.improve.copy()),
        "improve_names": list(IMPROVE_NAMES),
        "owner": _ro(state.owner.copy()),
        "units": {
            "id": _ro(u_id), "owner": _ro(u_owner), "type": _ro(u_type),
            "q": _ro(u_q), "r": _ro(u_r), "hp": _ro(u_hp), "mp": _ro(u_mp),
            "count": len(units_sorted),
        },
        "unit_names": list(tables.units),
        "unit_list": tuple(unit_list),
        "cities": {
            "id": _ro(c_id), "owner": _ro(c_owner), "q": _ro(c_q), "r": _ro(c_r),
            "pop": _ro(c_pop), "hp": _ro(c_hp), "food": _ro(c_food),
            "prod": _ro(c_prod), "queue": _ro(c_queue), "buildings": _ro(c_bld),
            "border": _ro(c_border), "capital": _ro(c_cap),
            "count": len(cities_sorted),
        },
        "building_names": bnames,
        "city_list": tuple(city_list),
        "gold": _ro(gold),
        "score": _ro(score),
        "alive": _ro(alive),
        "techs": _ro(techs),
        "tech_names": tnames,
        "researching": researching,
        "research_progress": _ro(rprog),
        "diplomacy": _ro(diplomacy),
        "production_items": item_names,
    }
    return obs


# ---------------------------------------------------------------------------
# flat encoder
# ---------------------------------------------------------------------------

def flat_layout(state: GameState) -> dict:
    """Offset/size table for the flat observation vector (deterministic)."""
    L = state.hexmap.n
    n_civs = len(state.civs)
    n_units = int(state.tables.constants["unit_cap"]["value"])
    n_cities = int(state.tables.constants["city_cap"]["value"])
    n_techs = len(state.tables.techs)
    off = 0
    layout = {}
    for name, size in (
        ("turn", 1), ("done", 1), ("winner", 1),
        ("terrain", L), ("resource", L), ("improve", L), ("owner", L),
        ("unit_id", n_units), ("unit_owner", n_units), ("unit_type", n_units),
        ("unit_q", n_units), ("unit_r", n_units), ("unit_hp", n_units),
        ("unit_mp", n_units),
        ("city_id", n_cities), ("city_owner", n_cities), ("city_q", n_cities),
        ("city_r", n_cities), ("city_pop", n_cities), ("city_hp", n_cities),
        ("city_food", n_cities), ("city_prod", n_cities),
        ("city_queue", n_cities), ("city_buildings", n_cities),
        ("city_border", n_cities), ("city_capital", n_cities),
        ("gold", n_civs), ("score", n_civs), ("alive", n_civs),
        ("research_progress", n_civs), ("techs", n_civs * n_techs),
        ("diplomacy", n_civs * n_civs),
    ):
        layout[name] = (off, size)
        off += size
    layout["total"] = off
    return layout


def encode_flat(obs: dict, layout: dict) -> np.ndarray:
    """Flatten an observation into one float32 vector per `layout`."""
    vec = np.zeros(layout["total"], dtype=np.float32)
    m = obs["meta"]
    vec[layout["turn"][0]] = m["turn"]
    vec[layout["done"][0]] = 1.0 if m["done"] else 0.0
    vec[layout["winner"][0]] = -1.0 if m["winner"] is None else float(m["winner"])
    for name in ("terrain", "resource", "improve", "owner"):
        o, _ = layout[name]
        vec[o : o + obs[name].shape[0]] = obs[name]
    u = obs["units"]
    cnt = u["count"]
    for name in ("id", "owner", "type", "q", "r", "hp", "mp"):
        o, _ = layout["unit_" + name]
        vec[o : o + cnt] = u[name][:cnt]
    c = obs["cities"]
    ccnt = c["count"]
    for name in ("id", "owner", "q", "r", "pop", "hp", "food", "prod",
                 "queue", "buildings", "border", "capital"):
        o, _ = layout["city_" + name]
        vec[o : o + ccnt] = c[name][:ccnt]
    for name in ("gold", "score", "alive", "research_progress"):
        o, _ = layout[name]
        v = obs[name]
        vec[o : o + v.shape[0]] = v
    o, _ = layout["techs"]
    vec[o : o + obs["techs"].size] = obs["techs"].ravel()
    o, _ = layout["diplomacy"]
    vec[o : o + obs["diplomacy"].size] = obs["diplomacy"].ravel()
    return vec
