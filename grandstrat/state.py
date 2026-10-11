"""World state: the full (hidden) game state, cleanly separated from
per-civ observations.

Design rule: keep hidden state and observation SEPARATE so a per-civ fog
mask drops in without refactoring.
`GameState` is the world; `observe.build_observation` derives per-civ views
from it.  Nothing in `grandstrat.env` hands out `GameState` internals:
observations are freshly built read-only numpy structures.

GameState supports:
  clone()          -- exact deep copy (no aliasing between the clone
                      and the original).  HexMap geometry is shared (immutable).
  state_hash()     -- sha256 over a canonical byte serialization.  Stable
                      across processes and PYTHONHASHSEED values.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .rng import Rng
    from .hexmap import HexMap
    from .tables import Tables


class Unit:
    # `wid` is the unit's WORLD IDENTITY (docs/DESIGN_NOTES.md §13): a
    # path-independent identity fixed at birth, of the form
    #   (owner, birth_turn, birth_key, ordinal)
    # where birth_key is the birth city's world identity (its founding hex
    # index; cities already have one) for produced units, the camp id for
    # barbarians, and -1 for ad-hoc spawns outside any city/camp.  The
    # integer `id` remains the array/indexing handle ONLY: two game arms that
    # diverge in production order assign different creation-counter ids to
    # corresponding units, but their world identities still match, so CRN
    # combat rolls keep pairing across arms.  Wid is immutable after birth.
    __slots__ = ("id", "owner", "type", "q", "r", "i", "hp", "mp", "acted", "wid")

    def __init__(self, uid: int, owner: int, utype: str, q: int, r: int, hp: int, mp: int):
        self.id = uid
        self.owner = owner  # civ id, or -1 for barbarian
        self.type = utype
        self.q = q
        self.r = r
        self.i = -1  # hex index; engine spawn/move keep it current
        self.hp = hp
        self.mp = mp
        self.acted = False  # moved / attacked / worked this turn (heal gate)
        self.wid = (owner, -1, -1, -1)  # set by spawn_unit

    def copy(self) -> "Unit":
        u = Unit.__new__(Unit)
        for s in Unit.__slots__:
            setattr(u, s, getattr(self, s))
        return u

    def __repr__(self) -> str:  # pragma: no cover
        return f"Unit({self.id},{self.owner},{self.type},{self.q},{self.r},hp={self.hp})"


class City:
    __slots__ = (
        "id", "owner", "q", "r", "pop", "food_bucket", "prod_bucket", "queue",
        "hp", "buildings", "culture", "border", "capital", "worked",
        "worked_stale", "attacked",
    )

    def __init__(self, cid: int, owner: int, q: int, r: int, capital: bool):
        self.id = cid
        self.owner = owner
        self.q = q
        self.r = r
        self.pop = 1
        self.food_bucket = 0
        self.prod_bucket = 0
        self.queue = None  # item name or None
        self.hp = 200  # set from tables at creation
        self.buildings = ()  # sorted tuple of building names
        self.culture = 0
        self.border = 1  # territory radius
        self.capital = capital  # capital flag is eternal (see docs/DESIGN_NOTES.md)
        self.worked = ()  # tuple of hex indices worked (center excluded)
        self.worked_stale = True
        self.attacked = False

    def copy(self) -> "City":
        c = City.__new__(City)
        for s in City.__slots__:
            setattr(c, s, getattr(self, s))
        return c

    def __repr__(self) -> str:  # pragma: no cover
        return f"City({self.id},o{self.owner},{self.q},{self.r},pop={self.pop})"


class Civ:
    """Per-civilisation bookkeeping (civ id == index in GameState.civs)."""

    __slots__ = ("gold", "techs", "researching", "research_bucket", "alive",
                 "score", "territory_count")

    def __init__(self, gold: int):
        self.gold = gold
        self.techs = ()  # sorted tuple of researched tech names
        self.researching = None  # current tech (auto-picked; see docs/DESIGN_NOTES.md)
        self.research_bucket = 0
        self.alive = True
        self.score = 0
        self.territory_count = 0

    def copy(self) -> "Civ":
        c = Civ.__new__(Civ)
        for s in Civ.__slots__:
            setattr(c, s, getattr(self, s))
        return c


class Camp:
    __slots__ = ("id", "q", "r", "alive")

    def __init__(self, cid: int, q: int, r: int):
        self.id = cid
        self.q = q
        self.r = r
        self.alive = True

    def copy(self) -> "Camp":
        c = Camp.__new__(Camp)
        for s in Camp.__slots__:
            setattr(c, s, getattr(self, s))
        return c


class GameState:
    """The full world.  Civ ids are 0..n_civs-1; barbarians are owner -1.

    Iteration order contract: `units`, `cities`, `camps` are plain dicts whose
    insertion order is id order (ids are assigned monotonically and never
    reused), so `for x in state.units.values()` is deterministic without
    sorting.  Hashing and observation building still sort explicitly where the
    order is load-bearing for outputs.
    """

    __slots__ = (
        "seed", "hexmap", "tables", "rng", "params", "dials",
        "terrain", "resource", "improve", "owner",
        "units", "cities", "civs", "camps",
        "occ_unit", "occ_city", "passable", "move_cost", "yf", "yp", "yg",
        "turn", "next_unit_id", "next_city_id", "next_camp_id",
        "done", "winner", "done_reason", "territory_stale",
        "birth_counts",
    )

    def __init__(self, hexmap: "HexMap", tables: "Tables", rng: "Rng",
                 params: dict, dials: dict, seed: int):
        self.hexmap = hexmap
        self.tables = tables
        self.rng = rng
        self.params = params  # scenario dict (name, R, civs, T, camps, start)
        self.dials = dials
        self.seed = int(seed)
        n = hexmap.n
        self.terrain = np.zeros(n, dtype=np.int8)    # index into tables.terrain
        self.resource = np.zeros(n, dtype=np.int8)   # 0 = none, else 1..k bonus id
        self.improve = np.zeros(n, dtype=np.int8)    # 0 = none, else improvement id
        self.owner = np.full(n, -1, dtype=np.int8)   # territory owner per hex
        self.units: dict[int, Unit] = {}
        self.cities: dict[int, City] = {}
        self.camps: dict[int, Camp] = {}
        self.civs: list[Civ] = []
        # Occupancy grids: hex index -> unit id / city id, or -1.  Maintained
        # by the engine helpers (spawn/move/remove/capture); derived data, so
        # excluded from state_hash.
        self.occ_unit = np.full(n, -1, dtype=np.int32)
        self.occ_city = np.full(n, -1, dtype=np.int32)
        # Static per-hex movement facts, filled by worldgen (terrain never
        # changes after generation).  passable[i] bool; move_cost[i] in 1..127,
        # 0 where impassable.
        self.passable = np.zeros(n, dtype=bool)
        self.move_cost = np.zeros(n, dtype=np.int8)
        # per-hex worked yields (terrain + resource + improvement), filled by
        # worldgen; engine._improve updates single hexes.  Derived data.
        self.yf = np.zeros(n, dtype=np.int16)
        self.yp = np.zeros(n, dtype=np.int16)
        self.yg = np.zeros(n, dtype=np.int16)
        self.turn = 0
        self.next_unit_id = 0
        self.next_city_id = 0
        self.next_camp_id = 0
        self.done = False
        self.winner = None
        self.done_reason = None
        self.territory_stale = True
        # ordinal bookkeeping for world identities: (owner, turn, birth_key)
        # -> units born so far with that signature this game.  Derived data
        # (wids themselves live on the units), but it MUST be cloned or a
        # cloned branch would re-emit ordinal 0.
        self.birth_counts: dict[tuple, int] = {}

    # -- helpers -----------------------------------------------------------
    def unit_at(self, q: int, r: int):
        uid = int(self.occ_unit[self.hexmap.index(q, r)])
        return self.units.get(uid) if uid >= 0 else None

    def city_at(self, q: int, r: int):
        cid = int(self.occ_city[self.hexmap.index(q, r)])
        return self.cities.get(cid) if cid >= 0 else None

    def camp_at(self, q: int, r: int):
        for c in self.camps.values():
            if c.alive and c.q == q and c.r == r:
                return c
        return None

    def clone(self) -> "GameState":
        """Exact deep copy.  Arrays copied; HexMap and Tables shared
        (immutable).  The rng bundle is stateless so sharing is exact, but a
        fresh bundle is made so per-clone override changes stay local."""
        s = GameState.__new__(GameState)
        for field in (
            "seed", "hexmap", "tables", "turn",
            "next_unit_id", "next_city_id", "next_camp_id", "done",
            "winner", "done_reason", "territory_stale",
        ):
            setattr(s, field, getattr(self, field))
        s.params = dict(self.params)
        s.dials = dict(self.dials)
        s.rng = self.rng.clone()
        s.terrain = self.terrain.copy()
        s.resource = self.resource.copy()
        s.improve = self.improve.copy()
        s.owner = self.owner.copy()
        s.units = {uid: u.copy() for uid, u in self.units.items()}
        s.cities = {cid: c.copy() for cid, c in self.cities.items()}
        s.camps = {kid: k.copy() for kid, k in self.camps.items()}
        s.civs = [c.copy() for c in self.civs]
        s.occ_unit = self.occ_unit.copy()
        s.occ_city = self.occ_city.copy()
        s.passable = self.passable.copy()
        s.move_cost = self.move_cost.copy()
        s.yf = self.yf.copy()
        s.yp = self.yp.copy()
        s.yg = self.yg.copy()
        s.birth_counts = dict(self.birth_counts)
        return s

    def state_hash(self) -> str:
        """sha256 over a canonical serialization.  Byte-stable across runs,
        processes, and PYTHONHASHSEED values (arrays via tobytes; objects in
        sorted id order; no dict/set iteration order enters the digest)."""
        h = hashlib.sha256()
        u = self.units
        h.update(b"gs1")
        h.update(str(self.seed).encode())
        h.update(str(self.turn).encode())
        h.update(str(self.next_unit_id).encode())
        h.update(str(self.next_city_id).encode())
        h.update(str(self.next_camp_id).encode())
        h.update(str(self.done).encode())
        h.update(str((self.winner, self.done_reason)).encode())
        for arr in (self.terrain, self.resource, self.improve, self.owner):
            h.update(arr.tobytes())
        for uid in sorted(u):
            x = u[uid]
            h.update(
                f"U{uid}|{x.owner}|{x.type}|{x.q}|{x.r}|{x.hp}|{x.mp}|{x.acted}|"
                f"{x.wid}".encode()
            )
        for cid in sorted(self.cities):
            x = self.cities[cid]
            h.update(
                f"C{cid}|{x.owner}|{x.q}|{x.r}|{x.pop}|{x.food_bucket}|"
                f"{x.prod_bucket}|{x.queue}|{x.hp}|{','.join(x.buildings)}|"
                f"{x.culture}|{x.border}|{x.capital}|{','.join(map(str, x.worked))}".encode()
            )
        for kid in sorted(self.camps):
            x = self.camps[kid]
            h.update(f"K{kid}|{x.q}|{x.r}|{x.alive}".encode())
        for i, c in enumerate(self.civs):
            h.update(
                f"V{i}|{c.gold}|{','.join(c.techs)}|{c.researching}|"
                f"{c.research_bucket}|{c.alive}|{c.score}|{c.territory_count}".encode()
            )
        h.update(repr(sorted(self.dials.items())).encode())
        return h.hexdigest()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GameState(seed={self.seed}, turn={self.turn}, "
            f"units={len(self.units)}, cities={len(self.civs) and len(self.cities)})"
        )
