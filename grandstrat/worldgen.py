"""World generation (design §1): terrain, bonus resources, civ spread,
starting state, barbarian camps.  All draws come from the `map` and `spread`
streams, keyed by world geometry (hex coordinates / placement slot), never by
generation call order alone.  Generation happens once at reset, pinned for
the whole game.

Implementation: `value_noise` (the only MapGen registered), a
bilinear value-noise field built from per-lattice-point hash draws, with a
radial falloff so the map centre is land.
"""

from __future__ import annotations

from .hexmap import HexMap, hex_count
from .rng import Rng
from .state import Camp, City, Civ, GameState, Unit
from .tables import Tables

_TERRAIN_ORDER = ("plains", "forest", "hills", "mountain", "water")


def _noise_at(hexmap: HexMap, rng: Rng, scale: int, gx: int, gy: int, tag: str) -> float:
    """Hashed lattice value for the noise grid (pure, counter-based)."""
    return rng.uniform("map", tag, scale, gx, gy)


def _value_noise(hexmap: HexMap, rng: Rng, scale: int, tag: str):
    """Bilinear value noise sampled at every hex centre, as a list of floats."""
    import numpy as np

    qs = hexmap.qs
    rs = hexmap.rs
    # lattice coordinates of the hex centre
    fx = qs.astype(np.float64) / scale
    fy = rs.astype(np.float64) * 0.8660254037844386 / scale  # hex vertical pitch
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = fx - x0
    ty = fy - y0
    sx = tx * tx * (3.0 - 2.0 * tx)  # smoothstep
    sy = ty * ty * (3.0 - 2.0 * ty)
    # Speed: neighbouring hexes share lattice
    # points, so the same (gx, gy) draw was repeated many times per layer.
    # Memoizing per lattice point is value-identical (the draw is a pure
    # function of the key) and cuts worldgen draws by ~4x.
    _memo: dict = {}

    def _lat(gx: int, gy: int) -> float:
        v = _memo.get((gx, gy))
        if v is None:
            v = _noise_at(hexmap, rng, scale, gx, gy, tag)
            _memo[(gx, gy)] = v
        return v

    xs0 = x0.tolist()
    ys0 = y0.tolist()
    n00 = np.array([_lat(xs0[i], ys0[i]) for i in range(hexmap.n)])
    n10 = np.array([_lat(xs0[i] + 1, ys0[i]) for i in range(hexmap.n)])
    n01 = np.array([_lat(xs0[i], ys0[i] + 1) for i in range(hexmap.n)])
    n11 = np.array([_lat(xs0[i] + 1, ys0[i] + 1) for i in range(hexmap.n)])
    a = n00 * (1 - sx) + n10 * sx
    b = n01 * (1 - sx) + n11 * sx
    return a * (1 - sy) + b * sy


class WorldGenerationError(RuntimeError):
    """A single worldgen attempt cannot host the requested start (e.g. a
    tiny map whose capital candidates sit on 1-hex islands).  generate_world
    retries with a deterministic derived seed; replay repeats the same loop."""


def generate_world(seed: int, params: dict, dials: dict, hexmap: HexMap,
                   tables: Tables, rng: Rng, max_attempts: int = 8) -> GameState:
    last = None
    for attempt in range(max_attempts):
        attempted = seed if attempt == 0 else (seed + attempt * 0x9E3779B97F4A7C15) & ((1 << 62) - 1)
        rng_local = Rng(attempted, dict(rng._overrides)) if attempt else rng
        try:
            return _generate_world(attempted, params, dials, hexmap, tables,
                                   rng, rng_local)
        except WorldGenerationError as e:
            last = e
    raise WorldGenerationError(
        f"no valid standard start in {max_attempts} attempts (seed {seed}): {last}"
    )


def _generate_world(seed: int, params: dict, dials: dict, hexmap: HexMap,
                    tables: Tables, rng: Rng, rng_local: Rng) -> GameState:
    # `rng` stays the run's master-seeded bundle (combat etc. draw from it);
    # only world-generation draws come from `rng_local` so a retry changes
    # the world without disturbing the run's other streams.
    state = GameState(hexmap, tables, rng, params, dials, seed)
    K = tables.constants
    n = hexmap.n
    R = params["R"]

    # --- terrain ---------------------------------------------------------
    import numpy as np

    # finer noise lattice on small maps so they don't become one blob
    scale = max(2, round(K["noise_scale"]["value"] * params["R"] / 12.0))
    elev = _value_noise(hexmap, rng_local, scale, "elev")
    moist = _value_noise(hexmap, rng_local, scale, "moist")
    dist0 = np.hypot(hexmap.qs.astype(np.float64), hexmap.rs.astype(np.float64)) / R
    land = (elev - K["land_falloff"]["value"] * dist0 ** 2) >= K["sea_level"]["value"]
    terr = np.full(n, tables.terrain_ids["water"], dtype=np.int8)
    ti = tables.terrain_ids
    hills = land & (moist >= K["terrain_hill_cut"]["value"])
    mount = hills & (moist >= K["terrain_mountain_cut"]["value"])
    forest = land & ~hills & (moist >= K["terrain_forest_cut"]["value"])
    plains = land & ~hills & ~forest
    terr[hills] = ti["hills"]
    terr[mount] = ti["mountain"]
    terr[forest] = ti["forest"]
    terr[plains] = ti["plains"]
    state.terrain = terr

    # --- bonus resources (resource_scarcity dial scales frequency) -------
    res = np.zeros(n, dtype=np.int8)
    chance = K["resource_chance"]["value"] * dials["resource_scarcity"]
    names = ("wheat", "iron", "gold_ore")
    resid = {nm: i + 1 for i, nm in enumerate(names)}
    for i in range(n):
        t = tables.terrain_names[terr[i]]
        if t == "plains" and rng_local.chance("map", chance, "res", int(hexmap.qs[i]), int(hexmap.rs[i])):
            res[i] = resid["wheat"]
        elif t == "hills" and rng_local.chance("map", chance, "res", int(hexmap.qs[i]), int(hexmap.rs[i])):
            res[i] = resid["iron"]
        elif t in ("forest",) and rng_local.chance("map", chance, "res", int(hexmap.qs[i]), int(hexmap.rs[i])):
            res[i] = resid["gold_ore"]
    state.resource = res

    # --- civilisation spread (the `spread` stream) ------------------------
    def settle_candidates(min_good: int) -> list[int]:
        out = []
        for i in range(n):
            if terr[i] in (ti["water"], ti["mountain"]):
                continue
            around = hexmap.within(i, 1)
            good = sum(1 for j in around if terr[j] not in (ti["water"], ti["mountain"]))
            if good >= min_good:
                out.append(i)
        return out

    n_civs = params["civs"]
    # deterministic relaxation ladder for tight maps: 5 good ring hexes,
    # then 3, then 1, then any land hex.
    candidates = next(
        (c for c in (settle_candidates(g) for g in (5, 3, 1, 0)) if len(c) >= n_civs),
        [],
    )
    if len(candidates) < n_civs:
        raise ValueError("map too small for civ count")
    shuffled = rng_local.stream("spread").shuffle(
        candidates, "civ_slots"
    )
    capitals: list[int] = []
    # Greedy max-min-distance spread: first pick via stream, then each next
    # pick maximises distance to the closest already-picked capital (tie:
    # candidate order from the shuffled stream => seeded, deterministic).
    cap0 = shuffled[0]
    capitals.append(cap0)
    while len(capitals) < n_civs:
        best, best_key = None, (-1, -1)
        for idx, c in enumerate(shuffled):
            if c in capitals:
                continue
            dmin = min(hexmap.distance(c, p) for p in capitals)
            key = (dmin, -idx)
            if key > best_key:
                best, best_key = c, key
        capitals.append(best)

    # --- starting state: capital city + settler + warrior + worker --------
    from .engine import spawn_unit

    for civ_id, cap_hex in enumerate(capitals):
        state.civs.append(Civ(K["start_gold"]["value"]))
        city = City(state.next_city_id, civ_id,
                    int(hexmap.qs[cap_hex]), int(hexmap.rs[cap_hex]), capital=True)
        city.hp = K["city_hp_max"]["value"]
        state.cities[city.id] = city
        state.next_city_id += 1
        state.occ_city[cap_hex] = city.id
        start_types = ("settler", "warrior", "worker")
        # placement spirals outward from the capital: ring 1, then ring 2,
        # then the capital hex itself.  (Small maps can put a capital on a
        # near-island; the standard start must not depend on ring-1 terrain.)
        spiral = []
        for rad in (1, 2):
            for dq, dr in hexmap.ring_offsets(rad):
                j = hexmap.idx.get((int(hexmap.qs[cap_hex]) + dq,
                                    int(hexmap.rs[cap_hex]) + dr))
                if j is not None:
                    spiral.append(j)
        spiral.append(cap_hex)
        for ut in start_types:
            placed = False
            for j in spiral:
                if terr[j] in (ti["water"], ti["mountain"]):
                    continue
                if int(state.occ_unit[j]) >= 0:
                    continue
                spawn_unit(state, civ_id, ut, int(hexmap.qs[j]), int(hexmap.rs[j]),
                           birth_key=hexmap.index(city.q, city.r))
                placed = True
                break
            if not placed:
                raise WorldGenerationError(
                    f"capital {city.id} cannot host the standard start"
                )

    # --- barbarian camps ---------------------------------------------------
    n_camps = int(params.get("camps", K["barb_camps_default"]["value"]))
    if n_camps > 0:
        pool = [
            i for i in range(n)
            if terr[i] not in (ti["water"], ti["mountain"])
            and all(hexmap.distance(i, p) >= 5 for p in capitals)
        ]
        pool = rng_local.stream("barbarians").shuffle(pool, "camp_slots")
        placed = 0
        for i in pool:
            if placed >= n_camps:
                break
            if any(
                hexmap.distance(i, hexmap.index(c.q, c.r)) < 4
                for c in state.camps.values()
            ):
                continue
            camp = Camp(state.next_camp_id, int(hexmap.qs[i]), int(hexmap.rs[i]))
            state.camps[camp.id] = camp
            state.next_camp_id += 1
            placed += 1

    recompute_territory(state)
    from .economy import recompute_scores
    from .engine import refresh_occ

    refresh_occ(state)
    refresh_unit_indices(state)

    # static per-hex movement facts (terrain is never modified after gen)
    passable = np.zeros(n, dtype=bool)
    move_cost = np.zeros(n, dtype=np.int8)
    for i in range(n):
        tname = tables.terrain_names[terr[i]]
        spec = tables.terrain[tname]
        passable[i] = bool(spec["passable"])
        mc = int(spec["move_cost"])
        move_cost[i] = mc if mc > 0 else 0
    state.passable = passable
    state.move_cost = move_cost

    # per-hex yields (terrain + resource; improvements come later one by one)
    ry = K["resource_yield"]["value"]
    for i in range(n):
        tname = tables.terrain_names[terr[i]]
        spec = tables.terrain[tname]
        f, pr, g = int(spec["food"]), int(spec["prod"]), int(spec["gold"])
        if res[i] != 0:
            for k, v in ry[names[res[i] - 1]].items():
                if k == "food":
                    f += int(v)
                elif k == "prod":
                    pr += int(v)
                elif k == "gold":
                    g += int(v)
        state.yf[i], state.yp[i], state.yg[i] = f, pr, g

    recompute_scores(state)
    return state


def refresh_unit_indices(state) -> None:
    hm = state.hexmap
    for u in state.units.values():
        u.i = hm.index(u.q, u.r)


def recompute_territory(state: GameState) -> None:
    """Assign every hex to the nearest city whose border radius reaches it
    (tie: smaller city id).  Maintains per-civ territory counts."""
    import numpy as np

    hm = state.hexmap
    owner = np.full(hm.n, -1, dtype=np.int8)
    best_d = np.full(hm.n, 1 << 30, dtype=np.int32)
    best_c = np.full(hm.n, 1 << 30, dtype=np.int32)
    for cid, city in state.cities.items():
        i0 = hm.index(city.q, city.r)
        for rad in range(city.border + 1):
            for dq, dr in (hm.ring_offsets(rad) if rad else [(0, 0)]):
                j = hm.idx.get((int(hm.qs[i0]) + dq, int(hm.rs[i0]) + dr))
                if j is None:
                    continue
                if rad < best_d[j] or (rad == best_d[j] and cid < best_c[j]):
                    best_d[j] = rad
                    best_c[j] = cid
                    owner[j] = city.owner
    state.owner = owner
    state.territory_stale = False
    counts = [0] * len(state.civs)
    for o in owner:
        if o >= 0:
            counts[o] += 1
    for civ, c in zip(state.civs, counts):
        civ.territory_count = c
