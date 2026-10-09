"""Search-agent interface (design §3).

* `SearchAgent.act(observation, belief, budget) -> actions` -- the plugin
  surface for MCTS/IS-MCTS/policy nets.
* `determinize(observation, belief, tables, key)` -- build a playable
  GameState from ONE civ's observation and a belief sample, never from the
  hidden state.  INFORMATION SAFETY BY CONSTRUCTION: the function does not
  receive a GameState, so it cannot insert the true hidden state.  Hidden
  foreign units exist in the determinized world only where the belief places
  them (currently visible units are pinned facts at their observed hex;
  memory tracks take a sampled candidate hex or GONE); units never seen do
  not exist (the belief's ontology); unknown terrain data uses a neutral
  prior (no bonus resource, no improvement).  The determinized world's own
    civ, cities, gold and techs are exact observation facts. Entity handles
    are mapped to private nonnegative engine ids; returned actions map back.
* `ForwardModel` -- steps CLONED or DETERMINIZED states through the real
  kernel (clone_state/step/legal_actions/observation/hidden_state_hash are
  the env-side paths; the forward model always runs the default full
  information module: a determinized state carries no fog bookkeeping, its
  "hidden" data is the sample itself).
* `DeterminizedSearchAgent` -- the bounded baseline: sample D
  determinizations, evaluate each root action by a scripted-pilot rollout of
  H turns, play the best-mean action.
* `BuyOrSearchAgent` -- the design's buy-or-search hook as a
  runnable baseline: each turn the agent buys an information action (gold,
  engine ledger) OR runs bounded search (compute budget), never both.

BUDGET ACCOUNTING: compute is charged in EXPANSIONS (one forward-model turn
step for one candidate) and DETERMINIZATIONS (one belief sample turned into
a state).  It is separate from gold by construction -- search never touches
the gold ledger and purchases never consume budget -- and it is exposed on
the Budget object, on the agent's `last_budget`/`total_*` fields, and (when
a runner logs it) in the record's per-turn `compute` column.
"""

from __future__ import annotations

import numpy as np

from .agents import DEFAULT_LINEUP, make_agents
from .agents.base import Agent, legal_set
from .engine import legal_actions
from .hexmap import HexMap
from .kernel import Kernel
from .observe import (InformationModule, build_base_observation)
from .rng import Rng, Stream, stream_seed
from .state import City, Civ, GameState, Unit
from .tables import load_tables


class Budget:
    """An explicitly bounded, separately-accounted compute budget (design §3:
    compute is a resource distinct from the game's gold)."""

    __slots__ = ("expansions_max", "expansions_used",
                 "determinizations_max", "determinizations_used")

    def __init__(self, expansions: int = 32, determinizations: int = 2):
        self.expansions_max = int(expansions)
        self.expansions_used = 0
        self.determinizations_max = int(determinizations)
        self.determinizations_used = 0

    @property
    def expansions_remaining(self) -> int:
        return max(0, self.expansions_max - self.expansions_used)

    @property
    def determinizations_remaining(self) -> int:
        return max(0, self.determinizations_max - self.determinizations_used)

    def charge_expansions(self, n: int = 1) -> None:
        self.expansions_used = min(self.expansions_max,
                                   self.expansions_used + int(n))

    def charge_determinization(self) -> None:
        self.determinizations_used = min(self.determinizations_max,
                                         self.determinizations_used + 1)

    def as_dict(self) -> dict:
        return {"expansions_used": self.expansions_used,
                "expansions_max": self.expansions_max,
                "determinizations_used": self.determinizations_used,
                "determinizations_max": self.determinizations_max}

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Budget(expansions {self.expansions_used}/{self.expansions_max},"
                f" determinizations {self.determinizations_used}/{self.determinizations_max})")


# ---------------------------------------------------------------------------
# determinization (from the observation + belief ONLY)
# ---------------------------------------------------------------------------

def _terrain_arrays(state: GameState, obs: dict, tables, visible) -> None:
    """Fill passability/move-cost/yield arrays from the PUBLIC terrain and
    the observation's known resource layer (truth where visible, memory
    where remembered, no bonus anywhere unknown)."""
    K = tables.constants
    L = state.hexmap.n
    terrain = np.asarray(obs["terrain"]).astype(np.int8)
    state.terrain = terrain.copy()
    names = tables.terrain_names
    passable = np.zeros(L, dtype=bool)
    move_cost = np.zeros(L, dtype=np.int8)
    yf = np.zeros(L, dtype=np.int16)
    yp = np.zeros(L, dtype=np.int16)
    yg = np.zeros(L, dtype=np.int16)
    mem = obs.get("memory")
    obs_res = np.asarray(obs["resource"])
    known_res = obs_res.copy()
    if mem is not None:
        unseen = ~np.asarray(visible, dtype=bool)
        known_res = np.where(unseen, np.asarray(mem["resource"]), obs_res)
    known_res = np.clip(known_res, 0, None).astype(np.int8)
    state.resource = known_res
    ry = K["resource_yield"]["value"]
    res_names = ("wheat", "iron", "gold_ore")
    for i in range(L):
        spec = tables.terrain[names[int(terrain[i])]]
        passable[i] = bool(spec["passable"])
        mc = int(spec["move_cost"])
        move_cost[i] = mc if mc > 0 else 0
        f, pr, g = int(spec["food"]), int(spec["prod"]), int(spec["gold"])
        r = int(known_res[i])
        if r:
            for k, v in ry[res_names[r - 1]].items():
                if k == "food":
                    f += int(v)
                elif k == "prod":
                    pr += int(v)
                elif k == "gold":
                    g += int(v)
        yf[i], yp[i], yg[i] = f, pr, g
    state.passable = passable
    state.move_cost = move_cost
    state.yf, state.yp, state.yg = yf, yp, yg


def _sample_configuration(belief, rng: Stream, turn: int):
    """One (tracks, positions) sample from a belief: aligned track lists and
    one position (or None = GONE) per track.  Fixed (pinned) tracks keep
    their known position."""
    from .belief import ParticleBelief, SupportResult, constraint_order

    if belief is None:
        return [], []
    if isinstance(belief, SupportResult):
        tracks = belief.tracks  # already constraint-ordered like the configs
        if not belief.configs:
            return tracks, [None] * len(tracks)
        cfg = belief.configs[rng.randint(0, len(belief.configs) - 1,
                                         "cfg", turn)]
        return tracks, list(cfg)
    if isinstance(belief, ParticleBelief):
        tracks = belief.tracks
        if not belief.particles:
            return tracks, [None] * len(tracks)
        cfg = belief.particles[rng.randint(0, len(belief.particles) - 1,
                                           "cfg", turn)]
        return tracks, list(cfg)
    raise TypeError(f"unsupported belief type {type(belief)!r}")


def model_id(observed_id: int) -> int:
    """Map observer-local negative handles into private engine-safe ids.

    Engine occupancy uses negative values for empty cells. These private ids
    are derived solely from the observation, never from hidden creation ids.
    Full-view observations already have nonnegative ids and retain them.
    """
    value = int(observed_id)
    return -value - 1 if value < 0 else value


def observation_action(action: tuple, obs: dict) -> tuple:
    """Return a model root action in the observation's id namespace."""
    units = {model_id(row[0]): int(row[0]) for row in obs["unit_list"]}
    cities = {model_id(row[0]): int(row[0]) for row in obs["city_list"]}
    kind = action[0]
    if kind == "ATTACK_U":
        return (kind, units[action[1]], units[action[2]])
    if kind == "ATTACK_C":
        return (kind, units[action[1]], cities[action[2]])
    if kind in ("MOVE", "FOUND", "IMPROVE"):
        return (kind, units[action[1]], *action[2:])
    if kind in ("PRODUCE", "SPY"):
        return (kind, cities[action[1]], *action[2:])
    return tuple(action)


def determinize(obs: dict, belief, tables, key: int = 0) -> GameState:
    """Build a playable GameState from one civ's observation and a belief
    sample (see the module docstring for the information-safety argument).
    `key` selects the sample; the same (observation, belief, key) always
    determinizes to the identical world."""
    from .worldgen import recompute_territory

    meta = obs["meta"]
    R = int(meta["radius"])
    hm = HexMap(R)
    L = hm.n
    params = {"R": R, "civs": int(meta["n_civs"]), "T": int(meta["horizon"]),
              "camps": 0, "start": "standard", "name": "determinized"}
    state = GameState(hm, tables, Rng(0), params, dict(meta["dials"]), seed=0)
    state.turn = int(meta["turn"])
    you = int(obs["you"])
    visible = obs.get("visible")
    if visible is None:
        visible = np.ones(L, dtype=bool)

    _terrain_arrays(state, obs, tables, visible)

    # -- own + pinned units: exact observation facts, private nonnegative ids.
    # Root actions are mapped back to observer handles before being returned.
    bnames = sorted(tables.buildings)
    item_names = list(tables.units) + bnames
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        u = Unit(model_id(uid), int(owner), str(utype), int(q), int(r),
                 int(hp), int(mp))
        u.wid = (int(owner), -1, -1, int(uid))
        u.i = hm.index(int(q), int(r))
        state.units[u.id] = u
        state.occ_unit[u.i] = u.id
        state.next_unit_id = max(state.next_unit_id, u.id + 1)

    # -- hidden foreign units: where the BELIEF places them, nowhere else
    rng = Stream("belief", stream_seed(int(key), "belief"))
    tracks, positions = _sample_configuration(belief, rng, state.turn)
    for t_idx, (tr, pos) in enumerate(zip(tracks, positions)):
        if tr.fixed or pos is None:
            continue  # pinned facts already placed; GONE means not placed
        pos = int(pos)
        if pos < 0 or pos >= L or int(state.occ_unit[pos]) >= 0:
            continue  # defensive: never stack a sampled unit
        spec = tables.units[tr.type_name]
        uid = state.next_unit_id
        u = Unit(uid, int(tr.owner), tr.type_name, int(hm.qs[pos]),
                 int(hm.rs[pos]), int(tables.constants["unit_hp_max"]["value"]),
                 int(spec["move"]))
        u.wid = (int(tr.owner), -1, -1, uid)
        u.i = pos
        state.units[uid] = u
        state.occ_unit[pos] = uid
        state.next_unit_id += 1

    # -- cities: own + visible from the city list, remembered ones from the
    # memory layer (stale contents = the civ's best information)
    added_cities: set[int] = set()
    for row in obs["city_list"]:
        (cid, owner, q, r, pop, hp, queue, buildings, border, capital) = row
        _add_city(state, hm, model_id(cid), int(owner), int(q), int(r), int(pop),
                  int(hp), queue, tuple(buildings), int(border),
                  bool(capital))
        added_cities.add(model_id(cid))
    mem = obs.get("memory")
    if mem is not None:
        # Presence is marked by hp, since a valid fog city handle is negative
        # and the first handle (-1) also equals the unknown-id sentinel.
        for i in np.flatnonzero(np.asarray(mem["city_hp"]) >= 0):
            hid = model_id(mem["city_id"][i])
            if hid in added_cities or int(state.occ_city[i]) >= 0:
                continue
            owner = int(mem["city_owner"][i])
            if owner < 0:
                continue
            qname = None
            qi = int(mem["city_queue"][i])
            if 0 <= qi < len(item_names):
                qname = item_names[qi]
            mask = int(mem["city_buildings"][i])
            buildings = tuple(b for k, b in enumerate(bnames) if mask >> k & 1)
            _add_city(state, hm, hid, owner, int(hm.qs[i]), int(hm.rs[i]),
                      max(1, int(mem["city_pop"][i])),
                      max(1, int(mem["city_hp"][i])), qname, buildings,
                      max(1, int(mem["city_border"][i])),
                      bool(mem["city_capital"][i]))
            added_cities.add(hid)

    # -- civ tables: exact for the observer, neutral priors elsewhere
    for i in range(int(meta["n_civs"])):
        gold = int(obs["gold"][i]) if obs["gold"][i] >= 0 else 0
        civ = Civ(gold)
        mask_row = obs["techs"][i]
        civ.techs = tuple(sorted(obs["tech_names"][k]
                                 for k in range(len(mask_row)) if mask_row[k]))
        civ.alive = bool(obs["alive"][i])
        if i == you:
            civ.researching = obs["researching"][i]
            civ.research_bucket = int(obs["research_progress"][i])
        state.civs.append(civ)

    # -- a determinized world is internally consistent: its territory derives
    # from ITS cities; no camps exist (no unseen barbarian reinforcement)
    recompute_territory(state)
    from .economy import recompute_scores
    recompute_scores(state)
    state.territory_stale = False
    return state


def _add_city(state: GameState, hm: HexMap, cid: int, owner: int, q: int,
              r: int, pop: int, hp: int, queue, buildings, border: int,
              capital: bool) -> None:
    c = City(cid, owner, q, r, capital=capital)
    c.pop = pop
    c.hp = hp
    c.queue = queue
    c.buildings = tuple(sorted(buildings))
    c.border = max(1, border)
    i = hm.index(q, r)
    state.cities[cid] = c
    state.occ_city[i] = cid
    state.next_city_id = max(state.next_city_id, cid + 1)


# ---------------------------------------------------------------------------
# forward model + search agents
# ---------------------------------------------------------------------------

class ForwardModel:
    """Steps cloned/determinized states through the real kernel.  Always the
    default full information module: a determinized state has no fog
    bookkeeping, and its hidden data IS the sample."""

    def __init__(self, tables):
        self.tables = tables
        self.kernel = Kernel(tables, {})

    def legal_actions(self, state: GameState, civ: int) -> list[tuple]:
        return legal_actions(state, civ)

    def step(self, state: GameState, actions_by_civ: dict):
        return self.kernel.step_turn(state, actions_by_civ)

    def observation(self, state: GameState, civ: int) -> dict:
        return InformationModule().build_observation(state, civ,
                                                     build_base_observation(state))


class SearchAgent:
    """The design §3 interface: act(observation, belief, budget) -> actions."""

    name = "search"

    def act(self, observation: dict, belief, budget: Budget) -> list[tuple]:
        raise NotImplementedError


class DeterminizedSearchAgent(SearchAgent):
    """Bounded search baseline (IS-MCTS-flavoured determinization): sample D
    determinizations of the world from the observation + belief, score each
    root action by the own-score delta of a scripted-pilot rollout (H turns,
    everyone including the pilot playing their scripted policies), and play
    the best-mean action.  Ties break by root-action order.  Every
    forward-model turn of a candidate charges one expansion."""

    def __init__(self, civ_id: int, params: dict | None = None):
        self.civ_id = int(civ_id)
        p = dict(params or {})
        self.determinizations = max(1, int(p.get("determinizations", 2)))
        self.horizon = max(1, int(p.get("horizon", 1)))
        self.tables = load_tables()
        self.fm = ForwardModel(self.tables)

    def act(self, observation: dict, belief, budget: Budget) -> list[tuple]:
        you = int(observation["you"])
        n = int(observation["meta"]["n_civs"])
        pilots = make_agents(DEFAULT_LINEUP, n)
        dmax = min(self.determinizations, budget.determinizations_remaining)
        if dmax <= 0:
            return []
        totals: dict[tuple, list[float]] = {}
        candidates: list[tuple] = []
        for d in range(dmax):
            dstate = determinize(observation, belief, self.tables, key=d)
            budget.charge_determinization()
            root_legal = self.fm.legal_actions(dstate, you)
            if d == 0:
                # Belief-only targets are not observed legal root targets.
                candidates = []
                for action in root_legal:
                    try:
                        observation_action(action, observation)
                    except KeyError:
                        continue
                    candidates.append(action)
            for a in candidates:
                if budget.expansions_remaining < self.horizon:
                    break  # honest budget: unevaluated candidates stay absent
                v = self._evaluate(dstate, a, you, pilots, budget)
                totals.setdefault(a, []).append(v)
        best, best_mean = None, None
        for a in candidates:
            vs = totals.get(a)
            if not vs:
                continue
            mean = sum(vs) / len(vs)
            if best is None or mean > best_mean:
                best, best_mean = a, mean
        return [observation_action(best, observation)] if best is not None else []

    def _evaluate(self, dstate: GameState, action: tuple, you: int,
                  pilots, budget: Budget) -> float:
        clone = dstate.clone()
        n = len(clone.civs)
        # A hidden occupancy hypothesis can make an observation-plausible
        # root order unavailable in this sample. Model it as a dropped order,
        # rather than crashing or obtaining a different free root action.
        available = action in self.fm.legal_actions(clone, you)
        acts: dict[int, list] = {you: [action] if available else []}
        for civ in range(n):
            if civ == you or not clone.civs[civ].alive:
                continue
            obs_c = self.fm.observation(clone, civ)
            acts[civ] = pilots[civ].act(obs_c, self.fm.legal_actions(clone, civ),
                                        clone.rng.stream(pilots[civ].stream_name))
        self.fm.step(clone, acts)
        budget.charge_expansions()
        for _ in range(self.horizon - 1):
            if clone.done or budget.expansions_remaining <= 0:
                break
            acts = {}
            for civ in range(n):
                if not clone.civs[civ].alive:
                    continue
                obs_c = self.fm.observation(clone, civ)
                acts[civ] = pilots[civ].act(obs_c, self.fm.legal_actions(clone, civ),
                                            clone.rng.stream(pilots[civ].stream_name))
            self.fm.step(clone, acts)
            budget.charge_expansions()
        v = float(clone.civs[you].score - dstate.civs[you].score)
        if clone.done:
            term = float(self.tables.constants["terminal_reward"]["value"])
            v += term if clone.winner == you else (-term if clone.winner is not None else 0.0)
        return v


class SearchAdapter(Agent):
    """Bridge from the scripted runner (`Agent.act(obs, legal, rng)`) to the
    design §3 interface: builds the belief for its civ, runs the wrapped
    SearchAgent under a fresh Budget, filters the proposal against the real
    legal-action list, and exposes `last_budget` plus running totals so a
    runner can log honest compute accounting."""

    name = "search_adapter"

    def __init__(self, civ_id: int, inner: SearchAgent, belief_fn=None,
                 budget_params: dict | None = None):
        super().__init__(civ_id, None)
        self.inner = inner
        self.belief_fn = belief_fn
        self.budget_params = dict(budget_params or {})
        self.last_budget: Budget | None = None
        self.total_expansions = 0
        self.total_determinizations = 0

    def reset(self) -> None:
        self.last_budget = None

    def plan(self, obs: dict, legal: set) -> list[tuple]:
        budget = Budget(**self.budget_params)
        belief = self.belief_fn(obs) if self.belief_fn is not None else None
        acts = self.inner.act(obs, belief, budget)
        self.last_budget = budget
        self.total_expansions += budget.expansions_used
        self.total_determinizations += budget.determinizations_used
        ls = legal_set(legal)
        return [a for a in acts if tuple(a) in ls]


class BuyOrSearchAgent(Agent):
    """The buy-or-search hook (design §3) as a runnable baseline: each
    turn the agent EITHER buys one information action (gold -- charged
    through the engine ledger) OR spends compute on bounded search -- never
    both.  mode:
      "info_first" (default): buy whenever a purchase is legal.
      "alternate":            buy on even turns, search on odd turns.
    Unit/city orders come from a wrapped scripted base agent; search proposals
    are merged in when compute is the chosen spend."""

    name = "buy_or_search"

    def __init__(self, civ_id: int, params: dict | None = None,
                 belief_fn=None, base: Agent | None = None,
                 inner: SearchAgent | None = None):
        super().__init__(civ_id, params)
        from .agents.expander import ExpanderAgent

        self.mode = str(self.params.get("mode", "info_first"))
        self.belief_fn = belief_fn
        self.base = base or ExpanderAgent(civ_id, self.params)
        self.inner = inner or DeterminizedSearchAgent(
            civ_id, self.params.get("search"))
        self.budget_params = dict(self.params.get("budget") or {})
        self.last_budget: Budget | None = None
        self.last_decision = None
        self.total_expansions = 0
        self.total_determinizations = 0

    def reset(self) -> None:
        self.last_budget = None
        self.last_decision = None

    def plan(self, obs: dict, legal: set) -> list[tuple]:
        turn = int(obs["meta"]["turn"])
        info_acts = [a for a in legal
                     if a[0] in ("SPY", "TRADE", "DIPLO", "ORACLE")]
        buy = False
        if info_acts:
            if self.mode == "info_first":
                buy = True
            elif self.mode == "alternate":
                buy = turn % 2 == 0
        if buy:
            self.last_decision = "reveal"
            return [info_acts[0]]
        self.last_decision = "compute"
        budget = Budget(**self.budget_params)
        belief = self.belief_fn(obs) if self.belief_fn is not None else None
        try:
            acts = self.inner.act(obs, belief, budget)
        finally:
            self.last_budget = budget
            self.total_expansions += budget.expansions_used
            self.total_determinizations += budget.determinizations_used
        base_orders = self.base.plan(obs, legal)
        return self._one_per_actor(list(base_orders)
                                   + [a for a in acts if a in legal])
