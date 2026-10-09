"""Gym/PettingZoo-style multi-agent environment (design §3, §0b).

Parallel-style API:
    env = GrandStratEnv(scenario="default", dials={...}, stream_seeds={...})
    obs = env.reset(seed)                    # dict civ_id -> observation
    obs, rewards, terminations, truncations, infos = env.step(actions)
                                             # actions: dict civ_id -> [tuples]
    env.legal_actions(civ)                   -> list[tuple]
    env.observation(civ)                     -> dict (read-only views)
    env.flat_observation(civ)                -> np.ndarray (float32)
    env.render_text(civ)                     -> {"json": str, "summary": str}
    env.clone_state()                        -> GameState (full hidden world;
                                    search-interface only, design §3)
    env.hidden_state_hash()                  -> str
    env.possible_agents                      -> range(n_civs)

Reward convention (see docs/DESIGN_NOTES.md): per-civ per-turn reward is the change
in that civ's per-turn score, plus +100/-100 at game end for winner/losers.
The terminal reward lands on the same step that sets `done` (so a learner
never misses it).

Observation lifetime: handed-out observations share read-only buffers that
stay valid until the NEXT call to step/reset (standard gym buffer semantics);
`flat_observation` and `render_text` copy immediately.  The full GameState is
never handed out through the agent-facing API.
"""

from __future__ import annotations

import numpy as np

from .diplomacy import make_diplomacy
from .engine import InvalidAction, legal_actions
from .hexmap import HexMap
from .kernel import Kernel
from .observe import (InformationModule, build_base_observation,
                      encode_flat, flat_layout, make_information)
from .rng import Rng, stream_seed
from .scenarios import validate_dials
from .state import GameState
from .tables import load_tables
from .worldgen import generate_world


class GrandStratEnv:
    def __init__(self, scenario: str = "default", scenario_overrides: dict | None = None,
                 dials: dict | None = None, stream_seeds: dict | None = None,
                 config: dict | None = None, tables=None):
        from .scenarios import scenario_params

        self.params = scenario_params(scenario, scenario_overrides)  # fail fast
        self._scenario_name = scenario
        self._scenario_overrides = dict(scenario_overrides or {})
        self.dials = validate_dials(dials)
        self.stream_seeds = dict(stream_seeds or {})
        self.config = dict(config or {})
        self.tables = tables or load_tables()
        self.kernel = Kernel(self.tables, self.config)
        # swappable subsystem modules (design §0b): information (fog) and
        # diplomacy are optional and default to "no structure"
        self.info = make_information(self.config, self.tables)
        self.diplomacy = make_diplomacy(self.config, self.tables)
        self._hexmaps: dict[int, HexMap] = {}
        self.state: GameState | None = None
        self._obs_base: dict | None = None
        self._prev_scores: list[int] | None = None
        self._layout: dict | None = None
        self._legal_cache: dict = {}

    def resolved_params(self) -> dict:
        """Effective scenario params (valid without reset; used by the CLI
        to size the agent lineup)."""
        from .scenarios import scenario_params

        return scenario_params(self._scenario_name, self._scenario_overrides)

    # -- properties ---------------------------------------------------------
    @property
    def possible_agents(self):
        return list(range(self.params["civs"])) if self.params else []

    @property
    def turn(self) -> int:
        return self.state.turn

    def _hexmap(self, R: int) -> HexMap:
        hm = self._hexmaps.get(R)
        if hm is None:
            hm = HexMap(R)
            self._hexmaps[R] = hm
        return hm

    # -- API ------------------------------------------------------------------
    def _matrix(self):
        """The public diplomacy matrix (or None: all-peace default)."""
        if self.diplomacy is None or self.state is None:
            return None
        return self.diplomacy.matrix(self.state)

    def reset(self, seed: int) -> dict:
        from .scenarios import scenario_params

        self.params = scenario_params(self._scenario_name, self._scenario_overrides)
        rng = Rng(seed, self.stream_seeds)
        hm = self._hexmap(self.params["R"])
        self.state = generate_world(seed, self.params, self.dials, hm,
                                    self.tables, rng)
        self._layout = None
        self._legal_cache = {}
        self._prev_scores = [c.score for c in self.state.civs]
        self.info.reset()
        if self.diplomacy is not None:
            self.diplomacy.reset()
        self._obs_base = build_base_observation(self.state, self._matrix())
        self._stamp_diplomacy_meta()
        return {civ: self.info.build_observation(self.state, civ, self._obs_base)
                for civ in range(self.params["civs"])}

    def _stamp_diplomacy_meta(self) -> None:
        """Live offers are public news: stamped into the shared base meta so
        every civ's observation carries them."""
        if self.diplomacy is not None and self._obs_base is not None:
            self._obs_base["meta"]["diplomacy_offers"] = \
                self.diplomacy.public_offers()

    def step(self, actions: dict):
        st = self.state
        if st is None:
            raise RuntimeError("call reset() first")
        if st.done:
            raise RuntimeError("game is over; reset to play again")
        report = self.kernel.step_turn(st, actions or {}, self._legal_cache,
                                       information=self.info,
                                       diplomacy=self.diplomacy)
        self._legal_cache = {}

        scores = [c.score for c in st.civs]
        # mistake_punishment dial: each of a civ's dropped (stale)
        # orders subtracts the dial value from that civ's reward.  Pure
        # reward shaping: scores, the ledger and the hidden state are
        # untouched, and 0 (the default) shapes nothing.
        penalty = float(self.dials.get("mistake_punishment", 0.0) or 0.0)
        rejected = [0] * len(st.civs)
        if penalty > 0.0:
            for e in report.events:
                if e.get("type") == "rejected" and isinstance(e.get("civ"), int) \
                        and 0 <= e["civ"] < len(rejected):
                    rejected[e["civ"]] += 1
        rewards = {}
        for i in range(len(st.civs)):
            r = float(scores[i] - self._prev_scores[i]) - penalty * rejected[i]
            if report.done:
                term = float(self.tables.constants["terminal_reward"]["value"])
                if st.winner is not None and i == st.winner:
                    r += term
                elif st.winner is not None:
                    r -= term
            rewards[i] = r
        self._prev_scores = scores

        self._obs_base = build_base_observation(st, self._matrix())
        self._stamp_diplomacy_meta()
        obs = {civ: self.info.build_observation(st, civ, self._obs_base)
               for civ in range(len(st.civs))}
        terminations = {civ: report.done for civ in range(len(st.civs))}
        truncations = {civ: False for civ in range(len(st.civs))}
        infos = {civ: {"report": report, "winner": st.winner,
                       "done_reason": st.done_reason}
                 for civ in range(len(st.civs))}
        return obs, rewards, terminations, truncations, infos

    def legal_actions(self, civ: int) -> list[tuple]:
        # per (turn, civ) memo: agents and validate share one enumeration;
        # the information module appends purchase actions when active (and
        # re-keys ids into observer-local handles); the diplomacy module
        # appends OFFER/ACCEPT/REJECT when active
        key = (self.state.turn, civ)
        lst = self._legal_cache.get(key)
        if lst is None:
            lst = legal_actions(self.state, civ, self.info, self.diplomacy)
            self._legal_cache[key] = lst
        return lst

    def observation(self, civ: int) -> dict:
        return self.info.build_observation(self.state, civ, self._obs_base)

    def flat_observation(self, civ: int) -> np.ndarray:
        if self._layout is None:
            self._layout = flat_layout(self.state)
        return encode_flat(self.observation(civ), self._layout)

    def render_text(self, civ: int) -> dict:
        from .textrender import render_text

        return render_text(self.observation(civ), self.legal_actions(civ))

    def belief(self, civ: int, particles: int | None = None, kind: str = "auto"):
        """Belief over the hidden state for `civ`, built from its observation
        alone (design §2): exact support enumeration on small maps, particle
        filter elsewhere.  Draws come from the named `belief` stream derived
        purely from the master seed, so calling this never perturbs the run's
        rng draw counters (records stay byte-identical).  Under the default
        full-observability module the belief is trivial: everything is
        visible."""
        from .belief import build_belief

        seed = stream_seed(self.state.seed, "belief")
        return build_belief(self.observation(civ), self.tables, seed=seed,
                            particles=particles, kind=kind)

    # -- search interface (design §3) ----------------------------------------
    def clone_state(self) -> GameState:
        return self.state.clone()

    def hidden_state_hash(self) -> str:
        return self.state.state_hash()
