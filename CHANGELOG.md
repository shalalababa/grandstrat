# Changelog

## 0.3.0

Tech-tree gating, diplomacy, observer-local handles, search interface.

- Improvements are tech-gated (`improvements.json` `tech` field enforced).
  Tech abilities: `civil_service` (+1 city sight), `guilds` (+3 gold/turn).
  A test checks `techs.json` unlocks against the content tables in both
  directions.
- Domination now triggers as soon as one civ holds every original capital;
  rivals surviving on secondary cities no longer block it.
- Unit and city caps are enforced where entities are created; the observation
  builder raises instead of truncating.
- Fog observations and legal actions use observer-local negative handles for
  every unit and city id, so engine creation counters no longer leak the
  number of entities in the world. Records store the submitted (handle) form;
  replay reproduces the mapping.
- Diplomacy module (`diplomacy: "simple"`): public open-borders and
  share-maps agreements, OFFER / ACCEPT / REJECT with gold side-payments,
  closed borders without an agreement, agreement breaking on attack, ledger
  `transfer` column, `merchant` agent.
- Search interface: `SearchAgent.act(observation, belief, budget)`,
  `determinize`, `ForwardModel`, `Budget`, `DeterminizedSearchAgent`,
  `SearchAdapter`, `BuyOrSearchAgent`; per-turn search spending in the
  record's `compute` column. `gs bench --mode default|fog|search`.
- Dials `mistake_punishment` (reward shaping for dropped orders) and
  `nonstationarity` (barbarian spawn ramp) are functional.
- `gs replay` accepts records played by agents outside the registry (agents
  are never executed during replay).

Fixed:

- Agent caches keyed by map shape carried world-content-derived scores, so
  two different-seeded worlds of the same shape in one process influenced
  each other. Keys now include a hash of the terrain (and resources).
- Several agent choices depended on set iteration order and therefore on
  `PYTHONHASHSEED`. All such choices are now explicit sorts.
- Determinization ignored its `key` argument (every sample was the same);
  distinct keys now give distinct samples where the support allows.
- Determinization copied negative observer handles into engine occupancy
  grids, where negative means "empty", which broke targeting and movement in
  the model. Observed handles are now mapped to private nonnegative ids and
  chosen actions are mapped back.
- Remembered fog cities were detected by a nonnegative id and so were missed
  once ids became negative handles; presence is now read from the HP field.

## 0.2.0

Fog of war, purchasable information, the information dial, belief.

- Units carry an immutable world identity
  `(owner, birth_turn, birth_key, ordinal)`; combat roll keys use it instead
  of creation-counter ids. Two games that differ in the information dial (and
  therefore in production order) now share combat rolls for the same fight;
  a test constructs such a pair and checks the rolls are byte-identical.
- Fog module (`information: "fog"`): per-civ visibility, last-seen memory,
  visibility history, masked per-civ observations; an adversarial probe test
  recomputes visibility independently and checks every layer and the legal
  action set every turn.
- Purchases SPY / TRADE / DIPLO / ORACLE with prices per dial rung, a
  `purchase` ledger column, an `orders_information` phase, and
  `info_purchase` record events with belief-support sizes on small maps.
- Exact belief-support enumeration and a particle belief; new `belief`
  random stream that never perturbs game draws.

Fixed:

- `hex_line` cube rounding.
- A kill-advance or capture-advance could stack two units on one hex.
- An agent attack memo keyed by `id()` of the legal list could reuse the
  previous turn's data when Python reused the address.
- The text renderer listed own cities among visible enemy cities.

## 0.1.0

World generation, units, cities, economy, combat with CRN rolls, barbarians,
four scripted agents, JSONL records, replay, CLI.

Fixed before release:

- A batch could contain two actions for the same unit and both executed (a
  unit could move twice, or attack one defender twice with the same roll).
  `engine.validate` now rejects any batch that binds one unit or city twice,
  and noisy agents filter their choices to one action per actor.
- Bankruptcy disbanded units without crediting their upkeep, so a single
  deficit turn could disband a civ's entire army (measured: in six default
  games, all 37 bankruptcies disbanded units until the treasury clamped at 0,
  320 units in total). A disbanded unit's upkeep is no longer charged, so
  disbanding stops at solvency.
- The speed test gated on an aggregate rate that varied 2x with machine load;
  it now times each game and gates on the best sample.
- One barbarian constant was hard-coded; it moved to `constants.json`, and a
  test now requires every data key to be documented.

## Verified properties

Checked by the test suite on every run:

- Same seed and actions give byte-identical records; final state hashes match
  across subprocesses with different `PYTHONHASHSEED`.
- Every action in `legal_actions` executes on sampled mid-game states across
  scenarios; sampled non-listed actions raise `InvalidAction`.
- Gold, research, food and production ledgers balance every turn, and an
  independent recount of upkeep from the unit set matches the ledger.
- Under fog, no civ sees a hex outside its sight unless purchased or shared;
  every listed legal action is computable from the observation alone.
- Each purchase reveals exactly its specified set and charges exactly its
  price.
- Observation arrays are read-only and never alias state arrays;
  `clone_state()` is a deep copy.
- The engine and agents perform no file I/O during play; the CLI writes only
  under `--out`.
- At the full-observability rung, the fog module produces dynamics identical
  to the default module.
- Speed gate: at least 200 default games (150 turns, 4 civs, R = 12, records
  written) per CPU-minute on the best single-game sample.

## Known issues

- Belief covers only previously seen units with a uniform prior; the particle
  filter is a per-request consistency filter, not a sequential Bayesian
  filter; multi-unit data association is heuristic.
- Determinization uses neutral values for unknown resources, treasuries and
  techs, keeps stale remembered cities, and omits unseen units and camps; the
  default forward model does not reproduce every fog and diplomacy mechanism.
- A move judged legal on the mover's last-known territory owner can still be
  dropped at execution (`closed_borders`) if the owner changed unseen. This
  is the documented stale-order behaviour.
- The fog module costs about 1.6x the default configuration per game.
