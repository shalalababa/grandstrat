# Design notes: choices where the specification is silent

[DESIGN.md](DESIGN.md) is the specification. Where it left a choice open, the
simplest defensible option was taken and recorded here. Section numbers are
stable (code comments cite them).

## 1. Turn structure and phases

One `env.step` resolves one full turn for all civs simultaneously. Submitted
action batches are validated upfront against `legal_actions`; resolution then
walks the phase list in this order:

1. `orders_produce`: PRODUCE queue sets (civ id order, submission order)
2. `orders_move`: MOVE for all civs, global unit-id order
3. `orders_attack`: ATTACK_U / ATTACK_C for all civs, attacker-id order
4. `orders_civilian`: FOUND / IMPROVE, global unit-id order
5. `orders_diplomacy`: ACCEPT / REJECT then OFFER, civ id order (diplomacy
   module only)
6. `orders_information`: SPY / TRADE / DIPLO / ORACLE, civ id order (fog
   module only)
7. `income`: borders, food/production, gold/science, research, upkeep,
   bankruptcy disbands, healing
8. `barbarians`: camp spawns, then barbarian actions
9. `cleanup`: camp clearing, eliminations, territory, victory, scores

**Stale vs illegal.** An action rejected by the upfront check raises
`InvalidAction` to the caller; the step is transactional and nothing
resolves. An action that was legal when validated but is stale at resolution
(its unit died to an earlier order, its target city was captured first, its
move hex filled up) is dropped and recorded as a `rejected` event with a
reason. It never crashes and never silently vanishes. Ownership is
re-checked at attack resolution, so two siege attacks on the same enemy city
in one turn resolve as capture + drop, never as a self-attack on the newly
captured city.

## 2. One action per unit per turn

Agents submit at most one action per unit or city per turn; `engine.validate`
rejects a batch that binds the same unit (MOVE / ATTACK_U / ATTACK_C / FOUND
/ IMPROVE) or the same city (PRODUCE) twice. A unit with move 2 (scout,
settler, horseman) therefore spends its movement on one expensive hex entry
per turn (hills cost 2): move points gate terrain cost, not double moves.

## 3. Research is automatic

Research always picks the cheapest available tech, ties broken by name;
`research_bucket` overflow carries into the next tech. There is no research
action in the legal-action space.

## 4. Workers and improvements

`IMPROVE` builds the terrain's improvement (per `improvements.json`)
instantly, consumes all remaining move points, and is legal only on
own-territory land without a city or existing improvement, and only once the
improvement's tech is known. One improvement per hex. No pillaging and no
roads.

## 5. Cities: capture, capitals, elimination, victory

- One garrison unit may share a city hex (one unit per hex otherwise;
  civilians count as units).
- A melee attack that reduces city HP to <= 0 captures the city. Ranged
  attacks never capture; city HP clamps at 1 if the melee attacker died to
  retaliation or is a barbarian. On capture: HP resets to half max,
  population halves, the old owner's garrison is destroyed, buildings are
  kept (walls included).
- The capital flag is permanent and attached to each civ's founding city. A
  civ with zero cities is eliminated and its units are removed.
- **Victory**: domination as soon as one civ owns every original capital
  location (rivals surviving on secondary cities do not prevent it); else
  last civ standing; else at turn T the highest score wins (tie: lower civ
  id). Winner -1 with reason `extinction` is a defensive edge that should be
  unreachable.

## 6. Barbarians

Camps are placed at world generation (>= 5 hexes from capitals). The spawn
roll per camp per turn comes from the `barbarians` stream, keyed
`(purpose, turn, camp_id)`. Barbarians are owner -1, never found cities,
never capture (city HP clamps at 1), respect no borders, and act with a
deterministic nearest-target greedy policy (ties by id, then direction
order). World generation retries (bounded, deterministic, derived seed) if a
standard start cannot be placed on a degenerate map; replay repeats the same
loop, so seeds stay pinned.

## 7. Rewards

Per-turn reward for each civ = **change in that civ's score**, plus terminal
+100 (winner) / -100 (each loser) on the step where `done` is set. The
terminal reward lands on the same step as `done`, so a learner never misses
it. The `mistake_punishment` dial adds reward shaping (§25).

## 8. Dials

| dial | effect |
|---|---|
| `fog` (0..K, K = 4) | functional with `information: "fog"`; selects fog depth and purchasable information (§16). Under the default `full` module it is logged but has no effect. |
| `resource_scarcity` | scales bonus-resource placement at world generation |
| `adversary_strength` | scales barbarian spawn chance |
| `mistake_punishment` | reward shaping for dropped orders (§25) |
| `nonstationarity` | barbarian spawn chance ramps up toward the horizon (§25) |

All dials are validated, logged in every record header, and listed by
`gs list`.

## 9. RNG streams

Counter-based draws: splitmix64 over `(stream_seed, key_tuple)`. There is no
generator state, so a draw is a pure function of its key. Streams: `map`,
`spread`, `combat`, `barbarians`, `diplomacy`, `belief`, and `agent_i` per
civ. Each stream can be re-seeded independently through the public API
(`--stream-seed name=seed`).

- `combat` and `barbarians` draw heavily in every full game; `map` and
  `spread` draw at reset.
- `agent_i` draws zero times at defaults: scripted agents are deterministic
  without draws. The stream is used only when an agent is constructed with
  `noise > 0` (the tests exercise this).
- `diplomacy` draws zero times: the shipped diplomacy module is
  deterministic (§23). The stream is reserved for user-supplied modules.
- `belief` draws only when a particle belief is requested, and never
  perturbs the game's own counters.

Every record footer carries per-stream draw counts so these claims can be
checked per game.

## 10. Observation contract

`build_base_observation` produces one shared, read-only view per step;
`InformationModule.build_observation` wraps it per civ with a `you` field.
Arrays are marked non-writeable, so sharing is safe. Buffers are valid until
the next `step`/`reset` (standard gym buffer semantics);
`flat_observation`/`render_text` copy immediately. The full `GameState` is
reachable only through the search interface (`clone_state` /
`hidden_state_hash`), never through the agent-facing API. The fog module
replaces the shared base with per-civ masked copies from the same builder.

## 11. Record ledger

Per-turn records carry the full conservation ledger: gold rows, research
rows, and per-city food/production rows with explicit
before/income/spent/after arithmetic. This is larger than a minimal record
(about 0.5-0.9 MB per default game) but makes conservation independently
checkable from records alone.

## 12. Speed approach

The speed target (DESIGN §5) is met with plain Python + numpy: counter-based
RNG, precomputed hex geometry, occupancy grids maintained by single-writer
engine helpers, per-hex yield arrays, per-(turn, civ) legal-action
memoization shared by agents and validation, vectorized agent scans, a
memoized per-shape distance field, and memoized value-noise lattice draws.

## 13. Unit world identity

`spawn_unit` assigns every unit a world identity
`wid = (owner, birth_turn, birth_key, ordinal)`. `birth_key` is the birth
city's founding-hex index for produced units, the camp id for barbarians, and
-1 for ad-hoc spawns; `ordinal` counts same-`(owner, turn, birth_key)` births
in the game (always 0 in engine play, since production queues are
single-item). The wid is immutable, cloned with the state, hashed into
`state_hash`, logged in `unit_built` / `unit_died` / `barb_spawn` events, and
used in every combat roll key `(purpose, turn, attacker_wid, defender_wid,
roll_index)`; a city's wid in city fights is `("city", founding_hex)`. The
integer creation-counter `id` is only an indexing handle.

Why: creation-counter ids are path-dependent. Two arms that diverge in
anything upstream of production (which is what the information dial does)
would give corresponding units different ids, and CRN pairing would silently
stop. With wids, the same fight between the same world-identified units draws
the same roll across arms. Births inside one `(owner, turn, city)` bucket are
still path-counted by design: arms whose birth sets differ there have no
corresponding units.

## 14. Fog: what is public, what is masked

- Terrain is public static geography (a printed map). Masking it would add
  no information structure and would complicate agents' terrain caches.
- Resource, improvement and owner layers show current truth only on
  currently visible hexes (sentinels -1 / -1 / -2 elsewhere; owner -1 =
  unowned is distinct from -2 = unknown). Own units are visible everywhere,
  foreign units only on visible hexes; cities likewise. Per-civ tables
  (gold, score, techs, research) are visible for self only; `alive` flags are
  public.
- Memory layer ("last seen"): per hex, the turn last seen plus
  resource / improvement / owner / unit summary / city summary as they were
  when seen, updated only on visible hexes. Exposed read-only as
  `obs["memory"]`; per-turn visibility history as `obs["vis_history"]` (used
  by the belief module's reachability filter).
- Visibility = sight radius of own units (`units.json` `sight`) plus
  `city_sight` of own cities (+ tech bonus), recomputed per turn with cached
  sight-reach matrices.
- At dial K and after an ORACLE purchase the shared full-observation base is
  handed out unmasked.
- `legal_actions` is computed by the engine from the hidden state, but every
  listed action is computable from the observation alone (every attack/move
  target is within the acting unit's sight; the civ's own purchase history is
  `obs["meta"]["info_state"]`). The fog probe test asserts set equality every
  turn.
- Entity ids in fog observations are observer-local handles (§22).

## 15. Purchases

- SCOUT is priced in kind (a unit's production + upkeep + exposure): no
  action and no gold price. Its information is the fog module's ordinary
  unit sight.
- SPY / TRADE / DIPLO / ORACLE are actions `("SPY", city_id)`,
  `("TRADE", partner)`, `("DIPLO", partner)`, `("ORACLE",)`. They are legal
  only under the fog module at a rung that enables them
  (`data/information.json`) and only when affordable; SPY additionally needs
  the gate tech and a target city the buyer has seen (visible now or in
  memory); DIPLO not twice with the same partner; ORACLE once. At most one
  purchase per civ per turn (batch rule in `engine.validate`).
- Revealed sets: SPY = live visibility of the target city's hex for
  `spy_duration` turns (queue observable through the city row). TRADE =
  one-shot memory write of the corridor within `trade_corridor_radius` of the
  hex line between the two capitals (applied at the next observation build,
  so contents are post-turn). DIPLO = live mutual union of the two civs'
  visible sets from then on, plus a one-shot merge of the partner's memory
  into hexes unknown to the buyer. ORACLE = the buyer's observations become
  the full state, permanently.
- Resolution happens in the `orders_information` phase. The charge lands in
  the ledger's gold rows as an explicit `purchase` column:
  `before - purchase + income - upkeep + bounty + transfer == after`.
  `info_purchase` record events name the buyer, the price, what was
  revealed, and (R <= 4 only) the belief-support size before and after; the
  "after" size is measured at the next observation build.

## 16. The dial

W in {0..K}, K = `FOG_K` = 4: W = 0 full fog and no purchases; W = 1 adds
SPY; W = 2 adds TRADE and DIPLO; W = 3 adds ORACLE and lowers prices; W = 4
is full observability with nothing purchasable. The dial is validated
(int 0..K) and logged in every record header and in observation meta
(`fog_dial`). Under `information: "fog"`, W = K is dynamics-identical to the
full module (asserted by a test).

## 17. Belief

The hypothesis space is the joint position of foreign units the observer has
seen at least once (currently visible ones are pinned facts), plus, per
memory track, a GONE hypothesis (the unit may have died unseen). Consistency
is exact: a track's candidates are the time-expanded reachable set from its
last sighting (per-turn BFS under the unit's movement rules, never standing
on a hex the observer could see at that turn, since it would have been
re-seen), excluding currently visible and known-occupied hexes; joint
configurations additionally require distinct positions. A memory record is
resolved when exactly one visible unit of the same owner and type sits in its
candidate set.

Exact enumeration (small maps) and the particle filter (fixed N =
`belief_particles`, seeded from the `belief` stream, systematic resampling,
ESS reported) range over the same space. Belief structures are built on
request only and never drawn during play, so records stay byte-identical.

Scope, stated plainly: this is a support over **tracked-unit
configurations**, not an exact posterior over every possible game history.
Units never seen do not exist in the hypothesis space; city capture,
production and economy histories are not hypothesized; the prior over
configurations is uniform. The particle filter is a static consistency filter
per request, not a sequential Bayesian filter. Data association for multiple
same-type units is heuristic (single-match resolution; ambiguous sightings
stay as separate tracks).

## 18. Record format

Schema `gs.record/v1`, one JSON object per line:

- **header**: seed, scenario, resolved params, dials, stream seeds, kernel
  config, agents, sha256 of each data table, initial state hash.
- **turn** lines: submitted actions (handle forms under fog), events, all
  combat rolls with their keys, the conservation ledger, scores, gold, the
  per-turn state hash, done/winner; an optional `compute` column (per-civ
  search expansions/determinizations, present only when a search agent
  plays).
- **footer**: final scores, winner, reason, per-stream draw counts.

There are no wall-clock or CPU timing fields, so identical seeds and actions
give byte-identical files. Additions since the first version are all
backward compatible (gold-row `purchase` and `transfer` columns,
`info_purchase` and `dipl_*` events, `wid` on unit events, world-identity
roll keys, the `compute` column); replay uses only the seed and the action
log.

## 19. Tech tree

- `unlocks.improvements` is enforced: `improvements.json` entries carry a
  `tech` gate and `can_improve` checks it.
- Techs may carry an `abilities` object, summed by `economy.ability_sum`.
  Two exist: `civil_service` -> `city_sight_bonus: 1` and `guilds` ->
  `gold_per_turn: 3` (visible in the ledger's income column).
- A consistency test pins `techs.json` `unlocks` against the content tables'
  `tech` fields in both directions, and requires every tech to unlock
  something or be a prereq of another tech.

## 20. Domination

`_cleanup` ends the game as `domination` as soon as one civ owns every
original capital location. A rival surviving on secondary cities does not
prevent it.

## 21. Entity caps

`unit_cap` / `city_cap` are enforced at the transitions that create
entities: `spawn_unit` refuses to overflow (production holds; barbarian
spawns are skipped before their roll, so draw counts stay honest), and
`build_base_observation` raises `ValueError` rather than silently truncating
if a cap were ever exceeded by another route.

## 22. Observer-local handles

Every unit and city id in a fog observation (unit rows, city rows, memory
`city_id`, and the agent-facing legal-action list) is an observer-local
handle: a stable, per-civ negative integer assigned at first sighting in
deterministic (id-sorted) order. Engine creation counters would otherwise
leak how many entities the whole world has spawned.

- `FogInformation.present_actions` re-keys the engine's true-id legal list
  into handles; `resolve_action` maps them back right after validation, so
  every kernel phase resolves in true-id space. Records quote the submitted
  (handle) forms; replay rebuilds the same mapping because it is a pure
  function of the trajectory.
- Handles persist for the whole game. DIPLO-merged memory entries get handles
  lazily at observation build.
- At dial K and after an ORACLE the module hands out the shared base view
  with true ids and the mapping is the identity.
- Belief construction is unaffected: tracks are keyed by (owner, type, hex,
  turn) facts, never by ids.

## 23. Diplomacy protocol

Kernel config `diplomacy: "simple"`; default `"none"`. See
`grandstrat/diplomacy.py` for the full semantics.

- Agreements are bilateral and public (observation matrix: 1.0 share_maps /
  0.5 open_borders / 0.0 none; live offers in
  `obs["meta"]["diplomacy_offers"]`).
- A bid is `("OFFER", partner, kind, gold)` with gold from a data-driven grid
  (0 / 10 / 25), paid by the proposer to the acceptor on acceptance. Offers
  live `diplomacy_offer_ttl` turns; one diplomacy order per civ per turn;
  ACCEPT / REJECT are legal only for the addressee.
- open_borders permits entering and healing in the partner's territory. With
  the module active, entering a foreign civ's territory without open borders
  is illegal. Legality uses the mover's knowledge of the hex owner (truth on
  visible hexes, last-seen owner from memory, unknown if never seen);
  execution re-checks the true owner and drops a stale move as
  `closed_borders`. This keeps "legal actions are computable from the
  observation" exact under fog. share_maps never opens borders; barbarians
  respect no borders; attacks ignore borders.
- share_maps reuses the fog module's live map exchange.
- Any attack on a partner breaks the agreement (`dipl_broken`); eliminations
  clear both sides.
- Gold transfers appear in the ledger's `transfer` column.
- Scripted acceptance = data reservation + public pseudo-jitter
  (`(pid*7 + acceptor*13) mod jitter`), with no RNG draws. The `merchant`
  scripted agent bids top-of-grid every `offer_every` turns so negotiations
  actually happen in scripted games.

## 24. Search interface, determinization, buy-or-search

- `grandstrat.search.SearchAgent.act(observation, belief, budget)` is the
  plugin surface. `SearchAdapter` bridges it into the scripted runner.
- `determinize(observation, belief, tables, key)` builds a playable state
  from one civ's observation plus a belief sample. It never receives the true
  hidden state: visible units sit at their observed hexes, memory tracks take
  a sampled candidate or GONE, never-seen units do not exist, unknown terrain
  data uses a neutral prior, and territory is recomputed from the
  determinized world's own cities. Distinct keys select distinct samples
  where the support allows; the same key is deterministic.
- Observed handles are negative, while engine occupancy uses negative values
  for "empty". Determinization therefore maps observed handles to private
  nonnegative model ids, and search maps chosen root actions back to
  observation-space actions, so planned actions are directly submittable to
  the real env. Remembered cities are detected by their HP marker, not by id
  sign. Targets that exist only in a belief sample are never returned as
  root actions.
- `ForwardModel` steps cloned or determinized states through the real kernel
  with the full information module.
- `DeterminizedSearchAgent` is a bounded baseline: D determinizations,
  scripted-pilot rollouts of H turns, best-mean root action. Compute is
  charged in expansions (forward-model turns) and determinizations against an
  explicit `Budget` and never touches the gold ledger. Runners log per-turn
  spending in the record's `compute` column.
- `BuyOrSearchAgent`: each turn it either buys one information action
  (gold) or spends compute on bounded search, never both. Modes `info_first`
  (default) and `alternate`.
- Known approximations: unknown resources, treasuries and techs are neutral;
  remembered cities keep stale pop/HP/owner; unseen camps and units do not
  exist; the default forward model does not reproduce every diplomacy/fog
  mechanism. Rollouts are therefore an approximate model of the hidden world.

## 25. mistake_punishment and nonstationarity

- `mistake_punishment` M (default 0): each of a civ's dropped (stale) orders
  in a turn subtracts M from that civ's reward. Pure reward shaping: scores,
  ledgers and dynamics are untouched (tested by identical state and events
  across M = 0 and M > 0). Invalid submissions raise at validation and are
  not reward events.
- `nonstationarity` N (default 0): barbarian spawn chance is scaled by
  `1 + N * turn / T`.

## 26. In-process determinism

Two classes of defect were found and fixed; tests guard both.

- **Cache contamination across worlds.** Agent caches keyed by map shape
  carried world-content-dependent arrays, so two different-seeded worlds of
  the same shape played in one process shared scores. Keys now include a
  sha1 of the terrain (and resources, where used), with a bounded cache size.
- **Hash-seed-dependent decisions.** Agents iterating sets of action tuples
  took "first seen" answers. All such choices are now total functions
  (min / max / sorted by explicit keys). The suite passes under different
  `PYTHONHASHSEED` values, and a test compares final state hashes across
  subprocesses with different hash seeds.
