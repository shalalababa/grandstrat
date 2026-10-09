# GRANDSTRAT design

This is the specification the simulator was built against. Section numbers
are stable: code comments cite them as "design §N". Where the specification
left a choice open, the choice made is recorded in
[DESIGN_NOTES.md](DESIGN_NOTES.md); the numbers are in
[CONSTANTS.md](CONSTANTS.md).

## 0. Purpose

A grand-strategy simulator in which **information is purchased with in-game
resources**. Fog of war, scouting, espionage and diplomacy are all ways of
paying to see. The simulator exposes the price and the exact effect of each
purchase, an information dial that controls how much can be seen and bought,
and belief tracking over what is hidden. Everything is observable from outside
the game and seedable, and every random stream is named. It is meant to be
fast enough for thousands of games per CPU-hour.

## 0b. General-purpose by construction

The engine is a general grand-strategy simulator. Architecture rules that keep
it reusable (multi-agent RL, LLM agents, planning under partial
observability, opponent modelling, negotiation):

- A rules **kernel** with data-driven content: units, buildings, techs,
  terrain and improvements live in JSON tables, not code.
- Modular subsystems behind interfaces: map generation, economy, combat,
  diplomacy, **information** (fog + purchases + dial), agents, recording.
  Modules are selected by config; the information module is optional and
  defaults to full observability.
- Two agent-facing APIs: (i) a Gym/PettingZoo-style multi-agent step API with
  array observations and a flat encoder; (ii) a **text renderer** that
  serialises a civ's observation and legal actions as structured text (JSON +
  a human-readable summary) for agents that act through a tool-call
  interface. Both are deterministic given the seed.
- A scenario and dial registry: named scenarios (map size, civ count, horizon,
  starting state) and dials (fog level, resource scarcity, adversary
  strength, punishment of single mistakes, non-stationarity) exposed as
  first-class parameters and logged per game.
- Belief support enumeration and CRN combat are modules; nothing in the
  kernel depends on them.
- Records are schema-versioned JSONL; a replay tool reconstructs any game
  from seed + action log.

## 1. World

- Hex map of radius R (default 12, 469 hexes); terrain from a seeded
  generator: plains / forest / hills / mountain / water; bonus resources
  (food, production, gold) per hex; 2-6 civilisations placed by a seeded
  spread rule; barbarian camps as neutral threats.
- Turn-based, simultaneous move resolution with deterministic ordering (a
  phase list); T turns (default 150).
- Units: settler, worker, scout, warrior, archer, spearman, horseman,
  catapult, plus a unit template table so techs can unlock new units without
  code. Attributes: move points, sight radius, strength, ranged strength and
  range, cost, upkeep.
- Cities: population, food/production/gold yields from worked hexes,
  buildings (granary, walls, barracks, library, market, and later ones),
  culture border growth, capture rules.
- Tech tree: 24 techs in 4 tiers, each unlocking units, buildings,
  improvements or abilities; research from science yield (automatic, see
  DESIGN_NOTES §3).
- Economy: gold income and upkeep; per-turn maintenance; bankruptcy disbands
  units deterministically.
- Combat: strength contest with terrain and flanking modifiers, damage
  formula, common-random-number (CRN) rolls keyed on **world identity**
  (purpose, turn, attacker, defender, roll index) and never on a path
  counter such as a fight index or call order.
- Victory and score: domination (one civ holds all original capitals), last
  civ standing, or highest score at T (cities + techs + territory + gold);
  a per-turn score is available for truncated games.

## 2. Information structure

- **Fog of war**: each civ sees only hexes within the sight of its units and
  cities, plus a "last seen" memory layer with staleness.
- **Purchasable information**, each with a price and exact semantics:
  - SCOUT: a unit's sight (paid in kind: the unit's production, upkeep and
    exposure to attack).
  - SPY (tech-gated): pay gold to reveal a chosen enemy city's contents and
    production queue for k turns.
  - TRADE: reveals the map within radius r of the route between the two
    capitals.
  - DIPLO: a live exchange of map layers with a partner.
  - ORACLE: reveal the full state to one civ at a gold price (the dial's top
    purchasable rung).
- **Information dial** W in {0..K}: 0 = full fog and no purchases; K = full
  observability; intermediate rungs enable purchase types and lower prices.
  The dial is a simulator parameter, logged in every game record.
- **Belief support**: on small maps (R <= 4, 2 civs) the engine enumerates
  the hidden configurations consistent with a civ's observations (exact
  support counting); on large maps it exposes a particle-filter belief with
  a fixed particle count. See DESIGN_NOTES §17 for the exact hypothesis space
  and its limits.
- **Diplomacy** (optional module): open-borders and share-maps agreements
  bargained through a simple bid-and-accept protocol (DESIGN_NOTES §23).

## 3. Agents

- Scripted baselines: expander, warmonger, turtle, scout-heavy, and a
  diplomacy-heavy merchant; each parameterised and deterministic given the
  seed.
- A search-agent interface `act(observation, belief, budget) -> actions` so
  MCTS, IS-MCTS or learned policies can plug in. The env exposes
  `clone_state`, `step`, `legal_actions(civ)`, `observation(civ)` and
  `hidden_state_hash` for determinization.
- A buy-or-search hook: at decision time an agent can spend gold on
  information or spend compute budget on search.

## 4. RNG streams

Named, independent, seedable streams: `map`, `spread`, `combat`
(world-keyed), `barbarians`, `diplomacy`, `belief`, and `agent_i` per civ.
Determinism requirement: same seeds + same actions give byte-identical
records. CRN requirement: two arms that differ only in the information dial
share `combat` rolls for identical fights.

## 5. Performance and interface

- Pure Python + numpy. Target: at least 200 full games (150 turns, 4 civs,
  R = 12, scripted agents, records written) per CPU-minute; checked by a test.
- Observation encoding: a per-civ dict of numpy arrays (map layers with fog
  mask, unit list, city list, tech bits, gold, diplomacy matrix) plus a flat
  vector encoder for RL.
- Game record: JSONL per game with per-turn scores, information purchases
  (what, price, what was revealed, belief support size before and after on
  small maps), combat rolls with their keys, and the dial values.
- CLI: `gs play --seed S --civs 4 --R 12 --dial W --agents expander,...` and
  `gs batch --n N ...` writing records; `gs replay` verifying them.

## 6. Tests

Determinism; CRN across dial arms; legal-action completeness (every legal
action executes, every illegal one is rejected); conservation (gold, food
and production accounting balances every turn); fog correctness (a civ never
observes a hex outside its sight unless purchased or shared, checked by an
adversarial probe); purchase semantics (each purchase reveals exactly what
its specification says and charges exactly its price); belief support
monotonicity on small maps; speed gate; record schema; no writes outside the
requested output directory.

## 7. Data constants

Data constants are ASSUMED (a plain fantasy setting, not a licensed game) and
listed in CONSTANTS.md, so nothing is presented as a reproduction of a
commercial title.

## 8. Non-goals

Graphics, real-time play, networking, a full commercial ruleset, balance for
human players.
