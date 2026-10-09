# Data constants and their provenance

GRANDSTRAT uses a **plain fantasy setting**. It does not reproduce any
commercial title. Every number in `grandstrat/data/*.json` is tagged
`"source": "ASSUMED"`: a value chosen so the simulator is playable, fast and
analytically simple, not a claim about any existing game.

The tables are the only home of content numbers. The engine reads them and
hard-codes none. Two tests keep this file and the tables in sync:
`test_constants_documented` fails if a key of `constants.json` or
`information.json` is not named here, and `test_every_constant_is_tagged_assumed`
fails if any table entry claims another source.

The tables are hashed (sha256) into every game record header, so a record
states exactly which content it was played with.

## Files

| file | contents |
|---|---|
| `terrain.json` | plains / forest / hills / mountain / water: passability, move cost, defense multiplier, base food/production/gold |
| `units.json` | 11 unit types (settler, worker, scout, warrior, archer, spearman, horseman, catapult, plus knight, crossbowman, musketman unlocked by tech): move, strength, ranged strength/range, production cost, gold upkeep, sight, tech gate, civilian flag |
| `buildings.json` | barracks, granary, walls, library, market, university, bank, laboratory: cost, tech gate, flat yield bonuses, city defense / extra HP |
| `techs.json` | 24 techs in 4 tiers: cost (science), explicit prereqs, unlocks (units/buildings/improvements), optional `abilities` |
| `improvements.json` | farm (plains), mine (hills), lumberyard (forest), trading_post (plains/forest/hills): worker-built flat yield bonuses, each with a tech gate |
| `constants.json` | combat formula, city growth/borders, barbarians, scoring, worldgen thresholds, caps, diplomacy |
| `information.json` | information subsystem: city sight, the fog dial's purchase rungs and gold prices, spy tech gate and duration, trade corridor radius, SCOUT's in-kind pricing, belief particle count and support cap |

## Combat

| constant | value | role |
|---|---|---|
| `damage_base` | 28 | expected damage at strength parity |
| `damage_exponent` | 0.75 | exponent on the (A/D) strength ratio |
| `roll_low` / `roll_high` | 0.8 / 1.2 | uniform CRN roll band around the mean |
| `unit_hp_max` | 100 | unit HP |
| `hp_strength_floor` | 0.5 | strength scales linearly 0.5x to 1.0x with HP |
| `flank_bonus` / `flank_max` | +0.15 / 2 | per friendly adjacent military unit of the defender, capped |
| `city_base_def` / `city_def_per_walls` | 6 / +4 | city defense strength |
| `city_hp_max` / `city_hp_per_walls` | 200 / +60 | city HP |
| `city_ranged_damage_factor` | 0.5 | ranged damage against cities is halved; cities do not retaliate against ranged attacks |
| `city_capture_hp_fraction` | 0.5 | a captured city resets to half max HP; population halves (min 1) |
| `heal_per_turn` | 5 | units that did not act heal in own territory; unattacked cities heal |

Damage per hit is `damage_base * (A/D)^damage_exponent * U[roll_low, roll_high)`.
Combat rolls come from the `combat` stream with world-keyed keys
`(purpose, turn, attacker_wid, defender_wid, roll_index)`, where the
participants' world identities are birth facts (see
[DESIGN_NOTES.md](DESIGN_NOTES.md) §13), never path counters or creation-order
ids.

## Economy and cities

| constant | value |
|---|---|
| `city_center_yield` | food 2, prod 2, gold 1, science 1 |
| `eat_per_pop` | 2 food per pop per turn |
| `growth_base` / `growth_per_pop` | 15 + 8 x pop food to grow |
| `border_culture` | border radii 1..5 at culture 0 / 15 / 50 / 120 / 250 |
| `base_culture_per_turn` | 1 culture per city per turn |
| `work_score_weights` | `{"food": 2.0, "prod": 1.5, "gold": 1.0}`: weights used to auto-pick worked hexes |
| `resource_yield` | `{"wheat": {"food": 2}, "iron": {"prod": 2}, "gold_ore": {"gold": 3}}` |
| gold from population | each city adds `pop // 2` gold (in `city_yields`) |
| `settler_min_pop` | 2 (a city needs pop >= 2 to complete a settler; founding consumes 1 pop) |
| `city_min_dist` | 2 hex distance between any two cities |
| `score_weights` | city 8, tech 6, territory hex 1, gold / 10 |
| `terminal_reward` | +/-100 at game end (winner +100, each loser -100) |
| `start_gold` | 20 |

## Barbarians and world generation

| constant | value |
|---|---|
| `barb_camps_default` | 6 camps (scenarios override this) |
| `barb_spawn_chance` | 0.06 per camp per turn (x `adversary_strength` dial, x the `nonstationarity` ramp) |
| `barb_horseman_chance` | 0.3: share of barbarian spawns that are horsemen (the rest are warriors) |
| `barb_max_per_camp` | 2 alive within one ring of the camp |
| `barb_spawn_radius` / `barb_aggression_radius` | 2 / 4 |
| `barb_clear_bounty` | 20 gold for ending a turn on a cleared camp |
| camp placement | >= 5 hexes from every capital, >= 4 from each other |
| `sea_level` / `land_falloff` | 0.45 / 0.25 (value-noise landmass with radial falloff) |
| `noise_scale` | 3 at R = 12, scaled `max(2, round(3R/12))` on smaller maps |
| `terrain_hill_cut` | 0.62: moisture at or above this is hills |
| `terrain_mountain_cut` | 0.85: moisture at or above this (within hills) is mountain |
| `terrain_forest_cut` | 0.4: moisture at or above this is forest; below is plains |
| `resource_chance` | 0.10 per land hex (x `resource_scarcity` dial); wheat / iron / gold_ore on plains / hills / forest |
| `unit_cap` / `city_cap` | 256 / 64 (observation array widths; production holds and barbarian spawns are skipped when the world is full) |

## Information subsystem

All in `data/information.json`. The fog dial W in {0..K}, K = 4 (`FOG_K` in
`scenarios.py`), selects rungs. These values only matter with the kernel
config `information: "fog"`.

| constant | value | role |
|---|---|---|
| `city_sight` | 2 | sight radius every city contributes to its owner's visible set |
| `spy_tech` | `"writing"` | SPY purchases require this tech (the tree has no espionage tech, so the gate reuses writing) |
| `spy_duration` | 10 | a spy reveals the target city's hex and production queue for this many turns |
| `trade_corridor_radius` | 2 | TRADE reveals the hexes within this radius of the hex line between the two capitals (one-shot memory write) |
| `rung_purchases` | `0: [], 1: [spy], 2: [spy, trade, diplo], 3: [spy, trade, diplo, oracle], 4: []` | purchase types enabled per rung; W = 4 is full observability, so nothing is for sale |
| `prices` | spy 80/60/40 (W=1/2/3), trade 50/40 (W=2/3), diplo 70/50 (W=2/3), oracle 150 (W=3) | gold price per purchase type per rung; a rung absent from a type's table means the type is not purchasable there |
| `scout_pricing` | `"in kind: unit production cost + per-turn upkeep + exposure risk"` | SCOUT is a unit's sight: no action and no gold price; it is paid for through the ordinary production/upkeep ledger |
| `belief_particles` | 64 | fixed particle count for the particle-filter belief on maps too large for exact enumeration |
| `belief_support_cap` | 200000 | support-size counting stops above this bound and reports the truncation (used in purchase record entries) |

SCOUT's "exposure" is the risk that the unit is attacked; it is part of the
description, not a charged number. Prices were chosen so that a civ with a
typical early-game income can afford roughly one purchase every few turns once
the rung enables it. No optimisation claim is made.

## Diplomacy

All in `data/constants.json`. Active only with kernel config
`diplomacy: "simple"` (the default `"none"` builds no diplomacy structure).

| constant | value | role |
|---|---|---|
| `diplomacy_offer_ttl` | 5 | an offer expires this many turns after it is made |
| `diplomacy_offer_grid` | `[0, 10, 25]` | gold side-payments an OFFER may propose (keeps the legal-action set finite); paid by the proposer to the acceptor on acceptance |
| `diplomacy_reservation` | open_borders 12, share_maps 20 | scripted acceptors demand at least this much gold per agreement kind |
| `diplomacy_threshold_jitter` | 8 | scripted acceptance threshold = reservation + (pid x 7 + acceptor x 13) mod 8: deterministic public pseudo-noise, no RNG draw |

## Tech abilities

`techs.json` entries may carry an `abilities` object. Two exist:
`civil_service` grants `city_sight_bonus: 1` (every city sees one hex farther)
and `guilds` grants `gold_per_turn: 3` (civ-wide income). With these, every
tech in the tree unlocks units, buildings, improvements, other techs, or an
ability. `improvements.json` entries carry their tech gate
(farm <- agriculture, mine <- mining, lumberyard <- trapping,
trading_post <- the_wheel), matching the tree's `unlocks.improvements`; a
test checks the two directions agree.

## Per-stream draw activity at defaults

Default scenario, four scripted agents:

| stream | used by | activity at defaults |
|---|---|---|
| `map` | terrain, resources | every reset |
| `spread` | capital placement | every reset |
| `combat` | every damage roll | every game with fights |
| `barbarians` | camp placement, spawns | every game with camps |
| `diplomacy` | user-supplied diplomacy modules | **0 draws**: the shipped diplomacy module is fully deterministic (data reservations plus public pseudo-jitter); the stream is reserved |
| `belief` | particle-filter belief draws | **0 draws unless `env.belief(..., kind="particles")` is requested**; derived from the master seed, so requesting beliefs never perturbs records |
| `agent_i` | scripted agents' `noise` parameter | **0 draws at defaults** (noise = 0); the tests exercise noise > 0 |

`stream_draws` is written into every record footer, so these counts can be
checked per game.
