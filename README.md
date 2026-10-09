# GRANDSTRAT

A headless, deterministic, turn-based grand-strategy simulator on a hex map,
written in pure Python + numpy. Civilisations expand, build, research, trade
and fight under fog of war. Information has a price: civs can pay gold to spy,
open trade routes, exchange maps or buy a full reveal, and an information dial
controls how much is visible and what is for sale. The engine tracks what each
civ could believe about hidden units, and exposes a search interface
(cloning, determinization, a forward model and a compute budget) for planning
agents.

There are no graphics. Agents interact through a Gym/PettingZoo-style
parallel API or a text renderer, and every game is written to a JSONL record
that a replay tool can verify turn by turn.

## Features

- **World**: axial hex map (radius 12 = 469 hexes by default), value-noise
  terrain (plains, forest, hills, mountain, water), bonus resources,
  2-6 civs, barbarian camps. Four scenarios: `default`, `duel`, `tiny`,
  `dense`.
- **Rules**: 11 unit types, 8 buildings, 4 worker improvements, a 24-tech
  tree in 4 tiers, city growth and culture borders, upkeep and bankruptcy,
  combat with terrain, flanking and ranged attacks, city capture, and
  domination / last-civ-standing / score-at-horizon victory. All content
  numbers live in JSON tables (`grandstrat/data/`).
- **Simultaneous turns**: every civ submits a batch of actions; one step
  resolves all of them through a fixed phase order. Illegal actions are
  rejected up front; orders that go stale mid-turn are dropped and logged.
- **Fog of war** (optional module): per-civ visibility from unit and city
  sight, a "last seen" memory layer, and observer-local ids so observations
  never leak global entity counts.
- **Purchasable information**: SPY (watch a city for 10 turns), TRADE (reveal
  a corridor between capitals), DIPLO (live map exchange), ORACLE (full
  reveal), each with a gold price, plus scouting paid in kind.
- **Information dial** `fog` W in 0..4: from full fog with nothing for sale,
  through rungs that enable purchases and lower prices, to full observability.
- **Belief**: exact enumeration of the hidden unit configurations consistent
  with a civ's observations on small maps, and a fixed-size particle belief on
  larger maps.
- **Diplomacy** (optional module): public open-borders and share-maps
  agreements, negotiated through a bid-and-accept protocol with gold
  side-payments.
- **Search interface**: `clone_state`, `hidden_state_hash`, a
  `determinize(observation, belief)` function that builds a playable world
  from one civ's observation without touching the true hidden state, a
  forward model, a bounded determinized-search baseline, and a
  buy-or-search agent that each turn either buys information or spends
  search budget.
- **Scripted agents**: `expander`, `warmonger`, `turtle`, `scout`
  (alias `scout_heavy`), `merchant`.
- **Dials**: `fog`, `resource_scarcity`, `adversary_strength`,
  `mistake_punishment` (reward penalty per dropped order), `nonstationarity`
  (barbarian pressure ramps up over the game).
- **Records and replay**: schema-versioned JSONL records with every action,
  event, combat roll and a full conservation ledger; `gs replay` re-runs a
  game from its seed and actions and checks every turn's state hash.

## Quickstart

Requires Python 3.11+ and numpy (tested with Python 3.13 and numpy 2.2).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # pinned numpy + pytest
pip install -e .                    # optional: installs the `gs` command

# bin/gs works without installing; it uses `python3` from PATH, or $GS_PYTHON
bin/gs play --seed 0                                         # default scenario, 4 scripted civs
bin/gs play --seed 0 --scenario duel --dial 2 --config information=fog --out records/fog
bin/gs play --seed 0 --scenario duel --dial 2 --config information=fog \
            --config diplomacy=simple --agents merchant,expander --out records/diplo
bin/gs batch --n 8 --seed-base 0 --out records/batch
bin/gs replay --record records/batch/game_seed3.jsonl
bin/gs bench --mode default --n 8    # also --mode fog, --mode search --scenario tiny
bin/gs list                          # scenarios and dials

python -m pytest -q
```

Example output:

```
$ bin/gs play --seed 0
seed 0: winner civ 0 (score) after 150 turns; scores [511, 149, 195, 264]
record: records/game_seed0.jsonl
$ bin/gs batch --n 8 --seed-base 0 --out records/batch
batch: 8 games in 1.64 s CPU (1.73 s wall)
speed: 292.1 games/CPU-minute (records in records/batch)
$ bin/gs replay --record records/batch/game_seed3.jsonl
OK: 150 turns replayed, all checks match (records/batch/game_seed3.jsonl)
```

`--dial` takes `name=value` or a bare integer (shorthand for `fog=W`);
`--config` sets kernel modules (`information=full|fog`,
`diplomacy=none|simple`); `--stream-seed name=seed` re-seeds one random
stream. The fog dial has an effect only with `--config information=fog`.

## Environment API

```python
import random

from grandstrat.env import GrandStratEnv

env = GrandStratEnv(scenario="duel", dials={"fog": 2},
                    config={"information": "fog"})
obs = env.reset(seed=0)                       # {civ: observation dict}
rng = random.Random(0)
done = False
while not done:
    actions = {}
    for civ in env.possible_agents:
        if not obs[civ]["alive"][civ]:
            continue
        legal = env.legal_actions(civ)        # list of action tuples
        actions[civ] = [rng.choice(legal)]    # at most one action per unit/city
    obs, rewards, terms, truncs, infos = env.step(actions)
    done = all(terms.values())

print("winner", infos[0]["winner"], "by", infos[0]["done_reason"])
print(env.render_text(0)["summary"].splitlines()[0])
belief = env.belief(0)                        # particle belief on this map size
```

- `env.observation(civ)`: dict of read-only numpy arrays (terrain, resource,
  owner, improvement layers; unit and city lists; visibility and memory under
  fog; techs, gold, score; diplomacy matrix; `meta`). Buffers stay valid
  until the next `step`/`reset`.
- `env.flat_observation(civ)`: one float32 vector for RL.
- `env.render_text(civ)`: `{"json": ..., "summary": ...}` with the
  observation and legal actions as text, for agents that act through tool
  calls.
- Actions are tuples such as `("MOVE", unit, direction)`,
  `("ATTACK_U", unit, target)`, `("PRODUCE", city, item)`,
  `("FOUND", unit)`, `("IMPROVE", unit)`, `("SPY", city)`, `("TRADE", civ)`,
  `("DIPLO", civ)`, `("ORACLE",)`, `("OFFER", civ, kind, gold)`,
  `("ACCEPT", offer)`, `("SKIP",)`. The full list is in
  `grandstrat/engine.py`.
- Reward per step = change in the civ's score, plus +100 / -100 to the
  winner / each loser on the final step.
- `env.clone_state()` and `env.hidden_state_hash()` give the full hidden
  world for search; they are not part of what an agent observes.
- `grandstrat.record.play_game(env, seed, agents, writer)` runs a full game
  with agents from `grandstrat.agents.make_agents`; `grandstrat.search` has
  `determinize`, `ForwardModel`, `Budget`, `DeterminizedSearchAgent`,
  `SearchAdapter` and `BuyOrSearchAgent`.

## Determinism and replay

- All randomness comes from named counter-based streams (`map`, `spread`,
  `combat`, `barbarians`, `diplomacy`, `belief`, `agent_i`). A draw is a pure
  function of the stream seed and a key, so there is no hidden generator
  state, and each stream can be re-seeded on its own.
- Same seed + same actions give byte-identical records. Records contain no
  timing fields. Results do not depend on `PYTHONHASHSEED` (tested across
  subprocesses) or on which games ran earlier in the same process (a game
  played alone and the same seed played inside a batch give identical
  records).
- Combat rolls are keyed on the world identity of the units involved
  (owner, birth turn, birth city, ordinal), not on creation order. Two games
  that differ only in the information dial therefore draw the same roll for
  the same fight (common random numbers), which makes paired comparisons
  across dial settings meaningful.
- Each record line carries a sha256 state hash; the header carries the
  sha256 of the data tables. `gs replay` rebuilds the game from the header
  and the action log and checks hashes, scores, gold, the done flag and the
  winner on every turn. Tampered seeds or actions are detected.
- Each turn's ledger satisfies
  `before - purchase + income - upkeep + bounty + transfer == after` for gold,
  with matching rows for research, food and production; tests check it
  every turn.

## Design overview

`GrandStratEnv` owns a `GameState` (the full hidden world) and a `Kernel`
that resolves one turn in fixed phases: produce, move, attack, civilian
orders, diplomacy, information purchases, income, barbarians, cleanup.
Subsystems sit behind small interfaces and are chosen by config: combat
(`crn`), information (`full` or `fog`), diplomacy (`none` or `simple`),
world generation (`value_noise`). Observations are derived read-only views
built by the information module; agents never receive the state object.
Content (units, buildings, techs, terrain, improvements, constants, prices)
is data in JSON.

More detail:

- [docs/DESIGN.md](docs/DESIGN.md): the specification.
- [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md): the concrete choices where
  the specification was silent (phase order, fog layers, purchase semantics,
  belief scope, handles, diplomacy, search, record format).
- [docs/CONSTANTS.md](docs/CONSTANTS.md): every data constant and what it
  does.
- [CHANGELOG.md](CHANGELOG.md): verified properties, fixed defects, and
  known issues.

## Layout

```
bin/gs                 CLI wrapper (play | batch | replay | bench | list)
grandstrat/
  rng.py               named counter-based random streams
  tables.py            loads data/*.json, sha256-stamped
  hexmap.py            axial hex geometry
  state.py             GameState: units, cities, civs; clone(), state_hash()
  worldgen.py          terrain, resources, civ placement, camps
  engine.py            action spec, legality, batch validation, spawn/move helpers
  kernel.py            turn phases and victory checks
  economy.py           yields, growth, production, research, upkeep, scoring
  combat.py            combat resolution with world-keyed CRN rolls
  barbarians.py        camp spawns and barbarian behaviour
  observe.py           information module interface, full observability, flat encoder
  fog.py               fog of war, memory, purchases, observer-local handles
  belief.py            exact support enumeration and particle belief
  diplomacy.py         agreements and the bid-and-accept protocol
  search.py            search interface, determinization, forward model, baselines
  textrender.py        JSON + text summary renderer
  scenarios.py         scenario and dial registry
  env.py               GrandStratEnv (parallel multi-agent API)
  record.py            JSONL record writer and game runner
  replay.py            record replay and verification
  cli.py               `gs` command
  agents/              scripted agents and registry
  data/*.json          content tables
tests/                 pytest suite
docs/                  design, design notes, constants
```

## Tests

`python -m pytest -q` runs 114 tests in about 12 seconds on a desktop CPU.
They cover determinism (including across processes with different hash
seeds), CRN pairing across dial settings, legal-action completeness on sampled
mid-game states, ledger conservation (including an independent upkeep
recount), fog correctness against an adversarial probe that recomputes
visibility, purchase semantics and prices, belief support properties, the
diplomacy protocol, determinization, search budget accounting, record schema
and replay, observation isolation, file-write sandboxing, and a speed gate
(at least 200 default games per CPU-minute, best single-game sample).

## Limitations

- The scripted agents are simple heuristics meant as baselines, not strong
  or balanced opponents. None of the registry agents buys information; only
  `BuyOrSearchAgent` (in `grandstrat.search`) does. The diplomacy policy
  exercises the protocol; it does not model negotiation skill.
- Belief is over the positions of foreign units the civ has already seen
  (plus "gone"), with a uniform prior. It is not a full posterior over game
  histories: never-seen units, city production and economies are not
  modelled, the particle filter is not sequential, and matching sightings of
  several same-type units is heuristic.
- Determinization is approximate: unknown resources, treasuries and techs are
  neutral, remembered cities are stale, unseen units and camps do not exist,
  and the forward model does not reproduce every fog/diplomacy mechanism.
- Research is automatic (cheapest available tech); there is no research
  action. A unit gets one action per turn.
- The `diplomacy` random stream is reserved and draws nothing with the
  shipped modules.
- The fog module costs roughly 1.6x the default per game, mostly from
  observer-local id mapping.
- Records are large (about 0.8 MB per default 150-turn game) because they
  carry the full ledger.
- All content numbers are invented for a plain fantasy setting and are not
  tuned for human play.

## License

MIT. See [LICENSE](LICENSE).
