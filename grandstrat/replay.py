"""Replay tool (design §0b): reconstruct any game from its record (seed +
action log) and verify every recorded state hash and every recorded score.

    python3 -m grandstrat.replay --record path/to/game.jsonl
    # or: gs replay --record ...

Exit code 0 = the replay reproduces every verifiable recorded field;
nonzero with a diff report otherwise. Agent compute and agent RNG draws cannot
be reconstructed from the action log alone.
"""

from __future__ import annotations

import json

from . import RECORD_SCHEMA
from .agents import make_agents
from .env import GrandStratEnv
from .tables import load_tables


def _action_from_json(lst) -> tuple:
    return tuple(lst)


def replay_record(path: str, env: GrandStratEnv | None = None) -> dict:
    tables = load_tables()
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty record")
    header = lines[0]
    if header.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"{path}: schema {header.get('schema')!r} != {RECORD_SCHEMA!r}")
    if lines[0].get("kind") != "header" or lines[-1].get("kind") != "footer" \
            or any(ln.get("kind") != "turn" for ln in lines[1:-1]):
        raise ValueError(f"{path}: expected header, turn lines, footer in order")
    turns = lines[1:-1]
    footer = lines[-1]

    env = env or GrandStratEnv(
        scenario=header["scenario"],
        scenario_overrides={
            k: v for k, v in header["scenario_params"].items() if k != "name"
        },
        dials=header["dials"],
        stream_seeds=header.get("stream_seeds", {}),
        config=header.get("config", {}),
        tables=tables,
    )
    # replay verifies seed + action log; scripted agents are not executed.
    # Records played by custom agents (e.g. the search adapter) name
    # non-registry agents: keep the header for provenance and step the log.
    agents_rebuilt = True
    try:
        agents = make_agents(header.get("agents") or ["expander"],
                             int(header["scenario_params"]["civs"]))
    except ValueError:
        agents = []
        agents_rebuilt = False
    del agents
    env.reset(header["seed"])

    mismatches = []

    def check(turn, field, recorded, actual):
        # JSONL converts tuples to lists and integer object keys to strings.
        # Compare against the value the writer would have serialized.
        actual = json.loads(json.dumps(actual))
        if recorded != actual:
            mismatches.append({"turn": turn, "field": field,
                               "record": recorded, "replay": actual})

    check("header", "tables_sha256", header.get("tables_sha256"), tables.sha256)
    check("initial", "hash", header.get("initial_hash"), env.hidden_state_hash())
    check("initial", "scores", header.get("initial_scores"),
          [c.score for c in env.state.civs])
    check("header", "civs", header.get("civs"), len(env.state.civs))

    for index, ln in enumerate(turns):
        check(index, "schema", ln.get("schema"), RECORD_SCHEMA)
        check(index, "turn", ln.get("turn"), env.state.turn)
        actions = {int(civ): [_action_from_json(a) for a in acts]
                   for civ, acts in ln["actions"].items()}
        # dead civs submitted nothing; their absent keys are fine
        obs, rewards, terms, truncs, infos = env.step(actions)
        report = infos[0]["report"]
        for field, actual in (
            ("hash", env.hidden_state_hash()),
            ("scores", report.scores), ("gold", report.gold),
            ("done", report.done), ("winner", report.winner),
            ("done_reason", report.done_reason),
            ("events", report.events), ("rolls", report.rolls),
            ("ledger", report.ledger),
        ):
            check(index, field, ln.get(field), actual)

    check("footer", "turns", footer.get("turns"), len(turns))
    check("footer", "schema", footer.get("schema"), RECORD_SCHEMA)
    check("footer", "final_scores", footer.get("final_scores"),
          [c.score for c in env.state.civs])
    check("footer", "winner", footer.get("winner"), env.state.winner)
    check("footer", "reason", footer.get("reason"), env.state.done_reason)
    check("footer", "game_complete", env.state.done, True)
    # Scripted/custom agents are not run during replay, so their named RNG
    # streams may have drawn while producing the record. World streams must
    # still match exactly.
    draws = footer.get("stream_draws", {})
    for name in sorted(set(draws) | set(env.state.rng.draw_counts)):
        if not name.startswith("agent_"):
            check("footer", f"stream_draws.{name}", draws.get(name),
                  env.state.rng.draw_counts.get(name))

    result = {
        "path": path,
        "turns_replayed": len(turns),
        "mismatches": mismatches,
        "agents_rebuilt": agents_rebuilt,
        "ok": not mismatches,
    }
    result["winner_match"] = (footer.get("winner") == env.state.winner)
    return result


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="gs replay",
                                description="Replay a GRANDSTRAT record and verify it.")
    p.add_argument("--record", required=True, help="path to a gs.record/v1 JSONL file")
    args = p.parse_args(argv)
    res = replay_record(args.record)
    if res["ok"]:
        print(f"OK: {res['turns_replayed']} turns replayed, all checks match "
              f"({res['path']})")
        return 0
    print(f"MISMATCH: {len(res['mismatches'])} field(s) differ ({res['path']})")
    for m in res["mismatches"][:20]:
        print(f"  turn {m['turn']} {m['field']}: record={m['record']!r} replay={m['replay']!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
