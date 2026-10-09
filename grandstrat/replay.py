"""Replay tool (design §0b): reconstruct any game from its record (seed +
action log) and verify every recorded state hash and every recorded score.

    python3 -m grandstrat.replay --record path/to/game.jsonl
    # or: gs replay --record ...

Exit code 0 = the replay reproduces the record byte-for-byte in every checked
field; nonzero with a diff report otherwise.
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
    turns = [ln for ln in lines if ln.get("kind") == "turn"]
    footer = next((ln for ln in lines if ln.get("kind") == "footer"), None)

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
    if header.get("initial_hash") != env.hidden_state_hash():
        mismatches.append({"turn": "initial", "field": "hash",
                           "record": header.get("initial_hash"),
                           "replay": env.hidden_state_hash()})

    for ln in turns:
        actions = {int(civ): [_action_from_json(a) for a in acts]
                   for civ, acts in ln["actions"].items()}
        # dead civs submitted nothing; their absent keys are fine
        obs, rewards, terms, truncs, infos = env.step(actions)
        h = env.hidden_state_hash()
        if h != ln["hash"]:
            mismatches.append({"turn": ln["turn"], "field": "hash",
                               "record": ln["hash"], "replay": h})
        if [c.score for c in env.state.civs] != ln["scores"]:
            mismatches.append({"turn": ln["turn"], "field": "scores",
                               "record": ln["scores"],
                               "replay": [c.score for c in env.state.civs]})
        if [c.gold for c in env.state.civs] != ln["gold"]:
            mismatches.append({"turn": ln["turn"], "field": "gold",
                               "record": ln["gold"],
                               "replay": [c.gold for c in env.state.civs]})
        if bool(env.state.done) != bool(ln["done"]):
            mismatches.append({"turn": ln["turn"], "field": "done",
                               "record": ln["done"], "replay": env.state.done})

    result = {
        "path": path,
        "turns_replayed": len(turns),
        "mismatches": mismatches,
        "agents_rebuilt": agents_rebuilt,
        "ok": not mismatches,
    }
    if footer is not None:
        result["winner_match"] = (footer.get("winner") == env.state.winner)
        if not result["winner_match"]:
            result["ok"] = False
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
