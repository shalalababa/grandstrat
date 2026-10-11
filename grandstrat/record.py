"""Schema-versioned JSONL game records (design §5, §0b) and the runner that
produces them.

File layout (one JSON object per line):
  line 1  {"schema": "gs.record/v1", "kind": "header", seed, scenario,
           scenario_params, dials, stream_seeds, agents, civs, tables_sha256,
           package_version, initial_scores, initial_hash}
  then    {"kind": "turn", turn, actions: {civ: [[...]...]}, events, rolls,
           ledger, scores, gold, hash, done, winner, done_reason}
           one per resolved turn, in order
  last    {"kind": "footer", final_scores, winner, reason, turns,
           stream_draws, wall_s, cpu_s}

Replay needs exactly: header (seed + scenario + dials + stream_seeds) and the
per-turn `actions`; world events, rolls, ledgers and results are verified by
grandstrat.replay. Agent provenance, compute and agent RNG draws cannot be
reconstructed from the action log alone.
"""

from __future__ import annotations

import json
import os
import time

from . import PACKAGE_NAME, PACKAGE_VERSION, RECORD_SCHEMA
from .agents import Agent
from .env import GrandStratEnv


def header_line(env: GrandStratEnv, seed: int, agent_names, initial_hash: str) -> dict:
    return {
        "schema": RECORD_SCHEMA,
        "kind": "header",
        "seed": int(seed),
        "scenario": env._scenario_name,
        "scenario_params": {k: v for k, v in env.params.items()},
        "dials": dict(env.dials),
        "stream_seeds": {k: int(v) for k, v in env.stream_seeds.items()},
        "config": dict(env.config),
        "agents": list(agent_names),
        "civs": len(env.state.civs),
        "tables_sha256": env.tables.sha256,
        "package": f"{PACKAGE_NAME} {PACKAGE_VERSION}",
        "initial_scores": [c.score for c in env.state.civs],
        "initial_hash": initial_hash,
    }


def turn_line(env: GrandStratEnv, report, actions_by_civ: dict, state_hash: str,
              compute: dict | None = None) -> dict:
    ln = {
        "schema": RECORD_SCHEMA,
        "kind": "turn",
        "turn": report.turn,
        "actions": {str(civ): [list(a) for a in acts]
                    for civ, acts in sorted(actions_by_civ.items())},
        "events": report.events,
        "rolls": report.rolls,
        "ledger": report.ledger,
        "scores": report.scores,
        "gold": report.gold,
        "hash": state_hash,
        "done": report.done,
        "winner": report.winner,
        "done_reason": report.done_reason,
    }
    if compute:
        # additive column: per-civ compute actually spent this turn
        # (search budget accounting; separate from the gold ledger).  Absent
        # entirely when no search agent played, so older records replay
        # unchanged.
        ln["compute"] = compute
    return ln


def footer_line(env: GrandStratEnv, turns: int, wall_s: float, cpu_s: float) -> dict:
    """NB: no wall/cpu time here -- same seeds + same actions must produce
    byte-identical records (design §4).  Timing belongs to the CLI/report."""
    return {
        "schema": RECORD_SCHEMA,
        "kind": "footer",
        "final_scores": [c.score for c in env.state.civs],
        "winner": env.state.winner,
        "reason": env.state.done_reason,
        "turns": turns,
        "stream_draws": dict(sorted(env.state.rng.draw_counts.items())),
    }


def play_game(env: GrandStratEnv, seed: int, agents: list[Agent],
              writer=None) -> dict:
    """Play one full game; optionally write a record; return a summary dict."""
    for a in agents:
        a.reset()
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    env.reset(seed)
    if writer is not None:
        writer.write(header_line(env, seed, [a.name for a in agents],
                                 env.hidden_state_hash()))
    turns = 0
    while not env.state.done:
        actions = {}
        for civ, agent in enumerate(agents):
            if not env.state.civs[civ].alive:
                continue
            obs = env.observation(civ)
            legal = env.legal_actions(civ)
            actions[civ] = agent.act(obs, legal, env.state.rng.stream(agent.stream_name))
        obs, rewards, terms, truncs, infos = env.step(actions)
        report = infos[0]["report"]
        turns += 1
        if writer is not None:
            compute = {}
            for civ, agent in enumerate(agents):
                b = getattr(agent, "last_budget", None)
                if b is not None:
                    compute[str(civ)] = b.as_dict()
                    agent.last_budget = None  # one turn per logged budget
            writer.write(turn_line(env, report, actions,
                                   env.hidden_state_hash(),
                                   compute=compute or None))
    if writer is not None:
        writer.write(footer_line(env, turns,
                                 time.perf_counter() - t0_wall,
                                 time.process_time() - t0_cpu))
    return {
        "seed": seed,
        "winner": env.state.winner,
        "reason": env.state.done_reason,
        "turns": turns,
        "scores": [c.score for c in env.state.civs],
        "stream_draws": dict(env.state.rng.draw_counts),
        "cpu_s": time.process_time() - t0_cpu,
        "wall_s": time.perf_counter() - t0_wall,
        "agent_names": [a.name for a in agents],
    }


class RecordWriter:
    """JSONL writer; creates the output file, nothing else."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":"), sort_keys=False) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def record_path(out_dir: str, seed: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"game_seed{seed}.jsonl")
