"""`gs` CLI: play, batch, replay, list.

    gs play --seed 0 --civs 4 --R 12 --dial fog=3 --agents expander,warmonger,turtle,scout
    gs batch --n 32 --seed-base 0 --out records/
    gs replay --record records/game_seed0.jsonl
    gs list
"""

from __future__ import annotations

import argparse
import sys
import time


def _parse_dials(items: list[str]) -> dict:
    dials = {}
    for it in items or []:
        if "=" in it:
            name, _, val = it.partition("=")
            try:
                dials[name] = int(val)
            except ValueError:
                dials[name] = float(val)
        else:
            # design's `--dial W` shorthand: the information (fog) dial
            dials["fog"] = int(it)
    return dials


def _parse_stream_seeds(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        name, _, val = it.partition("=")
        if not name or not val:
            raise SystemExit(f"--stream-seed expects name=seed, got {it!r}")
        out[name] = int(val)
    return out


def _add_game_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0, help="master seed")
    p.add_argument("--scenario", default="default")
    p.add_argument("--civs", type=int, default=None, help="override civ count")
    p.add_argument("--R", type=int, default=None, help="override map radius")
    p.add_argument("--T", type=int, default=None, help="override turn horizon")
    p.add_argument("--dial", action="append", default=[],
                   help="dial override: name=value or bare W (fog shorthand); repeatable")
    p.add_argument("--agents", default="expander,warmonger,turtle,scout",
                   help="comma-separated agent names, cycled to civ count")
    p.add_argument("--stream-seed", action="append", default=[],
                   help="re-seed one named stream: name=seed; repeatable")
    p.add_argument("--config", action="append", default=[],
                   help="kernel config key=value (e.g. information=fog)")
    p.add_argument("--out", default="records", help="record output directory")


def _build_env(args) -> "GrandStratEnv":
    from .env import GrandStratEnv

    overrides = {}
    if args.civs is not None:
        overrides["civs"] = args.civs
    if args.R is not None:
        overrides["R"] = args.R
    if args.T is not None:
        overrides["T"] = args.T
    config = {}
    for kv in args.config:
        k, _, v = kv.partition("=")
        config[k] = v
    return GrandStratEnv(
        scenario=args.scenario,
        scenario_overrides=overrides or None,
        dials=_parse_dials(args.dial),
        stream_seeds=_parse_stream_seeds(args.stream_seed),
        config=config or None,
    )


def cmd_play(args) -> int:
    from .agents import make_agents
    from .record import RecordWriter, play_game, record_path

    env = _build_env(args)
    agents = make_agents(args.agents.split(","), env.resolved_params()["civs"])
    path = record_path(args.out, args.seed)
    with RecordWriter(path) as w:
        summary = play_game(env, args.seed, agents, writer=w)
    print(
        f"seed {summary['seed']}: winner civ {summary['winner']} "
        f"({summary['reason']}) after {summary['turns']} turns; "
        f"scores {summary['scores']}"
    )
    print(f"record: {path}")
    return 0


def cmd_batch(args) -> int:
    from .agents import make_agents
    from .record import RecordWriter, play_game, record_path

    env = _build_env(args)
    agents = make_agents(args.agents.split(","), env.resolved_params()["civs"])
    t0 = time.process_time()
    t0w = time.perf_counter()
    games = 0
    for i in range(args.n):
        seed = args.seed_base + i
        path = record_path(args.out, seed)
        with RecordWriter(path) as w:
            play_game(env, seed, agents, writer=w)
        games += 1
    cpu_s = time.process_time() - t0
    wall_s = time.perf_counter() - t0w
    rate = games / (cpu_s / 60.0) if cpu_s > 0 else float("inf")
    print(f"batch: {games} games in {cpu_s:.2f} s CPU ({wall_s:.2f} s wall)")
    print(f"speed: {rate:.1f} games/CPU-minute (records in {args.out})")
    return 0


def cmd_replay(args) -> int:
    from .replay import main as replay_main

    return replay_main(["--record", args.record])


def cmd_list(args) -> int:
    from .scenarios import DIALS, SCENARIOS

    print("scenarios:")
    for name, s in sorted(SCENARIOS.items()):
        print(f"  {name}: {s}")
    print("dials:")
    for name, d in sorted(DIALS.items()):
        print(f"  {name}: default={d['default']} status={d['status']}")
    return 0


def cmd_bench(args) -> int:
    """Rate benchmarks for the three headline configurations (design §5:
    default / fog / search), reported SEPARATELY -- they contend for the box
    like any other jobs, so rates are comparable only within a quiet run."""
    from .agents import make_agents
    from .search import (Budget, DeterminizedSearchAgent, SearchAdapter)
    from .record import play_game

    if args.mode == "default":
        env = _build_env(args)
    else:
        if not args.dial:
            args.dial = ["0"]  # fog is only interesting with fog on
        args.config = list(args.config) + ["information=fog"]
        env = _build_env(args)
    n = env.resolved_params()["civs"]
    names = args.agents.split(",")
    agents = make_agents(names, n)
    if args.mode == "search" and agents:
        inner = DeterminizedSearchAgent(0, {"expansions": args.search_expansions,
                                            "determinizations": 2,
                                            "horizon": 1})
        agents[0] = SearchAdapter(0, inner,
                                  belief_fn=lambda obs: env.belief(obs["you"]),
                                  budget_params={"expansions": args.search_expansions,
                                                 "determinizations": 2})
    t0 = time.process_time()
    t0w = time.perf_counter()
    games = 0
    for i in range(args.n):
        play_game(env, args.seed_base + i, agents)
        games += 1
    cpu_s = time.process_time() - t0
    wall_s = time.perf_counter() - t0w
    rate = games / (cpu_s / 60.0) if cpu_s > 0 else float("inf")
    print(f"bench[{args.mode}] {args.scenario}: {games} games in {cpu_s:.2f} s CPU"
          f" ({wall_s:.2f} s wall)")
    print(f"  rate: {rate:.1f} games/CPU-minute"
          + (f" (civ0 search budget {args.search_expansions} expansions/turn)"
             if args.mode == "search" else ""))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gs", description="GRANDSTRAT simulator CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    play = sub.add_parser("play", help="play one game and write its record")
    _add_game_args(play)
    play.set_defaults(fn=cmd_play)
    batch = sub.add_parser("batch", help="play n games and write records")
    _add_game_args(batch)
    batch.add_argument("--n", type=int, default=8, help="number of games")
    batch.add_argument("--seed-base", type=int, default=0)
    batch.set_defaults(fn=cmd_batch)
    rep = sub.add_parser("replay", help="replay + verify a record")
    rep.add_argument("--record", required=True)
    rep.set_defaults(fn=cmd_replay)
    lst = sub.add_parser("list", help="list scenarios and dials")
    lst.set_defaults(fn=cmd_list)
    bench = sub.add_parser("bench",
                           help="rate benchmark: --mode default|fog|search")
    _add_game_args(bench)
    bench.add_argument("--n", type=int, default=8, help="number of games")
    bench.add_argument("--seed-base", type=int, default=0)
    bench.add_argument("--mode", default="default",
                       choices=("default", "fog", "search"),
                       help="default=full module; fog=information=fog at the "
                            "current dial; search=fog with civ 0 replaced by "
                            "a bounded determinized search agent")
    bench.add_argument("--search-expansions", type=int, default=24,
                       help="expansion budget per decision for --mode search")
    bench.set_defaults(fn=cmd_bench)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
