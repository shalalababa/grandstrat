"""Shared helpers for the grandstrat test suite."""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from grandstrat.agents import DEFAULT_LINEUP, make_agents  # noqa: E402
from grandstrat.env import GrandStratEnv  # noqa: E402
from grandstrat.record import RecordWriter, play_game  # noqa: E402

LINEUP = list(DEFAULT_LINEUP)


def make_env(scenario="default", **kw) -> GrandStratEnv:
    return GrandStratEnv(scenario=scenario, **kw)


def play_record(tmp_path, seed, scenario="default", agents=None, dials=None,
                name=None):
    """Play one game with the standard lineup, writing a record; return
    (env, summary, record_path)."""
    env = make_env(scenario, dials=dials)
    n = env.resolved_params()["civs"]
    ags = agents or make_agents(LINEUP[:n] if n <= 4 else LINEUP, n)
    path = str(tmp_path / (name or f"game_seed{seed}.jsonl"))
    with RecordWriter(path) as w:
        summary = play_game(env, seed, ags, writer=w)
    return env, summary, path


def occupancy_ok(state) -> bool:
    seen = {}
    for u in state.units.values():
        i = state.hexmap.index(u.q, u.r)
        if int(state.occ_unit[i]) != u.id or i in seen:
            return False
        seen[i] = u.id
    for i in range(state.hexmap.n):
        v = int(state.occ_unit[i])
        if v >= 0 and v not in state.units:
            return False
    for c in state.cities.values():
        if int(state.occ_city[state.hexmap.index(c.q, c.r)]) != c.id:
            return False
    return True


@pytest.fixture(autouse=True)
def _pin_cwd(tmp_path, monkeypatch):
    """Each test runs with cwd pinned to its tmp dir: accidental relative
    writes land in tmp, and the sandbox test verifies in-process I/O."""
    monkeypatch.chdir(tmp_path)
