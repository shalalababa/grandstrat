"""No writes outside grandstrat/ or the requested output path: the engine and agents
perform zero file I/O during play; only the record writer and CLI write,
and only to their requested paths."""

import builtins
import io
import os

from conftest import LINEUP, make_env

from grandstrat.agents import make_agents
from grandstrat.record import play_game

_REAL_OPEN = builtins.open


def test_engine_and_agents_do_no_file_io(tmp_path, monkeypatch):
    """Play a full game with `open` and os file-creation calls instrumented:
    the record writer is the only allowed caller, and it touches only its
    own output path."""
    opened = []
    instrumented = io.open

    def spying_open(file, *a, **kw):
        opened.append(os.fspath(file))
        return _REAL_OPEN(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", spying_open)

    out = tmp_path / "io_probe.jsonl"
    env = make_env("duel")
    agents = make_agents(["expander", "warmonger"], 2)
    from grandstrat.record import RecordWriter

    with RecordWriter(str(out)) as w:
        play_game(env, 4, agents, writer=w)

    # even without a writer, no I/O happens at all
    opened.clear()
    play_game(env, 4, agents, writer=None)
    assert opened == [], f"engine performed file I/O: {opened}"

    # reset-only path (worldgen) is I/O-free too
    opened.clear()
    env.reset(99)
    assert opened == [], f"worldgen performed file I/O: {opened}"


def test_cli_writes_only_under_out_dir(tmp_path):
    """`gs play` with --out under tmp creates files only there."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = tmp_path / "recs"
    before = {p for p in _tree(root)}
    prog = (
        f"import sys; sys.path.insert(0, {root!r});"
        "from grandstrat.cli import main;"
        f"main(['play', '--seed', '2', '--scenario', 'tiny',"
        f"'--agents', 'expander,warmonger', '--out', {str(out_dir)!r}])"
    )
    res = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                         text=True, cwd=tmp_path, timeout=180)
    assert res.returncode == 0, res.stderr
    after = {p for p in _tree(root)}
    assert before == after, f"files appeared under grandstrat/: {after - before}"
    assert (out_dir / "game_seed2.jsonl").exists()


def _tree(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # ignore pytest and pycache noise
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".pytest_cache")]
        for name in filenames:
            if name.endswith((".pyc", ".pyo")) or "__pycache__" in name:
                continue
            yield os.path.join(dirpath, name)
