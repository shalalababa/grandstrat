"""Determinism (design §4, §6): same seeds + same actions -> byte-identical
records and state hashes; different seeds differ; stable across processes
and PYTHONHASHSEED values."""

import hashlib
import os
import subprocess
import sys

from conftest import LINEUP, play_record

from grandstrat.env import GrandStratEnv
from grandstrat.agents import make_agents
from grandstrat.record import RecordWriter, play_game
from grandstrat.tables import load_tables


def _play(env, seed, agents, path):
    with RecordWriter(path) as w:
        return play_game(env, seed, agents, writer=w)


def test_same_seed_byte_identical_records(tmp_path):
    a = str(tmp_path / "a.jsonl")
    b = str(tmp_path / "b.jsonl")
    for out in (a, b):
        env = GrandStratEnv(scenario="duel")
        agents = make_agents(["expander", "warmonger"], 2)
        _play(env, 42, agents, out)
    da = hashlib.sha256(open(a, "rb").read()).digest()
    db = hashlib.sha256(open(b, "rb").read()).digest()
    assert da == db


def test_same_seed_same_hidden_hash_different_seed_differs(tmp_path):
    _, s1, _ = play_record(tmp_path, 7, scenario="duel")
    _, s2, _ = play_record(tmp_path, 7, scenario="duel")
    _, s3, _ = play_record(tmp_path, 8, scenario="duel")
    assert s1["scores"] == s2["scores"] and s1["winner"] == s2["winner"]
    assert s3["scores"] != s1["scores"] or s3["winner"] != s1["winner"]


def test_different_actions_diverge_deterministically():
    """Two identical scripts reach identical hashes; a different first action
    diverges immediately (checked at turn 1: greedy scripted agents may
    legitimately RE-converge later, so late divergence is not assumed)."""
    def script(skip_first_dir, turns):
        e = GrandStratEnv(scenario="duel")
        e.reset(5)
        for t in range(turns):
            actions = {}
            for civ in range(2):
                legal = e.legal_actions(civ)
                picks = [a for a in legal if a[0] == "MOVE"]
                if picks and not (t == 0 and civ == 0 and skip_first_dir):
                    actions[civ] = [picks[0]]
            e.step(actions)
        return e.hidden_state_hash()

    assert script(False, 1) != script(True, 1)
    assert script(False, 12) == script(False, 12)


def test_across_processes_and_hash_seeds(tmp_path):
    """Full short game inside subprocesses with different PYTHONHASHSEED
    values: the final hidden-state hash must be identical (no set/hash-order
    leakage anywhere in the engine)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prog = (
        "import sys; sys.path.insert(0, %r);"
        "from grandstrat.env import GrandStratEnv;"
        "from grandstrat.agents import make_agents;"
        "from grandstrat.record import play_game;"
        "env = GrandStratEnv(scenario='tiny');"
        "agents = make_agents(['expander', 'warmonger'], 2);"
        "s = play_game(env, 3, agents);"
        "print(env.hidden_state_hash())" % (root,)
    )
    hashes = []
    for hs in ("0", "1", "12345"):
        env_vars = dict(os.environ)
        env_vars["PYTHONHASHSEED"] = hs
        out = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                             text=True, env=env_vars, timeout=180)
        assert out.returncode == 0, out.stderr
        hashes.append(out.stdout.strip())
    assert len(set(hashes)) == 1
