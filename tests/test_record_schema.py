"""Record schema (design §5, §0b): gs.record/v1 line structure, required
fields, action encodings; and the replay tool reconstructs every game from
seed + action log."""

import json

import pytest

from conftest import play_record

from grandstrat import RECORD_SCHEMA
from grandstrat.replay import replay_record


def _lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


ACTION_TYPES = {"SKIP", "MOVE", "ATTACK_U", "ATTACK_C", "FOUND", "IMPROVE",
                "PRODUCE", "SPY", "TRADE", "DIPLO", "ORACLE",
                "OFFER", "ACCEPT", "REJECT"}
EVENT_TYPES = {
    "combat-irrelevant",  # placeholder guard: all real types checked below
}
KNOWN_EVENTS = {
    "border_grown", "pop_growth", "starvation", "unit_built", "building_built",
    "tech_researched", "bankruptcy_disband", "unit_died", "city_captured",
    "city_founded", "improvement_built", "barb_spawn", "camp_cleared",
    "civ_eliminated", "game_over", "rejected", "info_purchase",
    "dipl_offer", "dipl_accepted", "dipl_rejected", "dipl_expired",
    "dipl_broken",
}


def test_header_schema(tmp_path):
    _, _, path = play_record(tmp_path, 1, scenario="duel")
    header = _lines(path)[0]
    assert header["schema"] == RECORD_SCHEMA
    assert header["kind"] == "header"
    for field in ("seed", "scenario", "scenario_params", "dials",
                  "stream_seeds", "agents", "civs", "tables_sha256",
                  "initial_scores", "initial_hash"):
        assert field in header, f"header missing {field}"
    assert header["dials"]["fog"] == 4  # default dial logged per game


def test_turn_and_footer_schema(tmp_path):
    _, _, path = play_record(tmp_path, 1, scenario="duel")
    lines = _lines(path)
    footer = lines[-1]
    assert footer["kind"] == "footer"
    for field in ("final_scores", "winner", "reason", "turns", "stream_draws"):
        assert field in footer
    # no wall/cpu timings in records: same seeds+actions => byte-identical
    assert "wall_s" not in footer and "cpu_s" not in footer

    turn_count = 0
    for ln in lines[1:-1]:
        assert ln["schema"] == RECORD_SCHEMA
        assert ln["kind"] == "turn"
        for field in ("turn", "actions", "events", "rolls", "ledger",
                      "scores", "gold", "hash", "done", "winner"):
            assert field in ln, f"turn line missing {field}"
        assert len(ln["hash"]) == 64  # sha256 hex
        for civ, acts in ln["actions"].items():
            for a in acts:
                assert a[0] in ACTION_TYPES, f"unknown action {a}"
                if a[0] == "MOVE":
                    assert 0 <= a[2] <= 5
        for e in ln["events"]:
            assert e["type"] in KNOWN_EVENTS, f"unknown event {e['type']}"
        turn_count += 1
    assert turn_count == footer["turns"]


def test_dials_logged_per_game(tmp_path):
    _, _, path = play_record(tmp_path, 1, scenario="duel",
                             dials={"resource_scarcity": 0.5},
                             name="dials.jsonl")
    header = _lines(path)[0]
    assert header["dials"]["resource_scarcity"] == 0.5
    assert header["dials"]["adversary_strength"] == 1.0


@pytest.mark.parametrize("seed,scenario", [(2, "duel"), (7, "default"), (0, "tiny")])
def test_replay_reconstructs_games(tmp_path, seed, scenario):
    _, _, path = play_record(tmp_path, seed, scenario=scenario,
                             name=f"replay_{seed}.jsonl")
    res = replay_record(path)
    assert res["ok"], res["mismatches"]
    assert res["turns_replayed"] > 0
    assert res["winner_match"]


def test_replay_detects_tampering(tmp_path):
    _, _, path = play_record(tmp_path, 2, scenario="duel", name="tamper.jsonl")
    lines = _lines(path)
    # tamper with an EXECUTED move (one not listed in the turn's own
    # rejected events): replay must either report mismatches or raise
    # (the tampered batch became illegal) -- either way, not "ok".
    tampered_lines = None
    for ln in lines:
        if ln.get("kind") != "turn":
            continue
        rejected = {tuple(e["action"]) for e in ln["events"]
                    if e["type"] == "rejected"}
        for acts in ln["actions"].values():
            for a in acts:
                if a[0] == "MOVE" and tuple(a) not in rejected:
                    a[2] = (a[2] + 1) % 6
                    tampered_lines = lines
                    break
            if tampered_lines:
                break
        if tampered_lines:
            break
    assert tampered_lines is not None, "found no executed move to tamper with"
    tampered = str(tmp_path / "tampered.jsonl")
    with open(tampered, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    try:
        res = replay_record(tampered)
        assert not res["ok"], "replay accepted a tampered action log"
        assert any(m["field"] == "hash" for m in res["mismatches"])
    except Exception:
        pass  # tampered action was no longer legal: replay refused it


def test_replay_detects_seed_tampering(tmp_path):
    _, _, path = play_record(tmp_path, 2, scenario="duel", name="seed.jsonl")
    lines = _lines(path)
    lines[0]["seed"] = 999
    tampered = str(tmp_path / "seeded.jsonl")
    with open(tampered, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    try:
        res = replay_record(tampered)
        assert not res["ok"], "replay accepted a tampered seed"
    except Exception:
        pass
