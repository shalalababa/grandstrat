"""Conservation (design §6): gold, research, and every city's food and
production buckets balance exactly per turn, re-derived from the ledger the
engine writes into the record."""

import json


def _turns(path):
    with open(path) as f:
        for line in f:
            ln = json.loads(line)
            if ln.get("kind") == "turn":
                yield ln


def test_ledger_conservation(tmp_path):
    for seed, scenario in ((2, "duel"), (5, "default")):
        from conftest import play_record

        _, _, path = play_record(tmp_path, seed, scenario=scenario,
                                 name=f"cons_{seed}.jsonl")
        for ln in _turns(path):
            led = ln["ledger"]
            for row in led["gold"]:
                raw = (row["before"] - row.get("purchase", 0) + row["income"]
                       - row["upkeep"] + row.get("bounty", 0)
                       + row.get("transfer", 0))
                if row.get("clamped"):
                    assert raw < 0 and row["after"] == 0, (
                        f"clamp flag inconsistent turn {ln['turn']}: {row}")
                else:
                    assert row["after"] == raw, (
                        f"gold imbalance turn {ln['turn']}: {row}")
            for row in led.get("city_food", []):
                spent = row["food_spent"]
                if row["food_reset"]:
                    # starvation clamp: bucket floors at 0 (pop may drop)
                    assert row["food_after"] == 0
                else:
                    assert row["food_after"] == (
                        row["food_before"] + row["food_income"]
                        - row["food_eaten"] - spent
                    ), f"food imbalance turn {ln['turn']}: {row}"
            for row in led.get("city_prod", []):
                assert row["prod_after"] == (
                    row["prod_before"] + row["prod_income"] - row["prod_spent"]
                ), f"production imbalance turn {ln['turn']}: {row}"
                if row["completed"] is not None:
                    assert row["prod_before"] + row["prod_income"] >= row["prod_spent"]
            for row in led.get("research", []):
                assert row["after"] == (
                    row["before"] + row["income"] - row["spent"]
                ), f"research imbalance turn {ln['turn']}: {row}"


def test_gold_series_matches_ledger(tmp_path):
    """The record's per-turn `gold` snapshot equals the ledger's `after`."""
    from conftest import play_record

    _, _, path = play_record(tmp_path, 3, scenario="duel", name="gold.jsonl")
    for ln in _turns(path):
        assert ln["gold"] == [row["after"] for row in ln["ledger"]["gold"]]


def test_population_bookkeeping(tmp_path):
    """Pop changes come only from growth, starvation, settler completion, or
    capture — cross-check the food ledger's pop columns against events."""
    from conftest import play_record

    _, _, path = play_record(tmp_path, 2, scenario="duel", name="pop.jsonl")
    pop = {}
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    turn_lines = [l for l in lines if l.get("kind") == "turn"]
    for ln in turn_lines:
        captured_this_turn = {e["city"] for e in ln["events"]
                              if e["type"] == "city_captured"}
        # combat (and captures) resolve before income, so a captured city's
        # pop halving lands before this turn's food row
        for cid in captured_this_turn:
            if cid in pop:
                pop[cid] = max(1, pop[cid] // 2)
        for row in ln["ledger"].get("city_food", []):
            cid = row["city"]
            before = pop.get(cid, 1)
            assert row["pop_before"] == before, (
                f"turn {ln['turn']} city {cid}: ledger pop_before "
                f"{row['pop_before']} != tracked {before}"
            )
            after = row["pop_after"]
            settlers = sum(
                1 for e in ln["events"]
                if e["type"] == "unit_built" and e.get("kind") == "settler"
                and e.get("city") == cid
            )
            # settler completion (production phase) consumes 1 pop after the
            # food row was written
            pop[cid] = after - settlers
