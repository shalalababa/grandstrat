"""Scenario and dial registry (design §0b): named scenarios and named dials
are first-class simulator parameters, validated here and logged per game.

Dial status (see docs/DESIGN_NOTES.md):
  fog                 -- FUNCTIONAL: with the `information: "fog"`
                         kernel module the rung W ∈ {0..K} controls fog
                         depth and which purchases are enabled/priced
                         (data/information.json).  With the default `full`
                         module every rung still plays identically, and
                         W = K is the fog module's full-observability rung.
  resource_scarcity   -- FUNCTIONAL: scales bonus-resource frequency in
                         worldgen (0 = none).
  adversary_strength  -- FUNCTIONAL: scales barbarian spawn probability
                         (0 = no spawns).
  mistake_punishment  -- FUNCTIONAL: reward shaping only -- each
                         of a civ's dropped (stale) orders in a turn
                         subtracts the dial value from that civ's reward;
                         scores, ledgers and dynamics are untouched (0 = off).
  nonstationarity     -- FUNCTIONAL: barbarian spawn chance is
                         scaled by 1 + dial * turn / T -- the world grows
                         linearly more dangerous toward the horizon (0 = off).
"""

from __future__ import annotations

FOG_K = 4  # top rung: full observability

SCENARIOS: dict[str, dict] = {
    # name -> map params.  R = hex radius (3R(R+1)+1 hexes), civs = number of
    # civilisations, T = turn horizon, camps = barbarian camps, start =
    # starting-state preset (only "standard" exists).
    "default": {"R": 12, "civs": 4, "T": 150, "camps": 6, "start": "standard"},
    "duel":    {"R": 8,  "civs": 2, "T": 120, "camps": 2, "start": "standard"},
    "tiny":    {"R": 4,  "civs": 2, "T": 60,  "camps": 0, "start": "standard"},
    "dense":   {"R": 6,  "civs": 6, "T": 100, "camps": 2, "start": "standard"},
}

DIALS: dict[str, dict] = {
    "fog": {
        "default": FOG_K,
        "type": "int",
        "min": 0,
        "max": FOG_K,
        "status": "functional",
    },
    "resource_scarcity": {
        "default": 1.0,
        "type": "float",
        "min": 0.0,
        "max": None,
        "status": "functional",
    },
    "adversary_strength": {
        "default": 1.0,
        "type": "float",
        "min": 0.0,
        "max": None,
        "status": "functional",
    },
    "mistake_punishment": {
        "default": 0.0,
        "type": "float",
        "min": 0.0,
        "max": None,
        "status": "functional",
    },
    "nonstationarity": {
        "default": 0.0,
        "type": "float",
        "min": 0.0,
        "max": None,
        "status": "functional",
    },
}


def default_dials() -> dict:
    return {name: spec["default"] for name, spec in DIALS.items()}


def validate_dials(overrides: dict | None) -> dict:
    dials = default_dials()
    if overrides:
        for name, val in overrides.items():
            if name not in DIALS:
                raise ValueError(f"unknown dial {name!r}; valid: {sorted(DIALS)}")
            spec = DIALS[name]
            if spec["type"] == "int":
                if not isinstance(val, int) or isinstance(val, bool):
                    raise ValueError(f"dial {name} expects int, got {val!r}")
            else:
                val = float(val)
            if val < spec["min"] or (spec["max"] is not None and val > spec["max"]):
                raise ValueError(f"dial {name}={val} outside [{spec['min']}, {spec['max']}]")
            dials[name] = val
    return dials


def scenario_params(name: str, overrides: dict | None = None) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; valid: {sorted(SCENARIOS)}")
    params = dict(SCENARIOS[name])
    if overrides:
        for key, val in overrides.items():
            if key not in params:
                raise ValueError(f"unknown scenario param {key!r}")
            params[key] = val
    params["name"] = name
    return params
