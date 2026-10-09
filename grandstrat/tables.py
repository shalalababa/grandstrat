"""Data-driven content tables (design §0b): units, buildings, techs, terrain,
improvements, constants.  All live in JSON under grandstrat/data/, never in
code.  Every numeric constant is tagged with a `source` field:
  "ASSUMED"     -- plain-fantasy setting number, chosen for this simulator
                   (nothing here reproduces a commercial title; see
                   docs/CONSTANTS.md).
A tables sha256 (over the file bytes) is logged into every game record header
so records name the exact content they were played with.
"""

from __future__ import annotations

import hashlib
import json
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TABLE_FILES = (
    "terrain.json",
    "units.json",
    "buildings.json",
    "techs.json",
    "improvements.json",
    "constants.json",
    "information.json",
)

_REQUIRED = {
    "units.json": ("move", "strength", "cost", "upkeep", "sight"),
    "buildings.json": ("cost",),
    "techs.json": ("tier", "cost", "prereqs", "unlocks"),
    "terrain.json": (),
    "improvements.json": ("terrain",),
}


class Tables:
    """Immutable view over the loaded JSON tables + derived lookup indexes."""

    __slots__ = (
        "terrain",
        "units",
        "buildings",
        "techs",
        "improvements",
        "constants",
        "information",
        "sha256",
        "terrain_ids",
        "terrain_names",
        "unit_ids",
        "unit_names",
        "tech_names",
        "tech_tier",
        "tier_count",
    )

    def __init__(self, d: dict):
        for k in (
            "terrain",
            "units",
            "buildings",
            "techs",
            "improvements",
            "constants",
            "information",
        ):
            setattr(self, k, d[k])
        self.sha256 = d["sha256"]
        self.terrain_ids = {name: i for i, name in enumerate(self.terrain)}
        self.terrain_names = list(self.terrain)
        self.unit_ids = {name: i for i, name in enumerate(self.units)}
        self.unit_names = list(self.units)
        self.tech_names = sorted(self.techs)
        self.tech_tier = {t: self.techs[t]["tier"] for t in self.techs}
        self.tier_count = 1 + max(self.tech_tier.values())

    # -- derived queries ---------------------------------------------------
    def unit_techs_ok(self, unit_name: str, techs_done) -> bool:
        need = self.units[unit_name].get("tech")
        return need is None or need in techs_done

    def buildings_available(self, techs_done):
        for name, b in self.buildings.items():
            need = b.get("tech")
            if need is None or need in techs_done:
                yield name

    def techs_available(self, techs_done):
        for name in self.tech_names:
            if name in techs_done:
                continue
            if all(p in techs_done for p in self.techs[name]["prereqs"]):
                yield name


def _validate(name: str, obj) -> None:
    req = _REQUIRED.get(name, ())
    def check(table: dict):
        for key, spec in table.items():
            if not isinstance(spec, dict):
                raise ValueError(f"{name}:{key}: entry must be an object")
            for field in req:
                if field not in spec:
                    raise ValueError(f"{name}:{key}: missing field {field!r}")

    if name in ("units.json", "buildings.json", "techs.json", "improvements.json"):
        check(obj)


def load_tables(data_dir: str | None = None) -> Tables:
    """Load + validate all content tables.  Cached per directory."""
    global _CACHE
    d = data_dir or _DATA_DIR
    cached = _CACHE.get(d)
    if cached is not None:
        return cached
    out = {}
    h = hashlib.sha256()
    for fname in TABLE_FILES:
        path = os.path.join(d, fname)
        with open(path, "rb") as f:
            raw = f.read()
        h.update(hashlib.sha256(raw).digest())
        obj = json.loads(raw.decode("utf-8"))
        _validate(fname, obj)
        out[fname.split(".")[0]] = obj
    out["sha256"] = h.hexdigest()
    tables = Tables(out)
    _CACHE[d] = tables
    return tables


_CACHE: dict[str, Tables] = {}
