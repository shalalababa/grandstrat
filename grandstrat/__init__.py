"""GRANDSTRAT: a headless, deterministic grand-strategy simulator.

Design: docs/DESIGN.md.  World, units, cities, economy, combat (WORLD-keyed
CRN rolls), scripted agents, game records and replay, CLI; fog of war with
purchasable information and an information dial; belief construction;
observer-local handles; a diplomacy protocol; a search-agent interface with
determinization and a buy-or-search hook.
"""

PACKAGE_NAME = "grandstrat"
PACKAGE_VERSION = "0.3.0"

# Bumped whenever the record line format changes incompatibly.
RECORD_SCHEMA = "gs.record/v1"
