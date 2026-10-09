"""Agent registry: the four scripted baselines plus the merchant
(design §3), parameterised."""

from __future__ import annotations

from .base import Agent
from .expander import ExpanderAgent
from .merchant import MerchantAgent
from .scout import ScoutHeavyAgent
from .turtle import TurtleAgent
from .warmonger import WarmongerAgent

AGENTS: dict[str, type] = {
    "expander": ExpanderAgent,
    "warmonger": WarmongerAgent,
    "turtle": TurtleAgent,
    "scout": ScoutHeavyAgent,
    "scout_heavy": ScoutHeavyAgent,  # alias, as named in the design
    "merchant": MerchantAgent,       # diplomacy-heavy expander
}

DEFAULT_LINEUP = ("expander", "warmonger", "turtle", "scout")


def make_agents(names, n_civs: int, params: dict | None = None) -> list[Agent]:
    """Build one agent per civ; `names` is cycled if shorter than n_civs."""
    if isinstance(names, str):
        names = [names]
    names = list(names)
    if not names:
        raise ValueError("empty agent list")
    agents = []
    for i in range(n_civs):
        name = names[i % len(names)]
        cls = AGENTS.get(name)
        if cls is None:
            raise ValueError(f"unknown agent {name!r}; valid: {sorted(AGENTS)}")
        p = (params or {}).get(name, {})
        agents.append(cls(i, p))
    return agents
