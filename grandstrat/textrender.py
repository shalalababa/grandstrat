"""Text renderer (design §0b): serialise a civ's observation and legal actions
as structured text so LLM agents can play through a tool-call interface.

render_text(obs, legal_actions) -> {"json": <canonical JSON string>,
                                    "summary": <human-readable string>}
Both are deterministic given the observation (sorted keys, stable orderings).
"""

from __future__ import annotations

import json


def _units_json(obs: dict, civ: int) -> list[dict]:
    names = obs["unit_names"]
    out = []
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        if owner != civ:
            continue
        out.append({
            "id": uid, "type": utype, "q": int(q), "r": int(r),
            "hp": int(hp), "mp": int(mp),
        })
    return out


def _units_visible_json(obs: dict) -> list[dict]:
    out = []
    for uid, owner, utype, q, r, hp, mp in obs["unit_list"]:
        if owner == obs["you"]:
            continue
        out.append({
            "id": uid, "owner": int(owner), "type": utype,
            "q": int(q), "r": int(r), "hp": int(hp),
        })
    return out


def _cities_json(obs: dict, civ: int | None) -> list[dict]:
    bnames = obs["building_names"]
    out = []
    for cid, owner, q, r, pop, hp, queue, buildings, border, capital in obs["city_list"]:
        if civ is not None and owner != civ:
            continue
        out.append({
            "id": cid, "owner": int(owner), "q": int(q), "r": int(r),
            "pop": int(pop), "hp": int(hp), "queue": queue,
            "buildings": list(buildings), "border": int(border),
            "capital": bool(capital),
        })
    return out


def _cities_visible_json(obs: dict) -> list[dict]:
    """Foreign cities in view: everything in city_list the observer does not
    own (under fog the list is already fog-limited by the information
    module)."""
    you = obs["you"]
    out = []
    for cid, owner, q, r, pop, hp, queue, buildings, border, capital in obs["city_list"]:
        if owner == you:
            continue
        out.append({
            "id": cid, "owner": int(owner), "q": int(q), "r": int(r),
            "pop": int(pop), "hp": int(hp), "queue": queue,
            "buildings": list(buildings), "border": int(border),
            "capital": bool(capital),
        })
    return out


def _civ_json(obs: dict, civ: int) -> dict:
    tnames = obs["tech_names"]
    return {
        "id": civ,
        "gold": int(obs["gold"][civ]),
        "score": int(obs["score"][civ]),
        "alive": bool(obs["alive"][civ]),
        "techs": [tnames[i] for i in np_nonzero(obs["techs"][civ])],
        "researching": obs["researching"][civ],
        "research_progress": int(obs["research_progress"][civ]),
    }


def np_nonzero(mask) -> list[int]:
    return [int(i) for i in range(len(mask)) if mask[i]]


def render_json(obs: dict, legal_actions: list[tuple]) -> str:
    civ = obs["you"]
    payload = {
        "turn": obs["meta"]["turn"],
        "horizon": obs["meta"]["horizon"],
        "you": civ,
        "your_civ": _civ_json(obs, civ),
        "other_civs": [_civ_json(obs, i) for i in range(obs["meta"]["n_civs"]) if i != civ],
        "your_units": _units_json(obs, civ),
        "your_cities": _cities_json(obs, civ),
        "visible_enemy_units": _units_visible_json(obs),
        "visible_enemy_cities": _cities_visible_json(obs),
        "gold": [int(g) for g in obs["gold"]],
        "scores": [int(s) for s in obs["score"]],
        "alive": [bool(a) for a in obs["alive"]],
        "legal_actions": [list(a) for a in legal_actions],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def render_summary(obs: dict, legal_actions: list[tuple]) -> str:
    civ = obs["you"]
    m = obs["meta"]
    lines = []
    lines.append(
        f"Turn {m['turn']}/{m['horizon']} — you are civilisation {civ} "
        f"(gold {int(obs['gold'][civ])}, score {int(obs['score'][civ])})"
    )
    lines.append("Your cities:")
    for c in _cities_json(obs, civ):
        lines.append(
            f"  city {c['id']} at ({c['q']},{c['r']}) pop {c['pop']} hp {c['hp']}"
            f" queue {c['queue']} buildings {','.join(c['buildings']) or '-'}"
        )
    lines.append("Your units:")
    for u in _units_json(obs, civ):
        lines.append(
            f"  unit {u['id']} {u['type']} at ({u['q']},{u['r']}) hp {u['hp']} mp {u['mp']}"
        )
    enemies = _units_visible_json(obs)
    if enemies:
        lines.append("Visible foreign units:")
        for u in enemies:
            lines.append(f"  unit {u['id']} ({u['type']} civ {u['owner']}) at ({u['q']},{u['r']}) hp {u['hp']}")
    lines.append(f"Legal actions ({len(legal_actions)}):")
    for a in legal_actions:
        lines.append("  " + str(list(a)))
    return "\n".join(lines)


def render_text(obs: dict, legal_actions: list[tuple]) -> dict:
    return {
        "json": render_json(obs, legal_actions),
        "summary": render_summary(obs, legal_actions),
    }
