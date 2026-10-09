"""Merchant: the diplomacy-heavy scripted baseline.

Plays expansion exactly like the expander and, every `offer_every` turns,
bids the top grid price for open_borders (early) or share_maps (later) to
the lowest-id other alive civ it does not yet have that agreement with
(top-of-grid so the default acceptance reservations can say yes).
Accepts incoming offers that meet the default reservation rule.  Fully
deterministic; the only extra behaviour over the expander is diplomacy
orders.
"""

from __future__ import annotations

from .expander import ExpanderAgent


class MerchantAgent(ExpanderAgent):
    name = "merchant"
    _diplomacy_needs = ("ACCEPT", "OFFER", "REJECT")

    def __init__(self, civ_id, params=None):
        super().__init__(civ_id, params)
        self.offer_every = max(1, int(self.params.get("offer_every", 4)))

    def _diplomacy_layer(self, obs, ls, legal, chosen):
        chosen = list(super()._diplomacy_layer(obs, ls, legal, chosen))
        if any(a[0] in ("ACCEPT", "REJECT", "OFFER") for a in chosen):
            return chosen  # one diplomacy order per turn
        turn = int(obs["meta"]["turn"])
        if turn % self.offer_every:
            return chosen
        # alternate intents: open borders first cycle, share maps second
        kind = "open_borders" if (turn // self.offer_every) % 2 == 0 \
            else "share_maps"
        by_partner: dict[int, list] = {}
        for a in ls:
            if a[0] == "OFFER" and a[2] == kind:
                by_partner.setdefault(a[1], []).append(a)
        if not by_partner:
            return chosen
        partner = min(by_partner)
        opts = by_partner[partner]
        # top grid price, picked by VALUE: `ls` is a set, so position in the
        # iteration order must never choose the bid (PYTHONHASHSEED lesson)
        chosen.append(max(opts, key=lambda a: a[3]))
        return chosen
