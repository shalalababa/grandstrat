"""Diplomacy subsystem (design §0b/§2): a kernel module seam next to
combat and information, selected by kernel config `{"diplomacy": "simple"}`
(the default `"none"` builds no module and no diplomacy structure exists at
all -- the observation matrix stays all-peace).

SEMANTICS (simple and documented; every number ASSUMED, data/constants.json):

* Relations are per-pair and PUBLIC.  The observation's `diplomacy` matrix
  carries the strongest agreement per pair (1.0 share_maps, 0.5 open_borders,
  0.0 none); live offers are public news in `obs["meta"]["diplomacy_offers"]`.
  Civ identities are public (the matrix is n x n for everyone).

* OFFER(partner, kind, gold): proposes `kind` in {"open_borders",
  "share_maps"} for a gold side-payment PAID BY THE PROPOSER TO THE ACCEPTOR
  on acceptance (a bid: more gold, more likely accepted).  Offers are legal
  from a data-driven grid (`diplomacy_offer_grid`), live for
  `diplomacy_offer_ttl` turns, one per civ per turn (batch rule).

* ACCEPT(pid) / REJECT(pid): legal for the addressee while the offer lives.
  Acceptance transfers the gold (the ledger's `transfer` column keeps
  conservation checkable: before - purchase + income - upkeep + bounty +
  transfer == after) and activates the agreement.  If the proposer can no
  longer afford the offer (bankruptcy, another transfer), the acceptance
  fails as a `rejected` event (`offer_unaffordable`) and the offer dies.

* open_borders: while active, each partner's units may enter the other's
  territory and heal there.  With the module active, entering a foreign
  civ's territory WITHOUT open borders is not a legal move: legality is
  decided on the acting civ's KNOWLEDGE of the hex owner
  (information.owner_known), so `legal_actions` stays computable from the
  observation; execution re-checks the true owner and drops a stale move
  (`closed_borders`).  share_maps never opens borders.  Barbarians respect
  no borders.  Attacks ignore borders (war bypasses them).

* share_maps: live mutual union of visible sets plus a one-shot memory merge
  -- the same mechanics as the fog module's DIPLO purchase, applied through
  information.apply_share_maps when the offer is accepted.

* Betrayal: any attack on a civ with an active agreement breaks that
  agreement (`dipl_broken`); eliminations clear both.

* Determinism: the engine's diplomacy is fully deterministic; acceptance
  thresholds for scripted civs are data reservations plus a PUBLIC
  pseudo-jitter derived from the offer id (no rng draws).  The named
  `diplomacy` stream stays reserved for user-supplied modules and
  still draws zero times at defaults -- disclosed, not silently inert.
"""

from __future__ import annotations

import numpy as np

KINDS = ("open_borders", "share_maps")
LEVEL = {"none": 0.0, "open_borders": 0.5, "share_maps": 1.0}


class Proposal:
    __slots__ = ("pid", "frm", "to", "kind", "offer", "expires")

    def __init__(self, pid: int, frm: int, to: int, kind: str, offer: int,
                 expires: int):
        self.pid = pid
        self.frm = frm
        self.to = to
        self.kind = kind
        self.offer = offer
        self.expires = expires

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Offer({self.pid}: {self.frm}->{self.to} {self.kind} "
                f"for {self.offer}g, expires {self.expires})")


class SimpleDiplomacy:
    """The default implementation: public bilateral agreements with a
    bid-and-accept protocol.  Offers are sequential public ids (the pid is
    public news; it is not a hidden-state counter)."""

    name = "simple"
    closed_borders = True  # module-on: foreign territory needs open borders

    def __init__(self, tables):
        self.tables = tables
        self._agreements: dict[frozenset, str] = {}
        self._offers: list[Proposal] = []
        self._next_pid = 0
        self._seen_state = None

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        self._agreements = {}
        self._offers = []
        self._next_pid = 0
        self._seen_state = None

    def _ensure(self, state) -> None:
        if self._seen_state is not state:
            self.reset()
            self._seen_state = state

    # -- public views --------------------------------------------------------
    def matrix(self, state) -> np.ndarray:
        """The public relation matrix handed to every observation."""
        n = len(state.civs)
        m = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(m, 1.0)
        for pair, kind in self._agreements.items():
            a, b = tuple(pair)
            m[a, b] = m[b, a] = LEVEL[kind]
        return m

    def public_offers(self) -> list[dict]:
        return [{"pid": o.pid, "from": o.frm, "to": o.to, "kind": o.kind,
                 "offer": o.offer, "expires": o.expires}
                for o in self._offers]

    def agreement(self, a: int, b: int):
        return self._agreements.get(frozenset((a, b)))

    def borders_open(self, a: int, b: int) -> bool:
        """True when b's territory is open to a's units (healing; movement
        checks use entry_ok on a per-hex owner)."""
        return a == b or self.agreement(a, b) == "open_borders"

    def entry_ok(self, owner_at, mover: int) -> bool:
        """May `mover` enter a hex whose (known) territory owner is
        `owner_at`?  Unknown (-2) and unowned (-1) land is free; own
        territory is free; foreign territory needs open borders."""
        if owner_at is None or owner_at < 0 or owner_at == mover:
            return True
        return self.agreement(mover, owner_at) == "open_borders"

    # -- agent-facing action space --------------------------------------------
    def diplomacy_actions(self, state, civ: int) -> list[tuple]:
        """OFFERs this civ can still make and live offers addressed to it.
        Deterministic order: partners ascending, KINDS order, offer grid
        ascending, then live pids ascending."""
        self._ensure(state)
        if not state.civs[civ].alive:
            return []
        K = self.tables.constants
        grid = [int(g) for g in K["diplomacy_offer_grid"]["value"]]
        gold = state.civs[civ].gold
        out: list[tuple] = []
        pending = {(o.frm, o.to, o.kind) for o in self._offers}
        for p in range(len(state.civs)):
            if p == civ or not state.civs[p].alive:
                continue
            for kind in KINDS:
                if self.agreement(civ, p) == kind:
                    continue
                if (civ, p, kind) in pending:
                    continue
                for g in grid:
                    if 0 <= g <= gold:
                        out.append(("OFFER", p, kind, g))
        for o in self._offers:
            if o.to == civ:
                out.append(("ACCEPT", o.pid))
                out.append(("REJECT", o.pid))
        return out

    # -- resolution (orders_diplomacy phase) ----------------------------------
    def resolve_orders(self, state, report, resolved, information=None) -> None:
        """ACCEPT/REJECT first (civ id order), then new OFFERs (civ id
        order), then expiry.  `resolved` actions are true-id space."""
        self._ensure(state)
        for civ in range(len(state.civs)):
            for a in resolved[civ]:
                if a[0] == "ACCEPT":
                    self._accept(state, report, civ, int(a[1]), information)
                elif a[0] == "REJECT":
                    self._reject(report, civ, int(a[1]))
        for civ in range(len(state.civs)):
            for a in resolved[civ]:
                if a[0] == "OFFER":
                    self._offer(state, report, civ, a)
        still = []
        for o in self._offers:
            if state.turn > o.expires:
                report.events.append({"type": "dipl_expired", "pid": o.pid})
            else:
                still.append(o)
        self._offers = still

    def _offer(self, state, report, civ: int, action) -> None:
        _, partner, kind, gold = action
        if (kind not in KINDS or partner == civ or partner < 0
                or not state.civs[partner].alive or not state.civs[civ].alive
                or int(gold) < 0 or int(gold) > state.civs[civ].gold
                or self.agreement(civ, partner) == kind
                or any(o.frm == civ and o.to == partner and o.kind == kind
                       for o in self._offers)):
            report.events.append({"type": "rejected", "civ": civ,
                                  "action": list(action),
                                  "reason": "offer_invalid"})
            return
        ttl = int(self.tables.constants["diplomacy_offer_ttl"]["value"])
        p = Proposal(self._next_pid, civ, partner, kind, int(gold),
                     state.turn + ttl)
        self._next_pid += 1
        self._offers.append(p)
        report.events.append({"type": "dipl_offer", "pid": p.pid,
                              "from": civ, "to": partner, "kind": kind,
                              "offer": p.offer, "expires": p.expires})

    def _accept(self, state, report, civ: int, pid: int, information) -> None:
        o = next((x for x in self._offers if x.pid == pid), None)
        if o is None or o.to != civ:
            report.events.append({"type": "rejected", "civ": civ,
                                  "action": ["ACCEPT", pid],
                                  "reason": "offer_unknown"})
            return
        self._offers.remove(o)
        if state.civs[o.frm].gold < o.offer:
            report.events.append({"type": "rejected", "civ": civ,
                                  "action": ["ACCEPT", pid],
                                  "reason": "offer_unaffordable"})
            return
        # the bid: proposer pays the acceptor
        state.civs[o.frm].gold -= o.offer
        state.civs[civ].gold += o.offer
        report.transfers[o.frm] = report.transfers.get(o.frm, 0) - o.offer
        report.transfers[civ] = report.transfers.get(civ, 0) + o.offer
        self._agreements[frozenset((o.frm, o.to))] = o.kind
        if o.kind == "share_maps" and information is not None:
            apply_maps = getattr(information, "apply_share_maps", None)
            if apply_maps is not None:
                apply_maps(state, o.frm, o.to)
        report.events.append({"type": "dipl_accepted", "pid": o.pid,
                              "from": o.frm, "to": o.to, "kind": o.kind,
                              "offer": o.offer, "turn": state.turn})

    def _reject(self, report, civ: int, pid: int) -> None:
        o = next((x for x in self._offers if x.pid == pid), None)
        if o is None or o.to != civ:
            report.events.append({"type": "rejected", "civ": civ,
                                  "action": ["REJECT", pid],
                                  "reason": "offer_unknown"})
            return
        self._offers.remove(o)
        report.events.append({"type": "dipl_rejected", "pid": o.pid,
                              "from": o.frm, "to": o.to, "kind": o.kind})

    # -- world events ---------------------------------------------------------
    def notify_attack(self, state, report, attacker: int, victim: int) -> None:
        """Any attack breaks the attacker's agreements with the victim."""
        if attacker is None or victim is None or attacker < 0 or victim < 0:
            return
        kind = self._agreements.pop(frozenset((attacker, victim)), None)
        if kind is not None:
            report.events.append({"type": "dipl_broken",
                                  "between": sorted([attacker, victim]),
                                  "kind": kind})
        before = len(self._offers)
        self._offers = [o for o in self._offers
                        if frozenset((o.frm, o.to)) != frozenset((attacker, victim))]
        # dropped pending offers are covered by the dipl_broken event

    def notify_elimination(self, civ: int) -> None:
        dead = {civ}
        self._agreements = {pair: kind for pair, kind in self._agreements.items()
                            if not (set(pair) & dead)}
        self._offers = [o for o in self._offers
                        if o.frm != civ and o.to != civ]


def make_diplomacy(config: dict | None, tables):
    """Build the diplomacy module named by kernel config (design §0b seam).
    Default: no module (None) -- no diplomacy structure exists."""
    name = (config or {}).get("diplomacy", None) or "none"
    if name == "none":
        return None
    if name == "simple":
        return SimpleDiplomacy(tables)
    raise ValueError(f"unknown diplomacy module {name!r}")
