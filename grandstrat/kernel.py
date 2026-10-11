"""Turn kernel: resolves one simultaneous-moves turn through the design's
deterministic phase list.  This is the module the env calls; combat,
barbarians, diplomacy, economy, and worldgen are swappable modules behind it
(design §0b).

Phases, in order (design §1 "simultaneous move resolution with deterministic
ordering"):
    1. orders_produce       PRODUCE queue sets, civ id order, submission order
    2. orders_move          MOVE for ALL civs, global unit-id order
    3. orders_attack        ATTACK_U/ATTACK_C for ALL civs, attacker unit-id order
    4. orders_civilian      FOUND/IMPROVE, global unit-id order
    5. orders_diplomacy     ACCEPT/REJECT then OFFER, civ id order (diplomacy;
                            gold transfers land in this turn's ledger)
    6. orders_information   SPY/TRADE/DIPLO/ORACLE, civ id order (gold is
                            charged here and lands in this turn's ledger)
    7. income               economy.income_phase (borders, growth, research, gold)
    8. barbarians           spawns + barbarian acts
    9. cleanup              camp clearing, eliminations, victory/score, turn++

Submitted actions arrive in the agent-facing space (observer-local handles
under the fog module); the information module maps them to true ids right
after validation, and every phase below resolves in true-id space.  Record
events quote the SUBMITTED forms (what the agent actually played).
"""

from __future__ import annotations

from . import economy, worldgen
from .barbarians import barbarian_phase, clear_camps
from .combat import CrnCombat
from .engine import (InvalidAction, TurnReport, can_found_city, can_improve,
                     can_move_to, legal_actions, move_unit, remove_unit,
                     terrain_move_cost, validate)
from .hexmap import DIRECTIONS, NUM_DIRS
from .state import City, GameState
from .tables import Tables


class Kernel:
    """Owns the module implementations selected by config.  The seams are
    real (config selects, interfaces documented); alternatives ship as
    separate modules (combat: crn; information: full|fog; diplomacy:
    none|simple)."""

    def __init__(self, tables: Tables, config: dict | None = None):
        self.tables = tables
        cfg = dict(config or {})
        self.config = cfg
        combat_name = cfg.get("combat", "crn")
        if combat_name != "crn":
            raise ValueError(f"unknown combat module {combat_name!r}")
        self.combat = CrnCombat()
        info_name = cfg.get("information", None)
        if info_name not in (None, "full", "fog"):
            # unknown names are rejected loudly rather than silently degrading
            # to full observability.  The module INSTANCE itself is built by
            # the env (observe.make_information) and passed per step, keeping
            # env.info the single swap point for custom modules.
            raise ValueError(f"unknown information module {info_name!r}")
        self.information_name = info_name or "full"
        diplo_name = cfg.get("diplomacy", None)
        if diplo_name not in (None, "none", "simple"):
            raise ValueError(f"unknown diplomacy module {diplo_name!r}")
        self.diplomacy_name = diplo_name or "none"

    # ------------------------------------------------------------------
    def step_turn(self, state: GameState, actions_by_civ: dict,
                  legal_cache: dict | None = None,
                  information=None, diplomacy=None) -> TurnReport:
        if state.done:
            raise RuntimeError("game is over; reset to play again")
        n_civs = len(state.civs)
        if set(actions_by_civ) - set(range(n_civs)):
            raise InvalidAction(f"actions keyed by unknown civ ids: {sorted(set(actions_by_civ) - set(range(n_civs)))}")
        for civ in range(n_civs):
            legal_list = legal_cache.get((state.turn, civ)) if legal_cache else None
            validate(state, civ, actions_by_civ.get(civ, []), legal_list,
                     information=information, diplomacy=diplomacy)

        report = TurnReport(state.turn)
        # submitted = the agent-facing forms (handles under fog; what the
        # record quotes); resolved = true-id forms every phase consumes.
        resolve_action = (getattr(information, "resolve_action", None)
                          if information is not None else None)
        submitted = {}
        resolved = {}
        for civ in range(n_civs):
            acts = [tuple(a) for a in actions_by_civ.get(civ, [])]
            submitted[civ] = acts
            resolved[civ] = ([resolve_action(civ, a) for a in acts]
                             if resolve_action else acts)

        # begin: refresh unit movement points / acted flags
        for u in state.units.values():
            spec = self.tables.units.get(u.type)
            u.mp = int(spec["move"]) if spec else 0
            u.acted = False

        self._orders_produce(state, report, submitted, resolved)
        self._orders_move(state, report, submitted, resolved, diplomacy)
        self._orders_attack(state, report, submitted, resolved, diplomacy)
        self._orders_civilian(state, report, submitted, resolved)
        if diplomacy is not None:
            diplomacy.resolve_orders(state, report, resolved,
                                     information=information)
        if information is not None:
            self._orders_information(state, report, submitted, resolved,
                                     information)
        economy.income_phase(state, report, diplomacy=diplomacy)
        barbarian_phase(state, report, self.combat)

        clear_camps(state, report)
        self._cleanup(state, report, diplomacy)

        for u in state.units.values():
            u.acted = False
        state.turn += 1
        economy.recompute_scores(state)
        report.scores = [c.score for c in state.civs]
        report.gold = [c.gold for c in state.civs]
        report.done = state.done
        report.winner = state.winner
        report.done_reason = state.done_reason
        return report

    # ------------------------------------------------------------------
    def _stale(self, report: TurnReport, civ: int, action, reason: str) -> None:
        report.events.append({
            "type": "rejected", "civ": civ, "action": list(action),
            "reason": reason,
        })

    def _collect(self, state, resolved, submitted, kinds):
        """All submitted actions of the given kinds as (civ, submitted,
        resolved) triples; phases consume the resolved form, rejection
        events quote the submitted one."""
        out = []
        for civ in range(len(state.civs)):
            for a_sub, a_res in zip(submitted[civ], resolved[civ]):
                if a_res[0] in kinds:
                    out.append((civ, a_sub, a_res))
        return out

    def _orders_produce(self, state, report, submitted, resolved) -> None:
        for civ, _, a in self._collect(state, resolved, submitted, ("PRODUCE",)):
            city = state.cities.get(a[1])
            if city is None or city.owner != civ or not state.civs[civ].alive:
                self._stale(report, civ, a, "city_missing")
                continue
            city.queue = a[2]

    def _orders_move(self, state, report, submitted, resolved, diplomacy) -> None:
        hm = state.hexmap
        for civ, a_sub, a in sorted(self._collect(state, resolved, submitted, ("MOVE",)),
                                    key=lambda ca: ca[2][1]):
            u = state.units.get(a[1])
            if u is None:
                self._stale(report, civ, a_sub, "unit_dead")
                continue
            j = int(hm.neighbors[hm.index(u.q, u.r), a[2]])
            if j < 0 or not can_move_to(state, u, j):
                self._stale(report, civ, a_sub, "move_blocked")
                continue
            if diplomacy is not None and diplomacy.closed_borders \
                    and not diplomacy.entry_ok(int(state.owner[j]), civ):
                # legality was decided on the mover's knowledge; the true
                # territory can differ (an unseen capture, a new border) --
                # the move is dropped, not executed (design §6 staleness rule)
                self._stale(report, civ, a_sub, "closed_borders")
                continue
            u.mp -= terrain_move_cost(state, j)
            u.acted = True
            move_unit(state, u, int(hm.qs[j]), int(hm.rs[j]))

    def _orders_attack(self, state, report, submitted, resolved, diplomacy) -> None:
        for civ, a_sub, a in sorted(self._collect(state, resolved, submitted,
                                                  ("ATTACK_U", "ATTACK_C")),
                                    key=lambda ca: ca[2][1]):
            u = state.units.get(a[1])
            if u is None:
                self._stale(report, civ, a_sub, "unit_dead")
                continue
            t = (state.units if a[0] == "ATTACK_U" else state.cities).get(a[2])
            if t is None:
                self._stale(report, civ, a_sub, "target_dead")
                continue
            # resolution-time ownership re-check: an earlier-resolved action in
            # the same simultaneous turn may have captured the target city, so
            # what was a legal attack at submission can be self-attack now.
            if t.owner == u.owner:
                self._stale(report, civ, a_sub, "target_now_friendly")
                continue
            if a[0] == "ATTACK_U":
                self.combat.resolve_attack_unit(state, report, u, t)
            else:
                self.combat.resolve_attack_city(state, report, u, t)
            if diplomacy is not None:
                diplomacy.notify_attack(state, report, u.owner, t.owner)

    def _orders_civilian(self, state, report, submitted, resolved) -> None:
        for civ, a_sub, a in sorted(self._collect(state, resolved, submitted,
                                                  ("FOUND", "IMPROVE")),
                                    key=lambda ca: ca[2][1]):
            u = state.units.get(a[1])
            if u is None:
                self._stale(report, civ, a_sub, "unit_dead")
                continue
            if a[0] == "FOUND":
                if not can_found_city(state, u):
                    self._stale(report, civ, a_sub, "found_blocked")
                    continue
                self._found_city(state, report, u)
            else:
                if not can_improve(state, u):
                    self._stale(report, civ, a_sub, "improve_blocked")
                    continue
                self._improve(state, report, u)

    def _found_city(self, state, report, u) -> None:
        K = self.tables.constants
        civ = state.civs[u.owner]
        capital = not any(c.owner == u.owner for c in state.cities.values())
        city = City(state.next_city_id, u.owner, u.q, u.r, capital=capital)
        city.hp = int(K["city_hp_max"]["value"])
        state.cities[city.id] = city
        state.next_city_id += 1
        state.occ_city[state.hexmap.index(u.q, u.r)] = city.id
        remove_unit(state, u)
        state.territory_stale = True
        report.events.append({
            "type": "city_founded", "city": city.id, "owner": u.owner,
            "q": city.q, "r": city.r, "capital": capital,
        })

    def _improve(self, state, report, u) -> None:
        from .economy import IMPROVE_NAMES

        i = state.hexmap.index(u.q, u.r)
        tname = self.tables.terrain_names[state.terrain[i]]
        iname = None
        for name, spec in self.tables.improvements.items():
            if tname in spec["terrain"]:
                iname = name
                break
        if iname is None:
            self._stale(report, u.owner, ("IMPROVE", u.id), "no_improvement")
            return
        # improvement ids follow IMPROVE_NAMES order
        state.improve[i] = IMPROVE_NAMES.index(iname)
        spec = self.tables.improvements[iname]
        for k, v in spec["yield"].items():
            if k == "food":
                state.yf[i] += int(v)
            elif k == "prod":
                state.yp[i] += int(v)
            elif k == "gold":
                state.yg[i] += int(v)
        u.mp = 0
        u.acted = True
        report.events.append({
            "type": "improvement_built", "improvement": iname, "owner": u.owner,
            "q": u.q, "r": u.r,
        })

    # ------------------------------------------------------------------
    def _orders_information(self, state, report, submitted, resolved,
                            information) -> None:
        """Resolve SPY/TRADE/DIPLO/ORACLE purchases, civ id order.  The
        information module charges gold, applies reveals, and writes the
        record entry; the spend is accumulated on the report so income_phase
        puts an explicit `purchase` column in the gold ledger rows."""
        for civ in range(len(state.civs)):
            for a_sub, a_res in zip(submitted[civ], resolved[civ]):
                if a_res[0] in ("SPY", "TRADE", "DIPLO", "ORACLE"):
                    if information.resolve_purchase(state, report, civ, a_res) is False:
                        self._stale(report, civ, a_sub, "city_missing")

    # ------------------------------------------------------------------
    def _cleanup(self, state, report, diplomacy=None) -> None:
        K = self.tables.constants
        # eliminations: a civ with no cities dies; its units are removed
        for civ_id, civ in enumerate(state.civs):
            if not civ.alive:
                continue
            if any(c.owner == civ_id for c in state.cities.values()):
                continue
            civ.alive = False
            gone = [u.id for u in state.units.values() if u.owner == civ_id]
            for uid in gone:
                remove_unit(state, state.units[uid])
            if diplomacy is not None:
                diplomacy.notify_elimination(civ_id)
            report.events.append({
                "type": "civ_eliminated", "owner": civ_id, "units_removed": gone,
            })
        if state.territory_stale:
            # capture / founding / border growth set the flag; recomputing the
            # owner layer every turn would cost ~10% of the runtime for nothing
            worldgen.recompute_territory(state)

        # victory checks (design §1): domination = one civ owns every original
        # capital location (rival secondary cities surviving adds nothing to
        # the design's definition; no rival needs to be dead).  Else
        # the last civ standing, else the turn horizon decides on score.
        capitals = [c for c in state.cities.values() if c.capital]
        owners = {c.owner for c in capitals}
        if len(owners) == 1:
            w = owners.pop()
            if w >= 0:
                self._finish(state, report, w, "domination")
        alive = [i for i, c in enumerate(state.civs) if c.alive]
        if len(alive) == 1 and not state.done:
            self._finish(state, report, alive[0], "last_civ_standing")
        elif not alive and not state.done:
            self._finish(state, report, -1, "extinction")
        if not state.done and state.turn + 1 >= int(state.params["T"]):
            scores = [c.score for c in state.civs]
            best = max(range(len(scores)), key=lambda i: (scores[i], -i))
            self._finish(state, report, best, "score")

    def _finish(self, state, report, winner: int, reason: str) -> None:
        state.done = True
        state.winner = int(winner)
        state.done_reason = reason
        report.events.append({
            "type": "game_over", "winner": int(winner), "reason": reason,
        })
