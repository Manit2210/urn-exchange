"""Urn Exchange (game id ``exchange``, 3-6 players, zero-sum).

The multiplayer variation of the solo "Two urns" game (``urns.py`` holds the shared maths): the same two
urns and the same contract (pays 100 if the urn is A), but there is no house. Every contract bought is sold by another player in a single-price call
auction, so all chips move between players. Each player sees one public ball plus its own private
balls; an optional paid research ball adds one more (the fee goes to the other players). Two sealed
rounds per hand: the round-1 clearing price is announced before round 2, so the market itself is
evidence. Design and grading: ``docs/rules.md``.

Duplicate play: every ball is a pure function of (deal, seed, hand, SEAT SLOT, ball index), never of
who sits in the slot, so rotating the models through the slots gives every model every slot's evidence.

There is no exact optimal strategy in a multiplayer game, so the ranking is by chips (duplicate BT in
``urn_exchange.ratings``). The graded measures are exact only given what they condition on: beliefs
against the exact posterior of the seat's own information, the round-2 belief against the pooled
posterior of every seat's balls, trade regret against the best Kelly trade at the realised price.
Research only, paper chips, not investment advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from urn_exchange import urns as U
from urn_exchange.core import DISCLAIMER, Bot, Game, MatchResult, View, stream

HANDS = 10
START_BANKROLL = 1000
MAX_SIZE = 10
PRICE_MIN, PRICE_MAX = 1, 99
NOTE_MAX = 200
TH = U.TH  # nats: gaps below this are not counted as pass/trade mistakes
MILLI = U.MILLI
FEES = (3, 8, 15, 25)  # research fee per hand in research families (chips, paid once)
INSIDER_BALLS = 3

# family -> (research offered, insider slot, noisy reports)
FAMILIES: dict[str, tuple[bool, bool, bool]] = {
    "clean": (False, False, False),
    "research": (True, False, False),
    "insider": (False, True, False),
    "noisy": (False, False, True),
    "insider_research": (True, True, False),
    "noisy_research": (True, False, True),
}
FAMILY_CYCLE = ("clean", "research", "insider", "noisy", "insider_research", "noisy_research")
FAMILY_TEXT = {
    "clean": "clean draws, no research",
    "research": "costly research: each hand you may buy one more private ball (the fee varies by hand)",
    "insider": "one insider seat sees 3 private balls each hand; everyone else sees 1",
    "noisy": "noisy reporting: every reported colour may be flipped",
    "insider_research": "one insider seat sees 3 private balls; costly research each hand",
    "noisy_research": "noisy reporting and costly research each hand",
}
PILOT = {"pilot-01": "clean", "pilot-02": "research", "pilot-03": "insider", "pilot-04": "noisy"}


# --------------------------------------------------------------------------- deals


def deal_params(deal_id: str) -> tuple[str, float]:
    """(family, noise) of a deal. pilot-01..04 = clean, research, insider, noisy (15%)."""
    if deal_id in PILOT:
        fam = PILOT[deal_id]
        return fam, (0.15 if FAMILIES[fam][2] else 0.0)
    i = int(deal_id.split("-")[1])
    fam = FAMILY_CYCLE[(i - 1) % len(FAMILY_CYCLE)]
    noise = stream("exchange-deal", "1", deal_id).choice((0.1, 0.15, 0.2)) if FAMILIES[fam][2] else 0.0
    return fam, noise


def insider_slot(deal_id: str, n_players: int) -> int | None:
    """The insider seat slot of a deal (public), or None. A function of (deal, table size) only."""
    fam, _ = deal_params(deal_id)
    if not FAMILIES[fam][1]:
        return None
    return stream("exchange-insider", "1", deal_id, n_players).randrange(n_players)


@dataclass(frozen=True)
class ExSpec:
    """The public, deal-determined parameters of one hand."""

    prior: float
    red_a: float
    red_b: float
    k_priv: int  # private balls per ordinary seat
    fee: int  # research fee (0 = no research this hand)
    noise: float

    @property
    def uspec(self) -> U.HandSpec:
        """The equivalent ``urns`` spec (fair fixed quote) so its exact maths can be reused."""
        return U.HandSpec(self.prior, self.red_a, self.red_b, 1, 1 if self.fee else 0, self.fee, self.noise, 0.0, 0)

    @property
    def ref(self) -> int:
        return int(round(100 * self.prior))


def hand_spec(deal_id: str, hand: int) -> ExSpec:
    """Hand ``hand`` (0-based): a pure function of (deal, hand), never of the seed or the seating."""
    fam, noise = deal_params(deal_id)
    research, insider, _ = FAMILIES[fam]
    r = stream("exchange-hand", "1", deal_id, hand)
    prior = r.choice(U.PRIORS)
    red_a, red_b = r.choice(U.COMPOSITIONS)
    k = r.choice((1, 2))
    fee = r.choice(FEES)
    return ExSpec(prior, red_a, red_b, 1 if insider else k, fee if research else 0, noise)


def urn_of(deal_id: str, seed: int, hand: int, prior: float) -> str:
    return "A" if stream("exchange-urn", "1", deal_id, seed, hand).random() < prior else "B"


def ball_report(deal_id: str, seed: int, hand: int, slot: Any, j: int, red: float, noise: float) -> bool:
    """Reported colour (True = red) of ball ``j`` of ``slot`` ("pub" or a seat slot). Keyed by slot."""
    r = stream("exchange-ball", "1", deal_id, seed, hand, slot, j)
    true_red = r.random() < red
    flip = r.random() < noise
    return true_red != flip


def own_posterior(spec: ExSpec, reports: list[bool]) -> float:
    """Exact P(A | prior, reported colours, noise) via ``urns.posterior`` (no quote information)."""
    reds = sum(reports)
    return U.posterior(spec.uspec, reds, len(reports) - reds, 0)


def kl(p: float, q: float) -> float:
    """KL(p || q) in nats for Bernoulli beliefs, q clipped to [0.001, 0.999] as in urns."""
    q = min(max(q, 1e-3), 1 - 1e-3)
    return p * math.log(max(p, 1e-12) / q) + (1 - p) * math.log(max(1 - p, 1e-12) / (1 - q))


def _red(r: bool) -> str:
    return "red" if r else "blue"


def _balls(xs: list[bool]) -> str:
    return ", ".join(_red(x) for x in xs) or "(none)"


# --------------------------------------------------------------------------- the call auction


def demand_supply(orders: list[tuple[int, str, int, int]], p: int) -> tuple[int, int]:
    d = sum(n for _, side, n, lim in orders if side == "buy" and lim >= p)
    s = sum(n for _, side, n, lim in orders if side == "sell" and lim <= p)
    return d, s


def clearing_price(orders: list[tuple[int, str, int, int]]) -> tuple[int | None, int]:
    """Single price of the call auction: (price, volume), or (None, 0) when nothing crosses.

    Orders are (seat, side, size, limit). The price maximises matched volume min(D(p), S(p)); ties go to
    the smallest imbalance |D(p) - S(p)|; remaining ties to the price nearest the middle of the tied
    range, and an exact half-way tie to the price nearer 50. The rule reads only prices and sizes (no
    seat), and is symmetric under p -> 100 - p with buy <-> sell."""
    best: tuple[int, int] | None = None
    tied: list[int] = []
    for p in range(PRICE_MIN, PRICE_MAX + 1):
        d, s = demand_supply(orders, p)
        key = (min(d, s), -abs(d - s))
        if best is None or key > best:
            best, tied = key, [p]
        elif key == best:
            tied.append(p)
    if best is None or best[0] <= 0:
        return None, 0
    mid = (min(tied) + max(tied)) / 2
    price = min(tied, key=lambda p: (abs(p - mid), abs(p - 50)))
    return price, best[0]


def allocate(orders: list[tuple[int, str, int, int]], price: int, volume: int,
             tiebreak: list[int]) -> dict[int, int]:
    """Signed fills per seat at ``price`` (buy > 0). The short side fills in full. The long side fills by
    price priority (more aggressive limits first); at the marginal limit the remainder is shared pro rata
    to size, rounded down, and the leftover units go one each by largest fractional remainder, then by
    ``tiebreak`` (a deal-stream permutation of seat slots, never the seat index alone)."""
    fills: dict[int, int] = {}
    rank = {s: i for i, s in enumerate(tiebreak)}
    for side, sign in (("buy", 1), ("sell", -1)):
        elig = [(s, n, lim) for s, sd, n, lim in orders if sd == side and n > 0
                and (lim >= price if side == "buy" else lim <= price)]
        rem = volume
        levels = sorted({lim for _, _, lim in elig}, reverse=(side == "buy"))
        for lim in levels:
            lvl = [(s, n) for s, n, l2 in elig if l2 == lim]
            tot = sum(n for _, n in lvl)
            if rem >= tot:
                for s, n in lvl:
                    fills[s] = fills.get(s, 0) + sign * n
                rem -= tot
                continue
            if rem > 0:
                base = {s: rem * n // tot for s, n in lvl}
                frac = {s: rem * n % tot for s, n in lvl}
                left = rem - sum(base.values())
                for s in sorted(base, key=lambda s: (-frac[s], rank[s]))[:left]:
                    base[s] += 1
                for s, q in base.items():
                    if q:
                        fills[s] = fills.get(s, 0) + sign * q
                rem = 0
            break
    return fills


# --------------------------------------------------------------------------- Kelly maths


def hand_pnl(legs: list[tuple[int, int]], settle: int) -> int:
    """Chips of a hand position: legs (signed qty, price), settled at 100 (A) or 0 (B)."""
    return sum(q * (settle - p) for q, p in legs)


def worst_case(legs: list[tuple[int, int]]) -> int:
    return min(hand_pnl(legs, 100), hand_pnl(legs, 0))


def legal(bankroll: int, legs: list[tuple[int, int]], side: str, n: int, limit: int) -> bool:
    """The bankroll rule: if the order filled in full at its limit, the hand's worst case keeps >= 1 chip."""
    q = n if side == "buy" else -n
    return bankroll + worst_case(legs + [(q, limit)]) >= 1


def legal_max(bankroll: int, legs: list[tuple[int, int]], side: str, limit: int) -> int:
    for n in range(MAX_SIZE, 0, -1):
        if legal(bankroll, legs, side, n, limit):
            return n
    return 0


def log_value(p: float, bankroll: int, legs: list[tuple[int, int]]) -> float:
    """E[ln bankroll] at settlement when P(A) = p."""
    a, b = bankroll + hand_pnl(legs, 100), bankroll + hand_pnl(legs, 0)
    return p * math.log(max(a, 1e-9)) + (1 - p) * math.log(max(b, 1e-9))


def best_at_price(p: float, bankroll: int, legs: list[tuple[int, int]], price: int) -> tuple[float, int]:
    """Best Kelly trade at ``price`` (any size 0..MAX_SIZE, either side, legal): (value, signed qty)."""
    best = (log_value(p, bankroll, legs), 0)
    for side, sign in (("buy", 1), ("sell", -1)):
        for n in range(1, legal_max(bankroll, legs, side, price) + 1):
            v = log_value(p, bankroll, legs + [(sign * n, price)])
            if v > best[0] + U.EPS:
                best = (v, sign * n)
    return best


# --------------------------------------------------------------------------- the game


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _num(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


PHASES = ("research", "r1", "r2")


class Exchange(Game):
    game_id: ClassVar[str] = "exchange"
    version: ClassVar[str] = "0.1.0"
    kind: ClassVar[str] = "multi"
    min_players: ClassVar[int] = 3
    max_players: ClassVar[int] = 6
    title: ClassVar[str] = "Urn Exchange"
    # Headline metrics for the ratings report: chips first (the ranking, duplicate BT), then luck-free measures.
    HEADLINE: ClassVar[dict[str, str]] = {"chips": "higher_is_better", "belief_logloss": "lower_is_better",
                                          "market_read": "lower_is_better", "trade_regret": "lower_is_better"}

    def deals(self) -> list[str]:
        return list(PILOT) + [f"full-{i:02d}" for i in range(1, 13)]

    # ---- setup -----------------------------------------------------------
    def setup(self) -> None:
        n = self.n_players
        self.family, self.noise = deal_params(self.deal_id)
        self.insider = insider_slot(self.deal_id, n)
        self.bankroll = [START_BANKROLL] * n
        self.fees_paid = [0] * n
        self.fees_received = [0] * n
        self.hand = -1
        self.records: list[list[dict]] = [[] for _ in range(n)]  # one per seat per hand
        self.hands_log: list[dict] = []
        self.transcript: list[dict] = []
        self.last_text = [""] * n
        self.pending: dict[int, dict] = {}
        self._begin_hand()

    def _begin_hand(self) -> None:
        self.hand += 1
        self.pending = {}
        if self.hand >= HANDS:
            return
        n, h = self.n_players, self.hand
        s = self.spec = hand_spec(self.deal_id, h)
        self.urn = urn_of(self.deal_id, self.seed, h, s.prior)
        red = s.red_a if self.urn == "A" else s.red_b
        self.public = ball_report(self.deal_id, self.seed, h, "pub", 0, red, s.noise)
        # Each slot's balls: its private ones, then (index n_private) its research ball.
        self.balls = [[ball_report(self.deal_id, self.seed, h, slot, j, red, s.noise)
                       for j in range(self.n_private(slot) + 1)] for slot in range(n)]
        self.researched = [False] * n
        self.research_fees = [0] * n  # this hand: paid (+) and received (shown in views)
        self.research_recv = [0] * n
        self.legs: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        self.rounds: list[dict] = []  # per auction round: price, volume, fills, indicative quotes
        self.cur = [{"hand": h + 1} for _ in range(n)]
        self.phase = "research" if s.fee > 0 else "r1"

    # ---- information sets ----------------------------------------------------
    def n_private(self, slot: int) -> int:
        return INSIDER_BALLS if self.insider is not None and slot == self.insider else self.spec.k_priv

    def own_reports(self, seat: int) -> list[bool]:
        """The public ball plus this seat's private balls (and its research ball once bought)."""
        k = self.n_private(seat) + (1 if self.researched[seat] else 0)
        return [self.public] + self.balls[seat][:k]

    def own_post(self, seat: int) -> float:
        return own_posterior(self.spec, self.own_reports(seat))

    def pooled_reports(self) -> list[bool]:
        """Every ball any seat has seen so far this hand (public, private, research bought)."""
        out = [self.public]
        for s in range(self.n_players):
            out += self.own_reports(s)[1:]
        return out

    def pooled_post(self) -> float:
        return own_posterior(self.spec, self.pooled_reports())

    def can_research(self, seat: int) -> bool:
        return self.phase == "research" and self.bankroll[seat] - self.spec.fee >= 1

    def tiebreak(self, rnd: int) -> list[int]:
        order = list(range(self.n_players))
        stream("exchange-ration", "1", self.deal_id, self.seed, self.hand, rnd).shuffle(order)
        return order

    def fee_shares(self, payer: int) -> dict[int, int]:
        """Split of one research fee among the other seats: equal whole-chip shares, the remainder one
        chip each to the first seats of a deal-stream permutation keyed by (hand, payer slot)."""
        others = [s for s in range(self.n_players) if s != payer]
        f = self.spec.fee
        base, rem = divmod(f, len(others))
        stream("exchange-fee", "1", self.deal_id, self.seed, self.hand, payer).shuffle(others)
        return {s: base + (1 if i < rem else 0) for i, s in enumerate(others)}

    # ---- model-visible text -------------------------------------------------
    def rules(self, seat: int) -> str:
        n = self.n_players
        fam = self.family
        research, insider, noisy = FAMILIES[fam]
        others = n - 1
        ins_txt = (f" The insider is S{self.insider + 1} (fixed for this deal, known to all): it sees "
                   f"{INSIDER_BALLS} private balls each hand, everyone else 1."
                   if self.insider is not None else
                   " Each player gets the same number of private balls (1 or 2, stated each hand).")
        res_txt = (f"RESEARCH (before round 1, simultaneous): you may pay the hand's fee (shown; one of "
                   f"{', '.join(map(str, FEES))}) for ONE more private ball. The fee is split equally among the "
                   f"other {others} players in whole chips; any remainder goes one chip each to other players "
                   "picked by a fixed draw of the deal. Everyone then learns how many players bought research.\n"
                   if research else "No research in this deal.\n")
        noise_txt = (f"Every reported colour (public and private) is flipped red<->blue with probability "
                     f"{_pct(self.noise)}, independently; you see only reports.\n" if noisy else
                     "Reported colours are exact.\n")
        return (
            f"URN EXCHANGE (exchange v{self.version}), deal {self.deal_id}. {n} players S1-S{n}; you are "
            f"S{seat + 1}. {HANDS} independent hands; everyone starts with {START_BANKROLL} chips. "
            f"This deal: {FAMILY_TEXT[fam]}.\n"
            "Each hand the dealer secretly picks urn A with the stated prior, else urn B. Each ball is red or "
            "blue; each urn's share of red is stated. Balls are drawn with replacement (independent given the "
            "urn). Everyone sees 1 public ball; each player also sees its own private balls, different balls "
            f"from everyone else's, never shown to others.{ins_txt}\n" + noise_txt + res_txt +
            "CONTRACT: pays 100 chips if the urn is A, 0 if B. There is no house: every contract bought is "
            "sold by another player, so all chips move between players (zero-sum).\n"
            "TWO ROUNDS per hand. Each round every player sends one sealed order at the same time: side "
            f"buy/sell/pass, size 0-{MAX_SIZE} (0 only for pass, 1-{MAX_SIZE} to trade), limit a whole "
            f"price {PRICE_MIN}-{PRICE_MAX}. A call auction then fills at ONE price for everyone: the price "
            "that matches the most contracts (buy orders with limit >= price against sell orders with limit "
            "<= price); ties go to the smallest buy/sell imbalance, then the middle of the tied prices "
            "(nearer 50 on an exact tie). The smaller side fills in full; the larger side fills best limit "
            "first and, at the last limit, pro rata to size, rounded down, leftover contracts one each by "
            "largest remainder, then a fixed draw of the deal. You never trade with yourself. After round 1 "
            "everyone sees its price and volume (or 'no trade' with the best bid and ask), and your own fill. "
            "Then round 2, at its own single price.\n"
            "Buying q at price c: +q*(100-c) if A, -q*c if B; selling the reverse. BANKROLL RULE: if your "
            "order filled in full at its limit, the worst case of your whole position this hand must leave "
            "you at least 1 chip (a larger size is rejected).\n"
            "After the hand: the urn, your fills and P&L, both clearing prices. Other players' balls and "
            "fills are never shown.\n"
            "Every action includes belief.p_a, your probability that the urn is A. Optional note (max "
            f"{NOTE_MAX} chars, never shown to others).\n"
            "OBJECTIVE: maximise the expected log of your bankroll, hand by hand (Kelly). The ranking is "
            "by chips, in duplicate: every player plays every seat of the deal. Also graded: research and "
            "round-1 beliefs against the exact posterior of YOUR balls; the round-2 belief against the "
            "posterior of ALL players' balls (how well you read the market); trade regret against the best "
            "Kelly trade at the realised price (round 1 on your balls, round 2 on all balls).\n"
            f"{DISCLAIMER}")

    def _header(self, seat: int) -> str:
        s = self.spec
        ins = (f" (the insider S{self.insider + 1}: {INSIDER_BALLS})" if self.insider is not None else "")
        lab = "Reported" if s.noise > 0 else "Draws"
        priv = self.own_reports(seat)[1:]
        tail = " (the last is your research ball)" if self.researched[seat] else ""
        return (f"Prior P(A)={s.prior:g}. Urn A: {_pct(s.red_a)} red. Urn B: {_pct(s.red_b)} red. "
                f"Private balls per player: {s.k_priv}{ins}.\n"
                f"{lab}: public ball {_red(self.public)}; your private balls ({len(priv)}): {_balls(priv)}{tail}.")

    def _round_text(self, rnd: dict, seat: int, label: str) -> str:
        if rnd["price"] is None:
            bb = rnd["best_bid"] if rnd["best_bid"] is not None else "-"
            ba = rnd["best_ask"] if rnd["best_ask"] is not None else "-"
            core = f"{label}: no trade (best bid {bb}, best ask {ba})."
        else:
            core = f"{label}: cleared at {rnd['price']}, {rnd['volume']} contracts."
        q = rnd["fills"].get(seat, 0)
        mine = (f" You {'bought' if q > 0 else 'sold'} {abs(q)} @{rnd['price']}." if q else " Your fill: none.")
        return core + mine

    def view(self, seat: int) -> View:
        if self.done:
            return View("Match over.")
        lines: list[str] = []
        if self.last_text[seat]:
            lines.append(self.last_text[seat])
        s, n = self.spec, self.n_players
        lines.append(f"Hand {self.hand + 1}/{HANDS}, "
                     + {"research": "research phase", "r1": "round 1 of 2", "r2": "round 2 of 2"}[self.phase]
                     + f". You are S{seat + 1} of {n}.")
        lines.append(self._header(seat))
        if s.fee > 0 and self.phase != "research":
            k = sum(self.researched)
            mine = f"you paid {s.fee} for one more ball; " if self.researched[seat] else ""
            lines.append(f"Research: {mine}{k} of {n} players bought research; you received "
                         f"{self.research_recv[seat]} chips.")
        if self.phase == "r2":
            lines.append(self._round_text(self.rounds[0], seat, "Round 1"))
        pos = sum(q for q, _ in self.legs[seat])
        lines.append(f"Bankroll {self.bankroll[seat]}" + (f", position this hand {pos:+d}" if pos else "") + ".")
        if self.phase == "research":
            lines.append(f"Research costs {s.fee} chips (split among the other {n - 1} players). "
                         "Send research true/false and belief.p_a.")
        else:
            lines.append(f"Send a sealed order: side buy/sell/pass, size 0-{MAX_SIZE}, limit {PRICE_MIN}-"
                         f"{PRICE_MAX}, belief.p_a.")
        return View("\n".join(lines))

    def action_schema(self, seat: int) -> dict:
        belief = {"type": "object", "required": ["p_a"], "additionalProperties": False,
                  "properties": {"p_a": {"type": "number", "minimum": 0, "maximum": 1}}}
        if not self.done and self.phase == "research":
            return {"type": "object", "required": ["research", "belief"], "additionalProperties": False,
                    "properties": {"research": {"type": "boolean"}, "belief": belief,
                                   "note": {"type": "string"}}}
        return {"type": "object", "required": ["side", "size", "belief"], "additionalProperties": False,
                "properties": {"side": {"type": "string", "enum": ["buy", "sell", "pass"]},
                               "size": {"type": "integer", "minimum": 0, "maximum": MAX_SIZE},
                               "limit": {"type": "integer", "minimum": PRICE_MIN, "maximum": PRICE_MAX},
                               "belief": belief, "note": {"type": "string"}}}

    def to_act(self) -> list[int]:
        if self.done:
            return []
        return [s for s in range(self.n_players) if s not in self.pending]

    @property
    def done(self) -> bool:
        return self.hand >= HANDS

    def default_action(self, seat: int) -> dict:
        b = {"p_a": self.spec.prior}
        if self.phase == "research":
            return {"research": False, "belief": b}
        return {"side": "pass", "size": 0, "belief": b}

    # ---- moves --------------------------------------------------------------
    def apply(self, seat: int, action: dict) -> str | None:
        if self.done:
            return "the match is over"
        if not 0 <= seat < self.n_players or seat in self.pending:
            return f"seat {seat} is not due to act"
        if not isinstance(action, dict):
            return "action must be a JSON object"
        belief = action.get("belief")
        if not isinstance(belief, dict) or set(belief) != {"p_a"}:
            return "belief must be {p_a}"
        p_a = belief["p_a"]
        if not _num(p_a) or not 0 <= p_a <= 1:
            return "belief.p_a must be a number in [0, 1]"
        note = action.get("note")
        if note is not None and (not isinstance(note, str) or len(note) > NOTE_MAX):
            return f"note must be a string of at most {NOTE_MAX} characters"
        if self.phase == "research":
            if set(action) - {"research", "belief", "note"}:
                return f"unexpected fields: {sorted(set(action) - {'research', 'belief', 'note'})}"
            r = action.get("research")
            if not isinstance(r, bool):
                return "research must be true or false"
            if r and not self.can_research(seat):
                return f"cannot afford the research fee of {self.spec.fee} (bankroll must stay at least 1)"
            self.pending[seat] = {"research": r, "p_a": float(p_a)}
        else:
            if set(action) - {"side", "size", "limit", "belief", "note"}:
                return f"unexpected fields: {sorted(set(action) - {'side', 'size', 'limit', 'belief', 'note'})}"
            side, size, limit = action.get("side"), action.get("size"), action.get("limit")
            if side not in ("buy", "sell", "pass"):
                return "side must be buy, sell or pass"
            if isinstance(size, bool) or not isinstance(size, int):
                return "size must be a whole number"
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                      or not PRICE_MIN <= limit <= PRICE_MAX):
                return f"limit must be a whole number {PRICE_MIN}-{PRICE_MAX}"
            if side == "pass":
                if size != 0:
                    return "size must be 0 for pass"
            else:
                if not 1 <= size <= MAX_SIZE:
                    return f"{side} size must be 1-{MAX_SIZE} (use pass to not trade)"
                if limit is None:
                    return f"a {side} order needs a limit price {PRICE_MIN}-{PRICE_MAX}"
                if not legal(self.bankroll[seat], self.legs[seat], side, size, limit):
                    lim = legal_max(self.bankroll[seat], self.legs[seat], side, limit)
                    return (f"size {size} too large: filled in full at {limit} your worst case this hand would "
                            f"not leave 1 chip of your bankroll {self.bankroll[seat]}; the most you can {side} "
                            f"at limit {limit} is {lim}")
            self.pending[seat] = {"side": side, "size": size, "limit": limit, "p_a": float(p_a)}
        if len(self.pending) == self.n_players:
            self._process()
        return None

    def _process(self) -> None:
        acts = [self.pending[s] for s in range(self.n_players)]
        self.pending = {}
        if self.phase == "research":
            self._process_research(acts)
            self.phase = "r1"
        elif self.phase == "r1":
            self._process_round(acts, 1)
            self.phase = "r2"
        else:
            self._process_round(acts, 2)
            self._settle()
            self._begin_hand()

    def _process_research(self, acts: list[dict]) -> None:
        n = self.n_players
        for s in range(n):
            p = self.own_post(s)  # before the research ball
            self.cur[s]["research"] = {"p_own": p, "stated": acts[s]["p_a"], "kl": kl(p, acts[s]["p_a"]),
                                       "brier": (acts[s]["p_a"] - p) ** 2, "bought": acts[s]["research"]}
        for s in range(n):
            if acts[s]["research"]:
                self.researched[s] = True
                self.bankroll[s] -= self.spec.fee
                self.fees_paid[s] += self.spec.fee
                self.research_fees[s] += self.spec.fee
                for o, c in self.fee_shares(s).items():
                    self.bankroll[o] += c
                    self.fees_received[o] += c
                    self.research_recv[o] += c
        self.transcript.append({"hand": self.hand + 1, "event": "research",
                                "bought": [s for s in range(n) if self.researched[s]], "fee": self.spec.fee})

    def _process_round(self, acts: list[dict], rnd: int) -> None:
        n = self.n_players
        orders = [(s, a["side"], a["size"], a["limit"]) for s, a in enumerate(acts) if a["side"] != "pass"]
        price, volume = clearing_price(orders)
        fills = allocate(orders, price, volume, self.tiebreak(rnd)) if price is not None else {}
        bids = [lim for _, sd, _, lim in orders if sd == "buy"]
        asks = [lim for _, sd, _, lim in orders if sd == "sell"]
        info = {"round": rnd, "price": price, "volume": volume, "fills": fills,
                "best_bid": max(bids) if bids else None, "best_ask": min(asks) if asks else None}
        pooled = self.pooled_post()
        for s in range(n):
            a = acts[s]
            p_own = self.own_post(s)
            legs0 = list(self.legs[s])
            q = fills.get(s, 0)
            rec: dict[str, Any] = {"side": a["side"], "size": a["size"], "limit": a["limit"], "fill": q,
                                   "price": price, "stated": a["p_a"], "p_own": p_own, "p_pooled": pooled}
            if rnd == 1:
                rec.update(kl=kl(p_own, a["p_a"]), brier=(a["p_a"] - p_own) ** 2)
                yard = p_own
            else:
                rec.update(market_read=kl(pooled, a["p_a"]),
                           market_gain=kl(pooled, p_own) - kl(pooled, a["p_a"]),
                           kl_own=kl(p_own, a["p_a"]))
                yard = pooled
            B = self.bankroll[s]
            legs1 = legs0 + ([(q, price)] if q else [])
            if price is not None:
                best_v, best_q = best_at_price(yard, B, legs0, price)
                got = log_value(yard, B, legs1)
                none = log_value(yard, B, legs0)
                rec.update(regret=max(0.0, best_v - got) * MILLI, best_q=best_q,
                           pass_when_should_trade=q == 0 and best_v - none > TH,
                           trade_when_should_pass=q != 0 and got < none - TH,
                           wrong_direction=(q > 0 and price > 100 * yard) or (q < 0 and price < 100 * yard))
                if rnd == 2:
                    bo, _ = best_at_price(p_own, B, legs0, price)
                    rec["regret_own"] = max(0.0, bo - log_value(p_own, B, legs1)) * MILLI
            else:
                rec.update(regret=0.0, best_q=0, pass_when_should_trade=False, trade_when_should_pass=False,
                           wrong_direction=False)
                if rnd == 2:
                    rec["regret_own"] = 0.0
            self.cur[s][f"r{rnd}"] = rec
            self.legs[s] = legs1
        self.rounds.append(info)
        self.transcript.append({"hand": self.hand + 1, "event": f"round{rnd}", "price": price, "volume": volume,
                                "fills": {str(k): v for k, v in sorted(fills.items())},
                                "best_bid": info["best_bid"], "best_ask": info["best_ask"]})

    def _settle(self) -> None:
        n = self.n_players
        settle = 100 if self.urn == "A" else 0
        p_final = self.pooled_post()
        pnls = []
        for s in range(n):
            pnl = hand_pnl(self.legs[s], settle)
            pnls.append(pnl)
            self.bankroll[s] += pnl
            ev = sum(q * (100 * p_final - c) for q, c in self.legs[s])
            self.cur[s].update(pnl=pnl, ev_chips=ev - self.research_fees[s] + self.research_recv[s],
                               fees_paid=self.research_fees[s], fees_received=self.research_recv[s])
            self.records[s].append(self.cur[s])
            r1, r2 = self.rounds
            fills = []
            for rd in (r1, r2):
                q = rd["fills"].get(s, 0)
                if q:
                    fills.append(f"{'bought' if q > 0 else 'sold'} {abs(q)} @{rd['price']}")
            px = "; ".join(f"round {rd['round']} " + (f"{rd['price']} x {rd['volume']}" if rd["price"] is not None
                                                      else "no trade") for rd in (r1, r2))
            fee_txt = ""
            if self.research_fees[s] or self.research_recv[s]:
                fee_txt = f"; research fees paid {self.research_fees[s]}, received {self.research_recv[s]}"
            self.last_text[s] = (f"Last hand ({self.hand + 1}): urn {self.urn}. Prices: {px}. You "
                                 + (", ".join(fills) if fills else "did not trade")
                                 + f": {pnl:+d}{fee_txt}.")
        self.hands_log.append({
            "hand": self.hand + 1, "urn": self.urn, "spec": {"prior": self.spec.prior, "red_a": self.spec.red_a,
                                                          "red_b": self.spec.red_b, "k_priv": self.spec.k_priv,
                                                          "fee": self.spec.fee, "noise": self.spec.noise},
            "public": _red(self.public),
            "balls": [[_red(x) for x in self.own_reports(s)[1:]] for s in range(n)],
            "researched": list(self.researched), "pooled_posterior": p_final,
            "rounds": [{**rd, "fills": {str(k): v for k, v in sorted(rd["fills"].items())}} for rd in self.rounds],
            "pnl": pnls, "bankroll": list(self.bankroll)})
        self.transcript.append({"hand": self.hand + 1, "event": "settle", "urn": self.urn, "pnl": pnls,
                                "bankroll": list(self.bankroll)})

    # ---- grading ------------------------------------------------------------
    def _seat_metrics(self, s: int) -> dict:
        recs = self.records[s]
        beliefs = [r["research"] for r in recs if "research" in r] + [r["r1"] for r in recs]
        r2 = [r["r2"] for r in recs]
        rounds = [r["r1"] for r in recs] + r2

        def mean(xs: list[float]) -> float | None:
            return float(sum(xs) / len(xs)) if xs else None

        return {
            "chips": float(self.bankroll[s] - START_BANKROLL),
            "belief_logloss": mean([b["kl"] for b in beliefs]),  # research + round 1 vs own-info posterior
            "belief_brier": mean([b["brier"] for b in beliefs]),
            "market_read": mean([b["market_read"] for b in r2]),  # round 2 vs pooled posterior (KL, nats)
            "market_gain": mean([b["market_gain"] for b in r2]),  # > 0: round-2 belief nearer pooled than own
            "r2_belief_vs_own": mean([b["kl_own"] for b in r2]),
            "trade_regret": float(sum(r["regret"] for r in rounds)),  # milli-nats: r1 own-info + r2 pooled
            "trade_regret_r1": float(sum(r["r1"]["regret"] for r in recs)),
            "trade_regret_r2": float(sum(r["regret"] for r in r2)),
            "trade_regret_r2_own": float(sum(r["regret_own"] for r in r2)),
            "wrong_direction": sum(bool(r["wrong_direction"]) for r in rounds),
            "pass_when_should_trade": sum(bool(r["pass_when_should_trade"]) for r in rounds),
            "trade_when_should_pass": sum(bool(r["trade_when_should_pass"]) for r in rounds),
            "draws": sum(1 for r in recs if r.get("research", {}).get("bought")),
            "fees_paid": self.fees_paid[s],
            "fees_received": self.fees_received[s],
            "ev_chips": float(sum(r["ev_chips"] for r in recs)),  # fills at the final pooled posterior, net fees
            "final_chips": self.bankroll[s],
            "volume": sum(abs(r[k]["fill"]) for r in recs for k in ("r1", "r2")),
            "trades": sum(1 for r in recs for k in ("r1", "r2") if r[k]["fill"]),
            "decisions": len(beliefs) + len(r2),
            "hands": len(recs),
            "insider": self.insider == s,
        }

    def result(self) -> MatchResult:
        if not self.done:
            raise RuntimeError("exchange: match not finished")
        n = self.n_players
        payoffs = [float(self.bankroll[s] - START_BANKROLL) for s in range(n)]
        metrics = [self._seat_metrics(s) for s in range(n)]
        info = {"deal": self.deal_id, "family": self.family, "noise": self.noise, "insider": self.insider,
                "urns": [h["urn"] for h in self.hands_log], "hands": self.hands_log}
        return MatchResult(payoffs, metrics, info, list(self.transcript))


HEADLINE = Exchange.HEADLINE
GAME = Exchange


# --------------------------------------------------------------------------- bots


def _belief(p: float) -> dict:
    return {"p_a": min(max(float(p), 0.0), 1.0)}


def kelly_order(g: Exchange, seat: int, p: float, ref: int, scale: float = 1.0) -> dict:
    """A one-sided limit order from belief ``p``: buy if 100p is above the reference price ``ref`` (sell if
    below), limit at the belief (any fill has edge >= 0), size = the Kelly size at the expected fill (half way
    between belief and reference), times ``scale``, clipped to the bankroll rule; pass when there is none."""
    B, legs = g.bankroll[seat], g.legs[seat]
    v = 100 * p
    if v > ref + 1:
        side, limit = "buy", max(PRICE_MIN, min(PRICE_MAX, math.floor(v)))
    elif v < ref - 1:
        side, limit = "sell", max(PRICE_MIN, min(PRICE_MAX, math.ceil(v)))
    else:
        return {"side": "pass", "size": 0, "belief": _belief(p)}
    exp_px = max(PRICE_MIN, min(PRICE_MAX, int(round((v + ref) / 2))))
    _, q = best_at_price(p, B, legs, exp_px)
    n = abs(q) if (q > 0) == (side == "buy") else 0
    if scale != 1.0 and n:
        n = max(1, int(n * scale))
    n = min(n, legal_max(B, legs, side, limit))
    if n < 1:
        return {"side": "pass", "size": 0, "belief": _belief(p)}
    return {"side": side, "size": n, "limit": limit, "belief": _belief(p)}


def research_worth_it(g: Exchange, seat: int, p: float) -> bool:
    """Heuristic: would ``urns`` pay this fee for one more ball against a fixed fair quote at the prior mid?
    (Exact in urns; only a rule of thumb here, where the price is set by the other players.)"""
    s = g.spec
    if not g.can_research(seat):
        return False
    return U.solve(s.uspec, s.ref, s.ref, p, g.bankroll[seat], 1)["action"] == "draw"


def _noisy_p(g: Exchange, seat: int, p: float, sd: float) -> float:
    if sd <= 0:
        return p
    r = stream("exchange-bot-noise", g.deal_id, g.seed, g.hand, g.phase, seat)
    x = math.log(max(p, 1e-6) / max(1 - p, 1e-6)) + r.gauss(0.0, sd)
    return round(1 / (1 + math.exp(-x)), 4)


def make_bayes(name: str = "bayes", *, noise_sd: float = 0.0, scale: float = 1.0, research: bool = True):
    """Own-information posterior (optionally distorted in log-odds by ``noise_sd``), Kelly order against the
    prior mid in both rounds (it ignores the market price), research by the urns heuristic."""

    def fn(g: Exchange, seat: int, view: View) -> dict:
        p = _noisy_p(g, seat, g.own_post(seat), noise_sd)
        if g.phase == "research":
            return {"research": research and research_worth_it(g, seat, p), "belief": _belief(p)}
        return kelly_order(g, seat, p, g.spec.ref, scale)

    return lambda game, seat: Bot(name, game, seat, fn)


def _pooled(g: Exchange, seat: int, view: View) -> dict:
    """Privileged oracle: the posterior of every seat's balls; Kelly order against the round-1 price when
    there was one, else the prior mid. Never buys research."""
    p = g.pooled_post()
    if g.phase == "research":
        return {"research": False, "belief": _belief(p)}
    ref = g.spec.ref
    if g.phase == "r2" and g.rounds and g.rounds[0]["price"] is not None:
        ref = g.rounds[0]["price"]
    return kelly_order(g, seat, p, ref)


def _follower(g: Exchange, seat: int, view: View) -> dict:
    """Plays bayes in round 1; in round 2 states the round-1 price as its belief and passes."""
    if g.phase == "r2":
        px = g.rounds[0]["price"]
        p = px / 100 if px is not None else g.own_post(seat)
        return {"side": "pass", "size": 0, "belief": _belief(p)}
    return make_bayes()(g, seat).fn(g, seat, view)


def _pass(g: Exchange, seat: int, view: View) -> dict:
    if g.phase == "research":
        return {"research": False, "belief": _belief(g.spec.prior)}
    return {"side": "pass", "size": 0, "belief": _belief(g.spec.prior)}


def _random(g: Exchange, seat: int, view: View) -> dict:
    r = stream("exchange-bot-random", g.deal_id, g.seed, g.hand, g.phase, seat)
    p = r.random()
    if g.phase == "research":
        return {"research": g.can_research(seat) and r.random() < 0.3, "belief": _belief(p)}
    side = r.choice(["buy", "sell", "pass"])
    limit = r.randint(PRICE_MIN, PRICE_MAX)
    if side == "pass":
        return {"side": "pass", "size": 0, "belief": _belief(p)}
    m = legal_max(g.bankroll[seat], g.legs[seat], side, limit)
    if m < 1:
        return {"side": "pass", "size": 0, "belief": _belief(p)}
    return {"side": side, "size": r.randint(1, m), "limit": limit, "belief": _belief(p)}


BOTS = {
    "random": lambda g, s: Bot("random", g, s, _random),
    "alwayspass": lambda g, s: Bot("alwayspass", g, s, _pass),
    "bayes": make_bayes("bayes"),
    "halfkelly": make_bayes("halfkelly", scale=0.5),
    "noisy": make_bayes("noisy", noise_sd=0.6),
    "follower": lambda g, s: Bot("follower", g, s, _follower),
    "pooled": lambda g, s: Bot("pooled", g, s, _pooled),
}
