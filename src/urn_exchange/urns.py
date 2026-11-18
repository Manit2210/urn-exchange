"""Exact urn maths shared with the solo "Two urns" game: posteriors, Kelly sizing, optimal stopping.

Urn Exchange reuses these helpers (``exchange.py``). In the solo game a house quotes a fixed bid/ask and
the player may pay for extra draws; Urn Exchange maps each hand onto the same ``HandSpec`` with a fair
fixed quote, so the exact posterior (``posterior``) and the research heuristic of the ``bayes`` bot
(``solve``) are the same code. The solo game itself is not part of this repository.

Everything graded is computed under the player's information set: the exact posterior uses only the
prior, the reported draws, the stated noise rate and (in the solo game's informed-quote deals) the quote.
Objective: expected log bankroll hand by hand (myopic Kelly), with a position limit; the optimal stopping
value over the remaining paid draws is computed by exact recursion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

HANDS = 10
START_BANKROLL = 1000
MAX_SIZE = 10
SHADE = 15  # solo variant: an informed dealer's mid is prior-mid +/- SHADE
OFFSETS = (-SHADE, 0, SHADE)  # solo variant: what an uninformed dealer picks from, uniformly
TH = 0.002  # nats: a gap smaller than this is not counted as a "should have traded/passed" error
MILLI = 1000.0
EPS = 1e-12

PRIORS = (0.5, 0.5, 0.5, 0.3, 0.7, 0.25, 0.75, 0.4, 0.6, 0.2, 0.8)
COMPOSITIONS = ((0.7, 0.3), (0.75, 0.25), (0.8, 0.4), (0.6, 0.4), (0.65, 0.35), (0.9, 0.5),
                (0.55, 0.45), (0.3, 0.7), (0.4, 0.8), (0.2, 0.6), (0.7, 0.3), (0.85, 0.35))


# --------------------------------------------------------------------------- pure maths


@dataclass(frozen=True)
class HandSpec:
    """The public, deal-determined parameters of one hand."""

    prior: float
    red_a: float
    red_b: float
    n0: int  # draws shown before the first decision
    max_extra: int  # paid extra draws allowed
    fee: int  # chips per extra draw
    noise: float  # each reported colour is flipped with this probability
    informed: float  # probability the quote comes from an informed dealer (0 = fixed fair quote)
    half: int  # half spread in chips

    @property
    def base_mid(self) -> int:
        return int(round(100 * self.prior))

    @property
    def total_balls(self) -> int:
        return self.n0 + self.max_extra


def report_red_prob(red: float, noise: float) -> float:
    """P(a draw is REPORTED red) for an urn whose share of red balls is ``red``."""
    return red * (1 - noise) + (1 - red) * noise


def quote_lik(spec: HandSpec, offset: int) -> tuple[float, float]:
    """(P(offset | urn A), P(offset | urn B)); the quote mid is base_mid + offset."""
    q = spec.informed
    if q <= 0:
        return (1.0, 1.0) if offset == 0 else (0.0, 0.0)
    u = (1 - q) / len(OFFSETS)
    return (u + (q if offset == SHADE else 0.0), u + (q if offset == -SHADE else 0.0))


def posterior(spec: HandSpec, reds: int, blues: int, offset: int) -> float:
    """Exact P(urn is A | prior, reported draws, noise, quote)."""
    ra, rb = report_red_prob(spec.red_a, spec.noise), report_red_prob(spec.red_b, spec.noise)
    qa, qb = quote_lik(spec, offset)
    wa = spec.prior * qa * ra ** reds * (1 - ra) ** blues
    wb = (1 - spec.prior) * qb * rb ** reds * (1 - rb) ** blues
    return wa / (wa + wb) if wa + wb > 0 else spec.prior


def max_size(side: str, bankroll: int, bid: int, ask: int, limit: int = MAX_SIZE) -> int:
    """Largest legal size: the position limit, and the worst-case loss must stay below the bankroll."""
    risk = ask if side == "buy" else 100 - bid
    return max(0, min(limit, (bankroll - 1) // risk))


def trade_value(p: float, bankroll: int, bid: int, ask: int, side: str, n: int) -> float:
    """E[ln bankroll] after trading n contracts when P(A) = p."""
    if n <= 0:
        return math.log(bankroll)
    if side == "buy":
        return p * math.log(bankroll + n * (100 - ask)) + (1 - p) * math.log(bankroll - n * ask)
    return p * math.log(bankroll - n * (100 - bid)) + (1 - p) * math.log(bankroll + n * bid)


def trade_chips(p: float, bid: int, ask: int, side: str, n: int) -> float:
    """Expected chips of a trade when P(A) = p."""
    if n <= 0:
        return 0.0
    return n * (100 * p - ask) if side == "buy" else n * (bid - 100 * p)


def best_terminal(p: float, bankroll: int, bid: int, ask: int) -> tuple[float, str, int]:
    """Best of pass / buy n / sell n under E[ln bankroll]: (value, side, n); pass is ('pass', 0)."""
    best = (math.log(bankroll), "pass", 0)
    for side in ("buy", "sell"):
        for n in range(1, max_size(side, bankroll, bid, ask) + 1):
            v = trade_value(p, bankroll, bid, ask, side, n)
            if v > best[0] + EPS:
                best = (v, side, n)
    return best


def solve(spec: HandSpec, bid: int, ask: int, p: float, bankroll: int, left: int) -> dict:
    """Optimal stopping by exact recursion over the remaining paid draws.

    Returns v_star (value of the best policy), v_imm (best immediate pass/trade), q_draw (value of
    drawing once and then playing optimally; None if no draw is possible), and the optimal action.
    """
    v_imm, side, n = best_terminal(p, bankroll, bid, ask)
    q_draw = None
    if left > 0 and bankroll - spec.fee >= 1:
        ra, rb = report_red_prob(spec.red_a, spec.noise), report_red_prob(spec.red_b, spec.noise)
        b2 = bankroll - spec.fee
        q_draw = 0.0
        for red in (True, False):
            la, lb = (ra, rb) if red else (1 - ra, 1 - rb)
            pr = p * la + (1 - p) * lb
            if pr <= 0:
                continue
            p2 = p * la / pr
            q_draw += pr * solve(spec, bid, ask, p2, b2, left - 1)["v_star"]
    if q_draw is not None and q_draw > v_imm + EPS:
        return {"v_star": q_draw, "v_imm": v_imm, "q_draw": q_draw, "action": "draw", "side": "draw", "size": 0}
    return {"v_star": v_imm, "v_imm": v_imm, "q_draw": q_draw, "action": side, "side": side, "size": n}
