"""Play Urn Exchange yourself in the terminal against scripted bots (free, no keys).

You see exactly what a model in your seat would see: the same rules and views. Type a move in short form
or as the JSON a model would send:

  research phase   yes 0.42      (buy research, belief P(A) = 0.42)    no 0.42
  trading rounds   buy 3 60 0.7  (buy 3 contracts, limit 60, belief 0.7)
                   sell 2 40 0.3
                   pass 0.5
  or JSON          {"side": "buy", "size": 3, "limit": 60, "belief": {"p_a": 0.7}, "note": "..."}
"""

from __future__ import annotations

import json
from typing import Callable

from urn_exchange import exchange as X
from urn_exchange.core import View, make_bot, play_match


def parse_move(text: str, research_phase: bool) -> dict:
    """Short form or JSON -> an action dict (validation is left to the game, like for a model)."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    parts = text.split()
    if not parts:
        return {}
    head = parts[0].lower()
    try:
        if research_phase:
            p = float(parts[1]) if len(parts) > 1 else 0.5
            return {"research": head in ("y", "yes", "true", "research"), "belief": {"p_a": p}}
        if head == "pass":
            p = float(parts[1]) if len(parts) > 1 else 0.5
            return {"side": "pass", "size": 0, "belief": {"p_a": p}}
        if head in ("buy", "sell") and len(parts) >= 4:
            return {"side": head, "size": int(parts[1]), "limit": int(parts[2]), "belief": {"p_a": float(parts[3])}}
    except ValueError:
        pass
    return {"unparsed": text}


class Human:
    def __init__(self, seat: int, game: X.Exchange, input_fn: Callable[[str], str] = input,
                 print_fn: Callable[[str], None] = print):
        self.name = "you"
        self.seat = seat
        self.game = game
        self.input = input_fn
        self.print = print_fn
        self.shown_rules = False

    def act(self, view: View, schema: dict) -> dict:
        if not self.shown_rules:
            self.print("=" * 72 + "\n" + self.game.rules(self.seat) + "\n" + "=" * 72)
            self.shown_rules = True
        self.print("\n" + view.text)
        research = self.game.phase == "research"
        hint = "yes|no <p_a>" if research else "buy|sell <size> <limit> <p_a>  or  pass <p_a>"
        return parse_move(self.input(f"your move ({hint}) > "), research)

    def feedback(self, error: str) -> None:  # the next view carries the error text
        pass


def play(deal: str = "pilot-01", seed: int = 0, bots: list[str] | None = None, seat: int = 1,
         input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print) -> dict:
    """One match with you in ``seat`` (1-based) and ``bots`` in the other seats; returns the result."""
    bots = list(bots or ["bayes", "follower"])
    n = len(bots) + 1
    if not 1 <= seat <= n:
        raise ValueError(f"seat must be 1-{n}")
    g = X.Exchange()
    players = []
    it = iter(bots)
    for s in range(n):
        players.append(Human(s, g, input_fn, print_fn) if s == seat - 1 else make_bot(g, X.BOTS, next(it), s))
    log = play_match(g, players, deal, seed)
    r = log.result
    me = seat - 1
    print_fn("\nMatch over. Chips by seat: " + ", ".join(
        f"S{s + 1} {'(you) ' if s == me else ''}{p:+.0f}" for s, p in enumerate(r.payoffs)))
    m = r.metrics[me]
    print_fn(f"You: chips {m['chips']:+.0f}, luck-free ev {m['ev_chips']:+.1f}, belief error (KL) "
             f"{m['belief_logloss']:.4f}, market read {m['market_read']:.4f}, trade regret {m['trade_regret']:.1f}.")
    return r.to_dict()
