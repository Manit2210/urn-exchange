"""Core contract: the match loop, validation, defaults and determinism (toy game)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from urn_exchange.core import MAX_INVALID, Bot, Game, MatchResult, View, game_rng, play_match


class Guess(Game):
    """Two seats each guess a hidden digit once; closest wins 1 chip from the other."""

    game_id: ClassVar[str] = "toy_guess"
    version: ClassVar[str] = "0.0.1"
    kind: ClassVar[str] = "duel"
    min_players: ClassVar[int] = 2
    max_players: ClassVar[int] = 2

    def deals(self):
        return ["pilot-01", "pilot-02"]

    def setup(self):
        self.secret = self.rng.randint(0, 9)
        self.guesses: dict[int, int] = {}

    def rules(self, seat):
        return "Guess a digit 0-9."

    def to_act(self):
        return [] if self.done else [s for s in range(2) if s not in self.guesses]

    def view(self, seat):
        return View("Your turn.", {"seat": seat})

    def action_schema(self, seat):
        return {"type": "object", "required": ["guess", "belief"], "additionalProperties": False,
                "properties": {"guess": {"type": "integer"}, "belief": {"type": "object"}}}

    def apply(self, seat, action):
        if not 0 <= action["guess"] <= 9:
            return "guess must be 0-9"
        self.guesses[seat] = action["guess"]
        return None

    def default_action(self, seat):
        return {"guess": 0, "belief": {}}

    @property
    def done(self):
        return len(self.guesses) == 2

    def result(self):
        d = [abs(self.guesses[s] - self.secret) for s in range(2)]
        p = [0.0, 0.0] if d[0] == d[1] else ([1.0, -1.0] if d[0] < d[1] else [-1.0, 1.0])
        return MatchResult(p, [{"err": x} for x in d], {"secret": self.secret})


def bot(fn):
    return lambda g, s: Bot("b", g, s, fn)


def test_loop_and_zero_sum():
    g = Guess()
    oracle = bot(lambda game, seat, v: {"guess": game.secret, "belief": {}})
    rand = bot(lambda game, seat, v: {"guess": (game.secret + 5) % 10, "belief": {}})
    log = play_match(g, [oracle(g, 0), rand(g, 1)], "pilot-01")
    assert log.result.payoffs == [1.0, -1.0]
    assert sum(log.result.payoffs) == 0


def test_invalid_then_default():
    g = Guess()
    bad = bot(lambda game, seat, v: {"guess": 42, "belief": {}})
    ok = bot(lambda game, seat, v: {"guess": 3, "belief": {}})
    log = play_match(g, [bad(g, 0), ok(g, 1)], "pilot-01")
    assert log.invalid == [MAX_INVALID, 0] and log.defaulted == [1, 0]


def test_schema_errors_count_as_invalid():
    g = Guess()
    nobelief = bot(lambda game, seat, v: {"guess": 1})
    ok = bot(lambda game, seat, v: {"guess": 3, "belief": {}})
    log = play_match(g, [nobelief(g, 0), ok(g, 1)], "pilot-02")
    assert log.invalid[0] == MAX_INVALID


def test_rng_independent_of_seating_and_deterministic():
    a = game_rng("x", "1", "pilot-01", 0).random()
    assert a == game_rng("x", "1", "pilot-01", 0).random()
    assert a != game_rng("x", "1", "pilot-01", 1).random()


def test_unknown_deal_and_player_count():
    g = Guess()
    with pytest.raises(KeyError):
        g.reset("nope")
    with pytest.raises(ValueError):
        g.reset("pilot-01", 0, 3)
