"""Toy games for the harness tests (runner, schedule, ratings). Not test files.

Each toy hides a token ``HIDDEN-<n>`` that must never reach a view, the rules or the play CLI before the
match is done (it is revealed in ``MatchResult.info``).
"""

from __future__ import annotations

from typing import ClassVar

from urn_exchange.core import Bot, Game, MatchResult, View, stream

SCHEMA = {"type": "object", "required": ["guess", "belief"], "additionalProperties": False,
          "properties": {"guess": {"type": "integer"}, "belief": {"type": "object"}}}


class _Base(Game):
    rounds: ClassVar[int] = 1

    def deals(self):
        return [f"pilot-{i:02d}" for i in range(1, 5)] + [f"full-{i:02d}" for i in range(1, 9)]

    def setup(self):
        self.secret = self.rng.randint(0, 9)
        self.token = f"HIDDEN-{self.rng.randint(100000, 999999)}"
        self.guesses: dict[int, list[int]] = {s: [] for s in range(self.n_players)}
        self.seen: dict[int, int] = {s: 0 for s in range(self.n_players)}

    def rules(self, seat):
        return f"Guess a digit 0-9 ({self.rounds} round(s)). Closest guess wins chips."

    def to_act(self):
        if self.done:
            return []
        r = min(len(g) for g in self.guesses.values())
        return [s for s in range(self.n_players) if len(self.guesses[s]) == r]

    def view(self, seat):
        n = len(self.guesses[seat])
        return View(f"Round {n + 1}. Your turn.", {"round": n + 1, "seat": seat})

    def action_schema(self, seat):
        return SCHEMA

    def apply(self, seat, action):
        if not 0 <= action["guess"] <= 9:
            return "guess must be 0-9"
        self.guesses[seat].append(action["guess"])
        return None

    def default_action(self, seat):
        return {"guess": 0, "belief": {}}

    @property
    def done(self):
        return all(len(g) >= self.rounds for g in self.guesses.values())

    def errs(self):
        return [sum(abs(x - self.secret) for x in self.guesses[s]) for s in range(self.n_players)]


class ToyDuel(_Base):
    game_id: ClassVar[str] = "toy_duel"
    version: ClassVar[str] = "0.0.1"
    kind: ClassVar[str] = "duel"
    min_players: ClassVar[int] = 2
    max_players: ClassVar[int] = 2
    rounds: ClassVar[int] = 2

    def result(self):
        e = self.errs()
        p = [0.0, 0.0] if e[0] == e[1] else ([1.0, -1.0] if e[0] < e[1] else [-1.0, 1.0])
        return MatchResult(p, [{"err": x} for x in e], {"secret": self.secret, "token": self.token})


class ToyRoles(ToyDuel):
    game_id: ClassVar[str] = "toy_roles"
    ROLES: ClassVar[list[str]] = ["maker", "trader"]


class ToySolo(_Base):
    game_id: ClassVar[str] = "toy_solo"
    version: ClassVar[str] = "0.0.1"
    kind: ClassVar[str] = "solo"
    rounds: ClassVar[int] = 3
    HEADLINE: ClassVar[dict] = {"err": "lower_is_better"}

    def result(self):
        e = self.errs()[0]
        return MatchResult([float(10 - e)], [{"err": float(e), "note": None}], {"secret": self.secret,
                                                                                "token": self.token})


class ToyPit(_Base):
    game_id: ClassVar[str] = "toy_pit"
    version: ClassVar[str] = "0.0.1"
    kind: ClassVar[str] = "multi"
    min_players: ClassVar[int] = 3
    max_players: ClassVar[int] = 4

    def result(self):
        e = self.errs()
        best = min(e)
        winners = [s for s in range(self.n_players) if e[s] == best]
        n = self.n_players
        if len(winners) == n:
            p = [0.0] * n
        else:
            p = [(n - len(winners)) / len(winners) if s in winners else -1.0 for s in range(n)]
        return MatchResult(p, [{"err": x} for x in e], {"secret": self.secret, "token": self.token})


def _oracle(game, seat, view):
    return {"guess": game.secret, "belief": {"p": 1.0}}


def _near(game, seat, view):
    return {"guess": (game.secret + 1) % 10, "belief": {"p": 0.5}}


def _random(game, seat, view):
    r = stream("toy-random", game.deal_id, game.seed, seat, len(game.guesses[seat]))
    return {"guess": r.randint(0, 9), "belief": {"p": r.random()}}


def _bad(game, seat, view):
    return {"guess": 42, "belief": {}}


BOTS = {
    "oracle": lambda g, s: Bot("oracle", g, s, _oracle),
    "near": lambda g, s: Bot("near", g, s, _near),
    "random": lambda g, s: Bot("random", g, s, _random),
    "bad": lambda g, s: Bot("bad", g, s, _bad),
}
TOYS = {c.game_id: c for c in (ToyDuel, ToyRoles, ToySolo, ToyPit)}


def toy_registry() -> tuple[dict, dict]:
    return dict(TOYS), {g: dict(BOTS) for g in TOYS}
