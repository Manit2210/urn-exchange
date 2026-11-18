"""The game contract: a game, its players and the match loop.

A game module exports:

  GAME    a ``Game`` subclass (``game_id``, ``version``, ``min_players``, ``max_players``, ``kind``)
  BOTS    ``{name: factory(game, seat) -> Player}`` scripted players (usually ``Bot``); bots may read the
          game state (privileged), models never can

The match loop (``play_match``) is the only way a game is driven, by bots in tests and by models in
the runner (``urn_exchange.runner``), so both see byte-identical views and the same validation:

    game.reset(deal_id, seed, n_players)
    while not game.done:
        for seat in game.to_act():            # one or several seats (simultaneous moves)
            obs = game.view(seat)             # NEW private information since this seat last acted
            action = player.act(obs, game.action_schema(seat))
            err = game.apply(seat, action)    # None, or an error the player sees and may fix
    result = game.result()

Rules for every game:

* Deterministic: all randomness comes from ``self.rng`` (``game_rng`` of game id, version, deal and
  seed) or counter-based streams derived from it. No wall clock, no global ``random``. The same
  (deal, seed) gives the same draws for every seating (duplicate play).
* Hidden state never appears in a view of a seat that may not know it. ``rules(seat)`` and views are
  the only model-visible surfaces, plus ``action_schema``.
* Every action schema has a required ``belief`` property. ``result().metrics`` grades beliefs against
  the exact posterior where the game can compute one.
* ``apply`` returns an error string for an illegal action and leaves the state unchanged; after
  ``MAX_INVALID`` errors in a row the loop applies ``default_action(seat)`` (a pass).
* Payoffs are chips. Multiplayer games are zero-sum across seats. ``metrics`` per seat holds skill
  measures that do not depend on luck (belief error, trade regret).
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Protocol

from urn_exchange.util import canonical_json, sha256_hex, validate_schema

MAX_INVALID = 3
KINDS = ("solo", "duel", "multi")
DISCLAIMER = "This is a game with paper chips for research and fun; it is not investment advice."


def game_rng(game_id: str, version: str, deal_id: str, seed: int) -> random.Random:
    """The per-deal stream. Seating does not enter it, so every seating faces the same deal."""
    return random.Random(int(sha256_hex(f"games|{game_id}|{version}|{deal_id}|{seed}")[:16], 16))


def stream(key: str, *parts: Any) -> random.Random:
    """Counter-based stream: the same (key, parts) always yields the same draws."""
    return random.Random(int(sha256_hex("|".join([key, *(str(p) for p in parts)]))[:16], 16))


@dataclass
class View:
    """What one seat is shown before it acts: text for the model, data for tools/bots/transcripts."""

    text: str
    data: dict = field(default_factory=dict)

    def to_content(self) -> str:
        return self.text if not self.data else canonical_json({"text": self.text, "data": self.data})


@dataclass
class MatchResult:
    payoffs: list[float]  # chips won (+) or lost (-) per seat
    metrics: list[dict]  # per-seat skill measures (belief error, regret, ...); JSON-serialisable
    info: dict = field(default_factory=dict)  # deal facts revealed after the match (hidden values, ...)
    transcript: list[dict] = field(default_factory=list)  # public log for the replay viewer

    def to_dict(self) -> dict:
        return {"payoffs": self.payoffs, "metrics": self.metrics, "info": self.info,
                "transcript": self.transcript}


class Game(abc.ABC):
    game_id: ClassVar[str]
    version: ClassVar[str]
    kind: ClassVar[str]  # "solo" | "duel" | "multi"
    min_players: ClassVar[int] = 1
    max_players: ClassVar[int] = 1
    title: ClassVar[str] = ""

    def __init__(self) -> None:
        self.deal_id: str | None = None
        self.seed: int | None = None
        self.n_players = 0
        self.rng: random.Random = random.Random(0)

    # ---- hooks -------------------------------------------------------------
    @abc.abstractmethod
    def deals(self) -> list[str]:
        """Deal ids (``pilot-NN`` for the small set, ``full-NN`` for the tournament)."""

    @abc.abstractmethod
    def setup(self) -> None:
        """Build the deal from ``self.rng`` (called by ``reset``)."""

    @abc.abstractmethod
    def rules(self, seat: int) -> str:
        """The rules as this seat must know them (role-dependent; no hidden values)."""

    @abc.abstractmethod
    def to_act(self) -> list[int]:
        """Seats whose decision is needed now (several = simultaneous); [] when done."""

    @abc.abstractmethod
    def view(self, seat: int) -> View:
        """New private information for ``seat`` since it last acted (the first call is the opening state)."""

    @abc.abstractmethod
    def action_schema(self, seat: int) -> dict:
        """JSON schema of this seat's next action; must require ``belief``."""

    @abc.abstractmethod
    def apply(self, seat: int, action: dict) -> str | None:
        """Apply a schema-valid action. Return an error string (state unchanged) if it is illegal."""

    @abc.abstractmethod
    def default_action(self, seat: int) -> dict:
        """The pass/fold/no-op applied after ``MAX_INVALID`` errors in a row (must be legal)."""

    @property
    @abc.abstractmethod
    def done(self) -> bool: ...

    @abc.abstractmethod
    def result(self) -> MatchResult: ...

    # ---- shared ------------------------------------------------------------
    def reset(self, deal_id: str, seed: int = 0, n_players: int | None = None) -> None:
        if deal_id not in self.deals():
            raise KeyError(f"{self.game_id}: unknown deal {deal_id!r}")
        n = self.min_players if n_players is None else int(n_players)
        if not self.min_players <= n <= self.max_players:
            raise ValueError(f"{self.game_id}: {n} players outside [{self.min_players}, {self.max_players}]")
        self.deal_id, self.seed, self.n_players = deal_id, int(seed), n
        self.rng = game_rng(self.game_id, self.version, deal_id, int(seed))
        self.setup()

    def check(self, seat: int, action: Any) -> str | None:
        """Schema check then ``apply``; the loop and the runner call this, never ``apply`` directly."""
        if seat not in self.to_act():
            return f"seat {seat} is not due to act"
        if not isinstance(action, dict):
            return "action must be a JSON object"
        errors = validate_schema(action, self.action_schema(seat))
        if errors:
            return "; ".join(errors[:5])
        return self.apply(seat, action)


class Player(Protocol):
    name: str

    def act(self, view: View, schema: dict) -> dict: ...

    def feedback(self, error: str) -> None:  # an illegal action; act() is called again
        ...


@dataclass
class Bot:
    """A scripted player. ``fn(game, seat, view) -> action``; bots may read the game (privileged)."""

    name: str
    game: Game
    seat: int
    fn: Callable[[Game, int, View], dict]
    errors: list[str] = field(default_factory=list)

    def act(self, view: View, schema: dict) -> dict:
        return self.fn(self.game, self.seat, view)

    def feedback(self, error: str) -> None:
        self.errors.append(error)


@dataclass
class MatchLog:
    result: MatchResult
    steps: list[dict]  # {seat, view, action, error, defaulted}
    invalid: list[int]  # invalid actions per seat
    defaulted: list[int]  # default actions applied per seat


def play_match(game: Game, players: list[Player], deal_id: str, seed: int = 0, *,
               max_steps: int = 5000) -> MatchLog:
    """Drive one match with ``players[seat]`` (bots or the runner's model players)."""
    game.reset(deal_id, seed, len(players))
    steps: list[dict] = []
    invalid = [0] * len(players)
    defaulted = [0] * len(players)
    n = 0
    while not game.done:
        due = game.to_act()
        if not due:
            raise RuntimeError(f"{game.game_id}: not done but no seat to act")
        for seat in due:
            if game.done or seat not in game.to_act():
                continue
            view = game.view(seat)
            schema = game.action_schema(seat)
            streak = 0
            while True:
                n += 1
                if n > max_steps:
                    raise RuntimeError(f"{game.game_id}: exceeded {max_steps} steps")
                action = players[seat].act(view, schema)
                err = game.check(seat, action)
                steps.append({"seat": seat, "view": view.text if streak == 0 else None, "action": action,
                              "error": err, "defaulted": False})
                if err is None:
                    break
                invalid[seat] += 1
                streak += 1
                if streak >= MAX_INVALID:
                    fallback = game.default_action(seat)
                    ferr = game.check(seat, fallback)
                    if ferr is not None:
                        raise RuntimeError(f"{game.game_id}: default action illegal: {ferr}")
                    defaulted[seat] += 1
                    steps.append({"seat": seat, "view": None, "action": fallback, "error": None, "defaulted": True})
                    break
                players[seat].feedback(err)
                view = View(f"ERROR: {err}. Send a legal action.", {})
    return MatchLog(game.result(), steps, invalid, defaulted)


# --------------------------------------------------------------------------- bots


def make_bot(game: Game, game_id_bots: dict[str, Callable], name: str, seat: int) -> Player:
    try:
        factory = game_id_bots[name]
    except KeyError:
        raise KeyError(f"{game.game_id}: unknown bot {name!r}; known: {sorted(game_id_bots)}") from None
    return factory(game, seat)
