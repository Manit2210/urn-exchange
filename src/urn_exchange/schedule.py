"""Run configuration and the match schedule.

A schedule is a deterministic list of ``MatchSpec`` built from a ``Config`` and the game catalogue. Urn
Exchange is a multiplayer game: for each (deal, seed) the roster is split into groups of ``n_players``
models chosen greedily so that every model plays about equally often and with every other model about
equally often (pair co-occurrence balance, below). Each group plays ``n_players`` matches on that deal with
the models rotated cyclically through the seats, so every model plays every seat's evidence once
(duplicate play). The rotations share a ``unit_id`` (the group id). With fewer models than seats the group
is padded with ``bot:<fill_bot>`` seats (scripted, free). In v1 the six models sit at one six-seat table, so
there is one group per (deal, seed) and six rotations.

Greedy group design, per (deal, seed) block: ``ceil(N / k)`` groups; each seat is filled with the model
minimising, in order, (uses in this block, total appearances so far, co-occurrences with the models already
in the group, a hash of (game, deal, seed, trial, group, model)); 16 hash orders are tried per block and the
grouping with the smallest appearance spread, then the smallest sum of squared pair counts, is kept and then
improved by swapping members between groups while that sum drops. Nothing here depends on wall clock or
process state. The solo and duel kinds are kept for games that use them (the core contract supports them).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from urn_exchange.util import sha256_hex

BOT_PREFIX = "bot:"

# The same reply cap and reasoning effort for every model (the protocol); applied after each model's own
# params. A per-model thinking cap (``reasoning: {max_tokens: N}``) replaces the effort for that model.
DEFAULT_SHARED_PARAMS = {"max_tokens": 16000, "reasoning_effort": "high"}

ProviderKind = Literal["openai_compatible", "openai", "openai_responses", "gemini_api", "anthropic", "fake"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str  # label in results
    provider: ProviderKind
    model: str  # the provider's model id
    params: dict = Field(default_factory=dict)  # max_tokens, reasoning_effort, reasoning, cache_control, ...
    api_key_env: str | None = None  # environment variable NAME, never the key itself
    base_url: str | None = None
    price_override: dict | None = None  # {input_per_m, output_per_m, cached_input_per_m}
    # OpenRouter upstream routing (the request's ``provider`` object), e.g. {order: [anthropic],
    # allow_fallbacks: false}. Part of the cache key.
    routing: dict | None = None
    # Client-side pacing: at most this many calls per minute for this model (per process).
    rpm: float | None = Field(default=None, gt=0)
    # Dry-run cost estimate profile, measured in v1: input_tokens_per_call, cached_fraction,
    # output_tokens_per_call. Never sent.
    estimate: dict | None = None


class GameEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game: str = "exchange"
    deals: list[str] | None = None  # explicit deal ids; otherwise ``split`` + ``n_deals``
    split: Literal["pilot", "full", "all"] = "pilot"
    n_deals: int | None = Field(default=None, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0])
    n_players: int | None = Field(default=None, ge=1)  # table size; default the game's minimum
    fill_bot: str = "random"  # pads groups when the roster has fewer models than seats
    # Play only the first N seat rotations of each group (a smoke test). Incomplete groups are not rated.
    rotations: int | None = Field(default=None, ge=1)
    max_steps: int = Field(default=5000, ge=1)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "urn-exchange"
    run_id: str | None = None
    # Hard ceiling on what this run (all resumes included) may spend, in USD. Required; a paid run refuses
    # to start when it is 0, so nobody spends by accident.
    budget_usd: float = Field(ge=0)
    models: list[ModelConfig]
    # Applied to every model after its own params (same max_tokens and reasoning effort for all).
    params: dict = Field(default_factory=lambda: dict(DEFAULT_SHARED_PARAMS))
    games: list[GameEntry]
    concurrency: int = Field(default=4, ge=1)  # matches in parallel
    circuit_breaker: int = Field(default=3, ge=1)  # consecutive errored matches that trip a model
    ledger: str = "results/spend.sqlite"
    results_dir: str = "results/runs"
    cache_dir: str = "results/cache"
    # The cache key includes the run id, so a read only ever replays this run's own calls (a resumed match).
    cache_read: bool = True
    # Dry runs: model key -> bot name (or preference list). Unmapped models get strong bots in turn.
    dry_run_bots: dict[str, str | list[str]] = Field(default_factory=dict)
    # Estimate only, for models without an ``estimate`` profile: reply tokens per call (text + reasoning).
    estimate_output_tokens: int = Field(default=2000, ge=0)

    @model_validator(mode="after")
    def _unique(self):
        keys = [m.key for m in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate model keys in the roster")
        if any(k.startswith(BOT_PREFIX) for k in keys):
            raise ValueError(f"model keys may not start with {BOT_PREFIX!r}")
        return self

    def model_keys(self) -> list[str]:
        return [m.key for m in self.models]

    def effective_models(self) -> list[ModelConfig]:
        """Roster entries with the shared params applied last (they win over per-model params)."""
        return [m.model_copy(update={"params": {**m.params, **self.params}}) for m in self.models]


def load_config(path: str | Path) -> Config:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return Config.model_validate(data)


@dataclass(frozen=True)
class GameInfo:
    """What the scheduler needs to know about a game (built from its class)."""

    game_id: str
    version: str
    kind: str
    min_players: int
    max_players: int
    deals: tuple[str, ...]

    @staticmethod
    def from_class(cls) -> "GameInfo":
        g = cls()
        return GameInfo(cls.game_id, cls.version, cls.kind, int(cls.min_players), int(cls.max_players),
                        tuple(g.deals()))


@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    game_id: str
    game_version: str
    kind: str
    deal_id: str
    seed: int
    seating: tuple[str, ...]  # model key (or bot:<name>) per seat
    unit_id: str  # solo: (game, deal, seed); duel: pair id; multi: group id
    rotation: int = 0
    max_steps: int = 5000

    @property
    def n_players(self) -> int:
        return len(self.seating)

    def models(self) -> list[str]:
        return [s for s in self.seating if not s.startswith(BOT_PREFIX)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seating"] = list(self.seating)
        return d


def select_deals(info: GameInfo, entry: GameEntry) -> list[str]:
    if entry.deals:
        unknown = [d for d in entry.deals if d not in info.deals]
        if unknown:
            raise KeyError(f"{info.game_id}: unknown deals {unknown}")
        out = list(entry.deals)
    else:
        out = [d for d in info.deals if entry.split == "all" or d.startswith(entry.split + "-")]
    if entry.n_deals is not None:
        out = out[: entry.n_deals]
    if not out:
        raise ValueError(f"{info.game_id}: no deals selected (split={entry.split}, n_deals={entry.n_deals})")
    return out


def _h(*parts) -> str:
    return sha256_hex("|".join(str(p) for p in parts))


BLOCK_TRIALS = 16  # tie-break orders tried per block; the most balanced grouping is kept


def _improve(groups: list[list[str]], pair: Counter, models: list[str]) -> list[list[str]]:
    """Hill-climb by swapping members between groups while the sum of squared pair counts drops."""

    def score(gs: list[list[str]]) -> int:
        p2 = Counter(pair)
        for g in gs:
            for i, a in enumerate(g):
                for b in g[i + 1:]:
                    p2[frozenset((a, b))] += 1
        return sum(p2[frozenset((a, b))] ** 2 for i, a in enumerate(models) for b in models[i + 1:])

    groups = [list(g) for g in groups]
    cur = score(groups)
    improved = True
    while improved:
        improved = False
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                for x in range(len(groups[gi])):
                    for y in range(len(groups[gj])):
                        a, b = groups[gi][x], groups[gj][y]
                        if a == b or a in groups[gj] or b in groups[gi]:
                            continue
                        groups[gi][x], groups[gj][y] = b, a
                        new = score(groups)
                        if new < cur:
                            cur, improved = new, True
                        else:
                            groups[gi][x], groups[gj][y] = a, b
    return groups


def balanced_groups(models: list[str], k: int, block_key: str, counts: dict) -> list[list[str]]:
    """Groups of ``k`` distinct models for one (deal, seed) block; updates ``counts`` (appearances, pairs).

    Fewer models than seats: one group of all models (callers pad it with bots)."""
    n = len(models)
    app: Counter = counts.setdefault("appearances", Counter())
    pair: Counter = counts.setdefault("pairs", Counter())

    def pairs_of(group: list[str]):
        return [frozenset((a, b)) for i, a in enumerate(group) for b in group[i + 1:]]

    if n <= k:
        groups = [list(models)]
    else:
        best = None
        for trial in range(BLOCK_TRIALS):
            used: Counter = Counter()
            p2: Counter = Counter(pair)
            a2: Counter = Counter(app)
            groups = []
            for gi in range(math.ceil(n / k)):
                group: list[str] = []
                while len(group) < k:
                    cands = [m for m in models if m not in group]
                    group.append(min(cands, key=lambda m: (used[m], a2[m], sum(p2[frozenset((m, x))] for x in group),
                                                          _h(block_key, trial, gi, m))))
                for m in group:
                    used[m] += 1
                    a2[m] += 1
                for pr in pairs_of(group):
                    p2[pr] += 1
                groups.append(group)
            # Balance score: appearance spread first, then the sum of squared pair counts (lower = flatter).
            score = (max(a2[m] for m in models) - min(a2[m] for m in models),
                     sum(p2[frozenset((a, b))] ** 2 for i, a in enumerate(models) for b in models[i + 1:]))
            if best is None or score < best[0]:
                best = (score, groups)
        groups = _improve(best[1], pair, models)
    for group in groups:
        for m in group:
            app[m] += 1
        for pr in pairs_of(group):
            pair[pr] += 1
    return groups


def build_schedule(cfg: Config, catalog: dict[str, GameInfo], *, models: list[str] | None = None,
                   games: list[str] | None = None) -> list[MatchSpec]:
    """Every match of the config, in a deterministic order. ``models``/``games`` restrict the roster/games."""
    roster = [k for k in cfg.model_keys() if models is None or k in models]
    if models is not None and set(models) - set(cfg.model_keys()):
        raise KeyError(f"unknown model keys {sorted(set(models) - set(cfg.model_keys()))}")
    out: list[MatchSpec] = []
    for entry in cfg.games:
        if games is not None and entry.game not in games:
            continue
        if entry.game not in catalog:
            raise KeyError(f"unknown game {entry.game!r}; known: {sorted(catalog)}")
        info = catalog[entry.game]
        deals = select_deals(info, entry)
        base = dict(game_id=info.game_id, game_version=info.version, kind=info.kind, max_steps=entry.max_steps)
        if info.kind == "solo":
            for d in deals:
                for s in entry.seeds:
                    unit = f"{info.game_id}/{d}/s{s}"
                    for m in roster:
                        out.append(MatchSpec(f"{unit}/{m}", deal_id=d, seed=int(s), seating=(m,), unit_id=unit,
                                             **base))
        elif info.kind == "duel":
            for d in deals:
                for s in entry.seeds:
                    for i, a in enumerate(roster):
                        for b in roster[i + 1:]:
                            unit = f"{info.game_id}/{d}/s{s}/{a}~{b}"
                            for r, seating in enumerate(((a, b), (b, a))):
                                out.append(MatchSpec(f"{info.game_id}/{d}/s{s}/{'+'.join(seating)}", deal_id=d,
                                                     seed=int(s), seating=seating, unit_id=unit, rotation=r,
                                                     **base))
        elif info.kind == "multi":
            k = int(entry.n_players or info.min_players)
            if not info.min_players <= k <= info.max_players:
                raise ValueError(f"{info.game_id}: n_players {k} outside [{info.min_players}, {info.max_players}]")
            counts: dict = {}
            for d in deals:
                for s in entry.seeds:
                    block = f"{info.game_id}/{d}/s{s}"
                    for gi, group in enumerate(balanced_groups(roster, k, block, counts)):
                        seats = list(group) + [BOT_PREFIX + entry.fill_bot] * (k - len(group))
                        unit = f"{block}/g{gi}"
                        for r in range(min(k, entry.rotations or k)):
                            seating = tuple(seats[(j + r) % k] for j in range(k))
                            out.append(MatchSpec(f"{unit}/r{r}/{'+'.join(seating)}", deal_id=d, seed=int(s),
                                                 seating=seating, unit_id=unit, rotation=r, **base))
        else:  # pragma: no cover
            raise ValueError(f"{info.game_id}: unknown kind {info.kind!r}")
    ids = [m.match_id for m in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError("schedule produced duplicate match ids")
    return out


@dataclass
class ScheduleSummary:
    matches: int = 0
    by_game: dict = field(default_factory=dict)
    seats_by_model: dict = field(default_factory=dict)
    est_calls: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(matches: list[MatchSpec], decisions_per_seat: dict[str, float] | None = None) -> ScheduleSummary:
    """Match counts per game and model-seats per model. With ``decisions_per_seat`` (game -> mean decisions
    one seat makes per match, e.g. measured by a dry run) also an estimate of model calls (1 call per
    decision; nudges and illegal-action retries add to it)."""
    s = ScheduleSummary(matches=len(matches))
    calls = 0.0
    for m in matches:
        g = s.by_game.setdefault(m.game_id, {"kind": m.kind, "matches": 0, "model_seats": 0, "units": set(),
                                             "deals": set()})
        g["matches"] += 1
        g["units"].add(m.unit_id)
        g["deals"].add((m.deal_id, m.seed))
        for k in m.models():
            g["model_seats"] += 1
            s.seats_by_model[k] = s.seats_by_model.get(k, 0) + 1
            if decisions_per_seat and m.game_id in decisions_per_seat:
                calls += decisions_per_seat[m.game_id]
    for g in s.by_game.values():
        g["units"] = len(g["units"])
        g["deals"] = len(g["deals"])
    if decisions_per_seat:
        s.est_calls = int(round(calls))
    return s
