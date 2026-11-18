"""Runner: model players, tournaments, retries, spend guard, response cache and resume.

Every match is driven by ``core.play_match`` (validation through ``Game.check``) with a ``ModelPlayer``, a
scripted bot (``bot:<name>`` seats) or, in a dry run, a ``ModelPlayer`` whose FakeProvider is driven by one
of the game's bots, so a dry run exercises the real message shapes and sizes for free.

Message shape (GAMES_WIRE 1), one conversation per seat per match:

* system: ``ARENA_PREAMBLE`` (seat, players, the ``act`` rule, beliefs, scoring) + ``game.rules(seat)``.
  The preamble text is exactly what the v1 models saw (it names the arena "DeskBench Games"); changing it
  changes the experiment, so bump ``GAMES_WIRE`` if you do.
* the first view is a user message; tools = exactly one, ``act``, whose parameters are the game's
  ``action_schema(seat)`` as authored. One tool call per move; all seats of a round act simultaneously
  (each sees only its own information, and no seat sees another's order before the auction).
* after the model calls ``act``, the NEXT view is delivered as the tool result answering that call (the
  canonical ``{"role": "tool", "tool_call_id", "name": "act", "content"}``, which every adapter converts).
  Extra tool calls in the same turn are answered with ``IGNORED_RESULT`` so every call id has a result;
  the first ``act`` call is the one applied.
* text with no ``act`` call: one nudge (a user message, or the tool result of a wrongly named tool), then
  ``act`` returns ``{}`` and the game rejects it (``MAX_INVALID`` in a row -> the game's default action).
* the assistant turn is appended exactly as returned (``ProviderResponse.assistant_message``), so
  ``provider_state`` (thinking blocks, reasoning details, encrypted reasoning) is echoed back unmodified.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from urn_exchange.cache import ResponseCache
from urn_exchange.core import MAX_INVALID, Bot, Game, Player, View, make_bot, play_match
from urn_exchange.pricing import ModelPrice, require_paid_price, safe_price
from urn_exchange.providers import FakeProvider, Provider, ProviderError, ProviderResponse, make_provider
from urn_exchange.providers.base import retry_after_seconds
from urn_exchange.schedule import BOT_PREFIX, Config, GameInfo, MatchSpec, ModelConfig, build_schedule, summarize
from urn_exchange.spend import Ledger, SpendCapExceeded, SpendGuard
from urn_exchange.util import canonical_hash, canonical_json, now_iso, redact


def default_registry() -> tuple[dict[str, type[Game]], dict[str, dict[str, Callable[..., Player]]]]:
    """The games this package ships: Urn Exchange only."""
    from urn_exchange import exchange

    return {exchange.GAME.game_id: exchange.GAME}, {exchange.GAME.game_id: dict(exchange.BOTS)}


# --------------------------------------------------------------------------- retries and pacing

# A per-minute rate limit (429 "retry shortly") is transient: wait and retry the same call instead of
# failing the match. New OpenRouter accounts start with a low per-model request limit.
RATE_LIMIT_BACKOFF_S = (5, 10, 20, 40, 60, 60, 60, 60)
# Upstream hiccups (502 "Service temporarily unavailable", dropped connections, timeouts) are retried a
# few times with the same backoff before the match errors.
TRANSIENT_ERROR_CLASSES = frozenset({"gateway_error", "server_error", "connection", "timeout", "upstream_forbidden"})
TRANSIENT_MAX_RETRIES = 4
_pace_lock = threading.Lock()
_next_call_at: dict[int, float] = {}


def _pace(provider: Provider) -> None:
    """Client-side pacing: at most ``provider.rpm`` calls per minute per provider instance."""
    rpm = getattr(provider, "rpm", None)
    if not rpm:
        return
    gap = 60.0 / float(rpm)
    with _pace_lock:
        now = time.monotonic()
        at = max(now, _next_call_at.get(id(provider), 0.0))
        _next_call_at[id(provider)] = at + gap
    if at > now:
        time.sleep(at - now)


def complete_with_backoff(provider: Provider, messages, tools, system, params,
                          sleep=time.sleep) -> ProviderResponse:
    for attempt in range(len(RATE_LIMIT_BACKOFF_S) + 1):
        _pace(provider)
        try:
            return provider.complete(messages, tools, system, params)
        except ProviderError as e:
            transient = e.error_class in TRANSIENT_ERROR_CLASSES and attempt < TRANSIENT_MAX_RETRIES
            if not (e.error_class == "rate_limit" or transient) or attempt == len(RATE_LIMIT_BACKOFF_S):
                raise
            hint = retry_after_seconds(str(e))  # honour the server's suggested wait (capped)
            scale = float(os.environ.get("URN_EXCHANGE_BACKOFF_SCALE", "1"))  # tests set 0
            sleep(scale * min(120.0, max(RATE_LIMIT_BACKOFF_S[attempt], (hint or 0.0) + 1.0)))
    raise AssertionError("unreachable")  # pragma: no cover


def build_providers(models: list[ModelConfig], providers: dict[str, Provider] | None = None) -> dict[str, Provider]:
    """One provider per roster model (shared by every match thread, so pacing is per model)."""
    out = dict(providers or {})
    for m in models:
        if m.key in out:
            continue
        kw: dict[str, Any] = {"default_params": m.params}
        if m.api_key_env:
            kw["api_key_env"] = m.api_key_env
        if m.base_url:
            kw["base_url"] = m.base_url
        if m.price_override:
            kw["price"] = m.price_override
        if m.routing:
            if m.provider != "openai_compatible":
                raise ValueError(f"{m.key}: routing is an OpenRouter option (provider openai_compatible)")
            kw["routing"] = m.routing
        out[m.key] = make_provider(m.provider, m.model, **kw)
        out[m.key].rpm = m.rpm
    return out


# Bump whenever what a model sees or how the request is shaped changes (preamble, nudge, tool description,
# message shape). It is part of every cache key and recorded on every match.
GAMES_WIRE = "1"
ACT_TOOL = "act"
ACT_DESCRIPTION = ("Take your action in the game. Arguments must match this schema exactly; `belief` is "
                   "required. Call it once per turn.")
ARENA_PREAMBLE = (
    "You are playing DeskBench Games: an arena where AI models play small trading games for paper chips, for "
    "fun and research. No real money is involved and nothing here is investment advice.\n\n"
    "You are seat {seat} of {n_players} in \"{title}\".{others}\n\n"
    "How to play:\n"
    "- Act only by calling the `act` tool, once per turn, with arguments that match its schema. Text outside "
    "the tool call is not read by the game.\n"
    "- Every action has a required `belief`: put your honest current belief there. Beliefs are graded "
    "against the exact answer where the game can compute one.\n"
    "- An illegal action comes back as an ERROR; fix it and call `act` again. After {max_invalid} illegal "
    "actions in a row the game plays a default action (a pass, fold or no-op) for you.\n"
    "- You are scored on chips and on the quality of your decisions and beliefs.\n\n"
    "Game rules for your seat:\n"
)
OTHERS_TEXT = (" The other seats are played by other AI models or scripted bots; you see only what your seat "
               "may know.")
NUDGE = "Call the `act` tool now with your action; text outside the tool call is not read by the game."
UNKNOWN_TOOL = "Unknown tool {name!r}: the only tool is `act`. " + NUDGE
IGNORED_RESULT = "Ignored: only one `act` call per turn is applied (the first one)."
MAX_NUDGES = 1
TRUNCATION_STOPS = frozenset({"length", "max_tokens", "max_output_tokens", "incomplete"})
# Halt the whole run at once on these (no retry or other model fixes them).
FATAL_CLASSES = frozenset({"insufficient_credits", "auth", "missing_key", "quota"})


def system_prompt(game: Game, seat: int, title: str | None = None) -> str:
    n = game.n_players or game.min_players
    pre = ARENA_PREAMBLE.format(seat=seat, n_players=n, title=title or game.title or game.game_id,
                                others=OTHERS_TEXT if n > 1 else "", max_invalid=MAX_INVALID)
    return pre + game.rules(seat)


def act_tool(schema: dict) -> dict:
    return {"name": ACT_TOOL, "description": ACT_DESCRIPTION, "parameters": schema}


def view_content(view: View, error: str | None = None) -> str:
    """The exact text a model receives for a view (user message or ``act`` tool result)."""
    text = view.to_content()
    if error and error not in text:
        text = f"ERROR: {error}\n\n{text}"
    return text


def is_truncated(stop_reason: str | None) -> bool:
    return (stop_reason or "").lower() in TRUNCATION_STOPS


def est_input_tokens(messages: list[dict], tools: list[dict], system: str | None) -> int:
    """Same input-token proxy as ``Provider.estimate_max_cost`` (~2.5 chars/token + per-message overhead)."""
    chars = len(json.dumps(messages)) + len(json.dumps(tools)) + len(system or "")
    return math.ceil(chars / 2.5) + 50 * (len(messages) + len(tools))


class PlayerError(RuntimeError):
    """A provider failure inside one seat's call; carries the model so the runner can attribute it."""

    def __init__(self, model_key: str, seat: int, exc: BaseException):
        super().__init__(redact(f"{model_key} (seat {seat}): {type(exc).__name__}: {exc}"))
        self.model_key, self.seat, self.exc = model_key, seat, exc
        self.error_class = getattr(exc, "error_class", None) or ("spend_cap" if isinstance(exc, SpendCapExceeded)
                                                                 else type(exc).__name__)


@dataclass
class MatchContext:
    run_id: str  # also the cache namespace and the spend budget scope
    match_id: str
    game_id: str
    game_version: str
    deal_id: str
    seed: int


class ModelPlayer:
    """One model in one seat of one match (the ``Player`` protocol)."""

    def __init__(self, model_key: str, provider: Provider, game: Game, seat: int, ctx: MatchContext, *,
                 params: dict | None = None, cache: ResponseCache | None = None, cache_read: bool = True,
                 spend_guard: SpendGuard | None = None, projection_price: ModelPrice | None = None,
                 observer: Callable[[View, dict], None] | None = None, title: str | None = None):
        self.name = model_key
        self.model_key = model_key
        self.provider = provider
        self.game = game
        self.seat = seat
        self.ctx = ctx
        self.params = dict(params or {})
        self.cache = cache
        self.cache_read = cache_read
        self.guard = spend_guard
        self.projection_price = projection_price  # dry runs: the roster model's real price
        self.observer = observer
        self.title = title
        self.system: str | None = None
        self.messages: list[dict] = []
        self.calls: list[dict] = []
        self.errors: list[str] = []
        self._pending: list[tuple[str, str, bool]] = []  # (call id, tool name, applied) awaiting a result
        self._error: str | None = None

    # ---- Player protocol -------------------------------------------------------
    def feedback(self, error: str) -> None:
        self.errors.append(error)
        self._error = error

    def act(self, view: View, schema: dict) -> dict:
        if self.system is None:  # rules can depend on the deal/role: built after play_match's reset
            self.system = system_prompt(self.game, self.seat, self.title)
        if self.observer is not None:
            self.observer(view, schema)
        self._deliver(view_content(view, self._error))
        self._error = None
        tools = [act_tool(schema)]
        for attempt in range(1 + MAX_NUDGES):
            resp = self._call(tools, "act" if attempt == 0 else "nudge")
            self.messages.append(resp.assistant_message())
            acts = [tc for tc in resp.tool_calls if tc.name == ACT_TOOL]
            if acts:
                first = acts[0]
                self._pending = [(tc.id, tc.name, tc is first) for tc in resp.tool_calls]
                return dict(first.arguments) if isinstance(first.arguments, dict) else {}
            if attempt >= MAX_NUDGES:
                break
            if resp.tool_calls:  # only wrongly named tools: answer each, the answer carries the nudge
                for tc in resp.tool_calls:
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                          "content": UNKNOWN_TOOL.format(name=tc.name)})
            else:
                self.messages.append({"role": "user", "content": NUDGE})
        # No act call after the nudge: answer any dangling calls on the next delivery, return nothing.
        self._pending = [(tc.id, tc.name, False) for tc in resp.tool_calls]
        return {}

    # ---- internals ---------------------------------------------------------------
    def _deliver(self, content: str) -> None:
        if self._pending:
            # Every call id gets a result. If none was applied (only wrong tools after the nudge), the view
            # rides on the first result, so no user turn follows a tool turn (strict alternation).
            carrier = None if any(a for _, _, a in self._pending) else self._pending[0][0]
            for call_id, name, applied in self._pending:
                note = IGNORED_RESULT if name == ACT_TOOL else UNKNOWN_TOOL.format(name=name)
                text = content if applied else (f"{note}\n\n{content}" if call_id == carrier else note)
                self.messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": text})
            self._pending = []
        else:
            self.messages.append({"role": "user", "content": content})

    def key_params(self) -> dict:
        """Cache-key params (never sent): the game, deal, seed, seat, match and run, so mirrored matches,
        seats, seeds and runs never replay each other; plus the games wire. The provider adds its own
        ``wire_version`` (and routing) in ``request_hash``."""
        c = self.ctx
        return {**self.params, "__games_wire": GAMES_WIRE, "__game": c.game_id, "__game_version": c.game_version,
                "__deal": c.deal_id, "__seed": c.seed, "__seat": self.seat, "__match": c.match_id,
                "__run": c.run_id}

    def _call(self, tools: list[dict], kind: str) -> ProviderResponse:
        msgs, system = self.messages, self.system
        rh = self.provider.request_hash(msgs, tools, system, self.key_params())
        in_tok = est_input_tokens(msgs, tools, system)
        started, t0 = now_iso(), time.monotonic()
        resp: ProviderResponse | None = None
        hit = False
        if self.cache is not None and self.cache_read:
            resp = self.cache.get(rh)
            hit = resp is not None
        episode = f"{self.ctx.match_id}#{self.seat}"
        try:
            if resp is None:
                if self.guard is not None and self.guard.dry_run:
                    price = self.projection_price or self.provider.price
                    max_out = int(self.provider.effective_params(self.params).get("max_tokens", 16000))
                    self.guard.reserve(price.cost(in_tok, max_out), episode_id=episode, request_hash=rh,
                                       model=self.model_key)
                reservation = None
                if self.guard is not None and not self.guard.dry_run and self.provider.is_paid:
                    projected = self.provider.estimate_max_cost(msgs, tools, system, self.params)
                    reservation = self.guard.reserve(projected, episode_id=episode, request_hash=rh,
                                                     model=self.provider.model)
                try:
                    resp = complete_with_backoff(self.provider, msgs, tools, system, self.params)
                except ProviderError as e:
                    if reservation is not None and not e.billable:
                        self.guard.release(reservation)  # type: ignore[union-attr]
                    raise
                if reservation is not None:
                    self.guard.settle(reservation, resp.cost_usd)  # type: ignore[union-attr]
                if self.cache is not None:
                    resp.raw = {**(resp.raw or {}), "_games_provenance": {"run_id": self.ctx.run_id,
                                                                          "match_id": self.ctx.match_id,
                                                                          "seat": self.seat}}
                    self.cache.put(rh, resp)
        except (ProviderError, SpendCapExceeded) as e:
            self.calls.append({"kind": kind, "request_hash": rh, "error": redact(str(e)),
                               "error_class": getattr(e, "error_class", "spend_cap"), "started_at": started})
            raise PlayerError(self.model_key, self.seat, e) from e
        rec = {
            "kind": kind, "turn": len(self.calls) + 1, "request_hash": rh, "cache_hit": hit,
            "model": resp.model, "stop_reason": resp.stop_reason, "truncated": is_truncated(resp.stop_reason),
            "usage": resp.usage.to_dict(), "cost_usd": resp.cost_usd, "latency_s": resp.latency_s,
            "wall_s": round(time.monotonic() - t0, 4), "est_input_tokens": in_tok,
            "text": resp.text, "tool_calls": [t.to_dict() for t in resp.tool_calls], "started_at": started,
        }
        for k in ("billed_cost_usd", "upstream_provider"):
            if getattr(resp, k) is not None:
                rec[k] = getattr(resp, k)
        if resp.provider_state:  # full, for audit (never model-visible text)
            rec["provider_state"] = resp.provider_state
        self.calls.append(rec)
        return resp

    def summary(self) -> dict:
        ok = [c for c in self.calls if "error" not in c]
        tok = {k: sum(int(c["usage"].get(k, 0)) for c in ok) for k in
               ("input_tokens", "output_tokens", "cached_input_tokens")}
        return {"calls": len(self.calls), "nudges": sum(c["kind"] == "nudge" for c in self.calls),
                "truncated": sum(bool(c.get("truncated")) for c in ok),
                "text_only": sum(1 for c in ok if not c["tool_calls"]),
                "cache_hits": sum(bool(c.get("cache_hit")) for c in ok),
                "cost_usd": round(sum(float(c["cost_usd"]) for c in ok), 8),
                "billed_cost_usd": round(sum(float(c.get("billed_cost_usd") or 0.0) for c in ok), 8),
                "est_input_tokens": sum(int(c["est_input_tokens"]) for c in ok),
                "invalid_feedback": len(self.errors), **tok}


# --------------------------------------------------------------------------- dry-run players


# Dry-run defaults: bots that can play every seat, strong first, assigned to models in roster order.
DRY_RUN_BOT_ORDER = ("bayes", "halfkelly", "follower", "noisy", "pooled", "random")


class BotPolicy:
    """FakeProvider policy that answers with a game bot's action for the view the player just saw."""

    def __init__(self, bot: Player):
        self.bot = bot
        self.view: View | None = None
        self.schema: dict | None = None

    def observe(self, view: View, schema: dict) -> None:
        self.view, self.schema = view, schema

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        action = self.bot.act(self.view, self.schema or {})  # type: ignore[arg-type]
        return {"tool_calls": [{"name": ACT_TOOL, "arguments": action}]}


def dry_run_bot_name(cfg: Config, model_key: str, bots: dict) -> str:
    if not bots:
        raise KeyError("game has no BOTS for a dry run")
    pref = cfg.dry_run_bots.get(model_key)
    for name in ([pref] if isinstance(pref, str) else list(pref or [])):
        if name in bots:
            return name
    names = [b for b in DRY_RUN_BOT_ORDER if b in bots] or sorted(bots)
    return names[cfg.model_keys().index(model_key) % len(names)]


# --------------------------------------------------------------------------- one match


@dataclass
class RunEnv:
    """Everything a match needs from the run (built once by ``run_tournament``)."""

    cfg: Config
    run_id: str
    games: dict[str, type[Game]]
    bots: dict[str, dict]
    providers: dict[str, Provider]
    params: dict[str, dict]
    prices: dict[str, ModelPrice]
    dry_run: bool
    guard: SpendGuard | None
    cache: ResponseCache | None
    cache_read: bool


def _player_for(env: RunEnv, spec: MatchSpec, game: Game, seat: int, key: str) -> tuple[Player, dict]:
    bots = env.bots.get(spec.game_id, {})
    if key.startswith(BOT_PREFIX):
        name = key[len(BOT_PREFIX):]
        return make_bot(game, bots, name, seat), {"seat": seat, "bot": name}
    ctx = MatchContext(env.run_id, spec.match_id, spec.game_id, spec.game_version, spec.deal_id, spec.seed)
    if env.dry_run:
        bot_name = dry_run_bot_name(env.cfg, key, bots)
        policy = BotPolicy(make_bot(game, bots, bot_name, seat))
        prov = FakeProvider(f"dry:{key}", policy=policy)
        p = ModelPlayer(key, prov, game, seat, ctx, params=env.params[key], spend_guard=env.guard,
                        projection_price=env.prices[key], observer=policy.observe)
        orig = p.feedback

        def feedback(err: str, _orig=orig, _bot=policy.bot) -> None:
            _orig(err)
            _bot.feedback(err)

        p.feedback = feedback  # type: ignore[method-assign]
        return p, {"seat": seat, "model_key": key, "dry_run_bot": bot_name}
    prov = env.providers[key]
    p = ModelPlayer(key, prov, game, seat, ctx, params=env.params[key], cache=env.cache,
                    cache_read=env.cache_read, spend_guard=env.guard)
    return p, {"seat": seat, "model_key": key, "provider": prov.provider_name, "model": prov.model}


def run_match(env: RunEnv, spec: MatchSpec) -> dict:
    """Play one match through ``play_match``; returns its JSONL record (status ok or error)."""
    game = env.games[spec.game_id]()
    players: list[Player] = []
    meta: list[dict] = []
    for seat, key in enumerate(spec.seating):
        p, m = _player_for(env, spec, game, seat, key)
        players.append(p)
        meta.append(m)
    started, t0 = now_iso(), time.monotonic()
    rec: dict[str, Any] = {"record": "match", "run_id": env.run_id, "games_wire": GAMES_WIRE,
                           **spec.to_dict(), "dry_run": env.dry_run, "started_at": started}
    try:
        log = play_match(game, players, spec.deal_id, spec.seed, max_steps=spec.max_steps)
        rec.update(status="ok", result=log.result.to_dict(), steps=log.steps, invalid=log.invalid,
                   defaulted=log.defaulted)
    except PlayerError as e:
        rec.update(status="error", error=str(e), error_class=e.error_class, error_model=e.model_key,
                   error_seat=e.seat, fatal=isinstance(e.exc, SpendCapExceeded) or e.error_class in FATAL_CLASSES)
    except Exception as e:  # noqa: BLE001 - a game/harness bug: recorded, never silently dropped
        rec.update(status="error", error=redact(f"{type(e).__name__}: {e}"), error_class=f"harness:{type(e).__name__}",
                   error_model=None, fatal=False)
    for p, m in zip(players, meta):
        if isinstance(p, ModelPlayer):
            m.update(summary=p.summary(), calls=p.calls, system=p.system)
        elif isinstance(p, Bot):
            m.update(errors=list(p.errors))
    rec.update(players=meta, finished_at=now_iso(), wall_s=round(time.monotonic() - t0, 4))
    return rec


# --------------------------------------------------------------------------- tournament


class SpendNotConfirmed(PermissionError):
    pass


def catalog_from(games: dict[str, type[Game]]) -> dict[str, GameInfo]:
    return {gid: GameInfo.from_class(cls) for gid, cls in games.items()}


def game_meta(cls: type[Game]) -> dict:
    """Facts the ratings need without importing the game: kind, version, HEADLINE and ROLES (optional
    class or module attributes)."""
    mod = sys.modules.get(cls.__module__)
    headline = getattr(cls, "HEADLINE", None) or getattr(mod, "HEADLINE", None)
    roles = getattr(cls, "ROLES", None) or getattr(mod, "ROLES", None)
    return {"version": cls.version, "kind": cls.kind, "title": cls.title or cls.game_id,
            "min_players": cls.min_players, "max_players": cls.max_players,
            "headline": dict(headline) if headline else None, "roles": list(roles) if roles else None,
            # The first headline key is the game's primary score; stored by name because the manifest is
            # written with sorted keys.
            "primary": next(iter(headline)) if headline else None}


def load_records(run_dir: str | Path) -> list[dict]:
    """Match records of a run directory (``matches.jsonl``) or of a trace file (``.jsonl`` / ``.jsonl.gz``)."""
    p = Path(run_dir)
    if p.is_dir():
        p = p / "matches.jsonl"
    if not p.exists():
        return []
    if p.suffix == ".gz":
        import gzip

        text = gzip.open(p, "rt", encoding="utf-8").read()
    else:
        text = p.read_text()
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def latest_by_match(records: list[dict]) -> dict[str, dict]:
    """The last record per match id (a resumed run appends a fresh record for a re-played match). Records
    from a trace file carry a ``dataset`` label; the same match id in two datasets is two records."""
    out: dict[str, dict] = {}
    for r in records:
        if r.get("record") == "match":
            out[f"{r['dataset']}|{r['match_id']}" if r.get("dataset") else r["match_id"]] = r
    return out


def projection(cfg: Config, records: list[dict], prices: dict[str, ModelPrice], params: dict[str, dict]) -> dict:
    """Cost estimate from a dry run's measured call counts and message sizes.

    Expected: models with an ``estimate`` profile (measured in v1: input tokens per call including echoed
    reasoning, the cached fraction of those, output tokens per call) cost calls x that profile at the model's
    prices. Other models: the dry run's estimated input tokens (no cache discount) plus calls x
    ``estimate_output_tokens``. Worst case: no cache discount and every call writing ``max_tokens``.
    Real runs also re-send a few calls (retries, illegal-action feedback) that a bot dry run does not."""
    per: dict[str, dict] = {}
    for r in records:
        for p in r.get("players", []):
            k = p.get("model_key")
            if not k or "summary" not in p:
                continue
            a = per.setdefault(k, {"matches": 0, "calls": 0, "est_input_tokens": 0})
            a["matches"] += 1
            a["calls"] += p["summary"]["calls"]
            a["est_input_tokens"] += p["summary"]["est_input_tokens"]
    profiles = {m.key: m.estimate for m in cfg.models}
    total_exp = total_max = 0.0
    for k, a in per.items():
        price = prices[k]
        max_out = int(params[k].get("max_tokens", 16000))
        prof = profiles.get(k)
        if prof:
            in_call = float(prof["input_tokens_per_call"])
            cf = float(prof.get("cached_fraction", 0.0))
            out_call = float(prof.get("output_tokens_per_call", cfg.estimate_output_tokens))
            a["expected_usd"] = round(a["calls"] * (in_call * ((1 - cf) * price.input_per_m + cf * price.cached_input_per_m)
                                                     + out_call * price.output_per_m) / 1e6, 4)
            worst_in = max(a["est_input_tokens"], int(a["calls"] * in_call))
            a["estimate"] = "v1 profile"
        else:
            a["expected_usd"] = round(price.cost(a["est_input_tokens"], a["calls"] * cfg.estimate_output_tokens), 4)
            worst_in = a["est_input_tokens"]
            a["estimate"] = "flat"
        a["worst_case_usd"] = round(price.cost(worst_in, a["calls"] * max_out), 4)
        a["price_status"] = price.status
        total_exp += a["expected_usd"]
        total_max += a["worst_case_usd"]
    return {"per_model": per, "expected_usd": round(total_exp, 2), "worst_case_usd": round(total_max, 2),
            "budget_usd": cfg.budget_usd}


def run_tournament(cfg: Config, *, dry_run: bool = False, confirmed: bool = False,
                   run_id: str | None = None, resume: str | None = None,
                   providers: dict[str, Provider] | None = None,
                   registry: tuple[dict, dict] | None = None, models: list[str] | None = None,
                   events_stdout: bool = False, stop_event: threading.Event | None = None,
                   progress: Callable[[str], None] | None = None) -> dict:
    """Run (or dry-run) every scheduled match. Returns the run summary (also written to the manifest).

    A paid run (any non-fake provider, not a dry run) refuses without ``confirmed`` and a non-zero
    ``budget_usd``. ``resume`` = an earlier run id: its directory is reused, matches that finished ``ok``
    under the same game version and games wire are skipped, everything else is played (``cache_read`` lets
    a re-played match reuse that run's own cached calls)."""
    found, bots = registry if registry is not None else default_registry()
    wanted = {e.game for e in cfg.games}
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"games not found: {sorted(missing)}; known: {sorted(found)}")
    game_classes = {g: found[g] for g in wanted}
    catalog = catalog_from(game_classes)
    schedule = build_schedule(cfg, catalog, models=models)
    eff = {m.key: m for m in cfg.effective_models()}
    used = {k for sp in schedule for k in sp.models()}
    params = {k: dict(eff[k].params) for k in eff}
    prices = {k: safe_price(m.model, m.price_override) for k, m in eff.items()}

    provs: dict[str, Provider] = {}
    if not dry_run:
        provs = build_providers([eff[k] for k in eff if k in used], providers)
        paid = [k for k in cfg.model_keys() if k in used and provs[k].is_paid]
        if paid:
            if not cfg.budget_usd:
                raise SpendNotConfirmed("refusing: a paid run needs a non-zero budget_usd in the config")
            if not confirmed:
                raise SpendNotConfirmed(f"refusing: paid models {paid}; confirm the spend (budget_usd "
                                        f"{cfg.budget_usd:.2f}) after reading the dry-run estimate")
            for k in paid:
                require_paid_price(eff[k].model, eff[k].price_override)

    rid = resume or run_id or cfg.run_id or (
        f"{cfg.name}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
        f"{canonical_hash(cfg.model_dump())[:8]}" + ("-dry" if dry_run else ""))
    run_dir = Path(cfg.results_dir) / rid
    if resume and not run_dir.exists():
        raise FileNotFoundError(f"no run directory to resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    matches_path, events_path = run_dir / "matches.jsonl", run_dir / "events.jsonl"

    ledger = Ledger(cfg.ledger)
    guard = SpendGuard(ledger, cfg.budget_usd, run_id=rid, dry_run=dry_run)
    cache = None if dry_run else ResponseCache(cfg.cache_dir)
    env = RunEnv(cfg, rid, game_classes, bots, provs, params, prices, dry_run, guard, cache,
                 bool(cfg.cache_read) and not dry_run)

    prior = latest_by_match(load_records(run_dir)) if resume else {}
    done = {mid for mid, r in prior.items() if r.get("status") == "ok" and r.get("games_wire") == GAMES_WIRE
            and r.get("game_version") == catalog[r["game_id"]].version} if prior else set()
    todo = [sp for sp in schedule if sp.match_id not in done]

    manifest = {
        "run_id": rid, "name": cfg.name, "created_at": now_iso(), "dry_run": dry_run, "confirmed": bool(confirmed),
        "games_wire": GAMES_WIRE, "config": json.loads(cfg.model_dump_json()),
        "config_hash": canonical_hash(cfg.model_dump()), "resumed": bool(resume),
        "games": {g: game_meta(c) for g, c in game_classes.items()},
        "schedule": summarize(schedule).to_dict(), "skipped_resume": len(schedule) - len(todo),
        "python": sys.version.split()[0],
        "disclaimer": "Paper chips, research and fun only; not investment advice.",
    }
    _write_json(run_dir / "manifest.json", manifest)

    lock = threading.Lock()
    stop = threading.Event()
    ext = stop_event or threading.Event()
    halt: list[dict] = []
    streak: dict[str, int] = {}
    tripped: dict[str, str] = {}
    counts = {"ok": 0, "error": 0, "skipped": 0}
    fresh: list[dict] = []

    def emit(ev: dict) -> None:
        line = canonical_json({"ts": now_iso(), "run_id": rid, **ev})
        with lock:
            with events_path.open("a") as f:
                f.write(line + "\n")
            if events_stdout:
                print("EVENT " + line, flush=True)

    def write(rec: dict) -> None:
        line = json.dumps(rec, sort_keys=True, default=str)
        with lock:
            with matches_path.open("a") as f:
                f.write(line + "\n")
                f.flush()

    def one(sp: MatchSpec) -> None:
        if stop.is_set() or ext.is_set():
            with lock:
                counts["skipped"] += 1
            return
        bad = [k for k in sp.models() if k in tripped]
        if bad:
            write({"record": "match", "run_id": rid, "games_wire": GAMES_WIRE, **sp.to_dict(), "status": "skipped",
                   "error": f"circuit breaker open for {bad}", "dry_run": dry_run})
            with lock:
                counts["skipped"] += 1
            return
        rec = run_match(env, sp)
        write(rec)
        with lock:
            fresh.append({k: rec.get(k) for k in ("match_id", "status", "players", "game_id")})
            counts[rec["status"]] += 1
            if rec["status"] == "ok":
                for k in sp.models():
                    streak[k] = 0
            else:
                culprit = rec.get("error_model")
                if culprit:
                    streak[culprit] = streak.get(culprit, 0) + 1
                    if streak[culprit] >= cfg.circuit_breaker and culprit not in tripped:
                        tripped[culprit] = rec.get("error_class") or "error"
                if rec.get("fatal") or (not dry_run and str(rec.get("error_class", "")).startswith("harness:")):
                    if not halt:
                        halt.append({"reason": rec.get("error_class"), "match_id": sp.match_id,
                                     "error": rec.get("error")})
                    stop.set()
            n_done = counts["ok"] + counts["error"]
        cost = sum(p.get("summary", {}).get("cost_usd", 0.0) for p in rec.get("players", []))
        emit({"event": "match_end", "match_id": sp.match_id, "status": rec["status"],
              "error_class": rec.get("error_class"), "payoffs": (rec.get("result") or {}).get("payoffs"),
              "cost_usd": cost, "wall_s": rec.get("wall_s")})
        if progress is not None:
            progress(f"[{n_done}/{len(todo)}] {rec['status']:5s} {sp.match_id}"
                     + (f"  {rec.get('error_class')}" if rec["status"] != "ok" else f"  {rec.get('wall_s', 0):.0f}s"))

    emit({"event": "run_start", "matches": len(schedule), "todo": len(todo), "dry_run": dry_run})
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        for f in [pool.submit(one, sp) for sp in todo]:
            f.result()
    if halt:
        emit({"event": "halt", **halt[0]})
    if ext.is_set() and not halt:
        halt.append({"reason": "signal"})

    recs = list(latest_by_match(load_records(run_dir)).values())
    summary = {
        "run_id": rid, "run_dir": str(run_dir), "dry_run": dry_run, "matches": len(schedule),
        "played": len(fresh), "skipped_resume": len(schedule) - len(todo), **counts,
        "tripped": tripped, "halt": halt[0] if halt else None, "wall_s": round(time.monotonic() - t0, 3),
        "status": "halted" if halt else "finished",
    }
    if dry_run:
        summary["estimate"] = projection(cfg, [r for r in recs if r.get("status") == "ok"], prices, params)
        summary["guard_worst_case_usd"] = round(guard.dry_run_projected, 4)
        dec: list[float] = [p["summary"]["calls"] for r in recs for p in r.get("players", []) if "summary" in p]
        summary["calls_per_seat"] = round(sum(dec) / len(dec), 3) if dec else None
    else:
        summary["run_spent_usd"] = round(guard.spent(), 6)
    manifest["summary"] = summary
    manifest["finished_at"] = now_iso()
    _write_json(run_dir / "manifest.json", manifest)
    emit({"event": "run_end", "status": summary["status"], "ok": counts["ok"], "error": counts["error"]})
    return summary


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


__all__ = ["GAMES_WIRE", "ARENA_PREAMBLE", "ACT_TOOL", "NUDGE", "ModelPlayer", "MatchContext", "PlayerError",
           "BotPolicy", "RunEnv", "run_match", "run_tournament", "SpendNotConfirmed", "load_records",
           "latest_by_match", "system_prompt", "view_content", "act_tool", "projection", "game_meta",
           "default_registry", "complete_with_backoff", "build_providers"]
