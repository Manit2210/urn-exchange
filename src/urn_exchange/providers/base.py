"""Provider-agnostic chat-with-tools interface.

Canonical message format (what the runner builds and every provider converts):

* ``{"role": "user", "content": str}``
* ``{"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}],
  "provider_state": {...}}`` -- ``provider_state`` carries opaque provider-native reasoning state
  that must be echoed back unmodified, one key per owner: ``anthropic_content`` (thinking blocks),
  ``gemini_content`` (thought signatures), ``openrouter_reasoning_details`` (OpenRouter
  ``reasoning_details``), ``openai_reasoning_items`` (Responses reasoning items with
  ``encrypted_content``). Only the owning provider re-emits its key; the others ignore it. It is part
  of the request hash, comes from the (cached) response so a resumed match replays deterministically,
  is logged on the call records and never becomes model-visible text.
* ``{"role": "tool", "tool_call_id": str, "name": str, "content": str}``

Tools are ``{"name", "description", "parameters"}`` dicts (JSON schema parameters).
"""

from __future__ import annotations

import abc
import functools
import json
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from urn_exchange.pricing import ModelPrice, get_price, safe_price
from urn_exchange.util import ApiKey, canonical_hash, redact

DEFAULT_MAX_TOKENS = 16000


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input (includes cache writes)
    output_tokens: int = 0  # includes reasoning/thinking tokens
    cached_input_tokens: int = 0  # cache reads
    cache_write_tokens: int = 0  # subset of input billed at the cache-write rate

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolCallOut:
    id: str
    name: str
    arguments: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[ToolCallOut]
    usage: Usage
    cost_usd: float
    latency_s: float
    raw_request_hash: str
    model: str
    stop_reason: str
    raw: dict = field(default_factory=dict, repr=False)
    provider_state: dict = field(default_factory=dict, repr=False)
    # Gateway billing (OpenRouter ``usage.cost``) and the price-table cost of the same usage;
    # ``cost_usd`` is the billed cost when the gateway reports one. None for direct providers.
    billed_cost_usd: float | None = None
    price_cost_usd: float | None = None
    upstream_provider: str | None = None  # the host that served a gateway call

    def to_dict(self) -> dict:
        extra = {k: getattr(self, k) for k in ("billed_cost_usd", "price_cost_usd", "upstream_provider")
                 if getattr(self, k) is not None}
        return {**extra,
                "text": self.text,
                "tool_calls": [t.to_dict() for t in self.tool_calls],
                "usage": self.usage.to_dict(),
                "cost_usd": self.cost_usd,
                "latency_s": self.latency_s,
                "raw_request_hash": self.raw_request_hash,
                "model": self.model,
                "stop_reason": self.stop_reason,
                "raw": self.raw,
                "provider_state": self.provider_state}

    @classmethod
    def from_dict(cls, d: dict) -> "ProviderResponse":
        return cls(
            text=d["text"],
            tool_calls=[ToolCallOut(**t) for t in d["tool_calls"]],
            usage=Usage(**d["usage"]),
            cost_usd=d["cost_usd"],
            latency_s=d["latency_s"],
            raw_request_hash=d["raw_request_hash"],
            model=d["model"],
            stop_reason=d["stop_reason"],
            raw=d.get("raw", {}),
            provider_state=d.get("provider_state", {}),
            billed_cost_usd=d.get("billed_cost_usd"),
            price_cost_usd=d.get("price_cost_usd"),
            upstream_provider=d.get("upstream_provider"),
        )

    def assistant_message(self) -> dict:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": self.text,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
        }
        if self.provider_state:
            msg["provider_state"] = self.provider_state
        return msg


class ProviderError(RuntimeError):
    """Provider failure; the message is always redacted of secrets.

    ``billable`` is False only when the provider rejected the request before doing any work
    (HTTP 400/401/402/403/404/422/429), so the spend guard may release the reservation.
    ``error_class`` is a stable label (see ``classify_error``) used by retries and run halts.
    """

    def __init__(self, message: str, *, billable: bool = True, status: int | None = None,
                 error_class: str | None = None):
        super().__init__(message)
        self.billable = billable
        self.status = status
        self.error_class = error_class or classify_error(message, status)


_UNBILLED_STATUS = {400, 401, 402, 403, 404, 422, 429}

# (class, markers) in priority order; the first class with a matching marker wins.
_ERROR_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("insufficient_credits", ("402", "insufficient credits", "Insufficient credits", "requires more credits",
                              "PaymentRequired")),
    ("missing_key", ("is not set", "no api_key_env configured", "MissingCredentialError")),
    ("auth", ("401", "403", "AuthenticationError", "PermissionDenied", "PermissionDeniedError")),
    # Exhausted quota/billing does not recover by waiting; a per-minute rate limit does.
    ("quota", ("insufficient_quota", "exceeded your current quota", "check your plan and billing",
               "RESOURCE_EXHAUSTED", "quota")),
    ("rate_limit", ("429", "RateLimitError", "rate limit", "Rate limit", "rate_limit")),
    ("client_closed", ("client has been closed",)),
    ("not_found", ("404", "NotFoundError")),
    ("bad_request", ("400", "422", "BadRequestError", "UnprocessableEntityError", "INVALID_ARGUMENT")),
    ("timeout", ("Timeout", "timed out", "DEADLINE_EXCEEDED")),
    # A dropped connection (google-genai/httpx reports RemoteProtocolError "Server disconnected without
    # sending a response") is transient: retried instead of failing the whole match.
    ("connection", ("APIConnectionError", "ConnectionError", "ConnectError", "Connection reset", "RemoteProtocolError",
                    "Server disconnected", "ReadError")),
    ("server_error", ("500", "502", "503", "504", "529", "InternalServerError", "overloaded", "UNAVAILABLE",
                      "Bad Gateway", "Service Unavailable")),
    ("gateway_error", ("gateway_error",)),
)
_STATUS_CLASSES = {402: "insufficient_credits", 401: "auth", 403: "auth", 429: "rate_limit", 404: "not_found",
                   400: "bad_request", 422: "bad_request", 408: "timeout"}


def _marker_in(marker: str, text: str) -> bool:
    if marker.isdigit():  # a status code, not a digit run inside "4000" or a hash
        return re.search(rf"(?<![\w.]){marker}(?![\w.])", text) is not None
    return marker in text


_PER_MINUTE = re.compile(r"PerMinute|per minute|requests per minute|retry in [\d.]+\s*s", re.I)


def _is_per_minute_limit(text: str) -> bool:
    """A per-minute limit recovers by waiting, even when reported as RESOURCE_EXHAUSTED/quota
    (Gemini: GenerateRequestsPerMinutePerProjectPerModel, "Please retry in 46s"). Daily or free-tier
    exhaustion does not."""
    return bool(_PER_MINUTE.search(text)) and "PerDay" not in text and "per day" not in text.lower()


def retry_after_seconds(message: str | None) -> float | None:
    """Server-suggested wait from a rate-limit error ("retry in 46.4s", "retryDelay': '46s'")."""
    if "spend-based rate limit" in (message or ""):
        return 90.0  # rolling 10-minute spend window: wait long; 8 retries span about 12 minutes
    m = re.search(r"retry in ([\d.]+)\s*s", message or "", re.I) or \
        re.search(r"retryDelay'?\"?\s*:\s*'?\"?([\d.]+)s", message or "")
    return float(m.group(1)) if m else None


def classify_error(message: str | None, status: int | None = None) -> str:
    """Stable error class for an error string (redacted text) or a status code."""
    text = message or ""
    # Google AI Studio's "spend-based rate limit" (a spend cap per rolling 10 minutes on lower tiers) is
    # reported as 429 RESOURCE_EXHAUSTED; it clears by waiting, so it is a rate limit (long backoff), not
    # exhausted quota.
    if "spend-based rate limit" in text:
        return "rate_limit"
    # OpenRouter occasionally answers an Anthropic-routed call with 403 {"type": "forbidden", "message":
    # "Request not allowed"} that succeeds on retry; treated like a gateway hiccup, not bad auth.
    if "Request not allowed" in text and "forbidden" in text and status in (None, 403):
        return "upstream_forbidden"
    # OpenRouter 402 "would exceed your available credits given your current in-flight requests" reserves the
    # worst case of every in-flight request against the balance; it clears when those settle.
    if "in-flight requests" in text or "in_flight_bud" in text:
        return "rate_limit"
    # Google AI Studio sometimes answers 400 "User location is not supported" depending on the gateway's
    # egress region; the same request succeeds on retry.
    if "User location is not supported" in text:
        return "gateway_error"
    if _is_per_minute_limit(text) and (status in (None, 429)) and \
            ("429" in text or "RESOURCE_EXHAUSTED" in text or status == 429):
        return "rate_limit"
    if status is not None:
        if status == 429:  # quota exhaustion vs per-minute limit: decided by the message
            return "quota" if any(_marker_in(m, text) for m in _ERROR_CLASSES[3][1]) else "rate_limit"
        if status in _STATUS_CLASSES:
            return _STATUS_CLASSES[status]
        if 500 <= int(status) < 600:
            return "server_error"
    for cls, markers in _ERROR_CLASSES:
        if any(_marker_in(m, text) for m in markers):
            return cls
    head = text.split(":", 1)[0].strip()
    return f"other:{head[:40] or 'unknown'}"


def request_hash(provider: str, model: str, params: dict, messages: list[dict],
                 tools: list[dict], system: str | None) -> str:
    """Canonical request hash: key order, tuple/list and None-valued params do not matter."""
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    return canonical_hash({
        "provider": provider,
        "model": model,
        "params": clean_params,
        "system": system or "",
        "messages": messages,
        "tools": tools,
    })


# SDK clients are created lazily and shared by the match thread pool. Creation is serialised:
# google-genai closes its HTTP client when a replaced Client is garbage-collected, which kills
# in-flight requests made through the old one.
_CLIENT_LOCK = threading.RLock()


class Provider(abc.ABC):
    provider_name: str = "base"

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        make = cls.__dict__.get("_get_client")
        if make is not None:
            @functools.wraps(make)
            def locked(self, _make=make):
                with _CLIENT_LOCK:
                    return _make(self)
            cls._get_client = locked

    def __init__(self, model: str, *, api_key_env: str | None = None,
                 price: ModelPrice | dict | None = None, default_params: dict | None = None):
        self.model = model
        self.api_key_env = api_key_env
        self._key: ApiKey | None = None
        self.price = get_price(model, price) if price is not None else safe_price(model)
        self.default_params = dict(default_params or {})
        self.rpm: float | None = None  # client-side pacing, set by the runner

    # -- secrets
    def _api_key(self) -> ApiKey:
        if self._key is None:
            if not self.api_key_env:
                raise ProviderError("no api_key_env configured")
            self._key = ApiKey.from_env(self.api_key_env)
        return self._key

    def _raise(self, exc: BaseException) -> None:
        extra = [self._key.reveal()] if self._key is not None and self._key.present else []
        status = getattr(exc, "status_code", None)
        raise ProviderError(redact(f"{self.provider_name}/{self.model}: {type(exc).__name__}: {exc}", extra),
                            billable=status not in _UNBILLED_STATUS, status=status) from None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, api_key_env={self.api_key_env!r})"

    # -- API
    def effective_params(self, params: dict | None) -> dict:
        p = dict(self.default_params)
        p.update(params or {})
        return {k: v for k, v in p.items() if v is not None}

    # Bump when a provider changes how it shapes the wire request (tool strictness, reasoning
    # params, message conversion): the cache key must change with it.
    wire_version: str = "1"

    def request_hash(self, messages: list[dict], tools: list[dict], system: str | None,
                     params: dict | None) -> str:
        p = self.effective_params(params)
        if self.wire_version != "1":
            p = {**p, "__wire": self.wire_version}
        return request_hash(self.provider_name, self.model, p, messages, tools, system)

    def estimate_max_cost(self, messages: list[dict], tools: list[dict], system: str | None,
                          params: dict | None) -> float:
        """Conservative projection: ~2.5 chars/token for input, full max_tokens of output,
        no cache discount."""
        p = self.effective_params(params)
        chars = len(json.dumps(messages)) + len(json.dumps(tools)) + len(system or "")
        in_tok = math.ceil(chars / 2.5) + 50 * (len(messages) + len(tools))
        out_tok = int(p.get("max_tokens", DEFAULT_MAX_TOKENS))
        return self.price.cost(in_tok, out_tok)

    def cost(self, usage: Usage) -> float:
        return self.price.cost(usage.input_tokens - usage.cache_write_tokens, usage.output_tokens,
                               usage.cached_input_tokens, usage.cache_write_tokens)

    @abc.abstractmethod
    def complete(self, messages: list[dict], tools: list[dict], system: str | None = None,
                 params: dict | None = None) -> ProviderResponse: ...

    @property
    def is_paid(self) -> bool:
        # An explicitly zero-priced model (e.g. an OpenRouter ":free" endpoint) is not paid.
        p = getattr(self, "price", None)
        return not (p is not None and p.input_per_m == 0 and p.output_per_m == 0 and p.verified)


def parse_json_args(s: str | dict | None) -> dict:
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    try:
        v = json.loads(s)
    except ValueError:
        return {"__unparsed__": s}
    return v if isinstance(v, dict) else {"__unparsed__": s}
