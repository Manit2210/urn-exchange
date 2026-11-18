"""OpenAI Chat Completions provider with function tools, plus an OpenAI-compatible variant
(base_url + api key env var name) for OpenRouter and similar gateways."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from urn_exchange.providers.base import (
    _UNBILLED_STATUS,
    DEFAULT_MAX_TOKENS,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCallOut,
    Usage,
    parse_json_args,
)

_PASSTHROUGH = ("temperature", "top_p", "seed", "reasoning_effort", "parallel_tool_calls")


EMPTY_ASSISTANT_PLACEHOLDER = "(no text in this turn)"

# provider_state keys (see providers/base.py). Each is echoed back only by the provider that wrote
# it; every other provider ignores it, so a canonical history stays portable.
OPENROUTER_REASONING_KEY = "openrouter_reasoning_details"
OPENAI_REASONING_KEY = "openai_reasoning_items"


def to_openai_messages(messages: list[dict], system: str | None, *,
                       reasoning_details: bool = False) -> list[dict]:
    """Canonical messages -> Chat Completions messages. With ``reasoning_details`` (OpenRouter),
    an assistant turn re-emits the ``reasoning_details`` it was returned with, unmodified: OpenRouter
    requires them back on the assistant message for multi-turn tool use (Gemini thought signatures,
    Anthropic signed thinking); without them Gemini rejects later turns with 400 INVALID_ARGUMENT."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            # Some hosts reject an assistant turn with neither text nor tool calls (a reasoning-only
            # turn), so such a turn carries a neutral placeholder.
            content = m.get("content") or (None if m.get("tool_calls") else EMPTY_ASSISTANT_PLACEHOLDER)
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], sort_keys=True)}}
                    for tc in m["tool_calls"]
                ]
            details = (m.get("provider_state") or {}).get(OPENROUTER_REASONING_KEY) if reasoning_details else None
            if details:
                msg["reasoning_details"] = details
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            raise ValueError(f"unknown role {role!r}")
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t["parameters"]}} for t in tools]


class OpenAIProvider(Provider):
    provider_name = "openai"
    base_url: str | None = None

    def __init__(self, model: str, *, api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None,
                 client: Any = None, **kw):
        super().__init__(model, api_key_env=api_key_env, **kw)
        if base_url:
            self.base_url = base_url
        self._client = client

    def effective_params(self, params):
        p = super().effective_params(params)
        p.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        return p

    def build_request(self, messages, tools, system, params) -> dict:
        p = self.effective_params(params)
        req: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages, system),
            "max_completion_tokens": int(p["max_tokens"]),
        }
        if tools:
            req["tools"] = to_openai_tools(tools)
            req["tool_choice"] = "auto"
        for k in _PASSTHROUGH:
            if k in p:
                req[k] = p[k]
        return req

    def _get_client(self):
        if self._client is None:
            import openai  # lazy

            # Bound each request (a read timeout of 900 s covers a 16k-token answer at ~20 tokens/s; requests
            # that never return otherwise hang a match) and retry twice; the runner's transient-error retry
            # takes over after that.
            kwargs: dict[str, Any] = {"api_key": self._api_key().reveal(), "max_retries": 2,
                                      "timeout": openai.Timeout(float(os.environ.get("URN_EXCHANGE_HTTP_TIMEOUT_S", 900)),
                                                                connect=30.0)}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(self, messages, tools, system=None, params=None) -> ProviderResponse:
        rh = self.request_hash(messages, tools, system, params)
        req = self.build_request(messages, tools, system, params)
        t0 = time.monotonic()
        try:
            resp = self._get_client().chat.completions.create(**req)
            raw = resp.model_dump(mode="json", exclude_none=True)
        except Exception as e:  # noqa: BLE001
            self._raise(e)
        return self.parse_response(raw, rh, time.monotonic() - t0)

    def parse_response(self, raw: dict, rh: str, latency: float) -> ProviderResponse:
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = [
            ToolCallOut(tc.get("id") or f"call_{i}", tc["function"]["name"], parse_json_args(tc["function"].get("arguments")))
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        u = raw.get("usage") or {}
        cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        usage = Usage(
            input_tokens=int(u.get("prompt_tokens") or 0) - cached,
            output_tokens=int(u.get("completion_tokens") or 0),
            cached_input_tokens=cached,
        )
        stop = str(choice.get("finish_reason") or "")
        if msg.get("refusal"):
            stop = "refusal"
        return ProviderResponse(
            text=msg.get("content") or msg.get("refusal") or "",
            tool_calls=calls,
            usage=usage,
            cost_usd=self.cost(usage),
            latency_s=latency,
            raw_request_hash=rh,
            model=raw.get("model", self.model),
            stop_reason=stop,
            raw=raw,
            provider_state=self._reasoning_state(msg),
        )

    def _reasoning_state(self, msg: dict) -> dict:
        return {}  # OpenAI Chat Completions returns no reasoning state to echo back


OPENROUTER_HOST = "openrouter.ai"


class OpenAICompatibleProvider(OpenAIProvider):
    """Any OpenAI-compatible endpoint. Defaults to OpenRouter.

    On OpenRouter the request asks for usage accounting (``usage: {include: true}``) and the
    billed ``usage.cost`` becomes ``cost_usd`` (the price-table cost is kept alongside); the
    upstream host (response ``provider``) is recorded; ``routing`` pins the upstream provider
    (OpenRouter's ``provider`` object) and ``reasoning_effort`` / ``reasoning`` are sent as the
    ``reasoning`` object, only when configured. OpenRouter can return HTTP 200 with an
    ``error`` payload; that raises a ``ProviderError`` like any HTTP error.
    """

    provider_name = "openai_compatible"
    # 2: OpenRouter usage accounting, reasoning object, routing. 3: empty-assistant placeholder.
    # 4: reasoning_details echoed back on assistant turns. Changing the wire request must change the
    # cache key.
    wire_version = "4-openrouter-reasoning-details"

    def __init__(self, model: str, *, base_url: str = "https://openrouter.ai/api/v1",
                 api_key_env: str = "OPENROUTER_API_KEY", routing: dict | None = None, **kw):
        super().__init__(model, api_key_env=api_key_env, base_url=base_url, **kw)
        self.routing = dict(routing) if routing else None

    @property
    def is_openrouter(self) -> bool:
        return OPENROUTER_HOST in (self.base_url or "")

    def request_hash(self, messages, tools, system, params):
        # Include the endpoint so the same model name on two gateways never collides, and the
        # upstream pin when set (a pinned host is a different experimental condition).
        extra: dict[str, Any] = {"__base_url": self.base_url}
        if self.routing:
            extra["__routing"] = self.routing
        return super().request_hash(messages, tools, system, {**(params or {}), **extra})

    def _reasoning_state(self, msg: dict) -> dict:
        # Captured verbatim (a list of reasoning.text / .summary / .encrypted blocks); the plain
        # ``reasoning`` string is derived from it and is not sent back.
        details = msg.get("reasoning_details")
        return {OPENROUTER_REASONING_KEY: details} if isinstance(details, list) and details else {}

    def build_request(self, messages, tools, system, params) -> dict:
        req = super().build_request(messages, tools, system, params)
        req["messages"] = to_openai_messages(messages, system, reasoning_details=True)
        if not self.is_openrouter:
            return req
        p = self.effective_params(params)
        body: dict[str, Any] = {"usage": {"include": True}}
        reasoning = dict(p.get("reasoning") or {})
        effort = req.pop("reasoning_effort", None)
        if effort is not None and "max_tokens" not in reasoning:
            # A per-model thinking cap (``reasoning.max_tokens``) wins over the global effort: OpenRouter takes one
            # of the two. Without a cap, Gemini at effort high can spend the whole 16,000-token reply limit
            # thinking and never call the tool.
            reasoning.setdefault("effort", effort)
        if reasoning:
            body["reasoning"] = reasoning
        if self.routing:
            body["provider"] = dict(self.routing)
        if p.get("cache_control"):
            # Anthropic caches only on request: OpenRouter's top-level ``cache_control`` turns on its
            # automatic breakpoint, advanced every turn (without it a Claude model reads 0 cached tokens
            # over a whole match). Configured per model.
            body["cache_control"] = dict(p["cache_control"])
        req["extra_body"] = body
        return req

    def parse_response(self, raw: dict, rh: str, latency: float) -> ProviderResponse:
        self._raise_payload_error(raw)
        resp = super().parse_response(raw, rh, latency)
        if not self.is_openrouter:
            return resp
        table = resp.cost_usd
        usage = raw.get("usage") or {}
        billed = usage.get("cost")
        resp.price_cost_usd = table
        if billed is not None:
            try:
                resp.billed_cost_usd = float(billed)
                resp.cost_usd = resp.billed_cost_usd
            except (TypeError, ValueError):
                pass
        if usage.get("is_byok"):
            # Bring-your-own-key: OpenRouter bills only its fee; the model cost lands on your own
            # provider key. The ledger counts the true spend: upstream cost plus the fee.
            upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost")
            true_cost = float(upstream) if upstream is not None else table
            resp.cost_usd = true_cost + (resp.billed_cost_usd or 0.0)
        up = raw.get("provider")
        resp.upstream_provider = str(up) if up else None
        return resp

    def _raise_payload_error(self, raw: dict) -> None:
        """OpenRouter reports some failures (upstream errors, mid-generation failures, credit
        exhaustion) as HTTP 200 with an ``error`` object, at the top level or in a choice."""
        err = raw.get("error")
        if not err:
            for ch in raw.get("choices") or []:
                if ch.get("error"):
                    err = ch["error"]
                    break
                if ch.get("finish_reason") == "error":
                    err = {"message": "choice finished with finish_reason=error"}
                    break
        if not err:
            return
        if not isinstance(err, dict):
            err = {"message": str(err)}
        code = err.get("code")
        try:
            status = int(code) if code is not None else None
        except (TypeError, ValueError):
            status = None
        meta = err.get("metadata") or {}
        upstream = meta.get("provider_name") or raw.get("provider")
        msg = str(err.get("message") or "")[:300]
        text = f"{self.provider_name}/{self.model}: gateway_error: code={code} upstream={upstream}: {msg}"
        extra = [self._key.reveal()] if self._key is not None and self._key.present else []
        from urn_exchange.util import redact

        cls = None if status in (401, 402, 403, 429) else "gateway_error"
        raise ProviderError(redact(text, extra), billable=status not in _UNBILLED_STATUS, status=status,
                            error_class=cls)


def to_responses_input(messages: list[dict]) -> list[dict]:
    """Canonical harness messages -> Responses API input items. With ``store: false`` nothing is
    kept server-side, so an assistant turn re-emits the reasoning items it was returned with
    (``encrypted_content`` included), untouched and ahead of its message and function calls, as
    OpenAI recommends for stateless multi-turn function calling."""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            out.extend((m.get("provider_state") or {}).get(OPENAI_REASONING_KEY) or [])
            if m.get("content"):
                out.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                out.append({"type": "function_call", "call_id": tc["id"], "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"], sort_keys=True)})
        elif role == "tool":
            out.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        else:
            raise ValueError(f"unknown role {role!r}")
    return out


class OpenAIResponsesProvider(OpenAIProvider):
    """OpenAI Responses API (direct OpenAI route). Current reasoning models reject function tools
    with reasoning on /v1/chat/completions, so the direct route uses Responses."""

    provider_name = "openai_responses"
    # 2: strict: false tools. 3: include encrypted reasoning and echo reasoning items back.
    wire_version = "3-encrypted-reasoning"

    def build_request(self, messages, tools, system, params) -> dict:
        p = self.effective_params(params)
        req: dict[str, Any] = {
            "model": self.model,
            "input": to_responses_input(messages),
            "max_output_tokens": int(p["max_tokens"]),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if system:
            req["instructions"] = system
        if tools:
            # strict defaults to true on the Responses API, which makes every property required and
            # forces models to fill optional fields with placeholders. Send the schema as authored,
            # like every other provider.
            req["tools"] = [{"type": "function", "name": t["name"], "description": t.get("description", ""),
                             "parameters": t["parameters"], "strict": False} for t in tools]
            req["tool_choice"] = "auto"
        if "reasoning_effort" in p:
            req["reasoning"] = {"effort": p["reasoning_effort"]}
        for k in ("temperature", "top_p", "parallel_tool_calls"):
            if k in p:
                req[k] = p[k]
        return req

    def complete(self, messages, tools, system=None, params=None) -> ProviderResponse:
        rh = self.request_hash(messages, tools, system, params)
        req = self.build_request(messages, tools, system, params)
        t0 = time.monotonic()
        try:
            resp = self._get_client().responses.create(**req)
            raw = resp.model_dump(mode="json", exclude_none=True)
        except Exception as e:  # noqa: BLE001
            self._raise(e)
        return self.parse_response(raw, rh, time.monotonic() - t0)

    def parse_response(self, raw: dict, rh: str, latency: float) -> ProviderResponse:
        texts: list[str] = []
        calls: list[ToolCallOut] = []
        reasoning: list[dict] = []
        refused = False
        for i, item in enumerate(raw.get("output") or []):
            kind = item.get("type")
            if kind == "reasoning":
                reasoning.append(item)
            elif kind == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
                    elif c.get("type") == "refusal":
                        refused = True
                        texts.append(c.get("refusal", ""))
            elif kind == "function_call":
                calls.append(ToolCallOut(item.get("call_id") or f"call_{i}", item["name"],
                                         parse_json_args(item.get("arguments"))))
        u = raw.get("usage") or {}
        cached = int(((u.get("input_tokens_details") or {}).get("cached_tokens")) or 0)
        usage = Usage(
            input_tokens=int(u.get("input_tokens") or 0) - cached,
            output_tokens=int(u.get("output_tokens") or 0),
            cached_input_tokens=cached,
        )
        if refused:
            stop = "refusal"
        elif calls:
            stop = "tool_calls"
        elif raw.get("status") == "incomplete":
            stop = str((raw.get("incomplete_details") or {}).get("reason") or "incomplete")
        else:
            stop = "stop"
        return ProviderResponse(
            text="".join(texts), tool_calls=calls, usage=usage, cost_usd=self.cost(usage),
            latency_s=latency, raw_request_hash=rh, model=raw.get("model", self.model),
            stop_reason=stop, raw=raw,
            provider_state={OPENAI_REASONING_KEY: reasoning} if reasoning else {},
        )
