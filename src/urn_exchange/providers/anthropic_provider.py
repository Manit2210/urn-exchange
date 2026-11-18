"""Anthropic Messages API provider (direct route, ``ANTHROPIC_API_KEY``). Optional: v1 ran Claude through
OpenRouter. Install with ``pip install -e ".[anthropic]"``.

Request shape for current reasoning models: no ``temperature``/``top_p``/``top_k``, adaptive thinking
(``thinking={"type": "adaptive"}``) with ``output_config={"effort": ...}`` taken from ``effort`` or
``reasoning_effort``; ``tool_choice`` auto. Prompt caching via top-level ``cache_control``. Thinking
blocks are echoed back unmodified through ``provider_state``.
"""

from __future__ import annotations

import time
from typing import Any

from urn_exchange.providers.base import (
    DEFAULT_MAX_TOKENS,
    Provider,
    ProviderResponse,
    ToolCallOut,
    Usage,
)

_SAMPLING = ("temperature", "top_p", "top_k")


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
        elif role == "assistant":
            state = (m.get("provider_state") or {}).get("anthropic_content")
            if state:
                blocks = state
            else:
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            raise ValueError(f"unknown role {role!r}")
    return out


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]} for t in tools]


class AnthropicProvider(Provider):
    provider_name = "anthropic"

    def __init__(self, model: str, *, api_key_env: str = "ANTHROPIC_API_KEY", client: Any = None, **kw):
        kw.pop("base_url", None)
        kw.pop("routing", None)
        super().__init__(model, api_key_env=api_key_env, **kw)
        self._client = client

    def effective_params(self, params):
        p = super().effective_params(params)
        p.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        for k in _SAMPLING + ("budget_tokens",):
            p.pop(k, None)
        effort = p.pop("reasoning_effort", None)
        p.setdefault("effort", effort or "high")
        p.pop("reasoning", None)
        p.pop("cache_control", None)  # always on for this route
        return p

    def build_request(self, messages, tools, system, params) -> dict:
        p = self.effective_params(params)
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(p["max_tokens"]),
            "messages": to_anthropic_messages(messages),
            "thinking": {"type": "adaptive"},
            "extra_body": {"cache_control": {"type": "ephemeral"}, "output_config": {"effort": p["effort"]}},
        }
        if system:
            req["system"] = system
        if tools:
            req["tools"] = to_anthropic_tools(tools)
            req["tool_choice"] = {"type": "auto"}
        return req

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy, optional extra

            self._client = anthropic.Anthropic(api_key=self._api_key().reveal(), max_retries=3)
        return self._client

    def complete(self, messages, tools, system=None, params=None) -> ProviderResponse:
        rh = self.request_hash(messages, tools, system, params)
        req = self.build_request(messages, tools, system, params)
        t0 = time.monotonic()
        try:
            client = self._get_client()
            with client.messages.stream(**req) as stream:
                msg = stream.get_final_message()
            raw = msg.model_dump(mode="json", exclude_none=True)
        except Exception as e:  # noqa: BLE001
            self._raise(e)
        return self.parse_response(raw, rh, time.monotonic() - t0)

    def parse_response(self, raw: dict, rh: str, latency: float) -> ProviderResponse:
        text_parts, calls = [], []
        for b in raw.get("content", []):
            if b.get("type") == "text":
                text_parts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                calls.append(ToolCallOut(b["id"], b["name"], dict(b.get("input") or {})))
        u = raw.get("usage", {}) or {}
        cw = int(u.get("cache_creation_input_tokens") or 0)
        usage = Usage(
            input_tokens=int(u.get("input_tokens") or 0) + cw,
            output_tokens=int(u.get("output_tokens") or 0),
            cached_input_tokens=int(u.get("cache_read_input_tokens") or 0),
            cache_write_tokens=cw,
        )
        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=usage,
            cost_usd=self.cost(usage),
            latency_s=latency,
            raw_request_hash=rh,
            model=raw.get("model", self.model),
            stop_reason=str(raw.get("stop_reason") or ""),
            raw=raw,
            provider_state={"anthropic_content": raw.get("content", [])},
        )
