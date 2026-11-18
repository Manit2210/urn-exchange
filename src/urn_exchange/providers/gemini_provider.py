"""Gemini Developer API (Google AI Studio key, ``GEMINI_API_KEY``) via google-genai.

Optional route: v1 ran Gemini through OpenRouter by default; part of the v1 Gemini calls went direct
through this provider with the same thinking cap. Install with ``pip install -e ".[gemini]"``.

Model content (including thought signatures) is echoed back verbatim through ``provider_state``. The
thinking cap is ``thinking_budget`` (or the OpenRouter-style ``reasoning: {max_tokens: N}``, so a config
can move a model between routes unchanged). Without a cap Gemini can think through the whole reply limit
and never call the tool. ``reasoning_effort`` is not sent on this route.
"""

from __future__ import annotations

import time
from typing import Any

from urn_exchange.providers.base import (
    DEFAULT_MAX_TOKENS,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCallOut,
    Usage,
)


def to_gemini_contents(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif role == "assistant":
            state = (m.get("provider_state") or {}).get("gemini_content")
            if state:
                out.append(state)
                continue
            parts: list[dict] = []
            if m.get("content"):
                parts.append({"text": m["content"]})
            for tc in m.get("tool_calls", []):
                parts.append({"function_call": {"name": tc["name"], "args": tc["arguments"]}})
            out.append({"role": "model", "parts": parts})
        elif role == "tool":
            part = {"function_response": {"name": m["name"], "response": {"result": m["content"]}}}
            if out and out[-1]["role"] == "user" and all("function_response" in p for p in out[-1]["parts"]):
                out[-1]["parts"].append(part)
            else:
                out.append({"role": "user", "parts": [part]})
        else:
            raise ValueError(f"unknown role {role!r}")
    return out


class GeminiAPIProvider(Provider):
    provider_name = "gemini_api"

    def __init__(self, model: str, *, api_key_env: str = "GEMINI_API_KEY", client: Any = None, **kw):
        kw.pop("base_url", None)
        kw.pop("routing", None)
        super().__init__(model, api_key_env=api_key_env, **kw)
        self._client = client

    def effective_params(self, params):
        p = super().effective_params(params)
        p.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        return p

    def build_request(self, messages, tools, system, params) -> dict:
        p = self.effective_params(params)
        config: dict[str, Any] = {
            "max_output_tokens": int(p["max_tokens"]),
            "automatic_function_calling": {"disable": True},
        }
        if system:
            config["system_instruction"] = system
        if tools:
            config["tools"] = [{"function_declarations": [
                {"name": t["name"], "description": t.get("description", ""), "parameters_json_schema": t["parameters"]}
                for t in tools
            ]}]
        for k in ("temperature", "top_p", "seed"):
            if k in p:
                config[k] = p[k]
        if "thinking_level" in p:
            config["thinking_config"] = {"thinking_level": p["thinking_level"]}
        reasoning = p.get("reasoning") or {}
        if "thinking_budget" in p or "max_tokens" in reasoning:
            config["thinking_config"] = {"thinking_budget": int(p.get("thinking_budget", reasoning.get("max_tokens")))}
        return {"model": self.model, "contents": to_gemini_contents(messages), "config": config}

    def _get_client(self):
        if self._client is None:
            from google import genai  # lazy, optional extra

            self._client = genai.Client(api_key=self._api_key().reveal())
        return self._client

    def complete(self, messages, tools, system=None, params=None) -> ProviderResponse:
        rh = self.request_hash(messages, tools, system, params)
        req = self.build_request(messages, tools, system, params)
        t0 = time.monotonic()
        try:
            resp = self._get_client().models.generate_content(**req)
            raw = resp.model_dump(mode="json", exclude_none=True)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            self._raise(e)
        return self.parse_response(raw, rh, time.monotonic() - t0)

    def parse_response(self, raw: dict, rh: str, latency: float) -> ProviderResponse:
        cands = raw.get("candidates") or [{}]
        content = cands[0].get("content") or {"role": "model", "parts": []}
        text_parts, calls = [], []
        for i, part in enumerate(content.get("parts") or []):
            if part.get("text") and not part.get("thought"):
                text_parts.append(part["text"])
            fc = part.get("function_call")
            if fc:
                calls.append(ToolCallOut(fc.get("id") or f"call_{i}", fc["name"], dict(fc.get("args") or {})))
        u = raw.get("usage_metadata") or {}
        cached = int(u.get("cached_content_token_count") or 0)
        usage = Usage(
            input_tokens=int(u.get("prompt_token_count") or 0) - cached,
            output_tokens=int(u.get("candidates_token_count") or 0) + int(u.get("thoughts_token_count") or 0),
            cached_input_tokens=cached,
        )
        finish = str(cands[0].get("finish_reason") or "")
        if finish.upper() in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
            finish = "refusal"
        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=usage,
            cost_usd=self.cost(usage),
            latency_s=latency,
            raw_request_hash=rh,
            model=self.model,
            stop_reason=finish,
            raw=raw,
            provider_state={"gemini_content": content},
        )
