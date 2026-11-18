"""Providers, offline: OpenRouter usage accounting and BYOK cost, routing pins, reasoning pass-through and
echo, thinking caps, HTTP-200 error payloads, error classification and retry hints, key redaction."""

from __future__ import annotations

import copy

import pytest

from urn_exchange.providers import ProviderError, ProviderResponse, classify_error, make_provider, retry_after_seconds
from urn_exchange.providers.openai_provider import (EMPTY_ASSISTANT_PLACEHOLDER, OpenAICompatibleProvider,
                                                    OpenAIProvider, to_openai_messages, to_responses_input)
from urn_exchange.runner import TRANSIENT_ERROR_CLASSES, build_providers
from urn_exchange.schedule import ModelConfig

MSGS = [{"role": "user", "content": "Hand 1/10"}]
TOOLS = [{"name": "act", "description": "d", "parameters": {"type": "object", "properties": {}}}]
MODEL = "anthropic/claude-sonnet-5.5"
GEMINI = "google/gemini-4-argon"
FAKE_KEY = "test-not-a-real-key-0123456789"
DETAILS = [
    {"type": "reasoning.text", "text": "pooled posterior first", "signature": None, "id": "r1",
     "format": "google-gemini-v1", "index": 0},
    {"type": "reasoning.encrypted", "data": "c2lnbmF0dXJlLWJsb2I=", "id": "tool_c1", "format": "google-gemini-v1",
     "index": 1},
]


def raw_ok(cost=0.0123, provider="Anthropic", tool=True, details=None, content=""):
    msg = {"role": "assistant", "content": content}
    if tool:
        msg["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "act", "arguments": "{}"}}]
    if details is not None:
        msg["reasoning_details"] = details
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 400}}
    if cost is not None:
        usage["cost"] = cost
    return {"model": MODEL, "provider": provider, "choices": [{"message": msg, "finish_reason": "tool_calls"}],
            "usage": usage}


def test_request_asks_for_usage_and_a_thinking_cap_replaces_the_effort():
    p = OpenAICompatibleProvider(MODEL, client=object())
    req = p.build_request(MSGS, TOOLS, "sys", {})
    assert req["extra_body"] == {"usage": {"include": True}} and req["tool_choice"] == "auto"
    req = p.build_request(MSGS, TOOLS, "sys", {"reasoning_effort": "high"})
    assert req["extra_body"]["reasoning"] == {"effort": "high"} and "reasoning_effort" not in req
    # Gemini in v1: reasoning.max_tokens 8000 with the shared effort high -> only the cap is sent
    req = p.build_request(MSGS, TOOLS, "sys", {"reasoning_effort": "high", "reasoning": {"max_tokens": 8000}})
    assert req["extra_body"]["reasoning"] == {"max_tokens": 8000} and "reasoning_effort" not in req
    other = OpenAICompatibleProvider(MODEL, base_url="https://example.invalid/v1", client=object())
    assert "extra_body" not in other.build_request(MSGS, TOOLS, "sys", {})


def test_routing_pin_and_cache_control_are_sent_and_change_the_cache_key():
    free = OpenAICompatibleProvider(MODEL, client=object())
    pinned = OpenAICompatibleProvider(MODEL, client=object(), routing={"order": ["anthropic"], "allow_fallbacks": False})
    assert pinned.build_request(MSGS, TOOLS, "sys", {})["extra_body"]["provider"] == {
        "order": ["anthropic"], "allow_fallbacks": False}
    assert free.request_hash(MSGS, TOOLS, "sys", {}) != pinned.request_hash(MSGS, TOOLS, "sys", {})
    req = free.build_request(MSGS, TOOLS, "sys", {"cache_control": {"type": "ephemeral"}})
    assert req["extra_body"]["cache_control"] == {"type": "ephemeral"}
    m = ModelConfig(key="k", provider="openai_compatible", model=MODEL, routing={"order": ["anthropic"]})
    assert build_providers([m])["k"].routing == {"order": ["anthropic"]}
    with pytest.raises(ValueError, match="routing"):
        build_providers([m.model_copy(update={"provider": "gemini_api", "key": "g"})])


def test_billed_cost_and_byok_true_cost():
    p = OpenAICompatibleProvider(MODEL, client=object())
    r = p.parse_response(raw_ok(cost=0.0123), "h", 0.2)
    assert r.billed_cost_usd == pytest.approx(0.0123) and r.cost_usd == pytest.approx(0.0123)
    # price table (Sonnet 5.5: $2 in, $0.10 cached, $10 out): 600 x 2 + 400 x 0.1 + 100 x 10
    assert r.price_cost_usd == pytest.approx((600 * 2 + 400 * 0.1 + 100 * 10) / 1e6)
    assert r.upstream_provider == "Anthropic"
    back = ProviderResponse.from_dict(r.to_dict())
    assert (back.billed_cost_usd, back.price_cost_usd, back.upstream_provider) == (0.0123, r.price_cost_usd, "Anthropic")
    # BYOK: OpenRouter bills only its fee; the true cost adds the upstream inference cost
    raw = raw_ok(cost=0.0001)
    raw["usage"].update(is_byok=True, cost_details={"upstream_inference_cost": 0.003})
    r = p.parse_response(raw, "h", 0.1)
    assert r.cost_usd == pytest.approx(0.0031) and r.billed_cost_usd == 0.0001


@pytest.mark.parametrize("payload,cls,billable,status", [
    ({"error": {"code": 502, "message": "upstream down", "metadata": {"provider_name": "X"}}}, "gateway_error", True, 502),
    ({"error": {"code": 402, "message": "Insufficient credits"}}, "insufficient_credits", False, 402),
    ({"error": {"code": 429, "message": "rate limited"}}, "rate_limit", False, 429),
    ({"choices": [{"finish_reason": "error", "error": {"code": 500, "message": "mid-stream"}}]}, "gateway_error", True, 500),
])
def test_http_200_error_payload_raises(payload, cls, billable, status):
    p = OpenAICompatibleProvider(MODEL, client=object())
    with pytest.raises(ProviderError) as ei:
        p.parse_response({"model": MODEL, **payload}, "h", 0.1)
    assert (ei.value.error_class, ei.value.billable, ei.value.status) == (cls, billable, status)


def test_reasoning_details_captured_and_echoed_verbatim():
    p = OpenAICompatibleProvider(GEMINI, client=object())
    r = p.parse_response(raw_ok(details=copy.deepcopy(DETAILS)), "h", 0.1)
    assert r.provider_state == {"openrouter_reasoning_details": DETAILS} and "signature" not in r.text
    msgs = MSGS + [r.assistant_message(), {"role": "tool", "tool_call_id": "c1", "name": "act", "content": "1"}]
    sent = p.build_request(msgs, TOOLS, "sys", {})["messages"]
    assert sent[2]["role"] == "assistant" and sent[2]["reasoning_details"] == DETAILS
    other = copy.deepcopy(msgs)
    other[1]["provider_state"]["openrouter_reasoning_details"][1]["data"] = "b3RoZXI="
    assert p.request_hash(msgs, TOOLS, "sys", {}) != p.request_hash(other, TOOLS, "sys", {})
    # other providers ignore the OpenRouter key
    chat = OpenAIProvider("gpt-x", client=object()).build_request(msgs, TOOLS, "sys", {})["messages"]
    assert "reasoning_details" not in chat[2]
    assert [i.get("type", i.get("role")) for i in to_responses_input(msgs)] == ["user", "function_call",
                                                                                "function_call_output"]


def test_reasoning_only_assistant_turn_is_never_sent_empty():
    out = to_openai_messages([{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
                              {"role": "user", "content": "continue"}], None)
    assert out[1]["content"] == EMPTY_ASSISTANT_PLACEHOLDER


def test_gemini_direct_thinking_budget_and_key_redaction(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    p = make_provider("gemini_api", "gemini-4-argon", client=object())
    assert p.api_key_env == "GEMINI_API_KEY" and FAKE_KEY not in repr(p)
    req = p.build_request(MSGS, TOOLS, "sys", {})
    assert req["config"]["system_instruction"] == "sys" and "thinking_config" not in req["config"]
    req = p.build_request(MSGS, TOOLS, "sys", {"reasoning_effort": "high", "reasoning": {"max_tokens": 8000}})
    assert req["config"]["thinking_config"] == {"thinking_budget": 8000}
    p._api_key()
    with pytest.raises(ProviderError) as ei:
        p._raise(RuntimeError(f"bad key {FAKE_KEY}"))
    assert FAKE_KEY not in str(ei.value)
    raw = {"candidates": [{"content": {"role": "model", "parts": [
        {"text": "thinking", "thought": True}, {"function_call": {"name": "act", "args": {"side": "pass"}},
                                                "thought_signature": "c2ln"}]}, "finish_reason": "STOP"}],
           "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 10, "thoughts_token_count": 50}}
    r = p.parse_response(raw, "h", 0.1)
    assert r.text == "" and r.tool_calls[0].arguments == {"side": "pass"} and r.usage.output_tokens == 60
    assert r.provider_state["gemini_content"]["parts"][1]["thought_signature"] == "c2ln"  # echoed back verbatim


def test_anthropic_direct_request_shape():
    p = make_provider("anthropic", "claude-x", client=object())
    req = p.build_request(MSGS, TOOLS, "sys", {"reasoning_effort": "high", "max_tokens": 16000, "temperature": 1})
    assert req["thinking"] == {"type": "adaptive"} and req["extra_body"]["output_config"] == {"effort": "high"}
    assert "temperature" not in req and req["tool_choice"] == {"type": "auto"}


def test_dropped_connection_is_transient():
    msg = "gemini_api/gemini-4-carbon: RemoteProtocolError: Server disconnected without sending a response."
    assert classify_error(msg) == "connection" and "connection" in TRANSIENT_ERROR_CLASSES


def test_gemini_spend_based_limit_is_a_rate_limit_with_a_long_wait():
    msg = ("gemini_api/gemini-4-carbon: ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
           "\"You exceeded your spend-based rate limit. Your spending rate has exceeded the allowed limit\"}}")
    assert classify_error(msg) == "rate_limit" and retry_after_seconds(msg) == 90.0
    assert classify_error("429 RESOURCE_EXHAUSTED exceeded your current quota") == "quota"  # true quota unchanged


@pytest.mark.parametrize("msg,status,cls", [
    ("Error code: 402 - Insufficient credits", 402, "insufficient_credits"),
    ("would exceed your available credits given your current in-flight requests", 402, "rate_limit"),
    ("AuthenticationError: invalid key", 401, "auth"),
    ("403 {'type': 'forbidden', 'message': 'Request not allowed'}", 403, "upstream_forbidden"),
    ("Please retry in 46.4s. GenerateRequestsPerMinutePerProjectPerModel RESOURCE_EXHAUSTED 429", None, "rate_limit"),
    ("400 User location is not supported for the API use.", 400, "gateway_error"),
    ("Error code: 503 - overloaded", 503, "server_error"),
    ("max_tokens 4000 exceeded", None, "other:max_tokens 4000 exceeded"),
])
def test_error_classes(msg, status, cls):
    assert classify_error(msg, status) == cls


def test_rejected_request_is_not_billable():
    class E(Exception):
        status_code = 400

    class E5(Exception):
        status_code = 500

    p = OpenAIProvider("gpt-x", client=object())
    for exc, billable in ((E("bad"), False), (E5("boom"), True), (RuntimeError("net"), True)):
        with pytest.raises(ProviderError) as ei:
            p._raise(exc)
        assert ei.value.billable is billable


def test_missing_key_is_reported_by_name_not_value():
    p = make_provider("openai_compatible", MODEL)
    with pytest.raises(Exception) as ei:
        p._get_client()
    assert "OPENROUTER_API_KEY" in str(ei.value)
