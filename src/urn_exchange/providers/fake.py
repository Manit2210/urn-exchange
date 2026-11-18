"""FakeProvider: deterministic scripted responses for tests and dry runs. No network, no keys.

Stateless with respect to the conversation: the turn index is the number of assistant messages already
in ``messages``, so one instance is safe across threads and matches.
"""

from __future__ import annotations

import threading
from typing import Callable

from urn_exchange.pricing import ModelPrice
from urn_exchange.providers.base import Provider, ProviderError, ProviderResponse, ToolCallOut, Usage

# Scripted provider failures for offline tests: class -> (message, status, billable).
FAULTS: dict[str, tuple[str, int | None, bool]] = {
    "server_error": ("APIStatusError: Error code: 500 - upstream failure", 500, True),
    "quota": ("RateLimitError: Error code: 429 - rate limited", 429, False),
    "insufficient_credits": ("APIStatusError: Error code: 402 - Insufficient credits", 402, False),
    "bad_request": ("BadRequestError: Error code: 400 - invalid request", 400, False),
    "auth": ("AuthenticationError: Error code: 401 - invalid key", 401, False),
    "timeout": ("APITimeoutError: Request timed out.", None, True),
    "gateway_error": ("gateway_error: upstream provider error (HTTP 200 error payload)", None, True),
}

Policy = Callable[[list[dict], list[dict]], dict]


def idle_policy(messages: list[dict], tools: list[dict]) -> dict:
    return {"text": ""}


class FakeProvider(Provider):
    provider_name = "fake"

    def __init__(self, model: str = "fake-model", *, policy: Policy | None = None,
                 script: list[dict] | None = None, input_tokens: int = 1000,
                 output_tokens: int = 200, price: ModelPrice | dict | None = None,
                 default_params: dict | None = None, paid: bool = False, fault: str | None = None,
                 fault_every: int = 1):
        super().__init__(model, price=price, default_params=default_params)
        if fault is not None and fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; one of {sorted(FAULTS)}")
        if (policy is None) == (script is None):
            raise ValueError("pass exactly one of policy or script")
        self.fault = fault
        self.fault_every = max(1, int(fault_every))
        self._lock = threading.Lock()
        self.policy = policy
        self.script = script
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0  # informational only
        self.paid = paid  # True makes the spend guard treat it like a real paid model (tests)

    @property
    def is_paid(self) -> bool:
        return self.paid

    def complete(self, messages, tools, system=None, params=None) -> ProviderResponse:
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.fault is not None and n % self.fault_every == 0:
            msg, status, billable = FAULTS[self.fault]
            raise ProviderError(f"fake/{self.model}: {msg}", billable=billable, status=status)
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        if self.policy is not None:
            out = self.policy(messages, tools)
        else:
            assert self.script is not None
            out = self.script[min(turn, len(self.script) - 1)]
        calls = [
            ToolCallOut(tc.get("id") or f"call_{turn}_{i}", tc["name"], dict(tc.get("arguments", {})))
            for i, tc in enumerate(out.get("tool_calls", []))
        ]
        usage = Usage(self.input_tokens, self.output_tokens, 0, 0)
        stop = out.get("stop_reason") or ("tool_use" if calls else "end_turn")
        return ProviderResponse(
            text=out.get("text", ""),
            tool_calls=calls,
            usage=usage,
            cost_usd=self.cost(usage),
            latency_s=0.0,
            raw_request_hash=self.request_hash(messages, tools, system, params),
            model=self.model,
            stop_reason=stop,
            raw={"fake": True, "turn": turn},
        )
