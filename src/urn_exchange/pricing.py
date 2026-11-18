"""Per-model prices in USD per million tokens (the six models of the v1 run).

Prices drive the dry-run estimate and the spend guard's worst-case reservation only. On OpenRouter the
runner books the cost OpenRouter reports for each call (``usage.cost``, plus the upstream cost under BYOK),
so a stale price here never changes what you are actually billed; it only changes the estimate.

Checked on 18 Nov 2026 against OpenRouter's endpoint listing for the pinned upstream host (standard tier,
prompts below the long-context threshold). Prices change: check them before a paid run, and use
``price_override`` in the config for a model that is missing or has moved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

VERIFY = "VERIFY_BEFORE_RUN"
VERIFIED = "verified"


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float
    cache_write_per_m: float | None = None  # defaults to the input price when None
    status: str = VERIFIED

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
             cache_write_tokens: int = 0) -> float:
        cw = self.cache_write_per_m if self.cache_write_per_m is not None else self.input_per_m
        return (
            input_tokens * self.input_per_m
            + output_tokens * self.output_per_m
            + cached_input_tokens * self.cached_input_per_m
            + cache_write_tokens * cw
        ) / 1_000_000


def _p(inp: float, out: float, cached: float, cache_write: float | None = None) -> ModelPrice:
    return ModelPrice(inp, out, cached, cache_write)


PRICES: dict[str, ModelPrice] = {
    # OpenAI host on OpenRouter.
    "openai/gpt-6.2-astra": _p(8.0, 40.0, 0.40),
    "openai/gpt-6.2-sol": _p(2.0, 10.0, 0.10),
    # Anthropic host on OpenRouter; cache writes (5-minute TTL) bill at 1.25x input.
    "anthropic/claude-fable-5.2": _p(8.0, 40.0, 0.20, 10.0),
    "anthropic/claude-sonnet-5.5": _p(2.0, 10.0, 0.10, 2.5),
    # Google AI Studio standard tier on OpenRouter (thinking tokens bill as output).
    "google/gemini-4-argon": _p(2.0, 10.0, 0.10),
    "google/gemini-4-carbon": _p(2.0, 10.0, 0.10, 0.1875),
    # The same Gemini models called directly on the Gemini API (provider ``gemini_api``), standard tier.
    "gemini-4-argon": _p(2.0, 10.0, 0.10),
    "gemini-4-carbon": _p(2.0, 10.0, 0.10),
    # Offline tests and dry runs.
    "fake-model": _p(0.0, 0.0, 0.0),
}


class UnverifiedPriceError(RuntimeError):
    pass


class UnknownPriceError(KeyError):
    pass


def get_price(model: str, override: dict | ModelPrice | None = None) -> ModelPrice:
    """Look up a price; an override (dict with input_per_m/output_per_m[/cached_input_per_m]) replaces
    the table entry and counts as verified."""
    if isinstance(override, ModelPrice):
        return replace(override, status=VERIFIED)
    if override:
        inp = float(override["input_per_m"])
        return ModelPrice(
            inp,
            float(override["output_per_m"]),
            float(override.get("cached_input_per_m", inp * 0.1)),
            override.get("cache_write_per_m"),
            VERIFIED,
        )
    if model in PRICES:
        return PRICES[model]
    raise UnknownPriceError(f"no price for model {model!r}; add a price_override in the config")


def safe_price(model: str, override: dict | None = None) -> ModelPrice:
    """A price for estimates: unknown models are priced high so the spend guard stays conservative."""
    try:
        return get_price(model, override)
    except KeyError:
        return ModelPrice(30.0, 150.0, 30.0, status=VERIFY)


def require_paid_price(model: str, override: dict | None) -> ModelPrice:
    """A paid run refuses a model with no known price (set ``price_override`` in the config)."""
    price = get_price(model, override)
    if not price.verified:
        raise UnverifiedPriceError(f"price for {model!r} is {price.status}; set price_override in the config")
    return price
