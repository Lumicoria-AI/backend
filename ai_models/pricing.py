"""
Per-model token pricing — single source of truth for `cost_usd` on
LLMResponse + the agent_runs collection.

Prices are USD per 1,000 tokens (the industry-standard unit). Each
provider publishes their own table; we mirror the canonical SKUs and
their aliases.  When a model name doesn't match any entry, we fall back
to a conservative "unknown" rate so the cost column is never zero just
because a new model dropped.

Update cadence: whenever a provider changes a price.  Keep this file
boring — no business logic, just constants + one lookup function.
"""

from __future__ import annotations

from typing import Dict, Tuple


# ── Raw price table (USD per 1K tokens) ─────────────────────────────────
#
# Keys are normalised to lower-case, prefix-matched.  The first prefix
# that fits wins, so order them general → specific where it matters.

PRICES_PER_1K: Dict[str, Tuple[float, float]] = {
    # ── OpenAI ───────────────────────────────────────────────────────────
    # GPT-4.1 family
    "gpt-4.1":           (0.0030, 0.0120),
    "gpt-4.1-mini":      (0.0008, 0.0032),
    "gpt-4.1-nano":      (0.0002, 0.0008),
    # GPT-4o family
    "gpt-4o-mini":       (0.0015, 0.0060),
    "gpt-4o":            (0.0050, 0.0150),
    # GPT-4 turbo / legacy
    "gpt-4-turbo":       (0.0100, 0.0300),
    "gpt-4":             (0.0300, 0.0600),
    # GPT-3.5
    "gpt-3.5-turbo":     (0.0005, 0.0015),
    # o-series (reasoning)
    "o1-preview":        (0.0150, 0.0600),
    "o1-mini":           (0.0030, 0.0120),
    "o1":                (0.0150, 0.0600),
    "o3-mini":           (0.0011, 0.0044),
    "o3":                (0.0020, 0.0080),

    # ── Anthropic (Claude) ───────────────────────────────────────────────
    # Claude 4 / 4.x
    "claude-opus-4-8":   (0.0150, 0.0750),
    "claude-opus-4-7":   (0.0150, 0.0750),
    "claude-opus-4-6":   (0.0150, 0.0750),
    "claude-opus-4":     (0.0150, 0.0750),
    "claude-opus":       (0.0150, 0.0750),
    "claude-sonnet-4-6": (0.0030, 0.0150),
    "claude-sonnet-4":   (0.0030, 0.0150),
    "claude-sonnet":     (0.0030, 0.0150),
    "claude-haiku-4-5":  (0.0010, 0.0050),
    "claude-haiku":      (0.0010, 0.0050),
    # Claude 3.x legacy
    "claude-3-5-sonnet": (0.0030, 0.0150),
    "claude-3-5-haiku":  (0.0008, 0.0040),
    "claude-3-opus":     (0.0150, 0.0750),
    "claude-3-sonnet":   (0.0030, 0.0150),
    "claude-3-haiku":    (0.00025, 0.00125),

    # ── Google Gemini ────────────────────────────────────────────────────
    "gemini-2.5-pro":    (0.0125, 0.0500),
    "gemini-2.5-flash":  (0.0003, 0.0025),
    "gemini-2.0-flash":  (0.0001, 0.0004),
    "gemini-1.5-pro":    (0.0035, 0.0105),
    "gemini-1.5-flash":  (0.000075, 0.0003),
    "gemini-pro":        (0.0035, 0.0105),
    "gemini-flash":      (0.000075, 0.0003),
    "gemini":            (0.0035, 0.0105),

    # ── Perplexity Sonar ────────────────────────────────────────────────
    "sonar-large-online":    (0.0010, 0.0010),
    "sonar-medium-online":   (0.0006, 0.0006),
    "sonar-small-online":    (0.0002, 0.0002),
    "sonar-large":           (0.0010, 0.0010),
    "sonar-medium":          (0.0006, 0.0006),
    "sonar-small":           (0.0002, 0.0002),
    "sonar":                 (0.0010, 0.0010),
    "perplexity":            (0.0010, 0.0010),

    # ── Mistral ─────────────────────────────────────────────────────────
    "mistral-large":     (0.0020, 0.0060),
    "mistral-medium":    (0.0027, 0.0081),
    "mistral-small":     (0.0002, 0.0006),
    "codestral":         (0.0002, 0.0006),
    "pixtral-large":     (0.0020, 0.0060),
    "pixtral":           (0.0020, 0.0060),
    "mistral":           (0.0020, 0.0060),

    # ── Catch-all for self-hosted / unknown ─────────────────────────────
    "local":             (0.0, 0.0),
    "default":           (0.0010, 0.0030),
}


def get_price(model: str | None) -> Tuple[float, float]:
    """Return (input_per_1k, output_per_1k) for the given model name.

    Lookup is prefix-based and case-insensitive — `gpt-4o-2024-08-06`
    matches `gpt-4o`, `claude-sonnet-4-6-20251022` matches
    `claude-sonnet-4-6`, etc.  Falls back to the `default` row when
    nothing matches so a fresh model still produces a non-zero cost.
    """
    if not model:
        return PRICES_PER_1K["default"]
    name = model.strip().lower()
    # Try longest-prefix first so "claude-sonnet-4-6" beats "claude-sonnet"
    for key in sorted(PRICES_PER_1K.keys(), key=len, reverse=True):
        if key == "default" or key == "local":
            continue
        if name.startswith(key):
            return PRICES_PER_1K[key]
    if "local" in name or "embedding" in name:
        return PRICES_PER_1K["local"]
    return PRICES_PER_1K["default"]


def compute_cost(
    *,
    model: str | None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> float:
    """Compute USD cost for one LLM call.

    Returns `0.0` for self-hosted / unknown-but-priced-as-local models
    (so locally-served runs don't inflate the spend dashboard).
    """
    if not prompt_tokens and not completion_tokens:
        return 0.0
    in_rate, out_rate = get_price(model)
    cost = (
        (prompt_tokens / 1000.0) * in_rate
        + (completion_tokens / 1000.0) * out_rate
    )
    # Round to 6 decimal places — sub-cent precision, no float-noise tails.
    return round(cost, 6)
