"""
Three-tier model pricing + credits.

Resolution order for `(input_per_1k, output_per_1k)`:

  1. Env override         — settings.MODEL_PRICING_OVERRIDES_JSON
  2. Mongo override       — `model_pricing` collection (any doc keyed by model prefix)
  3. OpenRouter live cache — Redis-backed, 24 h TTL (refreshed at boot)
  4. Static fallback      — the hardcoded PRICES_PER_1K table below

This file is the single source of truth callers consult via
`compute_cost(model, prompt_tokens, completion_tokens)` and
`compute_credits(cost_usd)`.  Users never see USD; they see credits.
We always know the underlying USD for margin tracking.

Boot sequence: `warm_pricing_cache()` is awaited from FastAPI startup.
It loads the env override, snapshots the Mongo override doc set, and
fetches OpenRouter into Redis (if enabled).  Subsequent calls hit the
in-memory dicts; nothing reaches the network on the hot path.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


# ─── Static fallback table (USD per 1K tokens) ────────────────────────
#
# Keys are normalised to lower-case, prefix-matched.  Longest prefix
# wins (so "claude-sonnet-4-6" beats "claude-sonnet").  When nothing
# matches we fall back to `default`.

PRICES_PER_1K: Dict[str, Tuple[float, float]] = {
    # ── OpenAI ──────────────────────────────────────────────────────────
    "gpt-4.1":           (0.0030, 0.0120),
    "gpt-4.1-mini":      (0.0008, 0.0032),
    "gpt-4.1-nano":      (0.0002, 0.0008),
    "gpt-4o-mini":       (0.0015, 0.0060),
    "gpt-4o":            (0.0050, 0.0150),
    "gpt-4-turbo":       (0.0100, 0.0300),
    "gpt-4":             (0.0300, 0.0600),
    "gpt-3.5-turbo":     (0.0005, 0.0015),
    "o1-preview":        (0.0150, 0.0600),
    "o1-mini":           (0.0030, 0.0120),
    "o1":                (0.0150, 0.0600),
    "o3-mini":           (0.0011, 0.0044),
    "o3":                (0.0020, 0.0080),

    # ── Anthropic ───────────────────────────────────────────────────────
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
    "claude-3-5-sonnet": (0.0030, 0.0150),
    "claude-3-5-haiku":  (0.0008, 0.0040),
    "claude-3-opus":     (0.0150, 0.0750),
    "claude-3-sonnet":   (0.0030, 0.0150),
    "claude-3-haiku":    (0.00025, 0.00125),

    # ── Google Gemini ───────────────────────────────────────────────────
    "gemini-2.5-pro":    (0.0125, 0.0500),
    "gemini-2.5-flash":  (0.0003, 0.0025),
    "gemini-2.0-flash":  (0.0001, 0.0004),
    "gemini-1.5-pro":    (0.0035, 0.0105),
    "gemini-1.5-flash":  (0.000075, 0.0003),
    "gemini-pro":        (0.0035, 0.0105),
    "gemini-flash":      (0.000075, 0.0003),
    "gemini":            (0.0035, 0.0105),

    # ── Perplexity Sonar ───────────────────────────────────────────────
    "sonar-large-online":  (0.0010, 0.0010),
    "sonar-medium-online": (0.0006, 0.0006),
    "sonar-small-online":  (0.0002, 0.0002),
    "sonar-large":         (0.0010, 0.0010),
    "sonar-medium":        (0.0006, 0.0006),
    "sonar-small":         (0.0002, 0.0002),
    "sonar":               (0.0010, 0.0010),
    "perplexity":          (0.0010, 0.0010),

    # ── Mistral ────────────────────────────────────────────────────────
    "mistral-large":     (0.0020, 0.0060),
    "mistral-medium":    (0.0027, 0.0081),
    "mistral-small":     (0.0002, 0.0006),
    "codestral":         (0.0002, 0.0006),
    "pixtral-large":     (0.0020, 0.0060),
    "pixtral":           (0.0020, 0.0060),
    "mistral":           (0.0020, 0.0060),

    # ── Catch-all ──────────────────────────────────────────────────────
    "local":             (0.0, 0.0),
    "default":           (0.0010, 0.0030),
}


# ─── In-memory override caches (refreshed at boot + opportunistically) ─

_ENV_OVERRIDES: Dict[str, Tuple[float, float]] = {}
_MONGO_OVERRIDES: Dict[str, Tuple[float, float]] = {}
_OPENROUTER_OVERRIDES: Dict[str, Tuple[float, float]] = {}
_OVERRIDES_LOADED: bool = False


def _norm(model: str | None) -> str:
    return (model or "").strip().lower()


def _longest_prefix_match(name: str, table: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Find the longest key in `table` that is a prefix of `name`."""
    if not name:
        return None
    best: Optional[str] = None
    for key in table.keys():
        if key in ("default", "local"):
            continue
        if name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return table[best] if best else None


# ─── Override loaders ─────────────────────────────────────────────────


def _load_env_overrides() -> Dict[str, Tuple[float, float]]:
    """Parse settings.MODEL_PRICING_OVERRIDES_JSON.

    Expected shape:
        {"gpt-4o": {"input": 0.005, "output": 0.015}, ...}
    """
    from backend.core.config import settings

    raw = getattr(settings, "MODEL_PRICING_OVERRIDES_JSON", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:  # noqa: BLE001
        logger.warning("model_pricing_env_parse_failed", error=str(e))
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    if not isinstance(parsed, dict):
        return out
    for model, prices in parsed.items():
        try:
            inp = float(prices.get("input"))
            outp = float(prices.get("output"))
            out[_norm(model)] = (inp, outp)
        except Exception:  # noqa: BLE001
            continue
    logger.info("model_pricing_env_loaded", count=len(out))
    return out


async def _load_mongo_overrides() -> Dict[str, Tuple[float, float]]:
    """Snapshot the `model_pricing` Mongo collection if present.

    Documents are expected to look like:
        { "_id": "gpt-4o", "input_per_1k": 0.005, "output_per_1k": 0.015 }
    """
    try:
        from backend.db.mongodb.mongodb import MongoDB
        col = await MongoDB.get_collection("model_pricing")
        out: Dict[str, Tuple[float, float]] = {}
        async for doc in col.find({}):
            key = _norm(doc.get("_id") or doc.get("model"))
            inp = doc.get("input_per_1k") or doc.get("input")
            outp = doc.get("output_per_1k") or doc.get("output")
            if not key or inp is None or outp is None:
                continue
            try:
                out[key] = (float(inp), float(outp))
            except Exception:  # noqa: BLE001
                continue
        if out:
            logger.info("model_pricing_mongo_loaded", count=len(out))
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("model_pricing_mongo_load_skipped", error=str(e))
        return {}


# ─── OpenRouter live fetcher ──────────────────────────────────────────


_REDIS_KEY = "lumi:pricing:openrouter:v1"


async def _load_openrouter_cache() -> Dict[str, Tuple[float, float]]:
    """Read the OpenRouter map from Redis (if cached), otherwise fetch
    from the public endpoint and stash it for 24 h.

    Returns an empty dict on any error — callers fall back to the next
    pricing tier.
    """
    from backend.core.config import settings

    if not getattr(settings, "OPENROUTER_PRICING_ENABLED", True):
        return {}

    # Try Redis first.
    try:
        from backend.core.security import _get_redis
        redis = _get_redis()
        if redis:
            cached = redis.get(_REDIS_KEY)
            if cached:
                try:
                    raw_map = json.loads(cached)
                    return {
                        _norm(k): (float(v[0]), float(v[1]))
                        for k, v in raw_map.items()
                    }
                except Exception:  # noqa: BLE001
                    redis.delete(_REDIS_KEY)  # corrupt → refetch
    except Exception:  # noqa: BLE001
        pass

    # Fetch live.
    try:
        import httpx
    except Exception:
        logger.warning("openrouter_pricing_skipped_no_httpx")
        return {}

    url = getattr(
        settings,
        "OPENROUTER_PRICING_URL",
        "https://openrouter.ai/api/v1/models",
    )
    ttl_seconds = max(60, int(getattr(settings, "OPENROUTER_PRICING_TTL_HOURS", 24)) * 3600)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("openrouter_pricing_fetch_failed", error=str(e)[:200])
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    # OpenRouter shape: { "data": [{ "id": "openai/gpt-4o", "pricing": {"prompt": "0.0025", "completion": "0.01"}, ...}, ...] }
    # Their prices are USD per token, not per 1K.  Multiply by 1000.
    for entry in (data.get("data") or []):
        model_id = entry.get("id") or ""
        pricing = entry.get("pricing") or {}
        try:
            prompt_per_token = float(pricing.get("prompt") or 0)
            completion_per_token = float(pricing.get("completion") or 0)
        except Exception:  # noqa: BLE001
            continue
        if prompt_per_token == 0 and completion_per_token == 0:
            continue
        in_per_1k = prompt_per_token * 1000
        out_per_1k = completion_per_token * 1000
        # Record under both the full id ("openai/gpt-4o") and the bare
        # model slug ("gpt-4o") so prefix-matching catches either form.
        for k in (_norm(model_id), _norm(model_id.split("/", 1)[-1])):
            if k:
                out[k] = (in_per_1k, out_per_1k)

    # Cache.
    try:
        from backend.core.security import _get_redis
        redis = _get_redis()
        if redis and out:
            redis.setex(
                _REDIS_KEY,
                ttl_seconds,
                json.dumps({k: [v[0], v[1]] for k, v in out.items()}),
            )
    except Exception:  # noqa: BLE001
        pass

    logger.info("openrouter_pricing_loaded", count=len(out))
    return out


async def warm_pricing_cache() -> Dict[str, Any]:
    """Boot-time priming hook — call from FastAPI startup.

    Idempotent: safe to call multiple times.  Refreshes all three
    override tiers in parallel.
    """
    global _ENV_OVERRIDES, _MONGO_OVERRIDES, _OPENROUTER_OVERRIDES, _OVERRIDES_LOADED

    import asyncio

    _ENV_OVERRIDES = _load_env_overrides()
    mongo_task = asyncio.create_task(_load_mongo_overrides())
    openrouter_task = asyncio.create_task(_load_openrouter_cache())
    _MONGO_OVERRIDES = await mongo_task
    _OPENROUTER_OVERRIDES = await openrouter_task
    _OVERRIDES_LOADED = True
    return {
        "env": len(_ENV_OVERRIDES),
        "mongo": len(_MONGO_OVERRIDES),
        "openrouter": len(_OPENROUTER_OVERRIDES),
        "static": len(PRICES_PER_1K),
    }


def reset_overrides_for_test() -> None:
    """Test hook — clear in-memory override caches."""
    global _ENV_OVERRIDES, _MONGO_OVERRIDES, _OPENROUTER_OVERRIDES, _OVERRIDES_LOADED
    _ENV_OVERRIDES = {}
    _MONGO_OVERRIDES = {}
    _OPENROUTER_OVERRIDES = {}
    _OVERRIDES_LOADED = False


# ─── Public API ───────────────────────────────────────────────────────


def get_price(model: str | None) -> Tuple[float, float]:
    """Return `(input_per_1k, output_per_1k)` for `model`, consulting all
    override sources in priority order, then the static table, then a
    `default` fallback so unknown models still produce a non-zero cost.
    """
    name = _norm(model)
    if not name:
        return PRICES_PER_1K["default"]

    # 1. Env override (longest-prefix match)
    if _ENV_OVERRIDES:
        hit = _longest_prefix_match(name, _ENV_OVERRIDES) or _ENV_OVERRIDES.get(name)
        if hit:
            return hit
    # 2. Mongo override
    if _MONGO_OVERRIDES:
        hit = _longest_prefix_match(name, _MONGO_OVERRIDES) or _MONGO_OVERRIDES.get(name)
        if hit:
            return hit
    # 3. OpenRouter live cache
    if _OPENROUTER_OVERRIDES:
        # Try exact + provider-prefixed forms ("anthropic/claude-sonnet-4-6")
        direct = _OPENROUTER_OVERRIDES.get(name)
        if direct:
            return direct
        hit = _longest_prefix_match(name, _OPENROUTER_OVERRIDES)
        if hit:
            return hit
    # 4. Static fallback (longest-prefix)
    hit = _longest_prefix_match(name, PRICES_PER_1K)
    if hit:
        return hit
    if "local" in name or "embedding" in name:
        return PRICES_PER_1K["local"]
    return PRICES_PER_1K["default"]


def compute_cost(
    *,
    model: str | None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> float:
    """USD cost for one LLM call.  Always returns a non-negative float
    rounded to 6 decimals (sub-cent precision, no float-noise tails).
    """
    if not prompt_tokens and not completion_tokens:
        return 0.0
    in_rate, out_rate = get_price(model)
    cost = (
        (prompt_tokens / 1000.0) * in_rate
        + (completion_tokens / 1000.0) * out_rate
    )
    return round(max(0.0, cost), 6)


def compute_credits(cost_usd: float) -> int:
    """Convert a USD cost into user-facing credits.

    Formula: ceil(cost_usd × margin / unit_rate), minimum 1 credit when
    the call had any cost.  Both `margin` and `unit_rate` are
    configurable per-environment via settings.CREDIT_USD_RATE +
    settings.CREDIT_MARGIN_MULTIPLIER.

    Defaults: rate=$0.0003, margin=3.0 → a $0.001 LLM call costs the
    user 10 credits.  Adjust to match your plan economics.
    """
    if not cost_usd or cost_usd <= 0:
        return 0
    from backend.core.config import settings
    rate = float(getattr(settings, "CREDIT_USD_RATE", 0.0003)) or 0.0003
    margin = float(getattr(settings, "CREDIT_MARGIN_MULTIPLIER", 3.0)) or 1.0
    raw = (cost_usd * margin) / rate
    return max(1, int(math.ceil(raw)))


def get_pricing_snapshot() -> Dict[str, Any]:
    """Diagnostics — what's loaded right now.  Useful from an admin
    endpoint to verify overrides took effect."""
    return {
        "overrides_loaded": _OVERRIDES_LOADED,
        "env_count": len(_ENV_OVERRIDES),
        "mongo_count": len(_MONGO_OVERRIDES),
        "openrouter_count": len(_OPENROUTER_OVERRIDES),
        "static_count": len(PRICES_PER_1K),
    }
