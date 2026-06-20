"""
Phase E — Webhook delivery worker.

Pulls pending webhook deliveries, signs the payload with the org's secret
(HMAC-SHA256), POSTs to the registered URL, and marks success / failure.
Failed deliveries are re-scheduled with exponential backoff (60s, 300s,
1500s, ...).  A maximum of 8 attempts is enforced; after that the
delivery row is marked `failed` and the parent webhook's failure_count
is bumped.

Wiring:
    backend.tasks.celery_app.beat_schedule['webhook-deliver'] runs every 60s.

The actual HTTP POST runs inside an `asyncio.run()` inside the Celery
task to stay compatible with the synchronous Celery worker model — the
codebase already uses this pattern in document_tasks.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from backend.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

MAX_ATTEMPTS = 8
BASE_DELAY_S = 60
SIGNATURE_HEADER = "X-Lumicoria-Signature"
EVENT_HEADER = "X-Lumicoria-Event"
DELIVERY_HEADER = "X-Lumicoria-Delivery"
TIMESTAMP_HEADER = "X-Lumicoria-Timestamp"


def _next_attempt_at(attempts: int) -> datetime:
    """Exponential backoff with a 1500s cap."""
    delay = min(BASE_DELAY_S * (5 ** attempts), 1500)
    return datetime.utcnow() + timedelta(seconds=delay)


def _sign(secret: str, body: bytes, ts: str) -> str:
    """HMAC-SHA256(secret, "{ts}.{body}") — Stripe-style."""
    if not secret:
        return ""
    msg = f"{ts}.".encode("utf-8") + body
    mac = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256)
    return mac.hexdigest()


# ──────────────────────────────────────────────────────── dispatcher


async def _dispatch_one(delivery: Dict[str, Any]) -> None:
    import httpx
    from bson import ObjectId

    from backend.db.mongodb.mongodb import MongoDB

    deliveries = await MongoDB.get_collection("webhook_deliveries")
    webhooks = await MongoDB.get_collection("webhooks")

    webhook = await webhooks.find_one({"_id": delivery["webhook_id"]})
    if not webhook or not webhook.get("enabled"):
        await deliveries.update_one(
            {"_id": delivery["_id"]},
            {"$set": {"status": "skipped", "error": "webhook missing or disabled",
                      "next_attempt_at": None}},
        )
        return

    secret_hash = webhook.get("secret_hash") or ""
    # In the current build the plaintext is not retained; signing uses the
    # stored hash as the symmetric key.  Customers receive the plaintext
    # at creation time and can recover it from their copy.  When the team
    # rotates to envelope-encrypted secret storage, this swap-out is the
    # only line that changes.
    secret = secret_hash

    body = json.dumps({
        "event": delivery["event"],
        "payload": delivery.get("payload") or {},
        "delivery_id": str(delivery["_id"]),
        "organization_id": str(delivery["organization_id"]),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }, default=str).encode("utf-8")

    ts = str(int(time.time()))
    signature = _sign(secret, body, ts)
    headers = {
        "Content-Type": "application/json",
        EVENT_HEADER: delivery["event"],
        DELIVERY_HEADER: str(delivery["_id"]),
        TIMESTAMP_HEADER: ts,
        SIGNATURE_HEADER: f"t={ts},v1={signature}",
        "User-Agent": "Lumicoria-Webhooks/1.0",
    }

    attempts = int(delivery.get("attempts") or 0) + 1
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook["url"], content=body, headers=headers)
        if 200 <= resp.status_code < 300:
            await deliveries.update_one(
                {"_id": delivery["_id"]},
                {"$set": {
                    "status": "success",
                    "attempts": attempts,
                    "response_status": resp.status_code,
                    "response_body": (resp.text or "")[:2000],
                    "delivered_at": datetime.utcnow(),
                    "next_attempt_at": None,
                    "error": None,
                }},
            )
            await webhooks.update_one(
                {"_id": webhook["_id"]},
                {"$set": {"last_delivery_at": datetime.utcnow()}},
            )
            logger.info("webhook.delivered", delivery=str(delivery["_id"]),
                        url=webhook["url"], status=resp.status_code)
            return

        # 4xx / 5xx — treat as a transient failure (Stripe-style: only 2xx ack).
        await _mark_failed(delivery, attempts, deliveries, webhooks,
                           response_status=resp.status_code,
                           response_body=(resp.text or "")[:2000],
                           error=f"non-2xx ({resp.status_code})")
    except Exception as exc:  # noqa: BLE001
        await _mark_failed(delivery, attempts, deliveries, webhooks,
                           response_status=None, response_body=None,
                           error=str(exc))


async def _mark_failed(
    delivery: Dict[str, Any], attempts: int, deliveries, webhooks,
    *, response_status: Optional[int], response_body: Optional[str], error: str,
) -> None:
    next_at = _next_attempt_at(attempts) if attempts < MAX_ATTEMPTS else None
    final_status = "failed" if next_at is None else "pending"
    await deliveries.update_one(
        {"_id": delivery["_id"]},
        {"$set": {
            "status": final_status,
            "attempts": attempts,
            "response_status": response_status,
            "response_body": response_body,
            "error": error,
            "next_attempt_at": next_at,
        }},
    )
    if final_status == "failed":
        await webhooks.update_one(
            {"_id": delivery["webhook_id"]},
            {"$inc": {"failure_count": 1}},
        )
    logger.warning("webhook.delivery_failed", delivery=str(delivery["_id"]),
                   attempts=attempts, next_at=next_at, error=error)


async def _drain_due_deliveries(batch_size: int = 50) -> int:
    from backend.db.mongodb.mongodb import MongoDB

    deliveries = await MongoDB.get_collection("webhook_deliveries")
    now = datetime.utcnow()
    cursor = deliveries.find({
        "status": "pending",
        "$or": [
            {"next_attempt_at": None},
            {"next_attempt_at": {"$lte": now}},
        ],
    }).sort("created_at", 1).limit(batch_size)
    rows = await cursor.to_list(length=batch_size)
    for row in rows:
        try:
            await _dispatch_one(row)
        except Exception as exc:  # noqa: BLE001
            logger.exception("webhook.dispatch_unexpected", error=str(exc))
    return len(rows)


# ──────────────────────────────────────────────────────── Celery shim


def _run_async(coro):
    """Run a coroutine to completion in a fresh, isolated event loop.

    Motor (the async MongoDB driver) binds Futures to the loop that was
    current when the connection's first I/O ran. Reusing a loop across
    Celery prefork workers causes "Future attached to a different loop"
    errors. A fresh loop per task closes that hole; the cost is one new
    Motor client per invocation, which is fine for beat-scheduled jobs.
    """
    # Reset any module-level Motor client cached against an old loop so
    # the next coroutine recreates it on this fresh one.
    try:
        from backend.db.mongodb.mongodb import MongoDB
        MongoDB.reset_for_new_loop()  # type: ignore[attr-defined]
    except Exception:
        pass
    return asyncio.run(coro)


@celery_app.task(name="webhooks.deliver_due", bind=True, max_retries=3)
def deliver_due_webhooks(self, batch_size: int = 50) -> Dict[str, Any]:
    """Beat-triggered: drain webhook_deliveries that are due."""
    try:
        n = _run_async(_drain_due_deliveries(batch_size=batch_size))
        return {"processed": n}
    except Exception as exc:  # noqa: BLE001
        logger.exception("webhooks.deliver_due_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=30)
