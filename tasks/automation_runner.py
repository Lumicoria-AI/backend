"""
Phase C — Automation runner.

Out-of-process beat task that drives the automation engine.  Two
responsibilities:

1. Tick every `schedule`-triggered automation whose `next_run_at` is due,
   firing the matching event into the event bus (the engine picks it up
   inline and dispatches actions).
2. Retry `automation_runs` that ended in `error` with a backoff so a
   transient action failure doesn't poison the rule.

The Phase C event-driven automations are already handled in-process via
`automation_engine.install()` (the API process subscribes to event_bus).
This worker is the missing piece for `schedule`-trigger rules and
durability of action retries.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from backend.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

MAX_RUN_RETRIES = 5
BASE_RETRY_S = 120


def _next_retry_at(attempts: int) -> datetime:
    delay = min(BASE_RETRY_S * (3 ** attempts), 3600)
    return datetime.utcnow() + timedelta(seconds=delay)


def _matches_minute(cron: str, now: datetime) -> bool:
    """Tiny cron matcher — supports `*/N` and exact-minute fields, hour, dow.

    Format: `minute hour day month dow` (5 fields).  This is intentionally
    narrow so we don't shell out to croniter for the beat tick.
    """
    parts = cron.strip().split()
    if len(parts) != 5:
        return False
    minute, hour, dom, mon, dow = parts

    def _ok(field: str, value: int, lo: int, hi: int) -> bool:
        if field == "*":
            return True
        if field.startswith("*/"):
            try:
                n = int(field[2:])
                return n > 0 and (value % n == 0)
            except ValueError:
                return False
        # comma-separated list
        if "," in field:
            try:
                return value in {int(p) for p in field.split(",")}
            except ValueError:
                return False
        # range a-b
        m = re.match(r"^(\d+)-(\d+)$", field)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return a <= value <= b
        try:
            return value == int(field)
        except ValueError:
            return False

    return (
        _ok(minute, now.minute, 0, 59)
        and _ok(hour, now.hour, 0, 23)
        and _ok(dom, now.day, 1, 31)
        and _ok(mon, now.month, 1, 12)
        and _ok(dow, now.weekday(), 0, 6)
    )


async def _tick_scheduled_automations() -> Dict[str, int]:
    from backend.db.mongodb.mongodb import MongoDB
    from backend.services.event_bus import emit

    col = await MongoDB.get_collection("automations")
    runs = await MongoDB.get_collection("automation_runs")
    now = datetime.utcnow().replace(second=0, microsecond=0)
    cursor = col.find({"enabled": True, "trigger.type": "schedule"})
    fired = 0
    async for auto in cursor:
        cfg = (auto.get("trigger") or {}).get("config") or {}
        cron = cfg.get("cron")
        if not cron or not _matches_minute(cron, now):
            continue
        # Idempotency: skip if last_run_at is in the same minute.
        last = auto.get("last_run_at")
        if last and last.replace(second=0, microsecond=0) == now:
            continue
        org_id = str(auto.get("organization_id"))
        try:
            await emit(
                "automation.tick",
                organization_id=org_id,
                actor_id=None,
                source="schedule",
                payload={"automation_id": str(auto["_id"]),
                         "scheduled_minute": now.isoformat() + "Z"},
            )
            await col.update_one({"_id": auto["_id"]}, {"$set": {"last_run_at": now}})
            await runs.insert_one({
                "automation_id": auto["_id"],
                "organization_id": auto.get("organization_id"),
                "status": "completed",
                "trigger_payload": {"kind": "schedule", "cron": cron},
                "actions_executed": [],
                "started_at": now, "ended_at": datetime.utcnow(),
                "created_at": datetime.utcnow(),
            })
            fired += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("automation.schedule_tick_failed",
                             automation_id=str(auto["_id"]), error=str(exc))
    return {"fired": fired, "tick_minute": now.isoformat() + "Z"}


async def _retry_errored_runs(batch_size: int = 50) -> Dict[str, int]:
    """Retry automation_runs that errored, with exponential backoff."""
    from backend.db.mongodb.mongodb import MongoDB
    from backend.db.mongodb.repositories.automations_repository import automations_repository
    from backend.services.automation_engine import _dispatch_action, _eval_conditions
    from backend.services.event_bus import Event

    runs = await MongoDB.get_collection("automation_runs")
    automations = await MongoDB.get_collection("automations")
    now = datetime.utcnow()
    cursor = runs.find({
        "status": "error",
        "$or": [{"next_retry_at": {"$exists": False}},
                {"next_retry_at": {"$lte": now}}],
        "attempts": {"$lt": MAX_RUN_RETRIES},
    }).limit(batch_size)
    retried = 0
    async for run in cursor:
        attempts = int(run.get("attempts") or 0) + 1
        auto = await automations.find_one({"_id": run.get("automation_id")})
        if not auto:
            await runs.update_one({"_id": run["_id"]},
                                  {"$set": {"status": "abandoned"}})
            continue

        trigger_payload = run.get("trigger_payload") or {}
        synthetic_event = Event(
            type=trigger_payload.get("event_type") or "automation.retry",
            organization_id=str(auto.get("organization_id")) if auto.get("organization_id") else None,
            payload=trigger_payload.get("payload") or {},
            source="retry",
        )
        if not _eval_conditions(synthetic_event, auto.get("conditions") or []):
            await runs.update_one({"_id": run["_id"]},
                                  {"$set": {"status": "skipped", "attempts": attempts}})
            continue
        actions_executed = []
        err: Optional[str] = None
        try:
            for action in auto.get("actions") or []:
                actions_executed.append(await _dispatch_action(action, synthetic_event))
        except Exception as exc:  # noqa: BLE001
            err = str(exc)

        next_status = "completed" if not err else "error"
        await runs.update_one(
            {"_id": run["_id"]},
            {"$set": {
                "status": next_status,
                "attempts": attempts,
                "ended_at": datetime.utcnow(),
                "actions_executed": actions_executed,
                "error": err,
                "next_retry_at": _next_retry_at(attempts) if err else None,
            }},
        )
        retried += 1
    return {"retried": retried}


def _run_async(coro):
    """Run a coroutine in a fresh event loop per Celery task.

    Motor binds Futures to the loop that ran the connection's first I/O;
    reusing a loop across prefork workers causes 'Future attached to a
    different loop' errors. asyncio.run() creates and tears down a new
    loop, isolating each task.
    """
    try:
        from backend.db.mongodb.mongodb import MongoDB
        MongoDB.reset_for_new_loop()  # type: ignore[attr-defined]
    except Exception:
        pass
    return asyncio.run(coro)


@celery_app.task(name="automations.tick_scheduled", bind=True, max_retries=3)
def tick_scheduled_automations(self) -> Dict[str, Any]:
    try:
        return _run_async(_tick_scheduled_automations())
    except Exception as exc:  # noqa: BLE001
        logger.exception("automations.tick_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=30)


@celery_app.task(name="automations.retry_errored", bind=True, max_retries=3)
def retry_errored_runs(self) -> Dict[str, Any]:
    try:
        return _run_async(_retry_errored_runs())
    except Exception as exc:  # noqa: BLE001
        logger.exception("automations.retry_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=60)
