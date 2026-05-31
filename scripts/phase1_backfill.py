"""
Phase 1 backfill — idempotent, safe to run multiple times.

This script:
  1. Ensures every legacy `tasks` document has the new Phase 1 fields with
     sensible defaults (no destructive overwrites — only `$set` when missing).
  2. Creates indexes on the three new collections (calendar_events, invites,
     agent_runs) by simply *touching* their repositories, which lazily
     initialise indexes on first access.
  3. Reports counts so you can confirm what changed.

Run with:
    lumicoria_ai_venv/bin/python -m backend.scripts.phase1_backfill
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict

import structlog

from backend.db.mongodb.mongodb import init_mongodb, close_mongodb, MongoDB
from backend.db.mongodb.repositories.calendar_repository import calendar_repository
from backend.db.mongodb.repositories.invite_repository import invite_repository
from backend.db.mongodb.repositories.agent_run_repository import agent_run_repository

logger = structlog.get_logger(__name__)


async def backfill_tasks_collection() -> Dict[str, int]:
    """Set Phase 1 defaults on legacy task rows that don't have them yet.

    All operations use `$set` with `$exists: false` filters so we never
    overwrite existing data.  Safe to re-run.
    """
    tasks = await MongoDB.get_collection("tasks")
    stats: Dict[str, int] = {}

    # 1. assignee_kind — default to "user" when assigned_to exists, else null
    r = await tasks.update_many(
        {"assignee_kind": {"$exists": False}, "assigned_to": {"$ne": None}},
        {"$set": {"assignee_kind": "user"}},
    )
    stats["assignee_kind_user"] = r.modified_count

    r = await tasks.update_many(
        {"assignee_kind": {"$exists": False}},
        {"$set": {"assignee_kind": None}},
    )
    stats["assignee_kind_null"] = r.modified_count

    # 2. assigned_to_email / assigned_to_agent — default to null
    for field in ("assigned_to_email", "assigned_to_agent"):
        r = await tasks.update_many(
            {field: {"$exists": False}},
            {"$set": {field: None}},
        )
        stats[f"{field}_null"] = r.modified_count

    # 3. agent_proposal — default to null
    r = await tasks.update_many(
        {"agent_proposal": {"$exists": False}},
        {"$set": {"agent_proposal": None}},
    )
    stats["agent_proposal_null"] = r.modified_count

    # 4. reminder_state — fresh ReminderState with all None timestamps
    r = await tasks.update_many(
        {"reminder_state": {"$exists": False}},
        {
            "$set": {
                "reminder_state": {
                    "last_morning_sent": None,
                    "last_evening_sent": None,
                    "last_critical_push": None,
                    "last_digest": None,
                    "last_overdue_alert": None,
                    "notify_on_complete": True,
                }
            }
        },
    )
    stats["reminder_state_default"] = r.modified_count

    # 5. calendar_event_id / gcal_event_id — null
    for field in ("calendar_event_id", "gcal_event_id", "invite_id"):
        r = await tasks.update_many(
            {field: {"$exists": False}},
            {"$set": {field: None}},
        )
        stats[f"{field}_null"] = r.modified_count

    # 6. status_history — empty list (so $push works without guard)
    r = await tasks.update_many(
        {"status_history": {"$exists": False}},
        {"$set": {"status_history": []}},
    )
    stats["status_history_init"] = r.modified_count

    # 7. inferred_due_date — false
    r = await tasks.update_many(
        {"inferred_due_date": {"$exists": False}},
        {"$set": {"inferred_due_date": False}},
    )
    stats["inferred_due_date_false"] = r.modified_count

    return stats


async def backfill_user_settings_collection() -> Dict[str, int]:
    """Add task_reminder_settings to UserSettings rows that don't have it."""
    user_settings = await MongoDB.get_collection("user_settings")
    default_settings = {
        "daily_morning_enabled": True,
        "daily_morning_time": "08:00",
        "evening_critical_push": True,
        "evening_critical_time": "17:00",
        "critical_hour_warning": True,
        "weekly_digest_enabled": True,
        "weekly_digest_day": "friday",
        "weekly_digest_time": "09:00",
        "timezone": "UTC",
        "quiet_hours_enabled": False,
        "quiet_hours_start": "22:00",
        "quiet_hours_end": "07:00",
    }
    r = await user_settings.update_many(
        {"task_reminder_settings": {"$exists": False}},
        {"$set": {"task_reminder_settings": default_settings}},
    )
    return {"task_reminder_settings_default": r.modified_count}


async def ensure_new_collection_indexes() -> Dict[str, str]:
    """Touch each repository so its `_create_indexes` runs."""
    await calendar_repository._get_collection()
    await invite_repository._get_collection()
    await agent_run_repository._get_collection()
    return {
        "lumicoria_calendar_events": "indexes-ensured",
        "invites": "indexes-ensured",
        "agent_runs": "indexes-ensured",
    }


async def main() -> None:
    print("─" * 60)
    print(f"Phase 1 backfill starting at {datetime.utcnow().isoformat()}Z")
    print("─" * 60)

    await init_mongodb()
    try:
        task_stats = await backfill_tasks_collection()
        print("\n[tasks]")
        for k, v in task_stats.items():
            print(f"  {k:<32s} {v}")

        settings_stats = await backfill_user_settings_collection()
        print("\n[user_settings]")
        for k, v in settings_stats.items():
            print(f"  {k:<32s} {v}")

        index_stats = await ensure_new_collection_indexes()
        print("\n[indexes]")
        for k, v in index_stats.items():
            print(f"  {k:<32s} {v}")

        print("\n✓ Phase 1 backfill complete.")
    finally:
        await close_mongodb()


if __name__ == "__main__":
    asyncio.run(main())
