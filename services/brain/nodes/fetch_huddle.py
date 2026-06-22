"""Pull recent huddle transcripts (last 24 h) for evening recap +
morning follow-ups.

Joins HuddleSQL (the live-meeting row) with MeetingSQL (the
post-call MeetingAgent output via huddle.processed_meeting_id) so the
brain reasons over the structured summary, not the raw transcript.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import structlog
from sqlalchemy import select

from ..state import BrainState, HuddleSummaryRef
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


@traced_node("fetch_huddle")
async def fetch_huddle(state: BrainState) -> Dict[str, Any]:
    try:
        from backend.db.postgres import get_async_sessionmaker
        from backend.db.postgres_models import HuddleSQL, MeetingSQL
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_huddle.imports_failed", error=str(exc))
        return {
            "huddle_recents": [],
            "__payload_summary": {"count": 0, "reason": "imports_failed"},
            "__eval_score": 0.0,
        }

    since = datetime.utcnow() - timedelta(hours=24)

    huddles: List[HuddleSummaryRef] = []
    try:
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            q = (
                select(HuddleSQL, MeetingSQL)
                .outerjoin(MeetingSQL, MeetingSQL.id == HuddleSQL.processed_meeting_id)
                .where(
                    HuddleSQL.host_user_id == state.user_id,
                    HuddleSQL.ended_at.isnot(None),
                    HuddleSQL.ended_at >= since,
                    HuddleSQL.deleted_at.is_(None),
                )
                .order_by(HuddleSQL.ended_at.desc())
                .limit(10)
            )
            rows = (await session.execute(q)).all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_huddle.query_failed", error=str(exc))
        return {
            "huddle_recents": [],
            "__payload_summary": {"count": 0, "error": "query_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    for huddle, meeting in rows:
        summary_text = (meeting.summary if meeting else None) or (huddle.title or "")
        huddles.append(
            HuddleSummaryRef(
                huddle_id=str(huddle.id),
                title=huddle.title,
                ended_at=huddle.ended_at,
                summary=summary_text[:1000] if summary_text else None,
            )
        )

    return {
        "huddle_recents": huddles,
        "__payload_summary": {"count": len(huddles), "window_hours": 24},
        "__eval_score": 1.0,
    }
