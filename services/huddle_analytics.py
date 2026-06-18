"""
Lumicoria Huddle — speaker analytics.

Computes (from persisted transcript chunks):
  - talk_time_by_speaker        — seconds (estimated from word count)
  - turn_count_by_speaker       — how many separate turns each speaker had
  - interruption_count          — rough interruption signal: turns that
                                  arrived less than 1s after the previous
                                  speaker finished
  - sentiment_trend             — list of {ts, score} buckets (~30s)
                                  Uses simple lexicon scoring (positive
                                  / negative seeds). Replace with the
                                  Wellbeing agent later for accuracy.
  - longest_silence_sec         — gap between consecutive chunks

Stored on HuddleSQL.meta["speaker_analytics"] (we keep HuddleSQL.meta
JSONB rather than adding a new column — analytics shape evolves often).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import select, update as sa_update

from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import HuddleSQL, HuddleTranscriptChunkSQL

logger = structlog.get_logger(__name__)


WPM = 150  # average speaking rate

POSITIVE_TOKENS = {
    "great", "good", "awesome", "excellent", "love", "yes", "agree", "thanks",
    "perfect", "amazing", "happy", "fantastic", "brilliant", "win", "shipped",
    "fixed", "solved", "ready", "done", "ship", "exciting", "wonderful",
}
NEGATIVE_TOKENS = {
    "bad", "broken", "fail", "issue", "problem", "blocker", "stuck",
    "delay", "missing", "wrong", "concern", "worried", "frustrated",
    "block", "slow", "late", "error", "bug", "crash", "lost",
}


def _word_count(text: str) -> int:
    return len(text.split())


def _sentiment_score(text: str) -> int:
    """Returns +1 / 0 / -1 per chunk based on token presence."""
    tokens = {t.strip(".,!?:;").lower() for t in text.split()}
    pos = len(tokens & POSITIVE_TOKENS)
    neg = len(tokens & NEGATIVE_TOKENS)
    if pos == neg:
        return 0
    return 1 if pos > neg else -1


async def compute_for_huddle(huddle_id: str) -> Dict[str, Any]:
    """Pull all chunks for a huddle and return the analytics dict."""
    factory = get_async_sessionmaker()
    async with factory() as session:
        rows = (await session.execute(
            select(HuddleTranscriptChunkSQL)
            .where(HuddleTranscriptChunkSQL.huddle_id == huddle_id)
            .order_by(HuddleTranscriptChunkSQL.ts.asc())
        )).scalars().all()

    if not rows:
        return {
            "talk_time_by_speaker": {},
            "turn_count_by_speaker": {},
            "interruption_count": 0,
            "sentiment_trend": [],
            "longest_silence_sec": 0,
            "total_words": 0,
            "speakers_count": 0,
        }

    talk_time: Dict[str, float] = {}
    turn_count: Dict[str, int] = {}
    interruption_count = 0
    sentiment_buckets: List[Tuple[str, int, int]] = []  # (bucket_iso, score_sum, n)
    longest_silence = 0.0
    total_words = 0
    last_speaker: Optional[str] = None
    last_ts: Optional[datetime] = None

    bucket_window_sec = 30
    current_bucket_start: Optional[datetime] = None
    current_bucket_score = 0
    current_bucket_n = 0

    for c in rows:
        speaker = c.speaker_name or "Unknown"
        words = _word_count(c.text or "")
        total_words += words
        seconds = max(2.0, (words / WPM) * 60)
        talk_time[speaker] = talk_time.get(speaker, 0.0) + seconds

        # Turns + interruption detection
        if speaker != last_speaker:
            turn_count[speaker] = turn_count.get(speaker, 0) + 1
            if last_ts is not None and c.ts is not None:
                gap = (c.ts - last_ts).total_seconds()
                if gap < 1.0:
                    interruption_count += 1
                if gap > longest_silence:
                    longest_silence = gap
            last_speaker = speaker
        last_ts = c.ts

        # Sentiment bucketing
        if current_bucket_start is None:
            current_bucket_start = c.ts
        if c.ts and (c.ts - current_bucket_start).total_seconds() > bucket_window_sec:
            sentiment_buckets.append((current_bucket_start.isoformat(), current_bucket_score, current_bucket_n))
            current_bucket_start = c.ts
            current_bucket_score = 0
            current_bucket_n = 0
        current_bucket_score += _sentiment_score(c.text or "")
        current_bucket_n += 1

    if current_bucket_start and current_bucket_n:
        sentiment_buckets.append((current_bucket_start.isoformat(), current_bucket_score, current_bucket_n))

    sentiment_trend = [
        {"ts": ts, "score": (s / n) if n else 0, "samples": n}
        for ts, s, n in sentiment_buckets
    ]

    return {
        "talk_time_by_speaker": {k: round(v, 1) for k, v in talk_time.items()},
        "turn_count_by_speaker": turn_count,
        "interruption_count": interruption_count,
        "sentiment_trend": sentiment_trend,
        "longest_silence_sec": round(longest_silence, 1),
        "total_words": total_words,
        "speakers_count": len(talk_time),
    }


async def persist_for_huddle(huddle_id: str) -> Dict[str, Any]:
    """Compute + persist onto HuddleSQL.meta['speaker_analytics']."""
    analytics = await compute_for_huddle(huddle_id)
    factory = get_async_sessionmaker()
    async with factory() as session:
        row = (await session.execute(
            select(HuddleSQL).where(HuddleSQL.id == huddle_id)
        )).scalar_one_or_none()
        if not row:
            return analytics
        meta = dict(row.meta or {})
        meta["speaker_analytics"] = analytics
        await session.execute(
            sa_update(HuddleSQL)
            .where(HuddleSQL.id == huddle_id)
            .values(meta=meta, updated_at=datetime.utcnow())
        )
        await session.commit()
    return analytics
