"""Golden fixture set for the offline eval suite.

This file is intentionally hand-curated, not synthesised. The whole
point of a deploy-gating eval is to ensure that real-world emails the
team has seen before continue to classify and prioritise correctly.
Each fixture is one human's judgement call frozen in code.

Versioning: bump ``DATASET_VERSION`` when adding/removing fixtures so
trend rows in ``brain_evals`` stay comparable.

Distribution (20 emails):
  action_required:  6   (mixed urgency: 2 critical, 2 high, 2 medium)
  scheduling:       4   (3 high, 1 medium)
  informational:    3   (1 medium, 2 low)
  promotional:     4   (all low)
  spam:             2   (low)
  unknown:          1   (low — genuinely ambiguous)

Priority runs (5):
  1. busy_inbox    — 12 mixed-urgency items + 3 meetings — picks top-5 by stakes.
  2. quiet_morning — 2 emails, 0 meetings — should produce ≤2 actions.
  3. deadline_day  — multiple deadline-flagged items — critical first.
  4. evening_recap — yesterday's slipped + completed mixed.
  5. all_promo     — only promos + spam → all-clear digest expected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DATASET_VERSION = "v1.0"


# ─────────────────────────────────────────────────────────────────────
# Helpers (declared first so the run fixtures below can use them)
# ─────────────────────────────────────────────────────────────────────


def _ce(
    message_id: str,
    category: str,
    urgency: str,
    summary: str,
    *,
    suggested_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a ClassifiedEmail-shaped dict for fixture authoring."""
    return {
        "message_id": message_id,
        "category": category,
        "urgency": urgency,
        "confidence": 0.9,
        "summary": summary,
        "suggested_agent": suggested_agent,
    }


def _evt(event_id: str, summary: str, time_label: str) -> Dict[str, Any]:
    """Build a CalendarEventRef-shaped dict for fixture authoring."""
    return {
        "event_id": event_id,
        "summary": summary,
        "start": None,
        "end": None,
        "attendees": [],
        "location": None,
        "_time_label": time_label,  # for human readability only
    }


# ─────────────────────────────────────────────────────────────────────
# Classification fixtures — input → expected category/urgency
# ─────────────────────────────────────────────────────────────────────


GOLDEN_EMAILS: List[Dict[str, Any]] = [
    # ── action_required (6) ─────────────────────────────────────────
    {
        "id": "ar-001",
        "input": {
            "message_id": "msg-ar-001",
            "subject": "Re: Q4 budget — need your sign-off by Friday",
            "from_addr": "cfo@acme.com",
            "snippet": "Hey — attaching the revised Q4 plan. Please look it "
                       "over and sign off by EOD Friday so finance can lock it.",
            "label_ids": ["INBOX", "IMPORTANT"],
            "has_attachments": True,
        },
        "expected": {
            "category": "action_required",
            "urgency": "high",
            "suggested_agent": "document",
        },
    },
    {
        "id": "ar-002",
        "input": {
            "message_id": "msg-ar-002",
            "subject": "URGENT: production outage — investigate now",
            "from_addr": "ops-alerts@infra.lumicoria.ai",
            "snippet": "API p99 latency >2s for 12 min. Possible vector DB "
                       "saturation. On-call paged. Need lead-eng eyes immediately.",
            "label_ids": ["INBOX", "STARRED"],
            "has_attachments": False,
        },
        "expected": {
            "category": "action_required",
            "urgency": "critical",
            "suggested_agent": "executor",
        },
    },
    {
        "id": "ar-003",
        "input": {
            "message_id": "msg-ar-003",
            "subject": "Contract redlines — review and counter",
            "from_addr": "counsel@partnerfirm.com",
            "snippet": "Their counsel returned the MSA with 4 redlines. Two are "
                       "blockers (indemnity cap + exclusivity clause). Need your "
                       "response by Tuesday so we can close this week.",
            "label_ids": ["INBOX"],
            "has_attachments": True,
        },
        "expected": {
            "category": "action_required",
            "urgency": "high",
            "suggested_agent": "document",
        },
    },
    {
        "id": "ar-004",
        "input": {
            "message_id": "msg-ar-004",
            "subject": "Approve PR #1421 — autoscaler timeout fix",
            "from_addr": "noreply@github.com",
            "snippet": "Maya assigned you a review on lumicoria/backend#1421. "
                       "+187 -42. CI green. Description: ‘Wait for replica readiness.’",
            "label_ids": ["INBOX", "CATEGORY_UPDATES"],
            "has_attachments": False,
        },
        "expected": {
            "category": "action_required",
            "urgency": "medium",
            "suggested_agent": "code",
        },
    },
    {
        "id": "ar-005",
        "input": {
            "message_id": "msg-ar-005",
            "subject": "Customer escalation — Northwind threatening churn",
            "from_addr": "head-of-cs@lumicoria.ai",
            "snippet": "Northwind's CTO emailed me. They're frustrated with the "
                       "two open P1s. Need a strategy by tomorrow's exec sync.",
            "label_ids": ["INBOX", "IMPORTANT"],
            "has_attachments": False,
        },
        "expected": {
            "category": "action_required",
            "urgency": "critical",
        },
    },
    {
        "id": "ar-006",
        "input": {
            "message_id": "msg-ar-006",
            "subject": "Please complete H2 performance review for direct reports",
            "from_addr": "hr@lumicoria.ai",
            "snippet": "Reminder — H2 review forms due by next Friday. You have "
                       "3 direct reports outstanding.",
            "label_ids": ["INBOX"],
            "has_attachments": False,
        },
        "expected": {
            "category": "action_required",
            "urgency": "medium",
        },
    },

    # ── scheduling (4) ──────────────────────────────────────────────
    {
        "id": "sc-001",
        "input": {
            "message_id": "msg-sc-001",
            "subject": "Can we move tomorrow's 9am to 10:30?",
            "from_addr": "priya@acme.com",
            "snippet": "Hey — got a conflict at 9. Could we push to 10:30 same "
                       "day? Same agenda.",
            "label_ids": ["INBOX"],
            "has_attachments": False,
        },
        "expected": {
            "category": "scheduling",
            "urgency": "high",
            "suggested_agent": "meeting",
        },
    },
    {
        "id": "sc-002",
        "input": {
            "message_id": "msg-sc-002",
            "subject": "Meeting invitation: AI strategy roundtable",
            "from_addr": "events@techbrew.com",
            "snippet": "You're invited to a closed roundtable on agentic AI in "
                       "B2B SaaS. Wed Mar 12, 3pm PT. RSVP by Friday.",
            "label_ids": ["INBOX"],
            "has_attachments": False,
        },
        "expected": {
            "category": "scheduling",
            "urgency": "medium",
            "suggested_agent": "meeting",
        },
    },
    {
        "id": "sc-003",
        "input": {
            "message_id": "msg-sc-003",
            "subject": "Reschedule: 1:1 with Sam — pushed to Thursday",
            "from_addr": "calendar-noreply@google.com",
            "snippet": "Your 1:1 with Sam Bakshi has been moved from Tuesday "
                       "to Thursday at 4pm.",
            "label_ids": ["INBOX", "CATEGORY_UPDATES"],
            "has_attachments": False,
        },
        "expected": {
            "category": "scheduling",
            "urgency": "high",
        },
    },
    {
        "id": "sc-004",
        "input": {
            "message_id": "msg-sc-004",
            "subject": "Interview — Senior Engineer slot Tuesday",
            "from_addr": "recruiting@lumicoria.ai",
            "snippet": "We have Camille booked Tuesday 2pm for the system "
                       "design loop. Sending the candidate brief shortly.",
            "label_ids": ["INBOX"],
            "has_attachments": False,
        },
        "expected": {
            "category": "scheduling",
            "urgency": "high",
        },
    },

    # ── informational (3) ───────────────────────────────────────────
    {
        "id": "in-001",
        "input": {
            "message_id": "msg-in-001",
            "subject": "Your AWS bill — March: $4,217",
            "from_addr": "no-reply@aws.com",
            "snippet": "Your March 2026 AWS invoice is ready. $4,217.42, due "
                       "April 1. No action needed — auto-payment is on.",
            "label_ids": ["INBOX"],
            "has_attachments": True,
        },
        "expected": {
            "category": "informational",
            "urgency": "medium",
        },
    },
    {
        "id": "in-002",
        "input": {
            "message_id": "msg-in-002",
            "subject": "Weekly digest: Lumicoria engineering",
            "from_addr": "engineering-digest@lumicoria.ai",
            "snippet": "5 PRs merged, 1 hot-fix shipped, RAG pipeline p95 down "
                       "to 412ms.",
            "label_ids": ["INBOX"],
            "has_attachments": False,
        },
        "expected": {
            "category": "informational",
            "urgency": "low",
        },
    },
    {
        "id": "in-003",
        "input": {
            "message_id": "msg-in-003",
            "subject": "Stripe payout summary — $18,420 settled",
            "from_addr": "noreply@stripe.com",
            "snippet": "Your March payout of $18,420.10 was settled to your "
                       "linked bank account.",
            "label_ids": ["INBOX", "CATEGORY_UPDATES"],
            "has_attachments": False,
        },
        "expected": {
            "category": "informational",
            "urgency": "low",
        },
    },

    # ── promotional (4) ─────────────────────────────────────────────
    {
        "id": "pr-001",
        "input": {
            "message_id": "msg-pr-001",
            "subject": "🎉 30% off your next Notion subscription",
            "from_addr": "deals@notion.com",
            "snippet": "Limited time — upgrade to Notion AI for 30% off.",
            "label_ids": ["INBOX", "CATEGORY_PROMOTIONS"],
            "has_attachments": False,
        },
        "expected": {"category": "promotional", "urgency": "low"},
    },
    {
        "id": "pr-002",
        "input": {
            "message_id": "msg-pr-002",
            "subject": "Webinar this Thursday: Scaling RAG to enterprise",
            "from_addr": "marketing@vectordbco.com",
            "snippet": "Join us Thursday at 10am PT for a deep dive on RAG.",
            "label_ids": ["INBOX", "CATEGORY_PROMOTIONS"],
            "has_attachments": False,
        },
        "expected": {"category": "promotional", "urgency": "low"},
    },
    {
        "id": "pr-003",
        "input": {
            "message_id": "msg-pr-003",
            "subject": "New from Cursor: agentic coding workflows",
            "from_addr": "hello@cursor.com",
            "snippet": "Multi-file edits, terminal agents, MCP support.",
            "label_ids": ["INBOX", "CATEGORY_PROMOTIONS"],
            "has_attachments": False,
        },
        "expected": {"category": "promotional", "urgency": "low"},
    },
    {
        "id": "pr-004",
        "input": {
            "message_id": "msg-pr-004",
            "subject": "Last day to save — Black Friday deals end tonight",
            "from_addr": "deals@laptopmaker.com",
            "snippet": "Up to $1,200 off MacBook Pro M4. Ends midnight.",
            "label_ids": ["INBOX", "CATEGORY_PROMOTIONS"],
            "has_attachments": False,
        },
        "expected": {"category": "promotional", "urgency": "low"},
    },

    # ── spam (2) ────────────────────────────────────────────────────
    {
        "id": "sp-001",
        "input": {
            "message_id": "msg-sp-001",
            "subject": "Congratulations — you've won a $500 gift card!!!",
            "from_addr": "winnings@xj4r3.ru",
            "snippet": "Click here to claim your prize before midnight.",
            "label_ids": ["INBOX", "SPAM"],
            "has_attachments": False,
        },
        "expected": {"category": "spam", "urgency": "low"},
    },
    {
        "id": "sp-002",
        "input": {
            "message_id": "msg-sp-002",
            "subject": "Final notice: your account will be suspended",
            "from_addr": "support@arn3z-banking.tk",
            "snippet": "Verify your identity by clicking the link below to "
                       "prevent immediate account suspension.",
            "label_ids": ["INBOX", "SPAM"],
            "has_attachments": False,
        },
        "expected": {"category": "spam", "urgency": "low"},
    },

    # ── unknown (1) ─────────────────────────────────────────────────
    {
        "id": "un-001",
        "input": {
            "message_id": "msg-un-001",
            "subject": "hey",
            "from_addr": "old.friend@gmail.com",
            "snippet": "thinking of you",
            "label_ids": ["INBOX"],
            "has_attachments": False,
        },
        "expected": {"category": "unknown", "urgency": "low"},
    },
]


# ─────────────────────────────────────────────────────────────────────
# Priority-run fixtures — input bundle → expected top-K ranked actions
# ─────────────────────────────────────────────────────────────────────


GOLDEN_PRIORITY_RUNS: List[Dict[str, Any]] = [
    {
        "id": "run-busy",
        "mode": "morning",
        "input": {
            "classified_emails": [
                # Hard-coded classifications so we test prioritisation,
                # not classification (that's the classification F1 above).
                _ce("msg-ar-002", "action_required", "critical", "Production outage — vector DB saturation"),
                _ce("msg-ar-005", "action_required", "critical", "Northwind threatening churn"),
                _ce("msg-ar-001", "action_required", "high",     "Q4 budget sign-off by Friday"),
                _ce("msg-ar-003", "action_required", "high",     "Contract redlines, blocker clauses"),
                _ce("msg-sc-001", "scheduling",      "high",     "Move 9am to 10:30 — Priya request"),
                _ce("msg-sc-003", "scheduling",      "high",     "1:1 with Sam moved to Thursday"),
                _ce("msg-ar-004", "action_required", "medium",   "Approve PR #1421 — autoscaler fix"),
                _ce("msg-ar-006", "action_required", "medium",   "H2 review forms — 3 reports outstanding"),
                _ce("msg-in-001", "informational",   "medium",   "AWS bill — $4,217"),
                _ce("msg-pr-001", "promotional",     "low",      "30% off Notion"),
                _ce("msg-pr-002", "promotional",     "low",      "Webinar Thursday"),
                _ce("msg-sp-001", "spam",            "low",      "$500 gift card scam"),
            ],
            "calendar_events": [
                _evt("evt-001", "Acme review", "10:30"),
                _evt("evt-002", "1:1 with Sam", "16:00"),
                _evt("evt-003", "Standup", "09:00"),
            ],
            "open_tasks": [],
            "huddle_recents": [],
        },
        "expected_top_k": [
            {"title_contains": ["outage", "production", "investigate"], "priority": "critical"},
            {"title_contains": ["northwind", "churn", "escalation"], "priority": "critical"},
            {"title_contains": ["budget", "q4", "sign"], "priority": "high"},
            {"title_contains": ["contract", "redline", "msa"], "priority": "high"},
            {"title_contains": ["pr", "review", "1421", "approve"], "priority": "medium"},
        ],
    },
    {
        "id": "run-quiet",
        "mode": "morning",
        "input": {
            "classified_emails": [
                _ce("msg-in-002", "informational", "low",  "Weekly engineering digest"),
                _ce("msg-pr-003", "promotional",   "low",  "Cursor agentic workflows"),
            ],
            "calendar_events": [],
            "open_tasks": [],
            "huddle_recents": [],
        },
        # Quiet morning: at most 1-2 actions, none above medium priority.
        "expected_top_k": [],
        "max_high_priority": 0,
    },
    {
        "id": "run-deadlines",
        "mode": "morning",
        "input": {
            "classified_emails": [
                _ce("msg-ar-001", "action_required", "high",     "Q4 budget sign-off Friday"),
                _ce("msg-ar-003", "action_required", "critical", "MSA redlines — close by Tuesday"),
                _ce("msg-ar-006", "action_required", "medium",   "H2 reviews next Friday"),
                _ce("msg-sc-004", "scheduling",      "high",     "Interview Camille Tuesday 2pm"),
            ],
            "calendar_events": [
                _evt("evt-004", "Camille interview", "14:00"),
            ],
            "open_tasks": [],
            "huddle_recents": [],
        },
        "expected_top_k": [
            {"title_contains": ["msa", "redline", "contract"], "priority": "critical"},
            {"title_contains": ["q4", "budget"], "priority": "high"},
            {"title_contains": ["interview", "camille"], "priority": "high"},
            {"title_contains": ["h2", "review"], "priority": "medium"},
        ],
    },
    {
        "id": "run-evening",
        "mode": "evening",
        "input": {
            "classified_emails": [
                _ce("msg-ar-002", "action_required", "critical", "Production outage RESOLVED"),
                _ce("msg-in-003", "informational",   "low",      "Stripe payout settled"),
            ],
            "calendar_events": [],
            "open_tasks": [
                {"task_id": "t1", "title": "Q4 budget sign-off",
                 "priority": "high", "status": "in_progress", "due_date": None},
                {"task_id": "t2", "title": "Approve PR #1421",
                 "priority": "medium", "status": "in_progress", "due_date": None},
            ],
            "huddle_recents": [
                {"huddle_id": "h1", "title": "Exec sync",
                 "ended_at": None,
                 "summary": "Discussed Northwind retention plan. Sam owns followup."},
            ],
        },
        "expected_top_k": [
            {"title_contains": ["northwind", "retention", "follow"], "priority": "high"},
            {"title_contains": ["q4", "budget", "tomorrow"], "priority": "high"},
        ],
    },
    {
        "id": "run-allpromo",
        "mode": "morning",
        "input": {
            "classified_emails": [
                _ce("msg-pr-001", "promotional", "low", "30% off Notion"),
                _ce("msg-pr-002", "promotional", "low", "Webinar Thursday"),
                _ce("msg-pr-004", "promotional", "low", "Black Friday deals"),
                _ce("msg-sp-001", "spam",        "low", "Gift card scam"),
            ],
            "calendar_events": [],
            "open_tasks": [],
            "huddle_recents": [],
        },
        # All-promo morning: model should produce ZERO actions.
        "expected_top_k": [],
        "max_actions": 0,
    },
]


