"""Well-being service layer.

Modules:
  - sanitize:        length caps + control-char strip for user-facing text
  - productivity:    aggregates task / activity / agent-run data into
                     focus_minutes / completion_ratio / streak metrics
  - session_tracker: Redis-backed last-activity + last-break clock
                     that powers the live break countdown
  - digest:          weekly digest payload assembly + email send
  - orchestrator:    Gemini-locked agent wrapper for recommendations,
                     break advice, chat, and weekly reflection
"""
