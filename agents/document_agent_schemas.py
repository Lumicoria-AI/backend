"""Pydantic schemas for the Document Agent's 4-stage extraction pipeline.

Lives alongside ``document_agent.py`` (not a v2/parallel file). The
schemas are imported by:
  - ``agents/document_agent.py`` — the agent itself.
  - ``services/brain/evals/schema_eval.py`` — already generic; these
    classes plug in directly.
  - Any downstream consumer (compose node, frontend) that wants typed
    access to the agent's output.

Pipeline overview:

  Stage A: classify           → DocumentClassification
  Stage B: chunk + summarise  → list[ChunkSummary]
  Stage C: extract            → DocumentExtraction
                                 (action items, decisions, dates,
                                  people, summary, sentiment)
  Stage D: self-evaluate      → SelfEvalResult

The agent's public ``process_async`` return is backward-compatible:
  ``{"analysis": str, "tasks": list, "metadata": dict, ...}``
plus new fields ``extraction``, ``classification``, ``confidence``,
``low_confidence``, ``sources``.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────
# Stage A — classify
# ─────────────────────────────────────────────────────────────────────


DocumentType = Literal[
    "contract",
    "invoice",
    "meeting_notes",
    "proposal",
    "email_thread",
    "report",
    "spec",
    "policy",
    "presentation",
    "letter",
    "form",
    "research_paper",
    "other",
]

Sensitivity = Literal["public", "internal", "confidential", "restricted"]
Urgency = Literal["critical", "high", "medium", "low"]


class DocumentClassification(BaseModel):
    """Stage A output. One small LLM call; informs every later stage.

    ``estimated_action_count`` lets the agent skip Stage C entirely
    when 0 (e.g. an FYI newsletter) — cheaper, and the user's task
    list doesn't get polluted with non-actions.
    """
    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType = "other"
    language: str = "en"
    sensitivity: Sensitivity = "internal"
    urgency: Urgency = "low"
    estimated_action_count: int = Field(default=0, ge=0, le=50)
    short_title: Optional[str] = Field(default=None, max_length=120)
    detected_parties: List[str] = Field(default_factory=list, max_length=20)


# ─────────────────────────────────────────────────────────────────────
# Stage B — chunk summaries (one per chunk)
# ─────────────────────────────────────────────────────────────────────


class ChunkSummary(BaseModel):
    """One LLM-produced summary per chunk. The chunk_id is the sequential
    index in the original RecursiveCharacterTextSplitter output —
    Stage C cites it back so we can highlight provenance in the UI.
    """
    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    summary: str = Field(..., max_length=2000)
    key_terms: List[str] = Field(default_factory=list, max_length=12)
    has_action: bool = False


# ─────────────────────────────────────────────────────────────────────
# Stage C — structured extract
# ─────────────────────────────────────────────────────────────────────


Priority = Literal["critical", "high", "medium", "low"]


class ExtractedActionItem(BaseModel):
    """One concrete, verb-led action the document creates for someone.

    ``inferred_due_date`` mirrors the field on the existing Task model
    so callers can flag uncertain dates in the UI without re-deriving
    them.
    """
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., max_length=200)
    description: str = Field(default="", max_length=1000)
    priority: Priority = "medium"
    due_date: Optional[datetime] = None
    inferred_due_date: bool = False
    deadline_phrase: Optional[str] = Field(default=None, max_length=200)
    assignee: Optional[str] = Field(default=None, max_length=120)
    assigned_to_agent: Optional[str] = Field(default=None, max_length=64)
    cite_chunk_ids: List[int] = Field(default_factory=list, max_length=20)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractedDecision(BaseModel):
    """A decision recorded in the document (typical for meeting notes
    and proposals). Kept separate from action items because decisions
    don't always have an owner."""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., max_length=500)
    cite_chunk_ids: List[int] = Field(default_factory=list, max_length=20)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractedDate(BaseModel):
    """A named date — deadline, milestone, signing date."""
    model_config = ConfigDict(extra="forbid")

    date: Optional[datetime] = None
    raw_phrase: str = Field(..., max_length=200)
    what: str = Field(..., max_length=200)
    cite_chunk_ids: List[int] = Field(default_factory=list, max_length=20)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractedPerson(BaseModel):
    """A named person referenced in the document."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., max_length=120)
    role: Optional[str] = Field(default=None, max_length=120)
    email: Optional[str] = Field(default=None, max_length=240)
    cite_chunk_ids: List[int] = Field(default_factory=list, max_length=20)


Sentiment = Literal["positive", "neutral", "negative"]


class DocumentExtraction(BaseModel):
    """Stage C's full output — the structured view of the document."""
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., max_length=4000)
    action_items: List[ExtractedActionItem] = Field(default_factory=list)
    key_decisions: List[ExtractedDecision] = Field(default_factory=list)
    key_dates: List[ExtractedDate] = Field(default_factory=list)
    key_people: List[ExtractedPerson] = Field(default_factory=list)
    sentiment: Sentiment = "neutral"
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────
# Stage D — self-evaluation
# ─────────────────────────────────────────────────────────────────────


class SelfEvalResult(BaseModel):
    """LLM-as-judge evaluating the Stage C output against the source.

    ``score`` < 0.7 triggers one retry of Stage C with stricter
    settings (temperature=0, prompt asking for higher confidence
    threshold). If the retry also scores < 0.7, the agent flags the
    result as ``low_confidence=true`` so the digest can render a
    "review carefully" tag instead of a one-click approve.
    """
    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0.0, le=1.0)
    grounded: bool = True
    issues: List[str] = Field(default_factory=list, max_length=20)
    recommendations: List[str] = Field(default_factory=list, max_length=10)
    flagged_action_indices: List[int] = Field(default_factory=list, max_length=50)


# ─────────────────────────────────────────────────────────────────────
# Final pipeline result (what `process_async` augments its return with)
# ─────────────────────────────────────────────────────────────────────


class ExtractionResult(BaseModel):
    """Top-level result the agent returns alongside the backward-compatible
    ``analysis`` / ``tasks`` / ``metadata`` keys.

    Cached in MongoDB ``document_extractions`` keyed by
    ``sha256(content) + extractor_version``. Re-uploads of the same
    content return cache hits at zero LLM cost.
    """
    model_config = ConfigDict(extra="forbid")

    extraction_id: str
    content_hash: str
    extractor_version: str
    classification: DocumentClassification
    chunk_summaries: List[ChunkSummary] = Field(default_factory=list)
    extraction: DocumentExtraction
    self_eval: Optional[SelfEvalResult] = None
    confidence: float = 0.0
    low_confidence: bool = False
    chunk_count: int = 0
    duration_ms: Optional[int] = None
    cached: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
