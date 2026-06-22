"""
Phase 6 — Autonomous Task Executor.

A Celery-driven service that finds tasks assigned to one of the 21
Lumicoria agents and drafts a proposal for the human to review.

Flow per task:
    1. Pick up tasks with `assigned_to_agent != None` and either no
       `agent_proposal` or `agent_proposal.status == "revision"`.
    2. Call the agent's `context_summary(...)` to gather grounding
       (documents, prior notes, etc.).
    3. Call the agent's `process_async(...)` with the task + grounding.
    4. Persist the result on `Task.agent_proposal` with
       `status="pending_review"`.
    5. Notify the task owner (in-app + push) so they can review.

The executor is idempotent: a task already in PENDING_REVIEW is skipped
until the human approves / requests revision / rejects.

NOTE: Provider-agnostic.  Each agent class chooses its own LLM client
via `_resolve_provider()`.  This service does not pin a provider.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId

logger = structlog.get_logger(__name__)


# ── Local registry of agent_key → concrete agent class ────────────────
#
# We instantiate fresh agents per run to avoid coupling to the YAML-bound
# AgentService singleton in `backend/agents/agent_service.py` (which only
# loads agents declared in `config.yaml`).  This map matches the
# capability keys in `backend/agents/router.py:AGENT_REGISTRY`.

def _agent_class_map():
    from backend.agents.document_agent import DocumentAgent
    from backend.agents.wellbeing_agent import WellbeingAgent
    from backend.agents.meeting_agent import MeetingAgent
    from backend.agents.meeting_fact_checker_agent import MeetingFactCheckerAgent
    from backend.agents.creative_agent import CreativeAgent
    from backend.agents.student_agent import StudentAgent
    from backend.agents.rag_agent import RAGAgent
    from backend.agents.research_agent import ResearchAgent
    from backend.agents.research_mentor_agent import ResearchMentorAgent
    from backend.agents.social_media_agent import SocialMediaAgent
    from backend.agents.legal_document_agent import LegalDocumentAgent
    from backend.agents.learning_coach_agent import LearningCoachAgent
    from backend.agents.knowledge_graph_agent import KnowledgeGraphAgent
    from backend.agents.ethics_bias_agent import EthicsBiasAgent
    from backend.agents.focus_flow_agent import FocusFlowAgent
    from backend.agents.workspace_ergonomics_agent import WorkspaceErgonomicsAgent
    from backend.agents.customer_service_agent import CustomerServiceAgent
    from backend.agents.data_analysis_agent import DataAnalysisAgent
    from backend.agents.translation_agent import TranslationAgent
    from backend.agents.general_agent import GeneralAgent
    from backend.agents.brain_agent import BrainAgent
    try:
        from backend.agents.vision_agent import VisionAgent  # type: ignore
    except Exception:
        VisionAgent = None  # type: ignore

    mapping = {
        "document": DocumentAgent,
        "wellbeing": WellbeingAgent,
        "meeting": MeetingAgent,
        "meeting_fact_checker": MeetingFactCheckerAgent,
        "creative": CreativeAgent,
        "student": StudentAgent,
        "rag": RAGAgent,
        "research": ResearchAgent,
        "research_mentor": ResearchMentorAgent,
        "social_media": SocialMediaAgent,
        "legal_document": LegalDocumentAgent,
        "learning_coach": LearningCoachAgent,
        "knowledge_graph": KnowledgeGraphAgent,
        "ethics_bias": EthicsBiasAgent,
        "focus_flow": FocusFlowAgent,
        "workspace_ergonomics": WorkspaceErgonomicsAgent,
        "customer_service": CustomerServiceAgent,
        "data_analysis": DataAnalysisAgent,
        "translation": TranslationAgent,
        "general": GeneralAgent,
        # Autonomous brain — daily prioritisation. Not surfaced to chat;
        # invoked only by services/brain/nodes/prioritise.py.
        "brain": BrainAgent,
    }
    if VisionAgent:
        mapping["vision"] = VisionAgent
    return mapping


def instantiate_agent(agent_key: str):
    """Create a fresh instance of the agent identified by `agent_key`.

    Returns `None` if the key isn't registered.
    """
    mapping = _agent_class_map()
    agent_class = mapping.get(agent_key)
    if not agent_class:
        return None
    try:
        return agent_class({"type": agent_key, "model_config": {}})
    except Exception as e:  # noqa: BLE001
        logger.warning("agent_instantiation_failed", agent=agent_key, error=str(e))
        return None


# ── Per-task execution ────────────────────────────────────────────────


async def execute_task(task: Any) -> Dict[str, Any]:
    """Run the assigned agent against a single task and persist the result
    as an `agent_proposal` block on the task.

    Returns a small status dict suitable for logging / Celery return.
    """
    from backend.db.mongodb.repositories.task_repository import task_repository
    from backend.db.mongodb.repositories.agent_run_repository import agent_run_repository
    from backend.db.mongodb.models.agent_run import AgentRunCreate, AgentRunTrigger
    from backend.services.notification_service import notification_service
    from backend.db.mongodb.models.notification import (
        NotificationPriority,
        NotificationType,
    )
    from backend.services.activity_logger import log_activity

    task_id = str(getattr(task, "id", "") or "")
    agent_key = getattr(task, "assigned_to_agent", None)
    if not task_id or not agent_key:
        return {"status": "skipped", "reason": "no_agent_or_id"}

    org_id = str(getattr(task, "organization_id", None) or "")
    user_id = str(getattr(task, "assigned_to", None) or getattr(task, "created_by", "") or "")
    title = getattr(task, "title", "") or ""
    description = getattr(task, "description", "") or ""

    # Idempotency: if a proposal already exists in pending_review, skip.
    existing = getattr(task, "agent_proposal", None)
    existing_status = None
    if existing:
        existing_status = (
            existing.get("status") if isinstance(existing, dict)
            else getattr(existing, "status", None)
        )
        if existing_status in ("pending_review", "approved"):
            return {"status": "skipped", "reason": f"already_{existing_status}"}

    agent = instantiate_agent(agent_key)
    if agent is None:
        await task_repository.set_agent_proposal(
            task_id=task_id,
            organization_id=org_id or None,
            proposal={
                "status": "error",
                "content": f"Agent '{agent_key}' is not registered for autonomous execution.",
                "sources": [],
                "created_at": datetime.utcnow(),
            },
        )
        return {"status": "error", "reason": "agent_not_found", "agent": agent_key}

    # Reset usage so we capture only the tokens spent during THIS task run.
    try:
        agent.reset_usage()
    except Exception:
        pass

    query = (title + ("\n\n" + description if description else "")).strip() or "Draft an answer for this task."

    # Open an AgentRun row so analytics can track this invocation.
    run_id: Optional[str] = None
    try:
        payload = AgentRunCreate(
            agent_key=agent_key,
            agent_name=agent_key.replace("_", " ").title(),
            user_id=ObjectId(user_id) if ObjectId.is_valid(user_id) else ObjectId(),
            organization_id=ObjectId(org_id) if ObjectId.is_valid(org_id) else None,
            task_id=ObjectId(task_id) if ObjectId.is_valid(task_id) else None,
            trigger=AgentRunTrigger.TASK_EXECUTOR,
            input={"title": title, "description": description},
        )
        run = await agent_run_repository.start_run(payload)
        run_id = str(getattr(run, "id", "") or "")
    except Exception as e:  # noqa: BLE001
        logger.warning("agent_run_open_failed", task_id=task_id, error=str(e))

    # 1. Gather context via context_summary.
    try:
        context = await agent.context_summary(
            query=query,
            user_id=user_id or None,
            organization_id=org_id or None,
            task_id=task_id,
        )
    except Exception as e:  # noqa: BLE001
        context = {"context_snippets": [], "sources": [], "suggested_prompt": query}
        logger.warning("agent_context_summary_failed", task_id=task_id, error=str(e))

    sources = list(context.get("sources") or [])
    suggested_prompt = context.get("suggested_prompt") or query

    # 2. Run the agent.
    output_text = ""
    output_meta: Dict[str, Any] = {}
    error_message: Optional[str] = None
    try:
        # All BaseAgent subclasses implement process_async.  We pass a
        # uniform payload — each agent uses what it needs.
        agent_input = {
            "text": suggested_prompt,
            "query": query,
            "task_id": task_id,
            "user_id": user_id or None,
            "organization_id": org_id or None,
            "context_snippets": context.get("context_snippets", []),
            "metadata": {"task_title": title},
        }
        result = await agent.process_async(agent_input)
        if isinstance(result, dict):
            output_meta = result
            output_text = (
                result.get("response")
                or result.get("analysis")
                or result.get("summary")
                or result.get("content")
                or result.get("query_response")
                or ""
            )
            if not output_text:
                # Fallback: serialise the first reasonably-sized text-y field.
                for key, val in result.items():
                    if isinstance(val, str) and len(val.strip()) > 20:
                        output_text = val
                        break
        else:
            output_text = str(result)
    except Exception as e:  # noqa: BLE001
        error_message = str(e)[:400]
        logger.error("agent_execution_failed", task_id=task_id, agent=agent_key, error=error_message)

    proposal: Dict[str, Any] = {
        "status": "error" if error_message else "pending_review",
        "content": output_text or (error_message or "No output."),
        "sources": sources,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "agent_run_id": ObjectId(run_id) if run_id and ObjectId.is_valid(run_id) else None,
    }
    if error_message:
        proposal["error"] = error_message

    await task_repository.set_agent_proposal(
        task_id=task_id,
        organization_id=org_id or None,
        proposal=proposal,
    )

    # Read what the agent's wrapped LLM client recorded during this run
    # so we can stamp tokens + cost + credits onto the AgentRun row.
    try:
        usage = agent.consume_usage()
    except Exception:
        usage = {}

    # Convert internal USD cost into user-facing credits.  Users see
    # credits; we keep cost_usd as the internal margin reference.
    from backend.ai_models.pricing import compute_credits
    cost_amount = float(usage.get("cost_usd") or 0.0)
    credits = compute_credits(cost_amount)

    # Close the run.
    try:
        if run_id:
            if error_message:
                await agent_run_repository.fail_run(
                    run_id,
                    error_message,
                    metadata_patch={
                        "model_used": usage.get("model_used"),
                        "provider": usage.get("provider"),
                    } if usage else None,
                )
            else:
                await agent_run_repository.complete_run(
                    run_id,
                    output={
                        "content_preview": (output_text or "")[:600],
                        "sources_count": len(sources),
                    },
                    tokens_input=int(usage.get("prompt_tokens") or 0) or None,
                    tokens_output=int(usage.get("completion_tokens") or 0) or None,
                    cost_usd=cost_amount or None,
                    credits_used=credits or None,
                    metadata_patch={
                        "model_used": usage.get("model_used"),
                        "provider": usage.get("provider"),
                        "llm_calls": usage.get("calls", 0),
                    },
                )
    except Exception:  # noqa: BLE001
        pass

    # 3. Notify the human owner.
    try:
        owner_id = user_id or str(getattr(task, "created_by", "") or "")
        if owner_id and not error_message:
            # Build the deep-link + signed one-tap action URLs so the
            # in-app + push payloads both carry actionable links.
            from backend.core.config import settings as _settings
            from backend.services.task_action_tokens import (
                TaskAction,
                action_url,
            )
            base_url = _settings.PUBLIC_BASE_URL or _settings.FRONTEND_URL or "http://localhost:3000"
            review_url = f"{base_url.rstrip('/')}/tasks?task={task_id}&proposal=review"
            try:
                approve_url = action_url(
                    base_url=base_url,
                    user_id=owner_id,
                    task_id=task_id,
                    action=TaskAction.APPROVE_PROPOSAL,
                )
            except Exception:
                approve_url = None
            try:
                reject_url = action_url(
                    base_url=base_url,
                    user_id=owner_id,
                    task_id=task_id,
                    action=TaskAction.REJECT_PROPOSAL,
                )
            except Exception:
                reject_url = None

            await notification_service.create_in_app_notification(
                user_id=owner_id,
                title="Agent drafted a proposal",
                content=f"{agent_key.replace('_', ' ').title()} drafted a proposal for '{title[:60]}'. Review when ready.",
                notification_type=NotificationType.TASK,
                priority=NotificationPriority.HIGH,
                metadata={
                    "task_id": task_id,
                    "agent_key": agent_key,
                    "action": "agent_proposal_ready",
                    "review_url": review_url,
                    "approve_url": approve_url,
                    "reject_url": reject_url,
                },
            )
            try:
                from backend.services.push_notification_service import push_notification_service
                # Two action buttons + a default click destination.  The
                # service worker reads `event.action` ("review" / "approve")
                # and routes via the matching URL in the data payload.
                push_actions = [{"action": "review", "title": "Review", "url": review_url}]
                if approve_url:
                    push_actions.append({"action": "approve", "title": "Approve", "url": approve_url})

                await push_notification_service.send_to_user(
                    user_id=owner_id,
                    title="Agent proposal ready",
                    body=f"{agent_key.replace('_', ' ').title()}: {title[:60]}",
                    data={
                        "type": "agent_proposal_ready",
                        "task_id": task_id,
                        "agent_key": agent_key,
                        "review_url": review_url,
                        "approve_url": approve_url or "",
                        "reject_url": reject_url or "",
                    },
                    actions=push_actions,
                    click_action=review_url,
                )
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        logger.warning("agent_proposal_notify_failed", task_id=task_id, error=str(e))

    # 4. Activity log for analytics + audit.
    try:
        await log_activity(
            user_id=user_id or "",
            organization_id=org_id or "",
            activity_type="task.agent_proposal_drafted" if not error_message else "task.agent_proposal_error",
            details={
                "task_id": task_id,
                "agent_key": agent_key,
                "sources_count": len(sources),
                "error": error_message,
            },
            related_resource_type="TASK",
            related_resource_id=task_id,
            agent_name=agent_key,
        )
    except Exception:
        pass

    return {
        "status": "ok" if not error_message else "error",
        "task_id": task_id,
        "agent": agent_key,
        "sources": len(sources),
        "error": error_message,
    }


# ── Batch entry point used by Celery ─────────────────────────────────


async def run_pending_proposals(limit: int = 25) -> Dict[str, Any]:
    """Scan for tasks awaiting an agent draft and execute them.

    Designed to be called from a Celery beat task every few minutes.
    Returns a small dict for the Celery result backend.
    """
    from backend.db.mongodb.repositories.task_repository import task_repository

    tasks = await task_repository.find_tasks_for_executor(limit=limit)
    if not tasks:
        return {"picked": 0, "ok": 0, "errors": 0}

    ok = 0
    errors = 0
    for task in tasks:
        try:
            result = await execute_task(task)
            if result.get("status") == "ok":
                ok += 1
            elif result.get("status") == "error":
                errors += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            logger.error("task_executor_loop_error", error=str(e))

    return {"picked": len(tasks), "ok": ok, "errors": errors}


# ── Revision: re-run with human notes ─────────────────────────────────


async def re_run_with_revision_notes(task_id: str, organization_id: str, notes: str) -> Dict[str, Any]:
    """Re-run a task's assigned agent with the human's revision notes
    appended to the prompt.  Idempotent: bumps the proposal's
    `revision_notes` and re-stamps `status='revision'` so the next
    scheduled scan picks it up.
    """
    from backend.db.mongodb.repositories.task_repository import task_repository

    task = await task_repository.get_task_by_id(task_id, organization_id=organization_id)
    if not task:
        return {"status": "not_found"}

    existing_proposal = getattr(task, "agent_proposal", None) or {}
    if not isinstance(existing_proposal, dict):
        try:
            existing_proposal = existing_proposal.dict()
        except Exception:
            existing_proposal = {}

    existing_proposal["status"] = "revision"
    existing_proposal["revision_notes"] = (notes or "").strip()[:2000]
    existing_proposal["updated_at"] = datetime.utcnow()

    await task_repository.set_agent_proposal(
        task_id=task_id,
        organization_id=organization_id,
        proposal=existing_proposal,
    )

    return await execute_task(task)
