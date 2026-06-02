from .base_agent import BaseAgent
from backend.ai_models import LLMConfig
from typing import Dict, Any, List, Optional
import json
import structlog
import asyncio
from datetime import datetime, timedelta
import re

# Configure logger
logger = structlog.get_logger(__name__)

class DocumentAgent(BaseAgent):
    """Agent for processing documents using LLM providers.
    
    This agent extracts key information, tasks, dates, and insights from 
    documents using the provider-agnostic LLM interface.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default extraction targets if not specified in config
        self.extraction_targets = config.get("extraction_targets", [
            "tasks", "dates", "names", "organizations", 
            "monetary amounts", "action items", "key points"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the document with natural language questions.
        
        Args:
            query: The natural language query about the document
            context: Optional context including document content and metadata
            
        Returns:
            Dictionary containing the response and any relevant extracted information
        """
        if not context or not context.get("document_text"):
            return {"error": "No document content provided in context"}
            
        document_text = context["document_text"]
        prompt = (
            f"Using the following document content, answer this question: {query}\n\n"
            f"Document content:\n{document_text[:8000]}..."  # Limit document length
        )
        
        try:
            if self.llm_client:
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm_client.generate(messages)
                return {
                    "response": response.content,
                    "query": query,
                    "document_id": context.get("document_id"),
                    "confidence": 0.0
                }
            else:
                return {"error": "LLM client not initialized"}
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            return {"error": f"Failed to process query: {str(e)}"}

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process document data asynchronously.

        Extracts key information **and** structured tasks from the document.

        Args:
            data: Dictionary containing document text and metadata

        Returns:
            Dictionary with ``analysis``, ``tasks`` list, and metadata.
        """
        try:
            document_text = data.get("text", "")
            document_metadata = data.get("metadata", {})
            user_context = data.get("user_context", {})

            if not document_text:
                return {"error": "No document text provided"}

            if not self.llm_client:
                return {"error": "LLM client not initialized"}

            # Truncate to fit context window
            text_for_llm = document_text[:12000]

            # --- 1. General extraction ---
            extraction_prompt = (
                f"Analyze the following document and extract key information including "
                f"{', '.join(self.extraction_targets)}. For each extracted item, include "
                f"the exact text from the document and the relevant context.\n\n"
                f"Document content:\n{text_for_llm}"
            )

            # --- 2. Task extraction (structured JSON) ---
            # Phase 4: inject current datetime and tighten the date schema so
            # downstream reminder/calendar code never has to guess a format.
            now_utc = datetime.utcnow()
            user_tz = (user_context or {}).get("timezone", "UTC")
            now_iso = now_utc.replace(microsecond=0).isoformat() + "Z"
            week_from_now_iso = (now_utc + timedelta(days=7)).replace(microsecond=0).isoformat() + "Z"
            five_days_iso = (now_utc + timedelta(days=5)).replace(microsecond=0).isoformat() + "Z"

            # Phase 6: pull the agent capability registry so the LLM can
            # optionally assign each task to one of the 21 specialised agents.
            # One source of truth — backend/agents/router.py.
            try:
                from backend.agents.router import AGENT_REGISTRY as _AGENT_REGISTRY
                agent_lines = "\n".join(
                    f'    - "{k}": {v}' for k, v in _AGENT_REGISTRY.items()
                )
            except Exception:
                agent_lines = "    (agent registry unavailable)"

            task_prompt = (
                "You are extracting actionable tasks from a document for a user's "
                "task manager.  Return ONLY a JSON array (no prose, no markdown).\n\n"
                f"Current time (UTC):       {now_iso}\n"
                f"User timezone:            {user_tz}\n"
                f"Default if no due hint:   {five_days_iso}\n"
                f"Maximum allowed due_date: {week_from_now_iso}\n\n"
                "Schema per element:\n"
                '  "title": short, imperative task title (verb + object),\n'
                '  "description": fuller description of what needs to be done,\n'
                '  "priority": "low" | "medium" | "high" | "critical",\n'
                '  "due_date": ISO-8601 UTC datetime string OR null,\n'
                '  "deadline": exact phrase from the document if any (or null),\n'
                '  "inferred_due_date": true if due_date was NOT explicitly stated,\n'
                '  "assignee": person responsible if mentioned (or null),\n'
                '  "assigned_to_agent": one of the agent keys below OR null\n\n'
                "Available Lumicoria agents (key: capability):\n"
                f"{agent_lines}\n\n"
                "Rules for assigned_to_agent:\n"
                "  • Pick the single agent whose capability best fits the task action.\n"
                "  • If the task is a meeting / scheduling / action-item rollup, choose 'meeting'.\n"
                "  • If the task is contract / clause / legal review, choose 'legal_document'.\n"
                "  • If the task is summarise / extract from an uploaded doc, choose 'rag' or 'document'.\n"
                "  • If the task is research / fact-finding, choose 'research'.\n"
                "  • If the task is draft / write / brainstorm content, choose 'creative'.\n"
                "  • If the task is translate, choose 'translation'.\n"
                "  • If the task is data / CSV / analytics, choose 'data_analysis'.\n"
                "  • If the task requires only human judgment (calling a person, paying a bill, signing), set null.\n"
                "  • Set null when no agent clearly fits — DO NOT guess.\n\n"
                "Rules for due_date (MANDATORY — NEVER return null):\n"
                "  • Every task MUST have a non-null due_date in ISO-8601 UTC.\n"
                "  • If the document gives an absolute date, parse it into ISO-8601 UTC.\n"
                "  • If language signals urgency (urgent, ASAP, today) → within 24 hours.\n"
                "  • If language says 'this week' / 'by EOW' → end of the current week.\n"
                "  • If language says 'next week' → 7 days from now.\n"
                f"  • If NO date signal exists, set due_date to exactly {five_days_iso} (5 days from now).\n"
                f"  • due_date MUST NOT exceed {week_from_now_iso} (1-week cap).\n"
                "  • Set inferred_due_date=true unless an exact date appeared in the text.\n"
                "  • Returning null / empty / missing due_date is INVALID and the task will be rejected.\n\n"
                "Rules for priority:\n"
                "  • 'critical': blocks the business, security, regulatory, immediate.\n"
                "  • 'high': named owner + tight deadline.\n"
                "  • 'medium': default.\n"
                "  • 'low': nice-to-have, vague timing.\n\n"
                "Only include genuinely actionable items.  No status updates, no meta.\n"
                "If there are no tasks, return [].\n\n"
                f"Document content:\n{text_for_llm}"
            )

            # Run both prompts concurrently
            extraction_coro = self.llm_client.generate(
                [{"role": "user", "content": extraction_prompt}]
            )
            tasks_coro = self.llm_client.generate(
                [{"role": "user", "content": task_prompt}]
            )

            extraction_response, tasks_response = await asyncio.gather(
                extraction_coro, tasks_coro
            )

            # Parse the tasks JSON from the LLM response
            tasks = self._parse_tasks_json(tasks_response.content)

            return {
                "analysis": extraction_response.content,
                "tasks": tasks,
                "metadata": document_metadata,
                "extraction_targets": self.extraction_targets,
                "model_used": self.model_config.get("model", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Failed to process document: {str(e)}"}

    def _normalize_task_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: normalise one extracted task into the canonical schema.

        Coerces `due_date` to ISO-8601 UTC, falls back to a +5-day default
        when missing/unparseable, caps at +7 days, and lifts `deadline`
        into `due_date` when the LLM put a structured value there instead.
        """
        if not isinstance(raw, dict):
            return raw  # let the caller drop non-dict entries

        now = datetime.utcnow().replace(microsecond=0)
        max_due = now + timedelta(days=7)
        default_due = now + timedelta(days=5, hours=0)

        # 1. Resolve a candidate datetime from due_date OR deadline.
        candidate_str = raw.get("due_date") or raw.get("deadline")
        parsed_due: Optional[datetime] = None
        if candidate_str:
            parsed_due = self._coerce_to_datetime(str(candidate_str))

        inferred = bool(raw.get("inferred_due_date"))
        if parsed_due is None:
            parsed_due = default_due
            inferred = True

        # 2. Cap at +7 days (the system promises max-7-day due dates).
        if parsed_due > max_due:
            parsed_due = max_due
            inferred = True

        # 3. Never schedule in the past — bump to +1 hour from now.
        if parsed_due < now:
            parsed_due = now + timedelta(hours=1)
            inferred = True

        priority = str(raw.get("priority", "medium")).lower()
        if priority not in {"low", "medium", "high", "critical"}:
            priority = "medium"

        # Phase 6: validate `assigned_to_agent` against the registry; drop
        # anything the LLM hallucinated.
        suggested_agent = raw.get("assigned_to_agent")
        if isinstance(suggested_agent, str):
            suggested_agent = suggested_agent.strip().lower() or None
            try:
                from backend.agents.router import AGENT_REGISTRY as _AGENT_REGISTRY
                if suggested_agent not in _AGENT_REGISTRY:
                    suggested_agent = None
            except Exception:
                suggested_agent = None
        else:
            suggested_agent = None

        normalized: Dict[str, Any] = {
            "title": str(raw.get("title") or "Untitled task").strip()[:280],
            "description": str(raw.get("description") or "").strip(),
            "priority": priority,
            # `deadline` stays as the human-readable original phrase for UI
            "deadline": raw.get("deadline") if isinstance(raw.get("deadline"), str) else None,
            "due_date": parsed_due.isoformat() + "Z",
            "inferred_due_date": inferred,
            "assignee": raw.get("assignee") if isinstance(raw.get("assignee"), str) else None,
            "assigned_to_agent": suggested_agent,
        }
        return normalized

    def _coerce_to_datetime(self, value: str) -> Optional[datetime]:
        """Parse a wide range of date formats into a naive UTC datetime.

        Order of attempts:
          1. ISO-8601 (with or without trailing Z).
          2. python-dateutil best-effort (handles 'May 1, 2026', 'Mon 14:00', etc.).
          3. Embedded YYYY-MM-DD substring.
          4. None — caller will fall back to the +5-day default.
        """
        s = (value or "").strip()
        if not s:
            return None
        # Strip trailing 'Z' since fromisoformat (≤3.10) doesn't grok it
        iso_candidate = s[:-1] if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(iso_candidate)
            if dt.tzinfo is not None:
                # Drop tz and treat as UTC for downstream consistency.
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            pass
        # dateutil
        try:
            from dateutil import parser as date_parser  # type: ignore
            return date_parser.parse(s, fuzzy=True, default=datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0))
        except Exception:
            pass
        # Embedded YYYY-MM-DD
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 9, 0, 0)
            except ValueError:
                pass
        return None

    def _parse_tasks_json(self, raw_text: str) -> List[Dict[str, Any]]:
        """Best-effort parse of an LLM response into a list of task dicts.

        Phase 4: every returned task is normalised via `_normalize_task_record`
        so the downstream consumer (task creation + reminders) gets a
        consistent ISO `due_date` and bounded fields.
        """
        candidates: List[Any] = []

        # Try direct JSON parse first
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                candidates = parsed
            elif isinstance(parsed, dict) and "tasks" in parsed:
                candidates = parsed["tasks"]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from markdown code fences
        if not candidates:
            json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw_text)
            if json_match:
                try:
                    candidates = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        # Try finding any JSON array in the text
        if not candidates:
            bracket_match = re.search(r"\[[\s\S]*\]", raw_text)
            if bracket_match:
                try:
                    candidates = json.loads(bracket_match.group(0))
                except json.JSONDecodeError:
                    pass

        # Last resort: fall back to the existing regex parser, then normalise.
        if not candidates:
            candidates = self._parse_tasks(raw_text)

        # Normalise each entry — drops malformed dicts silently.
        normalized: List[Dict[str, Any]] = []
        for entry in candidates or []:
            if not isinstance(entry, dict):
                continue
            try:
                normalized.append(self._normalize_task_record(entry))
            except Exception:  # pragma: no cover — defensive
                continue
        return normalized

    # NB: a second copy of the original method body lives below for
    # backwards-compatibility with any caller that imports it directly.
    # The new implementation above supersedes it.
    def _parse_tasks_json_legacy(self, raw_text: str) -> List[Dict[str, Any]]:
        """Original parser preserved for back-compat (no normalisation)."""
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "tasks" in parsed:
                return parsed["tasks"]
        except json.JSONDecodeError:
            pass
        json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding any JSON array in the text
        bracket_match = re.search(r"\[[\s\S]*\]", raw_text)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: fall back to the existing regex parser
        return self._parse_tasks(raw_text)
            
    def process(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document to extract key information and generate tasks.
        
        Args:
            document_data: Dictionary containing document text, metadata, and context.
            
        Returns:
            Dictionary with extracted information, tasks, and insights.
        """
        # Extract document content and metadata
        document_text = document_data.get("text", "")
        document_metadata = document_data.get("metadata", {})
        user_context = document_data.get("user_context", {})
        
        if not document_text:
            return {"error": "No document text provided"}
            
        # Use the configured model to process the document
        prompt = (
            f"Analyze the following document and extract key information including "
            f"{', '.join(self.extraction_targets)}. For each extracted item, include "
            f"the exact text from the document and the relevant context."
        )
        
        try:
            # Get basic extraction first
            extraction_result = self._call_model(
                prompt=prompt, 
                document=document_text,
                model=self.model_config.get("model")
            )
            
            # Generate tasks separately for better specialization
            tasks_prompt = (
                f"Extract all tasks, deadlines, and action items from this document. "
                f"Format them as a list where each task includes a title, priority (High/Medium/Low), "
                f"deadline (if available), and responsible party (if mentioned). "
                f"Only include actionable items that require someone to do something."
            )
            
            # This could run in parallel with extraction in a real implementation
            tasks_result = self._call_model(
                prompt=tasks_prompt,
                document=document_text,
                model=self.model_config.get("model")
            )
            
            # Parse tasks into structured format
            parsed_tasks = self._parse_tasks(tasks_result)
            
            # Create comprehensive response
            result = {
                "document_id": document_metadata.get("id", ""),
                "extracted_info": self._parse_extraction(extraction_result),
                "tasks": parsed_tasks,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Document processing failed: {str(e)}"}

    def _parse_extraction(self, extraction_text: str) -> Dict[str, Any]:
        """Parse the extraction results into a structured format.
        
        Args:
            extraction_text: Raw text from the model extraction
            
        Returns:
            Structured dictionary with categorized extractions
        """
        try:
            # Try to parse as JSON first (in case model returned JSON)
            try:
                return json.loads(extraction_text)
            except json.JSONDecodeError:
                # If not JSON, use regex-based parsing
                pass
            
            # Initialize categories
            extraction_dict = {
                "dates": [],
                "names": [],
                "organizations": [],
                "monetary_amounts": [],
                "action_items": [],
                "key_points": [],
                "raw_extraction": extraction_text
            }
            
            # Extract dates with regex
            date_patterns = [
                r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",  # MM/DD/YYYY or DD/MM/YYYY
                r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",  # 1 Jan 2024
                r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4})\b",  # January 1st, 2024
            ]
            
            for pattern in date_patterns:
                for match in re.finditer(pattern, extraction_text, re.IGNORECASE):
                    if match.group(1) not in [d["text"] for d in extraction_dict["dates"]]:
                        extraction_dict["dates"].append({"text": match.group(1)})
            
            # Use section-based parsing for the rest
            sections = {
                "Names:": "names",
                "Organizations:": "organizations",
                "Monetary Amounts:": "monetary_amounts",
                "Action Items:": "action_items",
                "Key Points:": "key_points",
                "Tasks:": "action_items"
            }
            
            lines = extraction_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line is a section header
                for header, section_name in sections.items():
                    if line.startswith(header) or line == header:
                        current_section = section_name
                        break
                
                # If we're in a section and this isn't a header, it's an item
                if current_section and not any(line.startswith(h) for h in sections.keys()):
                    # Clean up bullet points and other markers
                    item_text = re.sub(r"^[-•*]\s*", "", line)
                    if item_text and len(item_text) > 3:  # Avoid very short items
                        if item_text not in [i["text"] for i in extraction_dict[current_section]]:
                            extraction_dict[current_section].append({"text": item_text})
            
            return extraction_dict
            
        except Exception as e:
            logger.error(f"Error parsing extraction: {str(e)}")
            return {"raw_extraction": extraction_text, "parsing_error": str(e)}

    def _parse_tasks(self, tasks_text: str) -> List[Dict[str, Any]]:
        """Parse the tasks extraction into a structured list.
        
        Args:
            tasks_text: Raw text from the model's task extraction
            
        Returns:
            List of structured task dictionaries
        """
        try:
            # Try to parse as JSON first
            try:
                parsed_json = json.loads(tasks_text)
                if isinstance(parsed_json, list):
                    return parsed_json
                elif isinstance(parsed_json, dict) and "tasks" in parsed_json:
                    return parsed_json["tasks"]
            except json.JSONDecodeError:
                # If not JSON, use regex-based parsing
                pass
                
            # Parse free-text format
            tasks = []
            current_task = {}
            
            # Detect bullet points or numbered items
            tasks_pattern = r"(?:^|\n)(?:\d+[\.\)]\s*|[-•*]\s*|Task\s*\d+:\s*|)([A-Z].*?)(?:\n|$)"
            
            for match in re.finditer(tasks_pattern, tasks_text, re.MULTILINE):
                task_text = match.group(1).strip()
                
                # Skip empty or very short tasks
                if not task_text or len(task_text) < 5:
                    continue
                    
                task = {"title": task_text}
                
                # Extract priority if present
                priority_match = re.search(r"\b(High|Medium|Low)\s+priority\b", task_text, re.IGNORECASE)
                if priority_match:
                    task["priority"] = priority_match.group(1).lower()
                
                # Extract deadline if present
                deadline_patterns = [
                    r"due\s+by\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"due\s+on\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                    r"deadline:\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"by\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                ]
                
                for pattern in deadline_patterns:
                    deadline_match = re.search(pattern, task_text, re.IGNORECASE)
                    if deadline_match:
                        task["deadline"] = deadline_match.group(1)
                        break
                
                # Extract responsible party if present
                assignee_patterns = [
                    r"assigned\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                    r"responsible:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                ]
                
                for pattern in assignee_patterns:
                    assignee_match = re.search(pattern, task_text, re.IGNORECASE)
                    if assignee_match:
                        task["assignee"] = assignee_match.group(1)
                        break
                
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error parsing tasks: {str(e)}")
            return [{"title": tasks_text, "parsing_error": str(e)}]
