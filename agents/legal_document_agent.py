from typing import Dict, Any, List, Optional
from enum import Enum
import json
import logging
import re
from datetime import datetime

from .base_agent import BaseAgent
# Removing circular import - agent_service already imports legal_document_agent

logger = logging.getLogger(__name__)

class LegalDocumentMode(str, Enum):
    """Enum for different legal document processing modes."""
    CLAUSE_EXTRACTION = "clause_extraction"
    RISK_ANALYSIS = "risk_analysis"
    VERSION_COMPARISON = "version_comparison"
    PLAIN_LANGUAGE = "plain_language"
    COMPLIANCE_CHECK = "compliance_check"

class LegalDocumentAgent(BaseAgent):
    """Agent specialized in processing legal documents, contracts, and regulatory content."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Legal Document Agent with specific capabilities."""
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "clause_extraction": True,
            "risk_analysis": True,
            "version_comparison": True,
            "plain_language_summary": True,
            "compliance_checking": True
        }
        
        # Configure analysis modes and parameters
        self.modes = {
            LegalDocumentMode.CLAUSE_EXTRACTION: {
                "description": "Extract key clauses, obligations, and deadlines",
                "parameters": {
                    "include_metadata": True,
                    "highlight_obligations": True,
                    "extract_dates": True
                }
            },
            LegalDocumentMode.RISK_ANALYSIS: {
                "description": "Analyze potential risks and unusual terms",
                "parameters": {
                    "risk_threshold": 0.7,
                    "include_recommendations": True,
                    "categorize_risks": True
                }
            },
            LegalDocumentMode.VERSION_COMPARISON: {
                "description": "Compare contract versions and identify changes",
                "parameters": {
                    "track_changes": True,
                    "highlight_additions": True,
                    "highlight_deletions": True,
                    "summarize_changes": True
                }
            },
            LegalDocumentMode.PLAIN_LANGUAGE: {
                "description": "Provide summaries in plain language",
                "parameters": {
                    "simplify_terms": True,
                    "include_examples": True,
                    "maintain_legal_accuracy": True
                }
            },
            LegalDocumentMode.COMPLIANCE_CHECK: {
                "description": "Check regulatory compliance",
                "parameters": {
                    "jurisdiction": "global",
                    "industry_specific": True,
                    "include_citations": True,
                    "severity_levels": ["critical", "high", "medium", "low"]
                }
            }
        }
        
        # Set default model configuration
        # 16k token ceiling because Gemini 2.5 spends part of its budget
        # on internal reasoning before any text reaches the wire, and
        # Claude's clause / risk outputs are typically long structured
        # JSON that gets truncated under a 4k cap.
        self.model_config.update({
            "temperature": 0.3,
            "max_tokens": 16384,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        })

    async def _process_with_model(
        self,
        system_prompt: str,
        user_payload: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Adapter onto BaseAgent._call_model_async with legal defaults.

        The agent's mode handlers call `self._process_with_model(
        system_prompt, document_text, parameters)`, but BaseAgent
        exposes `_call_model_async(prompt, system_prompt=..., ...)`.
        This thin shim bridges the two without rewriting every handler.
        """
        parameters = parameters or {}
        temperature = parameters.get(
            "temperature", self.model_config.get("temperature", 0.3)
        )
        max_tokens = parameters.get(
            "max_tokens", self.model_config.get("max_tokens", 16384)
        )
        response = await self._call_model_async(
            prompt=user_payload or "",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Stash the raw response so the summary / recommendation helpers
        # can read the model's own narrative fields (`summary`,
        # `recommendations`, etc.) without re-running the LLM.
        self._last_response = response if isinstance(response, str) else str(response or "")
        return response

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a legal document analysis request asynchronously."""
        try:
            mode = request.get("mode", LegalDocumentMode.CLAUSE_EXTRACTION.value)
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Validate mode
            if mode not in [m.value for m in LegalDocumentMode]:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Process based on mode
            if mode == LegalDocumentMode.CLAUSE_EXTRACTION.value:
                result = await self._extract_clauses(data, context, parameters)
            elif mode == LegalDocumentMode.RISK_ANALYSIS.value:
                result = await self._analyze_risks(data, context, parameters)
            elif mode == LegalDocumentMode.VERSION_COMPARISON.value:
                result = await self._compare_versions(data, context, parameters)
            elif mode == LegalDocumentMode.PLAIN_LANGUAGE.value:
                result = await self._generate_plain_language(data, context, parameters)
            elif mode == LegalDocumentMode.COMPLIANCE_CHECK.value:
                result = await self._check_compliance(data, context, parameters)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            return {
                "results": result,
                "metadata": {
                    "mode": mode,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": self.model_config.get("model", "sonar-large-online"),
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing legal document request: {str(e)}")
            return {"error": str(e)}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the legal document agent asynchronously."""
        return await self.process_async({
            "mode": LegalDocumentMode.PLAIN_LANGUAGE.value,
            "data": {"document": query},
            "context": context or {}
        })

    async def _extract_clauses(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key clauses, obligations, and deadlines from legal documents."""
        try:
            # Prepare system prompt for clause extraction
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.CLAUSE_EXTRACTION,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response.  The same JSON envelope
            # also carries obligations, deadlines, and free-form dates;
            # surface all of them so the frontend has a rich result.
            clauses = self._parse_clauses(response)
            obligations = self._parse_obligations(response)
            deadlines = self._parse_deadlines(response)
            dates = self._parse_dates(response)

            return {
                "clauses": clauses,
                "obligations": obligations,
                "deadlines": deadlines,
                "dates": dates,
                "metadata": {
                    "total_clauses": len(clauses),
                    "total_obligations": len(obligations),
                    "total_deadlines": len(deadlines),
                    "document_type": self._inferred_document_type(response, data.get("document_type")),
                    "extraction_confidence": self._calculate_confidence(response),
                },
            }

        except Exception as e:
            logger.error(f"Error extracting clauses: {str(e)}")
            raise

    async def _analyze_risks(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze potential risks and unusual terms in legal documents."""
        try:
            # Prepare system prompt for risk analysis
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.RISK_ANALYSIS,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response
            risks = self._parse_risks(response)
            
            return {
                "risks": risks,
                "summary": self._generate_risk_summary(risks),
                "recommendations": self._generate_risk_recommendations(risks),
                "metadata": {
                    "risk_level": self._calculate_risk_level(risks),
                    "document_type": self._inferred_document_type(response, data.get("document_type")),
                    "analysis_confidence": self._calculate_confidence(response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risks: {str(e)}")
            raise

    async def _compare_versions(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare different versions of legal documents and identify changes."""
        try:
            # Prepare system prompt for version comparison
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.VERSION_COMPARISON,
                context,
                parameters
            )
            
            # Process both versions
            old_version = data.get("old_version", "")
            new_version = data.get("new_version", "")
            
            response = await self._process_with_model(
                system_prompt,
                f"Old Version:\n{old_version}\n\nNew Version:\n{new_version}",
                parameters
            )
            
            # Parse and structure the response
            changes = self._parse_changes(response)
            
            return {
                "changes": changes,
                "summary": self._generate_change_summary(changes),
                "metadata": {
                    "total_changes": len(changes),
                    "document_type": self._inferred_document_type(response, data.get("document_type")),
                    "comparison_confidence": self._calculate_confidence(response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing versions: {str(e)}")
            raise

    async def _generate_plain_language(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate plain language summaries of legal documents."""
        try:
            # Prepare system prompt for plain language summary
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.PLAIN_LANGUAGE,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response
            summary = self._parse_summary(response)
            
            return {
                "summary": summary,
                "key_points": self._extract_key_points(summary),
                "metadata": {
                    "document_type": self._inferred_document_type(response, data.get("document_type")),
                    "summary_confidence": self._calculate_confidence(response),
                    "reading_level": self._calculate_reading_level(summary)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating plain language summary: {str(e)}")
            raise

    async def _check_compliance(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check regulatory compliance based on document content."""
        try:
            # Prepare system prompt for compliance check
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.COMPLIANCE_CHECK,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response
            compliance_issues = self._parse_compliance_issues(response)
            
            return {
                "compliance_issues": compliance_issues,
                "summary": self._generate_compliance_summary(compliance_issues),
                "recommendations": self._generate_compliance_recommendations(compliance_issues),
                "metadata": {
                    "jurisdiction": parameters.get("jurisdiction", "global"),
                    "industry": data.get("industry", "unknown"),
                    "compliance_confidence": self._calculate_confidence(response),
                    "severity_levels": self._calculate_severity_levels(compliance_issues)
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking compliance: {str(e)}")
            raise

    def _create_system_prompt(
        self,
        mode: LegalDocumentMode,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Build a strict-JSON system prompt per mode.  Every prompt
        ends with a schema the parser knows how to read; without this,
        the LLM picks its own shape and the frontend sees nothing."""
        ctx = (
            f"Document type: {context.get('document_type', 'unknown')}\n"
            f"Jurisdiction: {context.get('jurisdiction', 'global')}\n"
            f"Industry: {context.get('industry', 'general')}"
        )

        if mode == LegalDocumentMode.CLAUSE_EXTRACTION:
            return f"""You are a legal document analyst.  Read the user's document and extract its key clauses, obligations, and dates.

{ctx}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "inferred_document_type": "<NDA | service agreement | employment contract | lease | privacy policy | terms of service | constitution | statute | court ruling | other — infer from the content>",
  "clauses": [
    {{
      "name": "<short clause name>",
      "reference": "<section / article / paragraph reference, or empty>",
      "text": "<the clause text or a faithful summary>",
      "type": "<obligation | right | condition | definition | term | other>",
      "parties": ["<party names referenced, if any>"],
      "confidence": 0.0
    }}
  ],
  "obligations": [
    {{
      "party": "<who must act>",
      "duty": "<what they must do>",
      "deadline": "<deadline if present, or empty>",
      "source_clause": "<reference back to a clause name, or empty>"
    }}
  ],
  "deadlines": [
    {{"event": "<what is due>", "date": "<date or duration>"}}
  ],
  "dates": ["<every date mentioned, free form>"]
}}

Rules:
- Be exhaustive but concise.  Up to 25 clauses, 25 obligations, 30 deadlines.
- Confidence is a number between 0 and 1.
- If a section is empty, return an empty array — never omit a key.
- Always infer `inferred_document_type` from the content; never leave it empty.
"""

        if mode == LegalDocumentMode.RISK_ANALYSIS:
            return f"""You are a legal risk analyst.  Identify risky, unusual, or unfavourable terms in the document.

{ctx}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "inferred_document_type": "<infer from the content>",
  "risks": [
    {{
      "title": "<short risk name>",
      "severity": "<critical | high | medium | low>",
      "category": "<liability | indemnification | termination | payment | ip | data | confidentiality | other>",
      "description": "<plain-English explanation of why this is a risk>",
      "clause_reference": "<which clause it lives in, or empty>",
      "recommendation": "<concrete suggestion to mitigate>",
      "confidence": 0.0
    }}
  ],
  "summary": "<one paragraph overview of the risk profile>",
  "overall_risk_level": "<critical | high | medium | low>",
  "recommendations": ["<top-level recommendations, beyond per-risk advice>"]
}}

Rules:
- Up to 25 risks, ordered most severe first.
- Severity is one of the four values above, lowercase, exactly.
- Always infer `inferred_document_type` from the content; never leave it empty.
"""

        if mode == LegalDocumentMode.VERSION_COMPARISON:
            return f"""You compare two versions of a legal document and report what changed.

{ctx}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "inferred_document_type": "<infer from the content of either version>",
  "changes": [
    {{
      "type": "<addition | deletion | modification>",
      "section": "<section / clause reference>",
      "old_text": "<excerpt from the old version, or empty for additions>",
      "new_text": "<excerpt from the new version, or empty for deletions>",
      "impact": "<plain-English explanation of why this change matters>",
      "severity": "<critical | high | medium | low>"
    }}
  ],
  "summary": "<one paragraph overview of what changed and why it matters>",
  "additions_count": 0,
  "deletions_count": 0,
  "modifications_count": 0
}}

Rules:
- Up to 40 changes, ordered most impactful first.
- Use the same wording as the source documents where possible.
- Always infer `inferred_document_type`; never leave it empty.
"""

        if mode == LegalDocumentMode.PLAIN_LANGUAGE:
            return f"""You translate dense legal language into plain English without losing accuracy.

{ctx}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "inferred_document_type": "<infer from the content>",
  "summary": "<two to four paragraphs of plain-English summary>",
  "key_points": ["<short bullet points capturing the main ideas>"],
  "obligations_in_plain_english": [
    {{"who": "<party>", "what": "<duty in plain English>", "when": "<deadline or empty>"}}
  ],
  "watch_outs": ["<things the reader should pay extra attention to>"],
  "glossary": [
    {{"term": "<legal term>", "plain": "<plain English definition>"}}
  ]
}}

Rules:
- Aim for an 8th-grade reading level in the summary and key_points.
- Up to 10 key points, 15 obligations, 10 watch-outs, 12 glossary entries.
- Always infer `inferred_document_type`; never leave it empty.
"""

        if mode == LegalDocumentMode.COMPLIANCE_CHECK:
            return f"""You check a legal document for regulatory and contractual compliance issues.

{ctx}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "inferred_document_type": "<infer from the content>",
  "issues": [
    {{
      "title": "<short issue name>",
      "severity": "<critical | high | medium | low>",
      "regulation": "<the regulation / standard / clause that is being violated>",
      "description": "<plain-English explanation>",
      "clause_reference": "<which clause it relates to>",
      "recommendation": "<concrete fix>",
      "citation": "<formal citation or link, or omit the field entirely if there isn't one>"
    }}
  ],
  "compliant_areas": ["<areas where the document already meets the standard>"],
  "summary": "<one paragraph overview>",
  "recommendations": ["<top-level recommendations>"],
  "overall_compliance": "<compliant | partially_compliant | non_compliant>"
}}

Rules:
- Up to 20 issues, ordered most severe first.
- Always infer `inferred_document_type`; never leave it empty.
- Only include a `citation` when there is a real one — do NOT write "N/A" or "none"; just omit the field.
"""

        # Unknown mode (should be unreachable thanks to process_async's
        # validation) — minimal fallback.
        return f"You are a legal document analyst.\n{ctx}\nRespond in JSON."

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for the system prompt."""
        return "\n".join([f"- {k}: {v}" for k, v in parameters.items()])

    # ── JSON extractor ─────────────────────────────────────────

    @staticmethod
    def _extract_json(response: Any) -> Dict[str, Any]:
        """Best-effort JSON extraction.  Handles ```json``` fences,
        leading prose, and trailing prose.  Returns {} on failure."""
        if not response:
            return {}
        text = response.strip() if isinstance(response, str) else str(response)

        # 1. Plain parse.
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2. Strip ```json``` fences (with or without the language tag).
        fenced = re.search(
            r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

        # 3. Open-fence-only (truncated response): strip leading fence.
        stripped = re.sub(r"^```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(stripped)
        except Exception:
            pass

        # 4. Locate the first balanced {...} block.
        s = stripped.find("{")
        e = stripped.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(stripped[s : e + 1])
            except Exception:
                pass

        logger.warning(
            "legal_llm_non_json len=%s head=%r tail=%r",
            len(text),
            text[:300],
            text[-200:],
        )
        return {}

    def _inferred_document_type(
        self, response: str, fallback: Optional[str] = None
    ) -> str:
        """Read the LLM's inferred document type from the JSON envelope,
        falling back to the caller-supplied value (which is often
        'unknown' for users who didn't pick one) or 'document' as the
        last resort."""
        data = self._extract_json(response)
        for key in (
            "inferred_document_type",
            "document_type",
        ):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            inner = data.get("metadata") if isinstance(data.get("metadata"), dict) else None
            if inner:
                v2 = inner.get(key)
                if isinstance(v2, str) and v2.strip():
                    return v2.strip()
        if fallback and fallback.lower() not in ("", "unknown", "n/a"):
            return fallback
        return "document"

    @staticmethod
    def _coerce_severity(v: Any) -> str:
        if not v:
            return "medium"
        s = str(v).strip().lower()
        return s if s in ("critical", "high", "medium", "low") else "medium"

    @staticmethod
    def _as_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    @staticmethod
    def _as_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)):
            return str(v)
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return ""

    # ── Mode-specific parsers ──────────────────────────────────

    def _parse_clauses(self, response: str) -> List[Dict[str, Any]]:
        """Read clauses from the LLM JSON.  The prompt asks for top-level
        keys `clauses`, `obligations`, `deadlines`, `dates`, but the model
        sometimes wraps them in `extracted_data` or names the first key
        `key_clauses`; accept any of these gracefully."""
        data = self._extract_json(response)
        inner = data.get("extracted_data") or data
        raw = (
            inner.get("clauses")
            or inner.get("key_clauses")
            or inner.get("results")
            or []
        )
        out: List[Dict[str, Any]] = []
        for c in self._as_list(raw):
            if not isinstance(c, dict):
                continue
            out.append({
                "name": self._as_str(c.get("name") or c.get("clause_name") or c.get("title")),
                "reference": self._as_str(c.get("reference") or c.get("clause_reference")),
                "text": self._as_str(c.get("text") or c.get("clause_text") or c.get("content")),
                "type": self._as_str(c.get("type") or c.get("clause_type") or "other"),
                "parties": self._as_list(c.get("parties")),
                "obligation": self._as_str(c.get("obligation")),
                "confidence": float(c.get("confidence", 0.85) or 0.85),
            })
        return out

    def _parse_obligations(self, response: str) -> List[Dict[str, Any]]:
        data = self._extract_json(response)
        inner = data.get("extracted_data") or data
        raw = inner.get("obligations") or []
        out: List[Dict[str, Any]] = []
        for o in self._as_list(raw):
            if not isinstance(o, dict):
                continue
            out.append({
                "party": self._as_str(o.get("party")),
                "duty": self._as_str(o.get("duty") or o.get("description")),
                "deadline": self._as_str(o.get("deadline") or o.get("due")),
                "source_clause": self._as_str(o.get("source_clause") or o.get("clause")),
            })
        return out

    def _parse_deadlines(self, response: str) -> List[Dict[str, Any]]:
        data = self._extract_json(response)
        inner = data.get("extracted_data") or data
        raw = inner.get("deadlines") or []
        out: List[Dict[str, Any]] = []
        for d in self._as_list(raw):
            if isinstance(d, dict):
                out.append({
                    "event": self._as_str(d.get("event") or d.get("description")),
                    "date": self._as_str(d.get("date") or d.get("deadline")),
                })
            elif isinstance(d, str):
                out.append({"event": d, "date": ""})
        return out

    def _parse_dates(self, response: str) -> List[str]:
        data = self._extract_json(response)
        inner = data.get("extracted_data") or data
        raw = inner.get("dates") or []
        return [self._as_str(d) for d in self._as_list(raw) if d]

    def _parse_risks(self, response: str) -> List[Dict[str, Any]]:
        data = self._extract_json(response)
        raw = data.get("risks") or data.get("issues") or []
        out: List[Dict[str, Any]] = []
        for r in self._as_list(raw):
            if not isinstance(r, dict):
                continue
            out.append({
                "title": self._as_str(r.get("title") or r.get("name") or r.get("risk")),
                "severity": self._coerce_severity(r.get("severity")),
                "category": self._as_str(r.get("category") or "other"),
                "description": self._as_str(r.get("description") or r.get("detail")),
                "clause_reference": self._as_str(r.get("clause_reference") or r.get("clause")),
                "recommendation": self._as_str(r.get("recommendation") or r.get("mitigation")),
                "confidence": float(r.get("confidence", 0.85) or 0.85),
            })
        return out

    def _parse_changes(self, response: str) -> List[Dict[str, Any]]:
        data = self._extract_json(response)
        raw = data.get("changes") or data.get("diff") or []
        out: List[Dict[str, Any]] = []
        for c in self._as_list(raw):
            if not isinstance(c, dict):
                continue
            ctype = (c.get("type") or c.get("change_type") or "modification").strip().lower()
            if ctype not in ("addition", "deletion", "modification"):
                ctype = "modification"
            out.append({
                "type": ctype,
                "section": self._as_str(c.get("section") or c.get("clause")),
                "old_text": self._as_str(c.get("old_text") or c.get("before")),
                "new_text": self._as_str(c.get("new_text") or c.get("after")),
                "impact": self._as_str(c.get("impact") or c.get("description")),
                "severity": self._coerce_severity(c.get("severity")),
            })
        return out

    def _parse_summary(self, response: str) -> Dict[str, Any]:
        data = self._extract_json(response)
        return {
            "summary": self._as_str(data.get("summary")),
            "key_points": [self._as_str(k) for k in self._as_list(data.get("key_points"))],
            "obligations": [
                {
                    "who": self._as_str(o.get("who") or o.get("party")) if isinstance(o, dict) else "",
                    "what": self._as_str(o.get("what") or o.get("duty")) if isinstance(o, dict) else self._as_str(o),
                    "when": self._as_str(o.get("when") or o.get("deadline")) if isinstance(o, dict) else "",
                }
                for o in self._as_list(
                    data.get("obligations_in_plain_english") or data.get("obligations")
                )
            ],
            "watch_outs": [self._as_str(w) for w in self._as_list(data.get("watch_outs"))],
            "glossary": [
                {
                    "term": self._as_str(g.get("term")) if isinstance(g, dict) else "",
                    "plain": self._as_str(g.get("plain") or g.get("definition")) if isinstance(g, dict) else self._as_str(g),
                }
                for g in self._as_list(data.get("glossary"))
            ],
        }

    def _parse_compliance_issues(self, response: str) -> List[Dict[str, Any]]:
        data = self._extract_json(response)
        raw = data.get("issues") or data.get("compliance_issues") or []
        out: List[Dict[str, Any]] = []
        for i in self._as_list(raw):
            if not isinstance(i, dict):
                continue
            out.append({
                "title": self._as_str(i.get("title") or i.get("issue")),
                "severity": self._coerce_severity(i.get("severity")),
                "regulation": self._as_str(i.get("regulation") or i.get("standard")),
                "description": self._as_str(i.get("description")),
                "clause_reference": self._as_str(i.get("clause_reference") or i.get("clause")),
                "recommendation": self._as_str(i.get("recommendation") or i.get("fix")),
                "citation": self._as_str(i.get("citation")),
            })
        return out

    # ── Confidence + derived metrics ──────────────────────────

    def _calculate_confidence(self, response: str) -> float:
        """Average the per-item `confidence` fields when present; fall
        back to 0.85 if the LLM didn't emit any."""
        data = self._extract_json(response)
        all_items: List[Any] = []
        for key in ("clauses", "key_clauses", "risks", "issues", "changes"):
            all_items.extend(self._as_list(data.get(key) or (data.get("extracted_data") or {}).get(key)))
        scores = []
        for item in all_items:
            if isinstance(item, dict) and "confidence" in item:
                try:
                    scores.append(float(item["confidence"]))
                except Exception:
                    pass
        if not scores:
            return 0.85 if all_items else 0.0
        return round(sum(scores) / len(scores), 3)

    def _calculate_risk_level(self, risks: List[Dict[str, Any]]) -> str:
        if not risks:
            return "low"
        order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        highest = max(order.get(self._coerce_severity(r.get("severity")), 1) for r in risks)
        return {4: "critical", 3: "high", 2: "medium", 1: "low"}[highest]

    def _calculate_reading_level(self, summary: Dict[str, Any]) -> str:
        text = self._as_str(summary.get("summary"))
        if not text:
            return "n/a"
        words = re.findall(r"\b\w+\b", text)
        sentences = max(1, len(re.findall(r"[.!?]+", text)))
        if not words:
            return "n/a"
        avg_sentence = len(words) / sentences
        if avg_sentence < 14:
            return "easy"
        if avg_sentence < 20:
            return "intermediate"
        return "advanced"

    def _calculate_severity_levels(
        self, issues: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for i in issues or []:
            sev = self._coerce_severity(i.get("severity") if isinstance(i, dict) else None)
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def _generate_risk_summary(self, risks: List[Dict[str, Any]]) -> str:
        """Prefer the model's own summary if it gave one; otherwise
        derive a one-line summary from the parsed risk list."""
        # We store the raw response on `self._last_response` from the
        # mode handler so we can read the LLM's own "summary" field.
        raw = getattr(self, "_last_response", "")
        if raw:
            data = self._extract_json(raw)
            summary = self._as_str(data.get("summary"))
            if summary:
                return summary
        if not risks:
            return "No notable risks identified."
        counts = self._calculate_severity_levels(risks)
        parts = [f"{counts[k]} {k}" for k in ("critical", "high", "medium", "low") if counts.get(k)]
        return f"Identified {len(risks)} risks ({', '.join(parts)})."

    def _generate_risk_recommendations(
        self, risks: List[Dict[str, Any]]
    ) -> List[str]:
        raw = getattr(self, "_last_response", "")
        if raw:
            data = self._extract_json(raw)
            top = [self._as_str(r) for r in self._as_list(data.get("recommendations"))]
            if top:
                return top
        return [r["recommendation"] for r in (risks or []) if r.get("recommendation")][:8]

    def _generate_change_summary(self, changes: List[Dict[str, Any]]) -> str:
        raw = getattr(self, "_last_response", "")
        if raw:
            data = self._extract_json(raw)
            summary = self._as_str(data.get("summary"))
            if summary:
                return summary
        if not changes:
            return "No material differences detected."
        adds = sum(1 for c in changes if c.get("type") == "addition")
        dels = sum(1 for c in changes if c.get("type") == "deletion")
        mods = sum(1 for c in changes if c.get("type") == "modification")
        return f"{len(changes)} changes — {adds} additions, {dels} deletions, {mods} modifications."

    def _generate_compliance_summary(
        self, issues: List[Dict[str, Any]]
    ) -> str:
        raw = getattr(self, "_last_response", "")
        if raw:
            data = self._extract_json(raw)
            summary = self._as_str(data.get("summary"))
            if summary:
                return summary
        if not issues:
            return "No compliance issues detected."
        counts = self._calculate_severity_levels(issues)
        return (
            f"{len(issues)} compliance issues "
            f"({counts['critical']} critical, {counts['high']} high, "
            f"{counts['medium']} medium, {counts['low']} low)."
        )

    def _generate_compliance_recommendations(
        self, issues: List[Dict[str, Any]]
    ) -> List[str]:
        raw = getattr(self, "_last_response", "")
        if raw:
            data = self._extract_json(raw)
            top = [self._as_str(r) for r in self._as_list(data.get("recommendations"))]
            if top:
                return top
        return [i["recommendation"] for i in (issues or []) if i.get("recommendation")][:8]

    def _extract_key_points(self, summary: Dict[str, Any]) -> List[str]:
        return summary.get("key_points") or [] 