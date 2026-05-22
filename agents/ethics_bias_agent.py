from typing import Dict, Any, List, Optional
from enum import Enum
import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass
from uuid import uuid4

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class EthicsCategory(str, Enum):
    """Categories of ethical considerations."""
    FAIRNESS = "fairness"
    PRIVACY = "privacy"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    INCLUSION = "inclusion"
    SAFETY = "safety"
    SUSTAINABILITY = "sustainability"
    HUMAN_RIGHTS = "human_rights"

class BiasType(str, Enum):
    """Types of bias to detect."""
    GENDER = "gender"
    RACIAL = "racial"
    AGE = "age"
    RELIGIOUS = "religious"
    CULTURAL = "cultural"
    SOCIOECONOMIC = "socioeconomic"
    GEOGRAPHIC = "geographic"
    LANGUAGE = "language"
    ABILITY = "ability"
    OCCUPATIONAL = "occupational"

class IssueSeverity(str, Enum):
    """Severity levels for detected issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class EthicsIssue:
    """Represents an ethical issue found in content."""
    id: str
    category: EthicsCategory
    description: str
    location: Dict[str, Any]  # e.g., {"section": "introduction", "paragraph": 2}
    severity: IssueSeverity
    suggestions: List[str]
    citations: List[Dict[str, str]]  # e.g., [{"title": "...", "url": "...", "relevance": "..."}]
    confidence: float
    detected_at: datetime

@dataclass
class BiasIssue:
    """Represents a bias issue found in content."""
    id: str
    type: BiasType
    description: str
    location: Dict[str, Any]
    severity: IssueSeverity
    impact: str
    suggestions: List[str]
    citations: List[Dict[str, str]]
    confidence: float
    detected_at: datetime

class EthicsBiasAgent(BaseAgent):
    """Agent specialized in detecting ethical issues and bias in content."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Ethics & Bias Detector Agent."""
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "ethics_analysis": True,
            "bias_detection": True,
            "guideline_reference": True,
            "suggestion_generation": True,
            "citation_provision": True
        }
        
        # 16k token ceiling so Gemini 2.5 / Claude have room for the
        # full structured JSON without truncating in the middle of an
        # issue object.
        self.model_config.update({
            "temperature": 0.3,
            "max_tokens": 16384,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })

        # Load ethical guidelines and best practices
        self._load_guidelines()

    async def _process_with_model(
        self,
        system_prompt: str,
        user_payload: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Adapter onto BaseAgent._call_model_async with ethics-tuned
        defaults.  The agent's mode handlers call
        `self._process_with_model(system_prompt, payload, parameters)`
        but BaseAgent exposes `_call_model_async(prompt,
        system_prompt=..., ...)` — this shim bridges the two without
        rewriting every handler."""
        parameters = parameters or {}
        temperature = parameters.get(
            "temperature", self.model_config.get("temperature", 0.3)
        )
        max_tokens = parameters.get(
            "max_tokens", self.model_config.get("max_tokens", 16384)
        )
        return await self._call_model_async(
            prompt=user_payload or "",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── JSON extraction utilities ─────────────────────────────

    @staticmethod
    def _extract_json(response: Any) -> Dict[str, Any]:
        """Best-effort JSON extraction.  Handles ```json``` fences,
        leading / trailing prose, and naked top-level objects.
        Returns {} on failure rather than raising."""
        if not response:
            return {}
        text = response.strip() if isinstance(response, str) else str(response)

        # Plain parse.
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strip ```json``` fences.
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

        # Open-fence-only (truncated): drop the leading fence.
        stripped = re.sub(r"^```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(stripped)
        except Exception:
            pass

        # First balanced {...}.
        s = stripped.find("{")
        e = stripped.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(stripped[s : e + 1])
            except Exception:
                pass

        logger.warning(
            "ethics_bias_llm_non_json len=%s head=%r tail=%r",
            len(text),
            text[:300],
            text[-200:],
        )
        return {}

    @staticmethod
    def _coerce_severity(v: Any) -> "IssueSeverity":
        if not v:
            return IssueSeverity.MEDIUM
        s = str(v).strip().lower()
        try:
            return IssueSeverity(s)
        except Exception:
            return IssueSeverity.MEDIUM

    @staticmethod
    def _coerce_category(v: Any) -> Optional["EthicsCategory"]:
        if not v:
            return None
        try:
            return EthicsCategory(str(v).strip().lower().replace(" ", "_"))
        except Exception:
            return None

    @staticmethod
    def _coerce_bias_type(v: Any) -> Optional["BiasType"]:
        if not v:
            return None
        try:
            return BiasType(str(v).strip().lower().replace(" ", "_"))
        except Exception:
            return None

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

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an ethics and bias analysis request."""
        try:
            action = request.get("action", "analyze")
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Process based on action
            if action == "analyze":
                result = await self._analyze_content(data, context, parameters)
            elif action == "check_guidelines":
                result = await self._check_against_guidelines(data, context, parameters)
            elif action == "generate_suggestions":
                result = await self._generate_suggestions(data, context, parameters)
            elif action == "get_citations":
                result = await self._get_relevant_citations(data, context, parameters)
            else:
                raise ValueError(f"Unsupported action: {action}")
            
            return {
                "results": result,
                "metadata": {
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat(),
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing ethics and bias request: {str(e)}")
            return {"error": str(e)}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the ethics and bias agent asynchronously."""
        return await self.process_async({
            "action": "analyze",
            "data": {"content": query},
            "context": context or {}
        })

    async def _analyze_content(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze content for ethical issues and bias."""
        try:
            # Prepare system prompt for analysis
            system_prompt = self._create_analysis_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "content": data.get("content", ""),
                    "content_type": data.get("content_type", "text"),
                    "metadata": data.get("metadata", {}),
                    "context": context
                }),
                parameters
            )
            
            # Parse analysis results
            analysis = self._parse_analysis(response)
            
            return {
                "ethics_issues": [
                    {
                        "id": issue.id,
                        "category": issue.category.value,
                        "description": issue.description,
                        "location": issue.location,
                        "severity": issue.severity.value,
                        "suggestions": issue.suggestions,
                        "citations": issue.citations,
                        "confidence": issue.confidence,
                        "detected_at": issue.detected_at.isoformat()
                    }
                    for issue in analysis["ethics_issues"]
                ],
                "bias_issues": [
                    {
                        "id": issue.id,
                        "type": issue.type.value,
                        "description": issue.description,
                        "location": issue.location,
                        "severity": issue.severity.value,
                        "impact": issue.impact,
                        "suggestions": issue.suggestions,
                        "citations": issue.citations,
                        "confidence": issue.confidence,
                        "detected_at": issue.detected_at.isoformat()
                    }
                    for issue in analysis["bias_issues"]
                ],
                "summary": analysis["summary"],
                "overall_severity": analysis["overall_severity"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            raise

    async def _check_against_guidelines(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check content against ethical guidelines and best practices."""
        try:
            # Prepare system prompt for guideline checking
            system_prompt = self._create_guideline_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "content": data.get("content", ""),
                    "guidelines": self.guidelines,
                    "context": context
                }),
                parameters
            )
            
            # Parse guideline check results
            results = self._parse_guideline_check(response)
            
            return {
                "compliance": results["compliance"],
                "violations": results["violations"],
                "recommendations": results["recommendations"],
                "citations": results["citations"]
            }
            
        except Exception as e:
            logger.error(f"Error checking guidelines: {str(e)}")
            raise

    async def _generate_suggestions(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate suggestions for addressing ethical issues and bias."""
        try:
            # Prepare system prompt for suggestion generation
            system_prompt = self._create_suggestion_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "issues": data.get("issues", []),
                    "content": data.get("content", ""),
                    "context": context
                }),
                parameters
            )
            
            # Parse suggestions
            suggestions = self._parse_suggestions(response)
            
            return {
                "suggestions": suggestions["suggestions"],
                "implementation_steps": suggestions["implementation_steps"],
                "resources": suggestions["resources"],
                "citations": suggestions["citations"]
            }
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            raise

    async def _get_relevant_citations(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get relevant citations for ethical guidelines and best practices."""
        try:
            # Prepare system prompt for citation retrieval
            system_prompt = self._create_citation_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "topic": data.get("topic", ""),
                    "context": context,
                    "parameters": parameters
                }),
                parameters
            )
            
            # Parse citations
            citations = self._parse_citations(response)
            
            return {
                "citations": citations["citations"],
                "summary": citations["summary"],
                "relevance_scores": citations["relevance_scores"]
            }
            
        except Exception as e:
            logger.error(f"Error getting citations: {str(e)}")
            raise

    def _create_analysis_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Build a strict-JSON prompt for content analysis."""
        ethics_cats = ", ".join(c.value for c in EthicsCategory)
        bias_types = ", ".join(b.value for b in BiasType)
        severities = ", ".join(s.value for s in IssueSeverity)
        return f"""You are an ethics and bias detection engine.  Read the user's content and identify ethical issues and bias.

Context:
  Content type: {context.get('content_type', 'general')}
  Domain: {context.get('domain', 'general')}
  Audience: {context.get('audience', 'general')}
  Sensitivity: {parameters.get('sensitivity', 'standard')}

Allowed ethics categories: {ethics_cats}
Allowed bias types: {bias_types}
Allowed severity values: {severities}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "ethics_issues": [
    {{
      "category": "<one of the allowed ethics categories>",
      "description": "<plain-English explanation of the issue>",
      "location": {{"section": "<where in the content>", "excerpt": "<short quote, optional>"}},
      "severity": "<one of the allowed severity values>",
      "suggestions": ["<concrete fix>", "..."],
      "citations": [{{"title": "<source title>", "url": "<optional url>", "relevance": "<why this source matters>"}}],
      "confidence": 0.0
    }}
  ],
  "bias_issues": [
    {{
      "type": "<one of the allowed bias types>",
      "description": "<plain-English explanation>",
      "location": {{"section": "<where>", "excerpt": "<short quote, optional>"}},
      "severity": "<one of the allowed severity values>",
      "impact": "<who is affected and how>",
      "suggestions": ["<concrete fix>"],
      "citations": [{{"title": "...", "url": "...", "relevance": "..."}}],
      "confidence": 0.0
    }}
  ],
  "summary": "<one-paragraph overall summary>",
  "overall_severity": "<one of the allowed severity values>"
}}

Rules:
- Use only the allowed enum values, lowercase, exactly.
- Be exhaustive but concise.  Up to 15 ethics issues and 15 bias issues.
- If nothing is wrong, return empty arrays — never omit a key.
- Confidence is a number between 0 and 1.
"""

    def _create_guideline_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        severities = ", ".join(s.value for s in IssueSeverity)
        return f"""You evaluate content against ethical guidelines and best practices.

Context:
  Domain: {context.get('domain', 'general')}
  Focus areas: {context.get('guidelines_focus', 'all')}
  Compliance level: {parameters.get('compliance_level', 'standard')}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  Shape:

{{
  "compliance": {{
    "<guideline name>": {{"passed": true, "notes": "<why>"}},
    "<another guideline>": {{"passed": false, "notes": "<why it fails>"}}
  }},
  "violations": [
    {{
      "guideline": "<which guideline is violated>",
      "description": "<plain-English>",
      "severity": "<one of {severities}>",
      "evidence": "<short excerpt from the content>",
      "recommendation": "<concrete fix>"
    }}
  ],
  "recommendations": ["<top-level advice>"],
  "citations": [{{"title": "...", "url": "...", "relevance": "..."}}],
  "overall_compliance": "<compliant | partially_compliant | non_compliant>"
}}

Rules:
- Up to 15 violations, ordered most severe first.
- The `compliance` map should cover the relevant guidelines you evaluated (5 to 12 entries).
"""

    def _create_suggestion_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        return f"""You generate actionable suggestions for addressing ethics and bias issues.

Context:
  Content type: {context.get('content_type', 'general')}
  Domain: {context.get('domain', 'general')}
  Implementation level: {parameters.get('implementation_level', 'standard')}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  Shape:

{{
  "suggestions": [
    {{
      "issue_reference": "<which issue from the input this addresses>",
      "title": "<short suggestion name>",
      "description": "<what to do, plain English>",
      "priority": "<critical | high | medium | low>"
    }}
  ],
  "implementation_steps": ["<step 1>", "<step 2>", "..."],
  "resources": [{{"title": "...", "url": "...", "description": "..."}}],
  "citations": [{{"title": "...", "url": "...", "relevance": "..."}}]
}}

Rules:
- Each suggestion must be concrete and actionable.
- Up to 20 suggestions, 15 steps, 10 resources, 10 citations.
"""

    def _create_citation_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        return f"""You find relevant citations and resources on an ethics topic.

Context:
  Domain: {context.get('domain', 'general')}
  Citation types: {context.get('citation_types', 'all')}
  Quality threshold: {parameters.get('quality_threshold', 'high')}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  Shape:

{{
  "citations": [
    {{
      "title": "<title of paper / standard / case>",
      "authors": ["<author or org>"],
      "year": "<year if known>",
      "url": "<link if you have one>",
      "type": "<academic | standard | regulation | case_study | book | other>",
      "summary": "<one sentence on what it covers>",
      "relevance": "<why this matters for the topic>"
    }}
  ],
  "summary": "<one paragraph overview of what the citations cover>",
  "relevance_scores": {{"<citation title>": 0.0}}
}}

Rules:
- Up to 12 citations, most relevant first.
- Only return real, verifiable sources.  Skip anything you would not stand behind.
- relevance_scores values are numbers between 0 and 1.
"""

    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the LLM's analysis response.  Skips malformed items
        instead of failing the entire run."""
        data = self._extract_json(response)

        ethics_issues: List[EthicsIssue] = []
        for raw in self._as_list(data.get("ethics_issues")):
            if not isinstance(raw, dict):
                continue
            category = self._coerce_category(raw.get("category"))
            if not category:
                # Skip silently — unknown categories should not blow up
                # the whole run.
                continue
            ethics_issues.append(
                EthicsIssue(
                    id=str(uuid4()),
                    category=category,
                    description=self._as_str(raw.get("description")),
                    location=raw.get("location") or {},
                    severity=self._coerce_severity(raw.get("severity")),
                    suggestions=[self._as_str(s) for s in self._as_list(raw.get("suggestions")) if s],
                    citations=self._as_list(raw.get("citations")),
                    confidence=float(raw.get("confidence", 0.85) or 0.85),
                    detected_at=datetime.utcnow(),
                )
            )

        bias_issues: List[BiasIssue] = []
        for raw in self._as_list(data.get("bias_issues")):
            if not isinstance(raw, dict):
                continue
            btype = self._coerce_bias_type(raw.get("type"))
            if not btype:
                continue
            bias_issues.append(
                BiasIssue(
                    id=str(uuid4()),
                    type=btype,
                    description=self._as_str(raw.get("description")),
                    location=raw.get("location") or {},
                    severity=self._coerce_severity(raw.get("severity")),
                    impact=self._as_str(raw.get("impact")),
                    suggestions=[self._as_str(s) for s in self._as_list(raw.get("suggestions")) if s],
                    citations=self._as_list(raw.get("citations")),
                    confidence=float(raw.get("confidence", 0.85) or 0.85),
                    detected_at=datetime.utcnow(),
                )
            )

        return {
            "ethics_issues": ethics_issues,
            "bias_issues": bias_issues,
            "summary": self._as_str(data.get("summary")),
            "overall_severity": self._as_str(data.get("overall_severity")) or "low",
        }

    def _parse_guideline_check(self, response: str) -> Dict[str, Any]:
        data = self._extract_json(response)
        compliance_raw = data.get("compliance") or {}
        compliance: Dict[str, Any] = {}
        if isinstance(compliance_raw, dict):
            for k, v in compliance_raw.items():
                if not isinstance(v, dict):
                    # Tolerate `"<name>": true` shorthand from the LLM.
                    compliance[str(k)] = {"passed": bool(v), "notes": ""}
                else:
                    compliance[str(k)] = {
                        "passed": bool(v.get("passed")),
                        "notes": self._as_str(v.get("notes")),
                    }

        violations: List[Dict[str, Any]] = []
        for raw in self._as_list(data.get("violations")):
            if not isinstance(raw, dict):
                continue
            violations.append({
                "guideline": self._as_str(raw.get("guideline")),
                "description": self._as_str(raw.get("description")),
                "severity": self._coerce_severity(raw.get("severity")).value,
                "evidence": self._as_str(raw.get("evidence")),
                "recommendation": self._as_str(raw.get("recommendation")),
            })

        return {
            "compliance": compliance,
            "violations": violations,
            "recommendations": [self._as_str(r) for r in self._as_list(data.get("recommendations")) if r],
            "citations": self._as_list(data.get("citations")),
            "overall_compliance": self._as_str(data.get("overall_compliance")) or "partially_compliant",
        }

    def _parse_suggestions(self, response: str) -> Dict[str, Any]:
        data = self._extract_json(response)
        suggestions: List[Dict[str, Any]] = []
        for raw in self._as_list(data.get("suggestions")):
            if isinstance(raw, str):
                suggestions.append({
                    "title": raw[:80],
                    "description": raw,
                    "priority": "medium",
                    "issue_reference": "",
                })
            elif isinstance(raw, dict):
                suggestions.append({
                    "issue_reference": self._as_str(raw.get("issue_reference")),
                    "title": self._as_str(raw.get("title")) or "Suggestion",
                    "description": self._as_str(raw.get("description")),
                    "priority": self._coerce_severity(raw.get("priority")).value,
                })

        return {
            "suggestions": suggestions,
            "implementation_steps": [
                self._as_str(s) for s in self._as_list(data.get("implementation_steps")) if s
            ],
            "resources": self._as_list(data.get("resources")),
            "citations": self._as_list(data.get("citations")),
        }

    def _parse_citations(self, response: str) -> Dict[str, Any]:
        data = self._extract_json(response)
        citations: List[Dict[str, Any]] = []
        for raw in self._as_list(data.get("citations")):
            if not isinstance(raw, dict):
                continue
            citations.append({
                "title": self._as_str(raw.get("title")),
                "authors": [self._as_str(a) for a in self._as_list(raw.get("authors")) if a],
                "year": self._as_str(raw.get("year")),
                "url": self._as_str(raw.get("url")),
                "type": self._as_str(raw.get("type")) or "other",
                "summary": self._as_str(raw.get("summary")),
                "relevance": self._as_str(raw.get("relevance")),
            })

        scores_raw = data.get("relevance_scores") or {}
        scores: Dict[str, float] = {}
        if isinstance(scores_raw, dict):
            for k, v in scores_raw.items():
                try:
                    scores[str(k)] = float(v)
                except Exception:
                    pass

        return {
            "citations": citations,
            "summary": self._as_str(data.get("summary")),
            "relevance_scores": scores,
        }

    def _load_guidelines(self) -> None:
        """Load ethical guidelines and best practices."""
        # In a real implementation, this would load from a database or file
        self.guidelines = {
            "general": [
                {
                    "category": "fairness",
                    "guidelines": [
                        "Ensure equal treatment and opportunity",
                        "Avoid discriminatory language and practices",
                        "Consider diverse perspectives"
                    ]
                },
                {
                    "category": "privacy",
                    "guidelines": [
                        "Protect personal information",
                        "Obtain necessary consent",
                        "Follow data protection regulations"
                    ]
                }
                # Add more categories and guidelines
            ],
            "ai_specific": [
                {
                    "category": "transparency",
                    "guidelines": [
                        "Disclose AI system capabilities and limitations",
                        "Explain decision-making processes",
                        "Provide clear documentation"
                    ]
                },
                {
                    "category": "accountability",
                    "guidelines": [
                        "Establish clear responsibility",
                        "Implement monitoring and auditing",
                        "Enable human oversight"
                    ]
                }
                # Add more AI-specific guidelines
            ]
        } 