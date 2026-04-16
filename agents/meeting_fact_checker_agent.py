from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from uuid import uuid4

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ClaimType(str, Enum):
    """Types of claims that can be fact-checked."""
    STATISTIC = "statistic"
    FACT = "fact"
    REFERENCE = "reference"
    QUOTE = "quote"
    ASSERTION = "assertion"

class VerificationStatus(str, Enum):
    """Status of fact verification."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    DISPUTED = "disputed"
    UNVERIFIABLE = "unverifiable"
    PENDING = "pending"

class ClaimSeverity(str, Enum):
    """Severity of incorrect claims."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class Claim:
    """Represents a claim made during a meeting."""
    id: str
    timestamp: datetime
    speaker: str
    claim_type: ClaimType
    content: str
    context: Dict[str, Any]
    verification_status: VerificationStatus
    severity: ClaimSeverity
    confidence: float
    citations: List[Dict[str, str]]
    corrections: List[str]
    metadata: Dict[str, Any]

@dataclass
class MeetingSession:
    """Represents an active meeting session."""
    id: str
    start_time: datetime
    title: str
    participants: List[str]
    claims: List[Claim]
    summary: Optional[str]
    metadata: Dict[str, Any]

class MeetingFactCheckerAgent(BaseAgent):
    """Agent specialized in real-time fact checking during meetings."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Meeting Fact Checker Agent."""
        super().__init__(config)

        # Set default capabilities
        self.capabilities = {
            "real_time_verification": True,
            "claim_detection": True,
            "citation_provision": True,
            "correction_generation": True,
            "meeting_summary": True
        }

        # Force Perplexity as the provider — it's the only model with live
        # web search, which is essential for real-time fact verification.
        self.model_config.update({
            "provider": "perplexity",
            "model": "sonar",
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
        })

        # Re-initialize the LLM client now that model_config has perplexity set
        self.initialize_models()
        logger.info("Fact-checker agent initialized with Perplexity (sonar-large-online)")
        
        # Initialize active sessions and claim history
        self.active_sessions: Dict[str, MeetingSession] = {}
        self.claim_history: List[Claim] = []
        
        # Load verification guidelines
        self._load_verification_guidelines()

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a meeting fact checking request."""
        try:
            action = request.get("action", "verify")
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Process based on action
            if action == "verify":
                result = await self._verify_claim(data, context, parameters)
            elif action == "start_session":
                result = await self._start_session(data, context, parameters)
            elif action == "end_session":
                result = await self._end_session(data, context, parameters)
            elif action == "get_summary":
                result = await self._get_session_summary(data, context, parameters)
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
            logger.error(f"Error processing meeting fact checking request: {str(e)}")
            return {"error": str(e)}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the meeting fact checker agent asynchronously."""
        return await self.process_async({
            "action": "verify",
            "data": {
                "claim": query,
                "speaker": "user",
                "claim_type": "assertion"
            },
            "context": context or {}
        })

    async def _verify_claim(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify a claim made during a meeting."""
        try:
            # Prepare system prompt for claim verification
            system_prompt = self._create_verification_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "claim": data.get("claim"),
                    "speaker": data.get("speaker"),
                    "context": context,
                    "session_id": data.get("session_id")
                }),
                parameters
            )
            
            # Parse verification results
            verification = self._parse_verification(response)
            
            # Create claim record
            claim = Claim(
                id=str(uuid4()),
                timestamp=datetime.utcnow(),
                speaker=data.get("speaker", "unknown"),
                claim_type=ClaimType(data.get("claim_type", "assertion")),
                content=data.get("claim", ""),
                context=context,
                verification_status=verification["status"],
                severity=verification["severity"],
                confidence=verification["confidence"],
                citations=verification["citations"],
                corrections=verification["corrections"],
                metadata={
                    "session_id": data.get("session_id"),
                    "parameters": parameters
                }
            )
            
            # Store claim
            self.claim_history.append(claim)
            
            # Update session if active
            session_id = data.get("session_id")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].claims.append(claim)
            
            return {
                "claim_id": claim.id,
                "timestamp": claim.timestamp.isoformat(),
                "verification_status": claim.verification_status.value,
                "severity": claim.severity.value,
                "confidence": claim.confidence,
                "citations": claim.citations,
                "corrections": claim.corrections,
                "summary": verification.get("summary", "")
            }
            
        except Exception as e:
            logger.error(f"Error verifying claim: {str(e)}")
            raise

    async def _start_session(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start a new meeting session."""
        try:
            session_id = str(uuid4())
            
            session = MeetingSession(
                id=session_id,
                start_time=datetime.utcnow(),
                title=data.get("title", "Untitled Meeting"),
                participants=data.get("participants", []),
                claims=[],
                summary=None,
                metadata={
                    "context": context,
                    "parameters": parameters
                }
            )
            
            self.active_sessions[session_id] = session
            
            return {
                "session_id": session_id,
                "start_time": session.start_time.isoformat(),
                "title": session.title,
                "participants": session.participants
            }
            
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}")
            raise

    async def _end_session(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """End a meeting session and generate summary."""
        try:
            session_id = data.get("session_id")
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Generate session summary
            summary = await self._generate_session_summary(session, context, parameters)
            session.summary = summary
            
            # Prepare session report
            report = {
                "session_id": session.id,
                "title": session.title,
                "start_time": session.start_time.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "participants": session.participants,
                "total_claims": len(session.claims),
                "verification_stats": self._calculate_verification_stats(session.claims),
                "summary": summary,
                "claims": [
                    {
                        "id": claim.id,
                        "timestamp": claim.timestamp.isoformat(),
                        "speaker": claim.speaker,
                        "content": claim.content,
                        "verification_status": claim.verification_status.value,
                        "severity": claim.severity.value,
                        "citations": claim.citations,
                        "corrections": claim.corrections
                    }
                    for claim in session.claims
                ]
            }
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            return report
            
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            raise

    async def _get_session_summary(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get summary of an active session."""
        try:
            session_id = data.get("session_id")
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Generate or update summary
            summary = await self._generate_session_summary(session, context, parameters)
            session.summary = summary
            
            return {
                "session_id": session.id,
                "title": session.title,
                "start_time": session.start_time.isoformat(),
                "current_time": datetime.utcnow().isoformat(),
                "total_claims": len(session.claims),
                "verification_stats": self._calculate_verification_stats(session.claims),
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary: {str(e)}")
            raise

    async def _process_with_model(
        self,
        system_prompt: str,
        user_content: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call the LLM with a system prompt and user content, return parsed JSON dict."""
        import re

        response_text = await self._call_model_async(
            prompt=user_content,
            system_prompt=system_prompt + "\n\nRespond ONLY with valid JSON.",
            temperature=parameters.get("temperature", 0.3),
            max_tokens=parameters.get("max_tokens", 2000),
        )

        # Strip markdown fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, wrapping as summary")
            return {"summary": response_text.strip()}

    def _parse_verification(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured verification data from the LLM response."""
        status_raw = response.get("verification_status", response.get("status", "pending")).lower()

        # Normalize to valid enum values
        status_map = {
            "verified": "verified",
            "true": "verified",
            "partially_verified": "partially_verified",
            "partially true": "partially_verified",
            "partial": "partially_verified",
            "disputed": "disputed",
            "false": "disputed",
            "unverifiable": "unverifiable",
            "unverified": "unverifiable",
            "unknown": "unverifiable",
            "pending": "pending",
        }
        status = status_map.get(status_raw, "pending")

        severity_raw = response.get("severity", "medium").lower()
        severity_map = {
            "critical": "critical", "high": "high", "medium": "medium",
            "low": "low", "info": "info",
        }
        severity = severity_map.get(severity_raw, "medium")

        confidence = response.get("confidence", 0.5)
        if isinstance(confidence, (int, float)):
            # Normalize to 0-1 range if given as percentage
            if confidence > 1:
                confidence = confidence / 100.0
        else:
            confidence = 0.5

        return {
            "status": VerificationStatus(status),
            "severity": ClaimSeverity(severity),
            "confidence": confidence,
            "citations": response.get("citations", []),
            "corrections": response.get("corrections", []),
            "summary": response.get("summary", response.get("explanation", "")),
        }

    def _create_verification_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for claim verification."""
        return f"""You are a real-time fact-checking assistant with live web access. Your job is to make a deep research and verify claims made during meetings using current, up-to-date information from the internet.

Context:
- Meeting Type: {context.get('meeting_type', 'general')}
- Topic: {context.get('topic', 'unknown')}
- Participants: {context.get('participants', 'unknown')}
- Previous Claims: {context.get('previous_claims', 'none')}

IMPORTANT: You have live web access. Use it to:
1. Search for the most current data to verify or refute the claim
2. Find real, authoritative sources (company filings, news articles, official reports, press releases)
3. Provide actual URLs to the sources you found
4. Compare the claim against the latest available data
5. Suggest specific corrections with evidence if the claim is inaccurate

Return your response as JSON with these exact fields:
{{
  "verification_status": "verified" | "partially_verified" | "disputed" | "unverifiable",
  "confidence": 0.0 to 1.0,
  "severity": "critical" | "high" | "medium" | "low" | "info",
  "summary": "detailed explanation citing specific data points you found online",
  "citations": ["https://real-url-to-source-1", "https://real-url-to-source-2"],
  "corrections": ["specific correction with evidence if claim is inaccurate weather True or Not"]
}}

For citations, always provide real URLs when possible. If you cannot find a direct URL, describe the source precisely (e.g. "SEC 10-K filing, Q2 2025, page 12")."""

    async def _generate_session_summary(
        self,
        session: MeetingSession,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Generate a summary of the meeting session."""
        try:
            # Prepare system prompt for summary generation
            system_prompt = f"""Generate a comprehensive summary of the meeting session.
            
            Meeting Details:
            - Title: {session.title}
            - Participants: {', '.join(session.participants)}
            - Duration: {(datetime.utcnow() - session.start_time).total_seconds() / 60:.1f} minutes
            - Total Claims: {len(session.claims)}
            
            Include:
            1. Key points discussed
            2. Verified facts and statistics
            3. Important corrections or clarifications
            4. Action items or decisions
            5. Areas requiring further verification
            
            Provide a clear, concise summary with relevant citations.
            """
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "claims": [
                        {
                            "speaker": claim.speaker,
                            "content": claim.content,
                            "verification_status": claim.verification_status.value,
                            "citations": claim.citations,
                            "corrections": claim.corrections
                        }
                        for claim in session.claims
                    ],
                    "context": context
                }),
                parameters
            )
            
            return response.get("summary", "No summary available")
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            return "Error generating summary"

    def _calculate_verification_stats(self, claims: List[Claim]) -> Dict[str, Any]:
        """Calculate verification statistics for claims."""
        total = len(claims)
        if total == 0:
            return {
                "total_claims": 0,
                "verified": 0,
                "partially_verified": 0,
                "disputed": 0,
                "unverifiable": 0,
                "pending": 0
            }
        
        stats = {
            "total_claims": total,
            "verified": sum(1 for c in claims if c.verification_status == VerificationStatus.VERIFIED),
            "partially_verified": sum(1 for c in claims if c.verification_status == VerificationStatus.PARTIALLY_VERIFIED),
            "disputed": sum(1 for c in claims if c.verification_status == VerificationStatus.DISPUTED),
            "unverifiable": sum(1 for c in claims if c.verification_status == VerificationStatus.UNVERIFIABLE),
            "pending": sum(1 for c in claims if c.verification_status == VerificationStatus.PENDING)
        }
        
        # Calculate percentages
        for key in ["verified", "partially_verified", "disputed", "unverifiable", "pending"]:
            stats[f"{key}_percentage"] = (stats[key] / total) * 100
        
        return stats

    def _load_verification_guidelines(self) -> None:
        """Load verification guidelines and best practices."""
        # In a real implementation, this would load from a database or file
        self.verification_guidelines = {
            "statistics": {
                "guidelines": [
                    "Verify data sources and methodology",
                    "Check for recent updates",
                    "Consider context and scope",
                    "Look for peer-reviewed sources"
                ],
                "citations": [
                    {
                        "title": "Guidelines for Statistical Verification",
                        "author": "American Statistical Association",
                        "year": "2023",
                        "url": "https://www.amstat.org/guidelines"
                    }
                ]
            },
            "facts": {
                "guidelines": [
                    "Cross-reference multiple reliable sources",
                    "Check for consensus among experts",
                    "Consider temporal relevance",
                    "Verify primary sources when possible"
                ],
                "citations": [
                    {
                        "title": "Fact-Checking Standards",
                        "author": "International Fact-Checking Network",
                        "year": "2023",
                        "url": "https://www.poynter.org/ifcn"
                    }
                ]
            }
            # Add more categories and guidelines
        } 