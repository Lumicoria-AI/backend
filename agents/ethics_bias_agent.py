from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime
import json
from dataclasses import dataclass

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
        
        # Configure model for ethical analysis
        self.model_config.update({
            "temperature": 0.3,  # Lower temperature for more precise analysis
            "max_tokens": 4096,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })
        
        # Load ethical guidelines and best practices
        self._load_guidelines()

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
        """Create system prompt for content analysis."""
        return f"""You are a specialized ethics and bias detection AI. Your task is to analyze content for ethical issues and bias.
        
        Context:
        - Content Type: {context.get('content_type', 'general')}
        - Domain: {context.get('domain', 'general')}
        - Target Audience: {context.get('audience', 'general')}
        - Sensitivity Level: {parameters.get('sensitivity', 'standard')}
        
        Analyze for:
        1. Ethical issues across categories: fairness, privacy, transparency, accountability, inclusion, safety, sustainability, human rights
        2. Various types of bias: gender, racial, age, religious, cultural, socioeconomic, geographic, language, ability, occupational
        3. Potential impacts and severity
        4. Supporting evidence and citations
        
        Provide detailed analysis with specific examples, suggestions for improvement, and relevant citations.
        """

    def _create_guideline_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for guideline checking."""
        return f"""You are a specialized ethical guidelines checker. Your task is to evaluate content against established guidelines and best practices.
        
        Context:
        - Domain: {context.get('domain', 'general')}
        - Guidelines Focus: {context.get('guidelines_focus', 'all')}
        - Compliance Level: {parameters.get('compliance_level', 'standard')}
        
        Check against:
        1. Industry-specific ethical guidelines
        2. Best practices for content creation
        3. Regulatory requirements
        4. Cultural and social considerations
        
        Provide detailed compliance analysis with specific violations, recommendations, and citations.
        """

    def _create_suggestion_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for suggestion generation."""
        return f"""You are a specialized ethics and bias mitigation advisor. Your task is to generate actionable suggestions for addressing identified issues.
        
        Context:
        - Content Type: {context.get('content_type', 'general')}
        - Domain: {context.get('domain', 'general')}
        - Issue Types: {context.get('issue_types', 'all')}
        - Implementation Level: {parameters.get('implementation_level', 'standard')}
        
        Generate:
        1. Specific suggestions for each issue
        2. Step-by-step implementation guidance
        3. Relevant resources and tools
        4. Supporting citations and examples
        
        Ensure suggestions are practical, actionable, and well-supported by evidence.
        """

    def _create_citation_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for citation retrieval."""
        return f"""You are a specialized ethics and bias research assistant. Your task is to find relevant citations and resources.
        
        Context:
        - Topic: {context.get('topic', 'general')}
        - Domain: {context.get('domain', 'general')}
        - Citation Types: {context.get('citation_types', 'all')}
        - Quality Threshold: {parameters.get('quality_threshold', 'high')}
        
        Find:
        1. Academic papers and research
        2. Industry guidelines and standards
        3. Best practice documents
        4. Case studies and examples
        
        Provide detailed citations with relevance scores and summaries.
        """

    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured analysis results."""
        try:
            data = json.loads(response)
            return {
                "ethics_issues": [
                    EthicsIssue(
                        id=str(uuid4()),
                        category=EthicsCategory(issue["category"]),
                        description=issue["description"],
                        location=issue["location"],
                        severity=IssueSeverity(issue["severity"]),
                        suggestions=issue["suggestions"],
                        citations=issue["citations"],
                        confidence=issue.get("confidence", 1.0),
                        detected_at=datetime.utcnow()
                    )
                    for issue in data.get("ethics_issues", [])
                ],
                "bias_issues": [
                    BiasIssue(
                        id=str(uuid4()),
                        type=BiasType(issue["type"]),
                        description=issue["description"],
                        location=issue["location"],
                        severity=IssueSeverity(issue["severity"]),
                        impact=issue["impact"],
                        suggestions=issue["suggestions"],
                        citations=issue["citations"],
                        confidence=issue.get("confidence", 1.0),
                        detected_at=datetime.utcnow()
                    )
                    for issue in data.get("bias_issues", [])
                ],
                "summary": data.get("summary", ""),
                "overall_severity": data.get("overall_severity", "low")
            }
        except Exception as e:
            logger.error(f"Error parsing analysis: {str(e)}")
            raise

    def _parse_guideline_check(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into guideline check results."""
        try:
            data = json.loads(response)
            return {
                "compliance": data.get("compliance", {}),
                "violations": data.get("violations", []),
                "recommendations": data.get("recommendations", []),
                "citations": data.get("citations", [])
            }
        except Exception as e:
            logger.error(f"Error parsing guideline check: {str(e)}")
            raise

    def _parse_suggestions(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into suggestions."""
        try:
            data = json.loads(response)
            return {
                "suggestions": data.get("suggestions", []),
                "implementation_steps": data.get("implementation_steps", []),
                "resources": data.get("resources", []),
                "citations": data.get("citations", [])
            }
        except Exception as e:
            logger.error(f"Error parsing suggestions: {str(e)}")
            raise

    def _parse_citations(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into citations."""
        try:
            data = json.loads(response)
            return {
                "citations": data.get("citations", []),
                "summary": data.get("summary", ""),
                "relevance_scores": data.get("relevance_scores", {})
            }
        except Exception as e:
            logger.error(f"Error parsing citations: {str(e)}")
            raise

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