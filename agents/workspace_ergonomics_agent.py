from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from uuid import uuid4

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ErgonomicCategory(str, Enum):
    """Categories of ergonomic factors."""
    POSTURE = "posture"
    LIGHTING = "lighting"
    DESK_SETUP = "desk_setup"
    CHAIR_ADJUSTMENT = "chair_adjustment"
    MONITOR_POSITION = "monitor_position"
    KEYBOARD_MOUSE = "keyboard_mouse"
    AMBIENT_NOISE = "ambient_noise"
    TEMPERATURE = "temperature"

class IssueSeverity(str, Enum):
    """Severity levels for ergonomic issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ErgonomicIssue:
    """Represents an ergonomic issue detected in the workspace."""
    id: str
    category: ErgonomicCategory
    description: str
    severity: IssueSeverity
    location: Dict[str, Any]  # Coordinates or area in the image
    current_state: Dict[str, Any]
    recommended_state: Dict[str, Any]
    suggestions: List[str]
    citations: List[Dict[str, str]]
    confidence: float
    detected_at: datetime

@dataclass
class WorkspaceAnalysis:
    """Represents a complete workspace analysis."""
    id: str
    timestamp: datetime
    image_data: Optional[bytes]  # Optional captured image
    issues: List[ErgonomicIssue]
    overall_score: float
    recommendations: List[Dict[str, Any]]
    citations: List[Dict[str, str]]
    metadata: Dict[str, Any]

class WorkspaceErgonomicsAgent(BaseAgent):
    """Agent specialized in analyzing workspace ergonomics and providing recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Workspace Ergonomics Agent."""
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "posture_analysis": True,
            "lighting_analysis": True,
            "desk_setup_analysis": True,
            "real_time_monitoring": True,
            "recommendation_generation": True,
            "citation_provision": True
        }
        
        # Configure model for ergonomic analysis
        self.model_config.update({
            "temperature": 0.3,  # Lower temperature for more precise analysis
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        })
        
        # Initialize analysis history
        self.analysis_history: List[WorkspaceAnalysis] = []
        self.active_monitoring: Dict[str, Any] = {}
        
        # Load ergonomic guidelines and research
        self._load_ergonomic_guidelines()

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workspace ergonomics analysis request."""
        try:
            action = request.get("action", "analyze")
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Process based on action
            if action == "analyze":
                result = await self._analyze_workspace(data, context, parameters)
            elif action == "monitor":
                result = await self._monitor_workspace(data, context, parameters)
            elif action == "get_recommendations":
                result = await self._get_recommendations(data, context, parameters)
            elif action == "get_guidelines":
                result = await self._get_guidelines(data, context, parameters)
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
            logger.error(f"Error processing workspace ergonomics request: {str(e)}")
            return {"error": str(e)}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the workspace ergonomics agent asynchronously."""
        return await self.process_async({
            "action": "get_recommendations",
            "data": {"user_profile": {"query": query}},
            "context": context or {}
        })

    async def _analyze_workspace(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze workspace conditions from image data."""
        try:
            # Prepare system prompt for workspace analysis
            system_prompt = self._create_analysis_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "image_data": data.get("image_data"),
                    "current_conditions": data.get("current_conditions", {}),
                    "context": context
                }),
                parameters
            )
            
            # Parse analysis results
            analysis = self._parse_analysis(response)
            
            # Create workspace analysis record
            workspace_analysis = WorkspaceAnalysis(
                id=str(uuid4()),
                timestamp=datetime.utcnow(),
                image_data=data.get("image_data"),
                issues=analysis["issues"],
                overall_score=analysis["overall_score"],
                recommendations=analysis["recommendations"],
                citations=analysis["citations"],
                metadata={
                    "context": context,
                    "parameters": parameters
                }
            )
            
            # Store analysis
            self.analysis_history.append(workspace_analysis)
            
            return {
                "analysis_id": workspace_analysis.id,
                "timestamp": workspace_analysis.timestamp.isoformat(),
                "issues": [
                    {
                        "id": issue.id,
                        "category": issue.category.value,
                        "description": issue.description,
                        "severity": issue.severity.value,
                        "suggestions": issue.suggestions,
                        "citations": issue.citations,
                        "confidence": issue.confidence
                    }
                    for issue in workspace_analysis.issues
                ],
                "overall_score": workspace_analysis.overall_score,
                "recommendations": workspace_analysis.recommendations,
                "citations": workspace_analysis.citations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing workspace: {str(e)}")
            raise

    async def _monitor_workspace(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monitor workspace conditions in real-time."""
        try:
            # Start or update monitoring session
            session_id = data.get("session_id")
            if not session_id:
                session_id = str(uuid4())
                self.active_monitoring[session_id] = {
                    "start_time": datetime.utcnow(),
                    "last_analysis": None,
                    "issues_history": []
                }
            
            # Perform analysis
            analysis_result = await self._analyze_workspace(data, context, parameters)
            
            # Update monitoring session
            self.active_monitoring[session_id]["last_analysis"] = analysis_result
            self.active_monitoring[session_id]["issues_history"].append(
                analysis_result["issues"]
            )
            
            # Check for persistent issues
            persistent_issues = self._identify_persistent_issues(session_id)
            
            return {
                "session_id": session_id,
                "analysis": analysis_result,
                "persistent_issues": persistent_issues,
                "monitoring_duration": (
                    datetime.utcnow() - self.active_monitoring[session_id]["start_time"]
                ).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring workspace: {str(e)}")
            raise

    async def _get_recommendations(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get personalized ergonomic recommendations."""
        try:
            # Prepare system prompt for recommendations
            system_prompt = self._create_recommendation_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "analysis_history": [
                        {
                            "timestamp": analysis.timestamp.isoformat(),
                            "issues": [
                                {
                                    "category": issue.category.value,
                                    "description": issue.description,
                                    "severity": issue.severity.value
                                }
                                for issue in analysis.issues
                            ]
                        }
                        for analysis in self.analysis_history[-5:]  # Last 5 analyses
                    ],
                    "user_profile": data.get("user_profile", {}),
                    "context": context
                }),
                parameters
            )
            
            # Parse recommendations
            recommendations = self._parse_recommendations(response)
            
            return {
                "recommendations": recommendations["recommendations"],
                "priority_actions": recommendations["priority_actions"],
                "long_term_suggestions": recommendations["long_term_suggestions"],
                "product_recommendations": recommendations.get("product_recommendations", []),
                "citations": recommendations["citations"]
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    async def _get_guidelines(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get ergonomic guidelines for specific categories."""
        try:
            categories = data.get("categories", [cat.value for cat in ErgonomicCategory])
            guidelines = {}
            
            for category in categories:
                if category in self.ergonomic_guidelines:
                    guidelines[category] = {
                        "standards": self.ergonomic_guidelines[category]["standards"],
                        "best_practices": self.ergonomic_guidelines[category]["best_practices"],
                        "citations": self.ergonomic_guidelines[category]["citations"]
                    }
            
            return {
                "guidelines": guidelines,
                "last_updated": self.ergonomic_guidelines.get("last_updated", ""),
                "sources": self.ergonomic_guidelines.get("sources", [])
            }
            
        except Exception as e:
            logger.error(f"Error getting guidelines: {str(e)}")
            raise

    def _create_analysis_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for workspace analysis."""
        return f"""You are a specialized workspace ergonomics analyst. Analyze the workspace conditions and provide detailed insights.
        
        Context:
        - User Profile: {context.get('user_profile', 'general')}
        - Workspace Type: {context.get('workspace_type', 'office')}
        - Time of Day: {context.get('time_of_day', 'unknown')}
        - Previous Issues: {context.get('previous_issues', 'none')}
        
        Analyze:
        1. Posture and body positioning
        2. Lighting conditions
        3. Desk and chair setup
        4. Monitor and keyboard placement
        5. Ambient conditions
        
        Provide detailed analysis with specific issues, recommendations, and research citations.
        """

    def _create_recommendation_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for recommendations."""
        return f"""You are a specialized ergonomic advisor. Provide personalized recommendations for workspace improvement.
        
        Context:
        - User Profile: {context.get('user_profile', 'general')}
        - Workspace Type: {context.get('workspace_type', 'office')}
        - Budget: {context.get('budget', 'flexible')}
        - Constraints: {context.get('constraints', 'none')}
        
        Consider:
        1. Immediate actionable improvements
        2. Long-term workspace optimization
        3. Product recommendations if needed
        4. Research evidence and citations
        
        Provide detailed recommendations with implementation steps and expected benefits.
        """

    def _identify_persistent_issues(self, session_id: str) -> List[Dict[str, Any]]:
        """Identify issues that persist across multiple analyses."""
        if session_id not in self.active_monitoring:
            return []
        
        issues_history = self.active_monitoring[session_id]["issues_history"]
        if not issues_history:
            return []
        
        # Count occurrences of each issue type
        issue_counts = {}
        for analysis in issues_history:
            for issue in analysis:
                key = f"{issue['category']}:{issue['description']}"
                if key not in issue_counts:
                    issue_counts[key] = {
                        "count": 0,
                        "issue": issue
                    }
                issue_counts[key]["count"] += 1
        
        # Identify persistent issues (occurring in more than 50% of analyses)
        threshold = len(issues_history) * 0.5
        persistent_issues = [
            {
                "issue": data["issue"],
                "occurrence_count": data["count"],
                "persistence_score": data["count"] / len(issues_history)
            }
            for key, data in issue_counts.items()
            if data["count"] > threshold
        ]
        
        return persistent_issues

    def _load_ergonomic_guidelines(self) -> None:
        """Load ergonomic guidelines and research."""
        # In a real implementation, this would load from a database or file
        self.ergonomic_guidelines = {
            "posture": {
                "standards": [
                    "Neutral spine alignment",
                    "Feet flat on the floor",
                    "Knees at 90 degrees",
                    "Elbows at 90 degrees",
                    "Shoulders relaxed"
                ],
                "best_practices": [
                    "Take regular breaks to stretch",
                    "Alternate between sitting and standing",
                    "Use lumbar support if needed",
                    "Keep frequently used items within easy reach"
                ],
                "citations": [
                    {
                        "title": "Ergonomic Guidelines for Computer Workstations",
                        "author": "OSHA",
                        "year": "2023",
                        "url": "https://www.osha.gov/ergonomics"
                    }
                ]
            },
            "lighting": {
                "standards": [
                    "300-500 lux for general office work",
                    "500-750 lux for detailed work",
                    "Minimize glare and reflections",
                    "Use natural light when possible"
                ],
                "best_practices": [
                    "Position monitor perpendicular to windows",
                    "Use task lighting for detailed work",
                    "Adjust screen brightness to match ambient light",
                    "Use anti-glare filters if needed"
                ],
                "citations": [
                    {
                        "title": "Lighting Standards for Office Environments",
                        "author": "IESNA",
                        "year": "2023",
                        "url": "https://www.ies.org/standards"
                    }
                ]
            }
            # Add more categories and guidelines
        } 