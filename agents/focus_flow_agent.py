from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from uuid import uuid4

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class FocusState(str, Enum):
    """States of focus and flow."""
    DEEP_WORK = "deep_work"
    FLOW = "flow"
    DISTRACTED = "distracted"
    BREAK = "break"
    TRANSITION = "transition"

class DistractionType(str, Enum):
    """Types of digital distractions."""
    NOTIFICATION = "notification"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    MULTITASKING = "multitasking"
    ENVIRONMENTAL = "environmental"
    INTERNAL = "internal"

class ProductivityTechnique(str, Enum):
    """Types of productivity techniques."""
    POMODORO = "pomodoro"
    TIME_BLOCKING = "time_blocking"
    DEEP_WORK = "deep_work"
    FLOW_TRIGGERS = "flow_triggers"
    MINDFULNESS = "mindfulness"
    ENVIRONMENT_OPTIMIZATION = "environment_optimization"

@dataclass
class FocusSession:
    """Represents a focus session."""
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    state: FocusState
    duration: timedelta
    distractions: List[Dict[str, Any]]
    productivity_score: float
    technique_used: Optional[str]
    notes: Optional[str]

@dataclass
class DistractionEvent:
    """Represents a distraction event."""
    id: str
    timestamp: datetime
    type: DistractionType
    source: str
    duration: timedelta
    impact_score: float
    context: Dict[str, Any]

@dataclass
class ProductivityRecommendation:
    """Represents a productivity recommendation."""
    id: str
    technique: ProductivityTechnique
    description: str
    rationale: str
    implementation_steps: List[str]
    expected_benefits: List[str]
    citations: List[Dict[str, str]]
    confidence: float
    created_at: datetime

class FocusFlowAgent(BaseAgent):
    """Agent specialized in monitoring focus, analyzing work patterns, and providing productivity recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Focus & Flow Guardian Agent."""
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "focus_monitoring": True,
            "distraction_detection": True,
            "pattern_analysis": True,
            "technique_recommendation": True,
            "citation_provision": True
        }
        
        # Configure model for focus and productivity analysis
        self.model_config.update({
            "temperature": 0.7,  # Higher temperature for more creative recommendations
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })
        
        # Initialize session tracking
        self.active_sessions: Dict[str, FocusSession] = {}
        self.distraction_history: List[DistractionEvent] = []
        self.recommendation_history: List[ProductivityRecommendation] = []
        
        # Load productivity research and techniques
        self._load_productivity_research()

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a focus and flow monitoring request."""
        try:
            action = request.get("action", "monitor")
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Process based on action
            if action == "monitor":
                result = await self._monitor_focus(data, context, parameters)
            elif action == "analyze_patterns":
                result = await self._analyze_patterns(data, context, parameters)
            elif action == "get_recommendations":
                result = await self._get_recommendations(data, context, parameters)
            elif action == "track_distraction":
                result = await self._track_distraction(data, context, parameters)
            elif action == "end_session":
                result = await self._end_session(data, context, parameters)
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
            logger.error(f"Error processing focus and flow request: {str(e)}")
            return {"error": str(e)}

    async def _monitor_focus(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monitor current focus state and session."""
        try:
            # Prepare system prompt for focus monitoring
            system_prompt = self._create_monitoring_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "current_state": data.get("current_state", {}),
                    "session_data": data.get("session_data", {}),
                    "context": context
                }),
                parameters
            )
            
            # Parse monitoring results
            monitoring = self._parse_monitoring(response)
            
            # Update active session if needed
            if monitoring.get("should_start_session"):
                session = self._start_new_session(monitoring["focus_state"])
                self.active_sessions[session.id] = session
            
            return {
                "focus_state": monitoring["focus_state"],
                "session_id": monitoring.get("session_id"),
                "productivity_score": monitoring["productivity_score"],
                "recommendations": monitoring.get("immediate_recommendations", []),
                "citations": monitoring.get("citations", [])
            }
            
        except Exception as e:
            logger.error(f"Error monitoring focus: {str(e)}")
            raise

    async def _analyze_patterns(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze focus and productivity patterns."""
        try:
            # Prepare system prompt for pattern analysis
            system_prompt = self._create_analysis_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "session_history": data.get("session_history", []),
                    "distraction_history": data.get("distraction_history", []),
                    "context": context
                }),
                parameters
            )
            
            # Parse analysis results
            analysis = self._parse_analysis(response)
            
            return {
                "patterns": analysis["patterns"],
                "insights": analysis["insights"],
                "recommendations": analysis["recommendations"],
                "citations": analysis["citations"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            raise

    async def _get_recommendations(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get personalized productivity recommendations."""
        try:
            # Prepare system prompt for recommendations
            system_prompt = self._create_recommendation_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "user_profile": data.get("user_profile", {}),
                    "work_patterns": data.get("work_patterns", {}),
                    "context": context
                }),
                parameters
            )
            
            # Parse recommendations
            recommendations = self._parse_recommendations(response)
            
            # Store recommendations
            for rec in recommendations["recommendations"]:
                self.recommendation_history.append(rec)
            
            return {
                "recommendations": [
                    {
                        "id": rec.id,
                        "technique": rec.technique.value,
                        "description": rec.description,
                        "rationale": rec.rationale,
                        "implementation_steps": rec.implementation_steps,
                        "expected_benefits": rec.expected_benefits,
                        "citations": rec.citations,
                        "confidence": rec.confidence,
                        "created_at": rec.created_at.isoformat()
                    }
                    for rec in recommendations["recommendations"]
                ],
                "summary": recommendations["summary"],
                "citations": recommendations["citations"]
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    async def _track_distraction(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track a distraction event."""
        try:
            # Create distraction event
            event = DistractionEvent(
                id=str(uuid4()),
                timestamp=datetime.utcnow(),
                type=DistractionType(data["type"]),
                source=data["source"],
                duration=timedelta(seconds=data.get("duration_seconds", 0)),
                impact_score=data.get("impact_score", 0.0),
                context=data.get("context", {})
            )
            
            # Store event
            self.distraction_history.append(event)
            
            # Update active session if exists
            if data.get("session_id") in self.active_sessions:
                session = self.active_sessions[data["session_id"]]
                session.distractions.append({
                    "id": event.id,
                    "type": event.type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "duration": event.duration.total_seconds(),
                    "impact_score": event.impact_score
                })
            
            return {
                "event_id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "impact_assessment": self._assess_distraction_impact(event),
                "recommendations": self._get_distraction_recommendations(event)
            }
            
        except Exception as e:
            logger.error(f"Error tracking distraction: {str(e)}")
            raise

    async def _end_session(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """End a focus session and generate summary."""
        try:
            session_id = data.get("session_id")
            if not session_id or session_id not in self.active_sessions:
                raise ValueError("Invalid session ID")
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.utcnow()
            session.duration = session.end_time - session.start_time
            
            # Calculate final productivity score
            session.productivity_score = self._calculate_session_score(session)
            
            # Generate session summary
            summary = await self._generate_session_summary(session)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            return {
                "session_id": session.id,
                "duration": session.duration.total_seconds(),
                "productivity_score": session.productivity_score,
                "distraction_count": len(session.distractions),
                "summary": summary,
                "recommendations": summary.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            raise

    def _create_monitoring_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for focus monitoring."""
        return f"""You are a specialized focus and flow monitoring AI. Your task is to analyze the current state and provide insights.
        
        Context:
        - User Profile: {context.get('user_profile', 'general')}
        - Environment: {context.get('environment', 'general')}
        - Time of Day: {context.get('time_of_day', 'unknown')}
        - Previous State: {context.get('previous_state', 'unknown')}
        
        Analyze:
        1. Current focus state (deep work, flow, distracted, break, transition)
        2. Session progress and quality
        3. Immediate recommendations
        4. Supporting research and citations
        
        Provide detailed analysis with specific insights and actionable recommendations.
        """

    def _create_analysis_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for pattern analysis."""
        return f"""You are a specialized focus and productivity pattern analyst. Your task is to identify patterns and provide insights.
        
        Context:
        - Time Period: {context.get('time_period', 'recent')}
        - Focus Areas: {context.get('focus_areas', 'all')}
        - Analysis Depth: {parameters.get('analysis_depth', 'standard')}
        
        Analyze:
        1. Focus session patterns
        2. Distraction patterns
        3. Productivity trends
        4. Optimal work conditions
        
        Provide detailed analysis with specific patterns, insights, and research-backed recommendations.
        """

    def _create_recommendation_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for recommendations."""
        return f"""You are a specialized productivity technique advisor. Your task is to provide personalized recommendations.
        
        Context:
        - User Profile: {context.get('user_profile', 'general')}
        - Work Style: {context.get('work_style', 'general')}
        - Goals: {context.get('goals', 'general')}
        - Constraints: {context.get('constraints', 'none')}
        
        Consider:
        1. User's work patterns and preferences
        2. Current challenges and pain points
        3. Available techniques and their effectiveness
        4. Research evidence and citations
        
        Provide personalized recommendations with implementation steps and expected benefits.
        """

    def _start_new_session(self, initial_state: FocusState) -> FocusSession:
        """Start a new focus session."""
        session = FocusSession(
            id=str(uuid4()),
            start_time=datetime.utcnow(),
            end_time=None,
            state=initial_state,
            duration=timedelta(0),
            distractions=[],
            productivity_score=0.0,
            technique_used=None,
            notes=None
        )
        return session

    def _calculate_session_score(self, session: FocusSession) -> float:
        """Calculate productivity score for a session."""
        # Base score from duration and state
        base_score = min(session.duration.total_seconds() / 3600, 1.0)  # Normalize to 1 hour
        
        # Adjust for focus state
        state_multipliers = {
            FocusState.DEEP_WORK: 1.2,
            FocusState.FLOW: 1.5,
            FocusState.DISTRACTED: 0.5,
            FocusState.BREAK: 0.8,
            FocusState.TRANSITION: 0.9
        }
        state_score = base_score * state_multipliers.get(session.state, 1.0)
        
        # Penalize for distractions
        distraction_penalty = sum(
            d.get("impact_score", 0.0) * 0.1
            for d in session.distractions
        )
        
        return max(0.0, min(1.0, state_score - distraction_penalty))

    def _assess_distraction_impact(self, event: DistractionEvent) -> Dict[str, Any]:
        """Assess the impact of a distraction event."""
        # Calculate impact based on type and duration
        impact_factors = {
            DistractionType.NOTIFICATION: 0.3,
            DistractionType.SOCIAL_MEDIA: 0.8,
            DistractionType.EMAIL: 0.5,
            DistractionType.MULTITASKING: 0.7,
            DistractionType.ENVIRONMENTAL: 0.4,
            DistractionType.INTERNAL: 0.6
        }
        
        base_impact = impact_factors.get(event.type, 0.5)
        duration_factor = min(event.duration.total_seconds() / 300, 1.0)  # Normalize to 5 minutes
        
        return {
            "severity": base_impact * duration_factor,
            "recovery_time": duration_factor * 2,  # Estimated recovery time in minutes
            "focus_impact": base_impact * (1 + duration_factor)
        }

    def _get_distraction_recommendations(self, event: DistractionEvent) -> List[Dict[str, Any]]:
        """Get recommendations for handling a distraction."""
        recommendations = []
        
        if event.type == DistractionType.NOTIFICATION:
            recommendations.append({
                "type": "notification_management",
                "suggestion": "Configure notification settings to minimize interruptions",
                "steps": [
                    "Set up focus mode",
                    "Schedule notification batches",
                    "Prioritize important notifications"
                ]
            })
        elif event.type == DistractionType.SOCIAL_MEDIA:
            recommendations.append({
                "type": "digital_wellbeing",
                "suggestion": "Use website blockers during focus sessions",
                "steps": [
                    "Install a website blocker",
                    "Schedule blocking periods",
                    "Set up accountability measures"
                ]
            })
        # Add more recommendations for other distraction types
        
        return recommendations

    async def _generate_session_summary(self, session: FocusSession) -> Dict[str, Any]:
        """Generate a summary of a focus session."""
        try:
            # Prepare system prompt for summary generation
            system_prompt = """You are a specialized focus session analyst. Generate a detailed summary of the session."""
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "session": {
                        "duration": session.duration.total_seconds(),
                        "state": session.state.value,
                        "productivity_score": session.productivity_score,
                        "distractions": session.distractions
                    }
                }),
                {}
            )
            
            # Parse summary
            summary = json.loads(response)
            
            return {
                "overview": summary.get("overview", ""),
                "achievements": summary.get("achievements", []),
                "challenges": summary.get("challenges", []),
                "recommendations": summary.get("recommendations", []),
                "citations": summary.get("citations", [])
            }
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            return {
                "overview": "Session completed",
                "achievements": [],
                "challenges": [],
                "recommendations": [],
                "citations": []
            }

    def _load_productivity_research(self) -> None:
        """Load productivity research and techniques."""
        # In a real implementation, this would load from a database or file
        self.productivity_research = {
            "techniques": {
                ProductivityTechnique.POMODORO: {
                    "description": "Work in focused 25-minute intervals followed by short breaks",
                    "benefits": [
                        "Improved focus and concentration",
                        "Reduced mental fatigue",
                        "Better time management"
                    ],
                    "citations": [
                        {
                            "title": "The Pomodoro Technique: A Time Management Method",
                            "author": "Cirillo, F.",
                            "year": "2018",
                            "url": "https://francescocirillo.com/pages/pomodoro-technique"
                        }
                    ]
                },
                ProductivityTechnique.DEEP_WORK: {
                    "description": "Extended periods of focused, distraction-free work",
                    "benefits": [
                        "Higher quality output",
                        "Faster skill development",
                        "Greater satisfaction"
                    ],
                    "citations": [
                        {
                            "title": "Deep Work: Rules for Focused Success in a Distracted World",
                            "author": "Newport, C.",
                            "year": "2016",
                            "url": "https://www.calnewport.com/books/deep-work/"
                        }
                    ]
                }
                # Add more techniques and research
            }
        } 