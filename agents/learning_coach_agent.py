from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime
import json
import re

from .base_agent import BaseAgent
# Removing circular import - agent_service already imports learning_coach_agent

logger = logging.getLogger(__name__)

class LearningMode(str, Enum):
    """Enum for different learning support modes."""
    LEARNING_PATH = "learning_path"
    QUIZ_GENERATION = "quiz_generation"
    CONCEPT_EXPLANATION = "concept_explanation"
    PROGRESS_TRACKING = "progress_tracking"
    RESOURCE_RECOMMENDATION = "resource_recommendation"
    ADAPTIVE_LEARNING = "adaptive_learning"

class LearningCoachAgent(BaseAgent):
    """Agent specialized in providing comprehensive educational support and personalized learning experiences."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Learning Coach Agent with specific capabilities."""
        if "agent_model_config" not in config:
            config["agent_model_config"] = config.get("model_config", {})
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "learning_path_creation": True,
            "quiz_generation": True,
            "concept_explanation": True,
            "progress_tracking": True,
            "resource_recommendation": True,
            "adaptive_learning": True
        }
        
        # Configure learning modes and parameters
        self.modes = {
            LearningMode.LEARNING_PATH: {
                "description": "Create personalized learning paths based on user goals",
                "parameters": {
                    "difficulty_level": "intermediate",
                    "learning_style": "visual",
                    "time_commitment": "medium",
                    "include_assessments": True,
                    "adaptive_pacing": True
                }
            },
            LearningMode.QUIZ_GENERATION: {
                "description": "Generate quizzes and exercises for knowledge retention",
                "parameters": {
                    "question_types": ["multiple_choice", "short_answer", "true_false"],
                    "difficulty_distribution": {"easy": 0.3, "medium": 0.4, "hard": 0.3},
                    "include_explanations": True,
                    "adaptive_difficulty": True
                }
            },
            LearningMode.CONCEPT_EXPLANATION: {
                "description": "Provide explanations of complex concepts",
                "parameters": {
                    "explanation_depth": "detailed",
                    "include_examples": True,
                    "include_visuals": True,
                    "include_practice": True,
                    "language_level": "intermediate"
                }
            },
            LearningMode.PROGRESS_TRACKING: {
                "description": "Track learning progress and suggest improvements",
                "parameters": {
                    "tracking_metrics": ["completion", "accuracy", "time_spent", "engagement"],
                    "assessment_frequency": "weekly",
                    "include_recommendations": True,
                    "generate_reports": True
                }
            },
            LearningMode.RESOURCE_RECOMMENDATION: {
                "description": "Recommend learning resources across different formats",
                "parameters": {
                    "resource_types": ["video", "article", "book", "exercise", "interactive"],
                    "difficulty_match": True,
                    "include_ratings": True,
                    "personalized_recommendations": True
                }
            },
            LearningMode.ADAPTIVE_LEARNING: {
                "description": "Adapt learning content based on user performance and preferences",
                "parameters": {
                    "adaptation_frequency": "continuous",
                    "performance_threshold": 0.7,
                    "include_feedback": True,
                    "adjust_difficulty": True
                }
            }
        }
        
        # Set default model configuration
        self.model_config.update({
            "temperature": 0.7,  # Higher temperature for more creative and varied responses
            "max_tokens": 8192,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a learning support request asynchronously."""
        try:
            mode = request.get("mode", LearningMode.LEARNING_PATH.value)
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Validate mode
            if mode not in [m.value for m in LearningMode]:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Process based on mode
            if mode == LearningMode.LEARNING_PATH.value:
                result = await self._create_learning_path(data, context, parameters)
            elif mode == LearningMode.QUIZ_GENERATION.value:
                result = await self._generate_quiz(data, context, parameters)
            elif mode == LearningMode.CONCEPT_EXPLANATION.value:
                result = await self._explain_concept(data, context, parameters)
            elif mode == LearningMode.PROGRESS_TRACKING.value:
                result = await self._track_progress(data, context, parameters)
            elif mode == LearningMode.RESOURCE_RECOMMENDATION.value:
                result = await self._recommend_resources(data, context, parameters)
            elif mode == LearningMode.ADAPTIVE_LEARNING.value:
                result = await self._adapt_learning(data, context, parameters)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            return {
                "results": result,
                "metadata": {
                    "mode": mode,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": self.get_model_name(),
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing learning support request: {str(e)}")
            return {"error": str(e)}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the learning coach agent asynchronously."""
        return await self.process_async({
            "mode": LearningMode.CONCEPT_EXPLANATION.value,
            "data": {"concept": query},
            "context": context or {}
        })

    async def _process_with_model(
        self,
        system_prompt: str,
        user_content: str,
        parameters: Dict[str, Any],
    ) -> str:
        """Call the LLM with a system prompt and user content, return raw text response."""
        response_text = await self._call_model_async(
            prompt=user_content,
            system_prompt=system_prompt,
            temperature=parameters.get("temperature", 0.7),
            max_tokens=parameters.get("max_tokens", 8192),
        )
        return response_text

    async def _create_learning_path(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a personalized learning path based on user goals and preferences."""
        try:
            # Prepare system prompt for learning path creation
            system_prompt = self._create_system_prompt(
                LearningMode.LEARNING_PATH,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "goals": data.get("goals", []),
                    "current_level": data.get("current_level", "beginner"),
                    "preferences": data.get("preferences", {}),
                    "constraints": data.get("constraints", {})
                }),
                parameters
            )
            
            # Parse and structure the response
            learning_path = self._parse_learning_path(response)
            
            return {
                "learning_path": learning_path,
                "metadata": {
                    "estimated_duration": self._calculate_duration(learning_path),
                    "difficulty_level": parameters.get("difficulty_level", "intermediate"),
                    "adaptation_level": self._calculate_adaptation_level(learning_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating learning path: {str(e)}")
            raise

    async def _generate_quiz(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate quizzes and exercises for knowledge retention."""
        try:
            # Prepare system prompt for quiz generation
            system_prompt = self._create_system_prompt(
                LearningMode.QUIZ_GENERATION,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "topic": data.get("topic", ""),
                    "subtopics": data.get("subtopics", []),
                    "previous_performance": data.get("previous_performance", {}),
                    "learning_style": data.get("learning_style", "visual")
                }),
                parameters
            )
            
            # Parse and structure the response
            quiz = self._parse_quiz(response)
            
            return {
                "quiz": quiz,
                "metadata": {
                    "difficulty_distribution": self._calculate_difficulty_distribution(quiz),
                    "estimated_completion_time": self._calculate_quiz_duration(quiz),
                    "coverage_score": self._calculate_topic_coverage(quiz, data.get("topic", ""))
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            raise

    async def _explain_concept(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide explanations of complex concepts."""
        try:
            # Prepare system prompt for concept explanation
            system_prompt = self._create_system_prompt(
                LearningMode.CONCEPT_EXPLANATION,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "concept": data.get("concept", ""),
                    "prerequisites": data.get("prerequisites", []),
                    "learning_style": data.get("learning_style", "visual"),
                    "current_understanding": data.get("current_understanding", "basic")
                }),
                parameters
            )
            
            # Parse and structure the response
            explanation = self._parse_explanation(response)
            
            return {
                "explanation": explanation,
                "metadata": {
                    "complexity_level": self._calculate_complexity_level(explanation),
                    "prerequisite_coverage": self._calculate_prerequisite_coverage(explanation, data.get("prerequisites", [])),
                    "visual_aid_recommendations": self._generate_visual_aid_recommendations(explanation)
                }
            }
            
        except Exception as e:
            logger.error(f"Error explaining concept: {str(e)}")
            raise

    async def _track_progress(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track learning progress and suggest improvements."""
        try:
            # Prepare system prompt for progress tracking
            system_prompt = self._create_system_prompt(
                LearningMode.PROGRESS_TRACKING,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "learning_history": data.get("learning_history", []),
                    "assessment_results": data.get("assessment_results", []),
                    "goals": data.get("goals", []),
                    "time_spent": data.get("time_spent", {})
                }),
                parameters
            )
            
            # Parse and structure the response
            progress = self._parse_progress(response)
            
            return {
                "progress": progress,
                "recommendations": self._generate_progress_recommendations(progress),
                "metadata": {
                    "completion_rate": self._calculate_completion_rate(progress),
                    "mastery_level": self._calculate_mastery_level(progress),
                    "improvement_areas": self._identify_improvement_areas(progress)
                }
            }
            
        except Exception as e:
            logger.error(f"Error tracking progress: {str(e)}")
            raise

    async def _recommend_resources(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend learning resources across different formats."""
        try:
            # Prepare system prompt for resource recommendation
            system_prompt = self._create_system_prompt(
                LearningMode.RESOURCE_RECOMMENDATION,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "topics": data.get("topics", []),
                    "learning_style": data.get("learning_style", "visual"),
                    "difficulty_level": data.get("difficulty_level", "intermediate"),
                    "preferred_formats": data.get("preferred_formats", ["video", "article"])
                }),
                parameters
            )
            
            # Parse and structure the response
            recommendations = self._parse_recommendations(response)
            
            return {
                "recommendations": recommendations,
                "metadata": {
                    "format_distribution": self._calculate_format_distribution(recommendations),
                    "difficulty_match_score": self._calculate_difficulty_match(recommendations, data.get("difficulty_level", "intermediate")),
                    "personalization_score": self._calculate_personalization_score(recommendations, data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error recommending resources: {str(e)}")
            raise

    async def _adapt_learning(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt learning content based on user performance and preferences."""
        try:
            # Prepare system prompt for adaptive learning
            system_prompt = self._create_system_prompt(
                LearningMode.ADAPTIVE_LEARNING,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "performance_data": data.get("performance_data", {}),
                    "learning_style": data.get("learning_style", "visual"),
                    "current_content": data.get("current_content", {}),
                    "goals": data.get("goals", [])
                }),
                parameters
            )
            
            # Parse and structure the response
            adaptations = self._parse_adaptations(response)
            
            return {
                "adaptations": adaptations,
                "metadata": {
                    "adaptation_level": self._calculate_adaptation_level(adaptations),
                    "performance_impact": self._estimate_performance_impact(adaptations),
                    "learning_style_match": self._calculate_learning_style_match(adaptations, data.get("learning_style", "visual"))
                }
            }
            
        except Exception as e:
            logger.error(f"Error adapting learning: {str(e)}")
            raise

    def _create_system_prompt(
        self,
        mode: LearningMode,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create a system prompt based on the learning mode and parameters."""
        base_prompt = f"""You are a specialized learning coach AI assistant. Your task is to {self.modes[mode]['description']}.
        
        Context:
        - User Level: {context.get('user_level', 'intermediate')}
        - Learning Style: {context.get('learning_style', 'visual')}
        - Subject Area: {context.get('subject_area', 'general')}
        
        Parameters:
        {self._format_parameters(parameters)}
        
        Please provide comprehensive, engaging, and personalized learning support.
        Use markdown formatting with headers, bullet points, and bold text for clarity.
        Focus on creating an effective and enjoyable learning experience.
        """
        
        # Add mode-specific instructions
        if mode == LearningMode.LEARNING_PATH:
            base_prompt += """
            For learning path creation:
            1. Create a structured, personalized learning journey
            2. Include clear milestones and checkpoints
            3. Consider user's learning style and preferences
            4. Provide estimated time commitments
            5. Include assessment points
            """
        elif mode == LearningMode.QUIZ_GENERATION:
            base_prompt += """
            For quiz generation:
            1. Create diverse question types
            2. Include detailed explanations
            3. Match difficulty to user level
            4. Cover key concepts thoroughly
            5. Provide immediate feedback
            """
        elif mode == LearningMode.CONCEPT_EXPLANATION:
            base_prompt += """
            For concept explanation:
            1. Break down complex ideas
            2. Use clear, accessible language
            3. Include relevant examples
            4. Provide visual aids when helpful
            5. Connect to prior knowledge
            """
        elif mode == LearningMode.PROGRESS_TRACKING:
            base_prompt += """
            For progress tracking:
            1. Track multiple learning metrics
            2. Identify strengths and areas for improvement
            3. Provide actionable recommendations
            4. Celebrate achievements
            5. Adjust goals as needed
            """
        elif mode == LearningMode.RESOURCE_RECOMMENDATION:
            base_prompt += """
            For resource recommendation:
            1. Suggest diverse learning materials
            2. Match resources to learning style
            3. Consider difficulty level
            4. Include user ratings and reviews
            5. Provide clear learning objectives
            """
        elif mode == LearningMode.ADAPTIVE_LEARNING:
            base_prompt += """
            For adaptive learning:
            1. Analyze performance patterns
            2. Adjust content difficulty
            3. Modify learning approach
            4. Provide personalized feedback
            5. Update learning path dynamically
            """
        
        return base_prompt

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for the system prompt."""
        return "\n".join([f"- {k}: {v}" for k, v in parameters.items()])

    # ── Text extraction helper ─────────────────────────────────────────

    def _get_text(self, data) -> str:
        """Extract the raw text content from a parsed response."""
        if isinstance(data, dict):
            return data.get("content", "")
        if isinstance(data, list):
            return str(data)
        return str(data) if data else ""

    # ── Parse methods ────────────────────────────────────────────────

    def _parse_learning_path(self, response: str) -> Dict[str, Any]:
        return {"content": response}

    def _parse_quiz(self, response: str) -> Dict[str, Any]:
        return {"content": response}

    def _parse_explanation(self, response: str) -> Dict[str, Any]:
        return {"content": response}

    def _parse_progress(self, response: str) -> Dict[str, Any]:
        return {"content": response}

    def _parse_recommendations(self, response: str) -> List[Dict[str, Any]]:
        return [{"content": response}]

    def _parse_adaptations(self, response: str) -> Dict[str, Any]:
        return {"content": response}

    def _generate_progress_recommendations(self, progress: Dict[str, Any]) -> List[str]:
        """Extract recommendation keywords from progress text."""
        text = self._get_text(progress).lower()
        recs = []
        rec_keywords = {
            "practice more": "Increase practice frequency",
            "review": "Review previous material",
            "focus on": "Focus on weak areas",
            "spaced repetition": "Use spaced repetition",
            "active recall": "Practice active recall",
            "break": "Take regular breaks",
            "quiz": "Self-test with quizzes",
        }
        for keyword, label in rec_keywords.items():
            if keyword in text and label not in recs:
                recs.append(label)
        return recs if recs else ["Continue current learning pace"]

    # ── Learning Path helpers ────────────────────────────────────────

    def _calculate_duration(self, learning_path: Dict[str, Any]) -> str:
        text = self._get_text(learning_path).lower()
        duration_matches = re.findall(r"(\d+)\s*(weeks?|months?|days?|hours?)", text)
        if duration_matches:
            max_val, max_unit = 0, ""
            for val, unit in duration_matches:
                num = int(val)
                days = num * (30 if "month" in unit else 7 if "week" in unit else 1 if "day" in unit else 0)
                if days > max_val:
                    max_val = days
                    max_unit = f"{val} {unit}"
            return max_unit
        word_count = len(text.split())
        if word_count > 2000:
            return "4-8 weeks (estimated)"
        return "2-4 weeks (estimated)"

    def _calculate_adaptation_level(self, content: Dict[str, Any]) -> float:
        text = self._get_text(content).lower()
        indicators = ["adapt", "personalize", "adjust", "tailor", "custom",
                       "individual", "preference", "style", "pace", "level"]
        found = sum(1 for ind in indicators if ind in text)
        return round(min(found / 6, 1.0), 2)

    # ── Quiz helpers ─────────────────────────────────────────────────

    def _calculate_difficulty_distribution(self, quiz: Dict[str, Any]) -> Dict[str, float]:
        text = self._get_text(quiz).lower()
        easy = len(re.findall(r"\b(easy|basic|simple|beginner)\b", text))
        medium = len(re.findall(r"\b(medium|moderate|intermediate)\b", text))
        hard = len(re.findall(r"\b(hard|difficult|advanced|challenging)\b", text))
        total = easy + medium + hard
        if total == 0:
            return {"easy": 0.33, "medium": 0.34, "hard": 0.33}
        return {
            "easy": round(easy / total, 2),
            "medium": round(medium / total, 2),
            "hard": round(hard / total, 2),
        }

    def _calculate_quiz_duration(self, quiz: Dict[str, Any]) -> int:
        text = self._get_text(quiz)
        questions = len(re.findall(r"(?:question|Q)\s*\d+|^\s*\d+[\.\)]\s", text, re.MULTILINE))
        return max(questions * 2, 5)  # ~2 min per question, minimum 5 min

    def _calculate_topic_coverage(self, quiz: Dict[str, Any], topic: str) -> float:
        text = self._get_text(quiz).lower()
        if not topic:
            return 0.5
        topic_words = topic.lower().split()
        found = sum(1 for w in topic_words if w in text)
        word_coverage = found / max(len(topic_words), 1)
        questions = len(re.findall(r"(?:question|Q)\s*\d+|^\s*\d+[\.\)]\s", text, re.MULTILINE))
        depth = min(questions / 10, 1.0)
        return round((word_coverage * 0.4 + depth * 0.6), 2)

    # ── Concept Explanation helpers ──────────────────────────────────

    def _calculate_complexity_level(self, explanation: Dict[str, Any]) -> str:
        text = self._get_text(explanation)
        word_count = len(text.split())
        headers = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        if word_count > 2000 or headers > 8:
            return "advanced"
        elif word_count > 800 or headers > 4:
            return "intermediate"
        return "beginner"

    def _calculate_prerequisite_coverage(self, explanation: Dict[str, Any], prerequisites: List[str]) -> float:
        if not prerequisites:
            return 1.0
        text = self._get_text(explanation).lower()
        found = sum(1 for p in prerequisites if p.lower() in text)
        return round(found / len(prerequisites), 2)

    def _generate_visual_aid_recommendations(self, explanation: Dict[str, Any]) -> List[str]:
        text = self._get_text(explanation).lower()
        aids = []
        visual_keywords = {
            "diagram": "Flowchart or diagram",
            "graph": "Graph visualization",
            "chart": "Chart or infographic",
            "table": "Comparison table",
            "flowchart": "Process flowchart",
            "timeline": "Timeline visualization",
            "map": "Concept map",
            "hierarchy": "Hierarchy diagram",
            "formula": "Formula reference sheet",
            "code": "Code snippet examples",
        }
        for keyword, label in visual_keywords.items():
            if keyword in text and label not in aids:
                aids.append(label)
        return aids if aids else ["Concept map", "Summary infographic"]

    # ── Progress Tracking helpers ────────────────────────────────────

    def _calculate_completion_rate(self, progress: Dict[str, Any]) -> float:
        text = self._get_text(progress).lower()
        pct_matches = re.findall(r"(\d+)%", text)
        if pct_matches:
            values = [int(p) for p in pct_matches if 0 <= int(p) <= 100]
            if values:
                return round(sum(values) / len(values) / 100, 2)
        completed = len(re.findall(r"\b(completed|done|finished|mastered)\b", text))
        total_items = len(re.findall(r"^\s*[-*\d]", text, re.MULTILINE))
        if total_items > 0:
            return round(min(completed / max(total_items, 1), 1.0), 2)
        return 0.5

    def _calculate_mastery_level(self, progress: Dict[str, Any]) -> str:
        text = self._get_text(progress).lower()
        if any(w in text for w in ["expert", "mastery", "mastered", "proficient"]):
            return "expert"
        elif any(w in text for w in ["advanced", "strong", "solid"]):
            return "advanced"
        elif any(w in text for w in ["intermediate", "developing", "progressing"]):
            return "intermediate"
        return "beginner"

    def _identify_improvement_areas(self, progress: Dict[str, Any]) -> List[str]:
        text = self._get_text(progress)
        areas = []
        improve_section = re.search(
            r"(?:improv|weak|gap|struggle|need|focus|work on).*?(?=\n#{1,3}\s|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        if improve_section:
            bullets = re.findall(r"[-*]\s*(.{10,80})", improve_section.group())
            areas = [b.strip() for b in bullets[:5]]
        if not areas:
            keywords = re.findall(r"(?:improve|focus on|work on|strengthen)\s+(.{5,50}?)(?:[,\.\n])", text, re.IGNORECASE)
            areas = [k.strip() for k in keywords[:5]]
        return areas

    # ── Resource Recommendation helpers ──────────────────────────────

    def _calculate_format_distribution(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        text = self._get_text(recommendations[0]) if recommendations else ""
        text_lower = text.lower()
        formats = {
            "video": len(re.findall(r"\bvideo|youtube|course\b", text_lower)),
            "article": len(re.findall(r"\barticle|blog|post|read\b", text_lower)),
            "book": len(re.findall(r"\bbook|textbook\b", text_lower)),
            "exercise": len(re.findall(r"\bexercise|practice|hands-on|lab\b", text_lower)),
            "interactive": len(re.findall(r"\binteractive|simulation|tool\b", text_lower)),
        }
        total = sum(formats.values())
        if total == 0:
            return {"video": 0.2, "article": 0.2, "book": 0.2, "exercise": 0.2, "interactive": 0.2}
        return {k: round(v / total, 2) for k, v in formats.items()}

    def _calculate_difficulty_match(self, recommendations: List[Dict[str, Any]], target_difficulty: str) -> float:
        text = self._get_text(recommendations[0]).lower() if recommendations else ""
        if not target_difficulty:
            return 0.5
        target = target_difficulty.lower()
        mentions = len(re.findall(rf"\b{re.escape(target)}\b", text))
        return round(min(mentions / 3, 1.0), 2) if mentions else 0.3

    def _calculate_personalization_score(self, recommendations: List[Dict[str, Any]], user_data: Dict[str, Any]) -> float:
        text = self._get_text(recommendations[0]).lower() if recommendations else ""
        score = 0.3  # base
        if user_data.get("learning_style") and user_data["learning_style"].lower() in text:
            score += 0.25
        if user_data.get("difficulty_level") and user_data["difficulty_level"].lower() in text:
            score += 0.25
        topics = user_data.get("topics", [])
        if topics:
            found = sum(1 for t in topics if t.lower() in text)
            score += min(found / len(topics), 1.0) * 0.2
        return round(min(score, 1.0), 2)

    # ── Adaptive Learning helpers ────────────────────────────────────

    def _estimate_performance_impact(self, adaptations: Dict[str, Any]) -> float:
        text = self._get_text(adaptations).lower()
        positive = ["improve", "increase", "enhance", "boost", "accelerate",
                     "strengthen", "optimize", "effective", "efficient"]
        negative = ["decrease", "slower", "difficult", "struggle", "decline"]
        pos_count = sum(1 for ind in positive if ind in text)
        neg_count = sum(1 for ind in negative if ind in text)
        impact = 0.5 + min(pos_count / 6, 0.4) - min(neg_count / 4, 0.3)
        return round(max(min(impact, 1.0), 0.1), 2)

    def _calculate_learning_style_match(self, adaptations: Dict[str, Any], learning_style: str) -> float:
        text = self._get_text(adaptations).lower()
        if not learning_style:
            return 0.5
        style = learning_style.lower()
        style_keywords = {
            "visual": ["visual", "diagram", "chart", "image", "video", "infographic", "illustration"],
            "auditory": ["audio", "podcast", "lecture", "discussion", "verbal", "listen"],
            "kinesthetic": ["hands-on", "practice", "exercise", "lab", "interactive", "build", "experiment"],
            "reading": ["read", "text", "article", "book", "document", "written", "notes"],
        }
        keywords = style_keywords.get(style, [style])
        found = sum(1 for kw in keywords if kw in text)
        return round(min(found / 4, 1.0), 2)