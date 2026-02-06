from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime
import json

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
            "max_tokens": 4096,
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
                    "model": self.model_config.get("model", "sonar-large-online"),
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing learning support request: {str(e)}")
            return {"error": str(e)}

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

    # Helper methods for parsing and analysis
    def _parse_learning_path(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into a structured learning path."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_quiz(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into a structured quiz."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_explanation(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into a structured explanation."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_progress(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured progress data."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_recommendations(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response into structured resource recommendations."""
        # Implementation would parse the response into a structured format
        return []

    def _parse_adaptations(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured learning adaptations."""
        # Implementation would parse the response into a structured format
        return {}

    # Analysis and calculation methods
    def _calculate_duration(self, learning_path: Dict[str, Any]) -> str:
        """Calculate estimated duration for a learning path."""
        # Implementation would calculate duration based on learning path
        return ""

    def _calculate_adaptation_level(self, content: Dict[str, Any]) -> float:
        """Calculate the level of adaptation in learning content."""
        # Implementation would calculate adaptation level
        return 0.0

    def _calculate_difficulty_distribution(self, quiz: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the distribution of difficulty levels in a quiz."""
        # Implementation would calculate difficulty distribution
        return {"easy": 0.0, "medium": 0.0, "hard": 0.0}

    def _calculate_quiz_duration(self, quiz: Dict[str, Any]) -> int:
        """Calculate estimated completion time for a quiz."""
        # Implementation would calculate quiz duration
        return 0

    def _calculate_topic_coverage(self, quiz: Dict[str, Any], topic: str) -> float:
        """Calculate how well a quiz covers a topic."""
        # Implementation would calculate topic coverage
        return 0.0

    def _calculate_complexity_level(self, explanation: Dict[str, Any]) -> str:
        """Calculate the complexity level of an explanation."""
        # Implementation would calculate complexity level
        return ""

    def _calculate_prerequisite_coverage(self, explanation: Dict[str, Any], prerequisites: List[str]) -> float:
        """Calculate how well an explanation covers prerequisites."""
        # Implementation would calculate prerequisite coverage
        return 0.0

    def _generate_visual_aid_recommendations(self, explanation: Dict[str, Any]) -> List[str]:
        """Generate recommendations for visual aids."""
        # Implementation would generate visual aid recommendations
        return []

    def _calculate_completion_rate(self, progress: Dict[str, Any]) -> float:
        """Calculate the completion rate of learning activities."""
        # Implementation would calculate completion rate
        return 0.0

    def _calculate_mastery_level(self, progress: Dict[str, Any]) -> str:
        """Calculate the mastery level based on progress."""
        # Implementation would calculate mastery level
        return ""

    def _identify_improvement_areas(self, progress: Dict[str, Any]) -> List[str]:
        """Identify areas needing improvement."""
        # Implementation would identify improvement areas
        return []

    def _calculate_format_distribution(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the distribution of resource formats."""
        # Implementation would calculate format distribution
        return {}

    def _calculate_difficulty_match(self, recommendations: List[Dict[str, Any]], target_difficulty: str) -> float:
        """Calculate how well recommendations match target difficulty."""
        # Implementation would calculate difficulty match
        return 0.0

    def _calculate_personalization_score(self, recommendations: List[Dict[str, Any]], user_data: Dict[str, Any]) -> float:
        """Calculate how well recommendations are personalized."""
        # Implementation would calculate personalization score
        return 0.0

    def _estimate_performance_impact(self, adaptations: Dict[str, Any]) -> float:
        """Estimate the potential impact of learning adaptations."""
        # Implementation would estimate performance impact
        return 0.0

    def _calculate_learning_style_match(self, adaptations: Dict[str, Any], learning_style: str) -> float:
        """Calculate how well adaptations match learning style."""
        # Implementation would calculate learning style match
        return 0.0 