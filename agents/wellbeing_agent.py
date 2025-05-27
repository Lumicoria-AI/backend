from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional
import json
import structlog
import asyncio
from datetime import datetime
import re

# Configure logger
logger = structlog.get_logger(__name__)

# Well-being coach agent
class WellbeingAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"

    def process(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user wellbeing data and provide personalized recommendations.
        
        Args:
            user_data: Dictionary containing user activity data, metrics, and context
            
        Returns:
            Dictionary with wellbeing recommendations and insights
        """
        try:
            # Format user data for better prompt context
            formatted_context = self._format_user_data(user_data)
            
            prompt = (
                f"Based on the following user activity and wellbeing data, provide personalized wellbeing "
                f"recommendations. Include specific suggestions for breaks, focus techniques, stress management, "
                f"and overall wellbeing improvement. Format your response with clear sections and actionable items.\n\n"
                f"User data: {formatted_context}"
            )
            
            # Use the configured Perplexity model to generate suggestions
            model_response = self._call_model(prompt)
            
            # Parse the model response and extract structured recommendations
            wellbeing_suggestions = self._parse_wellbeing_advice(model_response)
            
            return {
                "wellbeing_advice": wellbeing_suggestions,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "raw_response": model_response
            }
            
        except Exception as e:
            logger.error(f"Error processing wellbeing data: {str(e)}")
            return {"error": f"Wellbeing analysis failed: {str(e)}"}
    
    async def process_async(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user wellbeing data asynchronously using Perplexity API.
        
        Args:
            user_data: Dictionary containing user activity data, metrics, and context
            
        Returns:
            Dictionary with wellbeing recommendations and insights
        """
        try:
            # Format user data for better prompt context
            formatted_context = self._format_user_data(user_data)
            
            # Ensure Perplexity client is initialized
            if not self.perplexity_client:
                self.initialize_models()
                
            if not self.perplexity_client:
                return {"error": "Perplexity client not initialized"}
            
            # Create system prompt for wellbeing advisor
            system_prompt = (
                "You are a compassionate wellbeing coach specializing in helping knowledge workers "
                "maintain balance and health. Provide personalized, evidence-based advice tailored to "
                "the user's specific situation. Focus on practical, actionable suggestions for breaks, "
                "physical activity, mental health, focus techniques, and healthy habits. Be encouraging "
                "and thoughtful, but respect the user's autonomy."
            )
            
            # Use Perplexity client directly for wellbeing advice
            response = await self.perplexity_client.generate_wellbeing_advice(
                user_data=user_data
            )
            
            # Parse the response into structured recommendations
            wellbeing_suggestions = self._parse_wellbeing_advice(response.content)
            
            return {
                "wellbeing_advice": wellbeing_suggestions,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "raw_response": response.content
            }
            
        except Exception as e:
            logger.error(f"Error in async wellbeing processing: {str(e)}")
            return {"error": f"Async wellbeing analysis failed: {str(e)}"}
    
    def _format_user_data(self, user_data: Dict[str, Any]) -> str:
        """Format user data for inclusion in prompts."""
        sections = []
        
        # Add activity data if available
        if "activity_log" in user_data:
            activities = user_data["activity_log"]
            if isinstance(activities, list) and activities:
                recent_activities = activities[-10:] if len(activities) > 10 else activities
                activity_str = "\n".join([f"- {a}" for a in recent_activities])
                sections.append(f"Recent activities:\n{activity_str}")
        
        # Add metrics if available
        metrics = user_data.get("metrics", {})
        if metrics:
            metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
            sections.append(f"Current metrics:\n{metrics_str}")
        
        # Add screen time if available
        if "screen_time" in user_data:
            sections.append(f"Daily screen time: {user_data['screen_time']} minutes")
        
        # Add breaks information
        if "breaks" in user_data:
            sections.append(f"Breaks taken today: {user_data['breaks']}")
        
        # Add goals if available
        goals = user_data.get("goals", [])
        if goals:
            goals_str = "\n".join([f"- {g.get('goal_type', 'Goal')}: {g.get('target_value', 'N/A')}" for g in goals])
            sections.append(f"Current goals:\n{goals_str}")
        
        # Add other relevant data
        for key, value in user_data.items():
            if key not in ["activity_log", "metrics", "screen_time", "breaks", "goals"] and not key.startswith("_"):
                if isinstance(value, (str, int, float, bool)):
                    sections.append(f"{key}: {value}")
        
        return "\n\n".join(sections)
    
    def _parse_wellbeing_advice(self, response_text: str) -> Dict[str, Any]:
        """
        Parse wellbeing advice from model response into structured format.
        
        Args:
            response_text: Raw text response from the model
            
        Returns:
            Structured wellbeing recommendations
        """
        try:
            # Attempt to extract sections using regex patterns
            sections = {}
            
            # Look for break recommendations
            break_match = re.search(
                r"(?:Breaks?|Rest) Recommendations?:(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                response_text, 
                re.IGNORECASE | re.DOTALL
            )
            if break_match:
                sections["break_recommendations"] = self._extract_list_items(break_match.group(1))
            
            # Look for focus techniques
            focus_match = re.search(
                r"(?:Focus|Concentration|Productivity) (?:Techniques?|Tips|Recommendations?):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                response_text, 
                re.IGNORECASE | re.DOTALL
            )
            if focus_match:
                sections["focus_techniques"] = self._extract_list_items(focus_match.group(1))
            
            # Look for stress management
            stress_match = re.search(
                r"(?:Stress|Anxiety) Management:(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                response_text, 
                re.IGNORECASE | re.DOTALL
            )
            if stress_match:
                sections["stress_management"] = self._extract_list_items(stress_match.group(1))
            
            # Look for physical health recommendations
            physical_match = re.search(
                r"(?:Physical|Health|Exercise) (?:Tips|Recommendations?):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                response_text, 
                re.IGNORECASE | re.DOTALL
            )
            if physical_match:
                sections["physical_health"] = self._extract_list_items(physical_match.group(1))
            
            # If we couldn't extract structured sections, fall back to simple bullet points
            if not sections:
                # Extract all bullet points
                all_bullets = self._extract_list_items(response_text)
                if all_bullets:
                    sections["general_recommendations"] = all_bullets
            
            # Add a priority recommendation if available
            priority_rec = self._extract_priority_recommendation(response_text)
            if priority_rec:
                sections["priority_recommendation"] = priority_rec
            
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing wellbeing advice: {str(e)}")
            return {"general_recommendations": [response_text]}
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract bullet points or numbered lists from text."""
        # Clean up the text
        text = text.strip()
        
        # Try to find bullet points or numbered items
        bullet_pattern = r"(?:^|\n)[-•*] +(.*?)(?:\n|$)"
        numbered_pattern = r"(?:^|\n)\d+\.? +(.*?)(?:\n|$)"
        
        bullet_items = re.findall(bullet_pattern, text, re.MULTILINE)
        numbered_items = re.findall(numbered_pattern, text, re.MULTILINE)
        
        items = bullet_items + numbered_items
        
        # If no bullet points found, try to split by newlines
        if not items and "\n" in text:
            items = [line.strip() for line in text.split("\n") if line.strip()]
        
        # If still empty and there's text, use the whole text
        if not items and text:
            items = [text]
            
        return items
    
    def _extract_priority_recommendation(self, text: str) -> Optional[str]:
        """Extract a priority recommendation if available."""
        patterns = [
            r"(?:Key|Main|Priority|Primary|Most Important) (?:Recommendation|Suggestion|Advice):\s*(.*?)(?:\n\n|\n[A-Z]|\Z)",
            r"(?:Based on your data|Right now|At this moment),\s*(?:I|we) (?:recommend|suggest|advise)[:\s]+(.*?)(?:\n\n|\n[A-Z]|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
