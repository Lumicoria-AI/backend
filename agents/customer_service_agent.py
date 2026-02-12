from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional, Union
import json
import structlog
import asyncio
from datetime import datetime, timedelta
import re

# Configure logger
logger = structlog.get_logger(__name__)

class CustomerServiceAgent(BaseAgent):
    """Agent for customer service and support using LLM providers.
    
    This agent helps with customer communication, feedback analysis,
    FAQ generation, response templates, and customer satisfaction strategies.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default agent capabilities if not specified in config
        self.capabilities = config.get("capabilities", [
            "response_generation",
            "feedback_analysis",
            "faq_generation",
            "template_creation",
            "satisfaction_strategy",
            "sentiment_analysis",
            "ticket_classification"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"  # Use Perplexity's Sonar model
        
        # Set response tone and style
        self.response_tone = config.get("response_tone", "professional_friendly")
        self.response_style = config.get("response_style", "clear_concise")
        
        # Set feedback analysis parameters
        self.feedback_categories = config.get("feedback_categories", [
            "product_quality",
            "service_experience",
            "pricing",
            "support_quality",
            "feature_requests",
            "bug_reports",
            "general_feedback"
        ])
        
        # Set template categories
        self.template_categories = config.get("template_categories", [
            "general_inquiry",
            "technical_support",
            "billing_issue",
            "feature_request",
            "complaint_handling",
            "thank_you",
            "follow_up"
        ])

    def process(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process customer service data to provide assistance.
        
        Args:
            customer_data: Dictionary containing customer context, feedback,
                          inquiries, and specific requests.
            
        Returns:
            Dictionary with customer service assistance, analysis, and responses.
        """
        # Extract customer context and request type
        customer_context = customer_data.get("context", {})
        request_type = customer_data.get("request_type", "general_assistance")
        content = customer_data.get("content", "")
        
        if not content:
            return {"error": "No customer content provided"}
            
        try:
            # Select appropriate prompt based on request type
            if request_type == "generate_response":
                prompt = self._create_response_prompt(content, customer_context)
            elif request_type == "analyze_feedback":
                prompt = self._create_feedback_analysis_prompt(content, customer_context)
            elif request_type == "generate_faq":
                prompt = self._create_faq_prompt(content, customer_context)
            elif request_type == "create_template":
                prompt = self._create_template_prompt(content, customer_context)
            elif request_type == "satisfaction_strategy":
                prompt = self._create_strategy_prompt(content, customer_context)
            else:
                # General assistance prompt
                prompt = f"As a customer service assistant, help with the following request. "
                prompt += f"Consider the customer context and provide professional, helpful guidance. "
                prompt += f"\n\nCustomer context: {json.dumps(customer_context)}\n\nRequest: {content}"
            
            # Use the configured model to process the request
            model_response = self._call_model(
                prompt=prompt,
                model=self.model_config.get("model")
            )
            
            # Parse response based on request type
            if request_type == "generate_response":
                parsed_result = self._parse_response(model_response)
            elif request_type == "analyze_feedback":
                parsed_result = self._parse_feedback_analysis(model_response)
            elif request_type == "generate_faq":
                parsed_result = self._parse_faq(model_response)
            elif request_type == "create_template":
                parsed_result = self._parse_template(model_response)
            elif request_type == "satisfaction_strategy":
                parsed_result = self._parse_strategy(model_response)
            else:
                parsed_result = self._parse_general_assistance(model_response)
            
            # Create comprehensive response
            result = {
                "response": parsed_result,
                "raw_response": model_response,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "request_type": request_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing customer service request: {str(e)}")
            return {"error": f"Customer service processing failed: {str(e)}"}
    
    async def process_async(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process customer service data asynchronously with optimized processing.
        
        Args:
            customer_data: Dictionary containing customer context, feedback,
                          inquiries, and specific requests.
            
        Returns:
            Dictionary with customer service assistance, analysis, and responses.
        """
        # Extract customer context and request type
        customer_context = customer_data.get("context", {})
        request_type = customer_data.get("request_type", "general_assistance")
        content = customer_data.get("content", "")
        
        if not content:
            return {"error": "No customer content provided"}
        
        try:
            # Create system prompt based on request type
            system_prompt = self._get_system_prompt(request_type, customer_context)
            
            # Create user prompt
            user_prompt = self._create_user_prompt(request_type, content, customer_context)
            
            # Call model asynchronously
            response = await self._call_model_async(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.model_config.get("model")
            )
            
            # Parse response based on request type
            if request_type == "generate_response":
                parsed_result = self._parse_response(response)
            elif request_type == "analyze_feedback":
                parsed_result = self._parse_feedback_analysis(response)
                
                # For feedback analysis, we might want to add sentiment analysis
                if "sentiment_analysis" in self.capabilities:
                    sentiment_messages = [
                        {"role": "system", "content": "You are an expert sentiment analyzer. Analyze the sentiment of the given text across the specified aspects."},
                        {"role": "user", "content": f"Analyze the sentiment of the following text across these aspects: {', '.join(self.feedback_categories)}.\n\nText: {content}"}
                    ]
                    sentiment_result = await self.llm_client.generate(sentiment_messages)
                    parsed_result["sentiment_analysis"] = sentiment_result.content
                    
            elif request_type == "generate_faq":
                parsed_result = self._parse_faq(response)
            elif request_type == "create_template":
                parsed_result = self._parse_template(response)
            elif request_type == "satisfaction_strategy":
                parsed_result = self._parse_strategy(response)
            else:
                parsed_result = self._parse_general_assistance(response)
            
            # Create comprehensive response
            result = {
                "response": parsed_result,
                "raw_response": response,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "request_type": request_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing customer service request asynchronously: {str(e)}")
            return {"error": f"Customer service processing failed: {str(e)}"}
    
    def _get_system_prompt(self, request_type: str, context: Dict[str, Any]) -> str:
        """Create system prompt based on request type and context."""
        base_prompt = "You are a professional customer service AI assistant. "
        base_prompt += f"Maintain a {self.response_tone} tone and {self.response_style} style. "
        
        if request_type == "generate_response":
            return base_prompt + "Generate clear, helpful, and empathetic responses to customer inquiries."
        elif request_type == "analyze_feedback":
            return base_prompt + "Analyze customer feedback for patterns, sentiment, and actionable insights."
        elif request_type == "generate_faq":
            return base_prompt + "Create clear, concise, and helpful FAQ entries based on common questions."
        elif request_type == "create_template":
            return base_prompt + "Create professional, reusable response templates for common scenarios."
        elif request_type == "satisfaction_strategy":
            return base_prompt + "Provide actionable strategies to improve customer satisfaction."
        else:
            return base_prompt + "Provide general customer service assistance and guidance."
    
    def _create_user_prompt(self, request_type: str, content: str, context: Dict[str, Any]) -> str:
        """Create user prompt based on request type and content."""
        if request_type == "generate_response":
            return f"Generate a customer service response for the following inquiry:\n\n{content}"
        elif request_type == "analyze_feedback":
            return f"Analyze the following customer feedback:\n\n{content}"
        elif request_type == "generate_faq":
            return f"Create FAQ entries for the following topic:\n\n{content}"
        elif request_type == "create_template":
            return f"Create a response template for the following scenario:\n\n{content}"
        elif request_type == "satisfaction_strategy":
            return f"Suggest strategies to improve customer satisfaction based on:\n\n{content}"
        else:
            return f"Provide customer service assistance for:\n\n{content}"
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse generated customer service response."""
        return {
            "response_text": response,
            "tone": self.response_tone,
            "style": self.response_style,
            "suggested_follow_up": self._extract_follow_up(response)
        }
    
    def _parse_feedback_analysis(self, response: str) -> Dict[str, Any]:
        """Parse customer feedback analysis."""
        return {
            "analysis": response,
            "categories": self._extract_categories(response),
            "sentiment": self._extract_sentiment(response),
            "action_items": self._extract_action_items(response)
        }
    
    def _parse_faq(self, response: str) -> Dict[str, Any]:
        """Parse generated FAQ content."""
        return {
            "faq_entries": self._extract_faq_entries(response),
            "categories": self._extract_categories(response),
            "suggested_updates": self._extract_suggested_updates(response)
        }
    
    def _parse_template(self, response: str) -> Dict[str, Any]:
        """Parse generated response template."""
        return {
            "template": response,
            "category": self._extract_template_category(response),
            "variables": self._extract_template_variables(response),
            "usage_notes": self._extract_usage_notes(response)
        }
    
    def _parse_strategy(self, response: str) -> Dict[str, Any]:
        """Parse customer satisfaction strategy suggestions."""
        return {
            "strategies": self._extract_strategies(response),
            "priority": self._extract_priority(response),
            "implementation_steps": self._extract_implementation_steps(response),
            "expected_impact": self._extract_expected_impact(response)
        }
    
    def _parse_general_assistance(self, response: str) -> Dict[str, Any]:
        """Parse general customer service assistance."""
        return {
            "assistance": response,
            "suggested_actions": self._extract_suggested_actions(response),
            "additional_resources": self._extract_additional_resources(response)
        }
    
    # Helper methods for extracting specific information from responses
    def _extract_follow_up(self, response: str) -> List[str]:
        """Extract suggested follow-up actions from response."""
        # Implementation would use regex or NLP to identify follow-up suggestions
        return []
    
    def _extract_categories(self, response: str) -> List[str]:
        """Extract relevant categories from response."""
        # Implementation would match against feedback_categories
        return []
    
    def _extract_sentiment(self, response: str) -> Dict[str, float]:
        """Extract sentiment scores from response."""
        # Implementation would analyze sentiment scores
        return {}
    
    def _extract_action_items(self, response: str) -> List[Dict[str, Any]]:
        """Extract actionable items from response."""
        # Implementation would identify specific actions to take
        return []
    
    def _extract_faq_entries(self, response: str) -> List[Dict[str, str]]:
        """Extract FAQ entries from response."""
        # Implementation would parse Q&A pairs
        return []
    
    def _extract_suggested_updates(self, response: str) -> List[str]:
        """Extract suggested FAQ updates from response."""
        # Implementation would identify update suggestions
        return []
    
    def _extract_template_category(self, response: str) -> str:
        """Extract template category from response."""
        # Implementation would match against template_categories
        return ""
    
    def _extract_template_variables(self, response: str) -> List[str]:
        """Extract template variables from response."""
        # Implementation would identify variable placeholders
        return []
    
    def _extract_usage_notes(self, response: str) -> str:
        """Extract template usage notes from response."""
        # Implementation would identify usage instructions
        return ""
    
    def _extract_strategies(self, response: str) -> List[Dict[str, Any]]:
        """Extract strategies from response."""
        # Implementation would parse strategy suggestions
        return []
    
    def _extract_priority(self, response: str) -> str:
        """Extract strategy priority from response."""
        # Implementation would identify priority level
        return ""
    
    def _extract_implementation_steps(self, response: str) -> List[str]:
        """Extract implementation steps from response."""
        # Implementation would parse step-by-step instructions
        return []
    
    def _extract_expected_impact(self, response: str) -> Dict[str, Any]:
        """Extract expected impact from response."""
        # Implementation would identify impact predictions
        return {}
    
    def _extract_suggested_actions(self, response: str) -> List[str]:
        """Extract suggested actions from response."""
        # Implementation would identify action items
        return []
    
    def _extract_additional_resources(self, response: str) -> List[Dict[str, str]]:
        """Extract additional resources from response."""
        # Implementation would identify resource references
        return [] 