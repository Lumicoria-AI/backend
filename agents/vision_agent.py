from .base_agent import BaseAgent
from backend.ai_models import LLMConfig
from typing import Dict, Any, List, Optional, Union, BinaryIO
import base64
import io
import json
import structlog
import asyncio
from datetime import datetime
import re
from pathlib import Path
import httpx

# Configure logger
logger = structlog.get_logger(__name__)

class VisionAgent(BaseAgent):
    """Agent for processing images and visual data using LLM providers.
    
    This agent analyzes images, extracts text, identifies objects, and provides
    detailed insights using the provider-agnostic LLM interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
            
        # Define default vision analysis components
        self.vision_tasks = config.get("vision_tasks", [
            "text_extraction", "object_detection", "scene_analysis", 
            "content_description", "safety_check"
        ])

    def process(self, vision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image data to extract visual information.
        
        Args:
            vision_data: Dictionary containing image data, context, and processing options.
            
        Returns:
            Dictionary with extracted visual information and insights.
        """
        # Extract image data and options
        image_content = vision_data.get("image_content")  # Binary image data
        image_path = vision_data.get("image_path")        # Path to image file
        image_url = vision_data.get("image_url")          # URL to image
        prompt = vision_data.get("prompt", "Analyze the provided image and describe its content in detail, including any text, objects, people, or noteworthy elements.")
        
        # Validate we have some image data
        if not any([image_content, image_path, image_url]):
            return {"error": "No image content provided"}
            
        try:
            # If we have binary data, use it directly
            if image_content:
                encoded_image = self._encode_image(image_content)
            # If we have a path, read and encode the file
            elif image_path:
                with open(image_path, "rb") as image_file:
                    encoded_image = self._encode_image(image_file.read())
            # If we have a URL, we'll use it directly in the prompt
            elif image_url:
                # Since Perplexity Sonar doesn't directly load images from URLs through API
                # We'll describe the URL in the prompt and use a special system instruction
                system_instruction = f"The user will provide an image URL. Please analyze the image at this URL: {image_url}"
                prompt = f"Analyze the image at this URL: {image_url}\n\n{prompt}"
                encoded_image = None
            
            # Call the model using our base agent's method
            # Special handling for image data
            model_params = {
                "model": self.model_config.get("model", "sonar-large-online"),
                "temperature": vision_data.get("temperature", 0.7),
                "max_tokens": vision_data.get("max_tokens", 4096),
                "top_p": vision_data.get("top_p", 0.9)
            }
            
            # For Perplexity API, we need to construct the prompt differently with image data
            if encoded_image:
                # Create message with image content
                full_prompt = self._format_image_prompt(prompt, encoded_image)
            else:
                # Just use the text prompt
                full_prompt = prompt
                
            # Call the model with the formatted prompt
            model_response = self._call_model(
                prompt=full_prompt,
                model_name=model_params["model"],
                temperature=model_params["temperature"],
                max_tokens=model_params["max_tokens"],
                top_p=model_params["top_p"],
                image_data=encoded_image  # Pass encoded image directly
            )

            # Extract structured information
            structured_analysis = self._extract_structured_information(model_response)
            
            # Create comprehensive response
            result = {
                "description": model_response,
                "structured_analysis": structured_analysis,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model")
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": f"Image processing failed: {str(e)}"}
    
    async def process_async(self, vision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image data asynchronously.
        
        Args:
            vision_data: Dictionary containing image data, context, and processing options.
            
        Returns:
            Dictionary with extracted visual information and insights.
        """
        # Extract image data and options
        image_content = vision_data.get("image_content")
        image_path = vision_data.get("image_path")
        image_url = vision_data.get("image_url")
        prompt = vision_data.get("prompt", "Analyze the provided image and describe its content in detail.")
        analysis_tasks = vision_data.get("analysis_tasks", self.vision_tasks)
        
        if not any([image_content, image_path, image_url]):
            return {"error": "No image content provided"}
            
        try:
            # Prepare image data
            if image_content:
                encoded_image = self._encode_image(image_content)
            elif image_path:
                with open(image_path, "rb") as image_file:
                    encoded_image = self._encode_image(image_file.read())
            elif image_url:
                # For URLs, we need to fetch the image asynchronously
                async with httpx.AsyncClient() as client:
                    response = await client.get(image_url)
                    response.raise_for_status()
                    encoded_image = self._encode_image(response.content)
            
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
            
            # Create a system prompt for image analysis
            system_prompt = (
                "You are an expert computer vision system. Analyze the provided image in detail and provide "
                "a comprehensive analysis. Focus on identifying objects, text, people, scene context, and "
                "any noteworthy elements. If you see text in the image, transcribe it accurately."
            )
            
            # Build messages using the LLMMessage format expected by all providers.
            # Images go in the `images` list (base64 strings or URLs), not in `content`.
            from backend.ai_models.base import LLMMessage, MessageRole

            messages = [
                LLMMessage(role=MessageRole.SYSTEM, content=system_prompt),
                LLMMessage(
                    role=MessageRole.USER,
                    content=prompt,
                    images=[f"data:image/jpeg;base64,{encoded_image}"],
                ),
            ]

            # Use default parameters
            config = LLMConfig(
                temperature=vision_data.get("temperature", 0.7),
                max_tokens=vision_data.get("max_tokens", 4096),
                top_p=vision_data.get("top_p", 0.9),
            )

            # Call LLM via provider-agnostic interface
            response = await self.llm_client.generate(messages, config=config)
            
            # Extract structured information
            structured_analysis = self._extract_structured_information(response.content)
            
            # Create comprehensive response
            result = {
                "description": response.content,
                "structured_analysis": structured_analysis,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "citations": [
                    {
                        "text": citation.text,
                        "url": citation.metadata.url,
                        "title": citation.metadata.title
                    } 
                    for citation in response.citations
                ] if hasattr(response, "citations") else []
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async image processing: {str(e)}")
            return {"error": f"Async image processing failed: {str(e)}"}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the vision agent asynchronously.
        
        Args:
            query: The query string about an image
            context: Optional context dictionary containing image data and analysis parameters
            
        Returns:
            Dictionary containing image analysis results
        """
        try:
            if not context or not (context.get("image_content") or context.get("image_path") or context.get("image_url")):
                return {"error": "No image data provided in context"}
            
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
            
            # Get image data from context
            image_content = context.get("image_content")
            image_path = context.get("image_path")
            image_url = context.get("image_url")
            
            # Prepare image data
            if image_content:
                encoded_image = self._encode_image(image_content)
            elif image_path:
                with open(image_path, "rb") as image_file:
                    encoded_image = self._encode_image(image_file.read())
            elif image_url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(image_url)
                    response.raise_for_status()
                    encoded_image = self._encode_image(response.content)
            
            # Create system prompt for image analysis
            system_prompt = (
                "You are an expert computer vision system. Analyze the provided image in detail and provide "
                "a comprehensive analysis. Focus on identifying objects, text, people, scene context, and "
                "any noteworthy elements. If you see text in the image, transcribe it accurately."
            )

            # Build messages using the LLMMessage format expected by all providers.
            from backend.ai_models.base import LLMMessage, MessageRole

            messages = [
                LLMMessage(role=MessageRole.SYSTEM, content=system_prompt),
                LLMMessage(
                    role=MessageRole.USER,
                    content=query,
                    images=[f"data:image/jpeg;base64,{encoded_image}"],
                ),
            ]

            # Use parameters from context or defaults
            config = LLMConfig(
                temperature=context.get("temperature", 0.7),
                max_tokens=context.get("max_tokens", 4096),
                top_p=context.get("top_p", 0.9),
            )

            # Call LLM via provider-agnostic interface
            response = await self.llm_client.generate(messages, config=config)
            
            # Extract structured information
            structured_analysis = self._extract_structured_information(response.content)
            
            # Create comprehensive response
            result = {
                "description": response.content,
                "structured_analysis": structured_analysis,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "citations": response.citations if response.citations else []
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying vision agent: {str(e)}")
            return {"error": f"Image analysis failed: {str(e)}"}

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64.
        
        Args:
            image_data: Raw binary image data
            
        Returns:
            Base64 encoded image string
        """
        return base64.b64encode(image_data).decode('utf-8')

    def _format_image_prompt(self, prompt: str, encoded_image: str) -> str:
        """Format prompt with image for model input.
        
        Args:
            prompt: Text prompt
            encoded_image: Base64 encoded image
            
        Returns:
            Formatted prompt with image data
        """
        # For Perplexity, we use a specific format to embed images in prompts
        return f"{prompt}\n\n[Image: data:image/jpeg;base64,{encoded_image}]"

    def _extract_structured_information(self, response_text: str) -> Dict[str, Any]:
        """Extract structured information from model response.
        
        Args:
            response_text: Raw text response from the model
            
        Returns:
            Structured dictionary with categorized information
        """
        try:
            # Try to parse as JSON first (in case model returned JSON)
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, use regex-based parsing
                pass
            
            # Initialize categories for extraction
            structured_data = {
                "detected_objects": [],
                "detected_text": [],
                "scene_type": "",
                "people_count": 0,
                "colors": [],
                "themes": [],
                "raw_description": response_text
            }
            
            # Extract objects with regex
            object_patterns = [
                r"(?:objects?|items?|elements?)(?:\s+detected)?(?:\s+include)?:?\s+((?:[^,.]+,?\s*)+)",
                r"(?:I\s+(?:can\s+)?see|There\s+(?:are|is)|The\s+image\s+(?:shows|contains))\s+((?:[^,.]+,?\s*)+)"
            ]
            
            for pattern in object_patterns:
                matches = re.finditer(pattern, response_text, re.IGNORECASE)
                for match in matches:
                    objects_text = match.group(1).strip()
                    objects = re.split(r',\s*|and\s+', objects_text)
                    for obj in objects:
                        obj = obj.strip()
                        if obj and obj not in structured_data["detected_objects"]:
                            structured_data["detected_objects"].append(obj)
            
            # Extract text — match quoted strings near text-related words,
            # plus common LLM patterns like "reads:", "says:", "labeled", etc.
            text_patterns = [
                # Quoted text near keywords: text "Hello", reads "World"
                r'(?:text|writing|inscription|label|title|heading|caption|reads?|says?|written|showing|displays?|reads)\s*[:—–-]?\s*["\u201c]([^"\u201d]+)["\u201d]',
                # Single-quoted
                r"(?:text|writing|reads?|says?|label|title|heading)\s*[:—–-]?\s*'([^']+)'",
                # **Bold label:** pattern (common in markdown LLM output)
                r'\*\*([^*]{2,80})\*\*',
                # "Text:" or "Title:" followed by content on same line
                r'(?:^|\n)\s*(?:Text|Title|Heading|Label|Caption|Name)\s*:\s*(.+?)(?:\n|$)',
            ]

            for pattern in text_patterns:
                matches = re.finditer(pattern, response_text, re.IGNORECASE)
                for match in matches:
                    text = match.group(1).strip()
                    # Filter out generic headings that aren't extracted text
                    skip_words = {"main content", "key objects", "overall scene", "readable text",
                                  "scene", "objects", "analysis", "description", "summary",
                                  "detected objects", "detected text", "here's"}
                    if (text
                        and len(text) > 1
                        and text.lower() not in skip_words
                        and text not in structured_data["detected_text"]):
                        structured_data["detected_text"].append(text)
            
            # Extract scene type
            scene_patterns = [
                r"(?:scene|setting|location)(?:\s+appears\s+to\s+be)?:?\s+([^.,]+)",
                r"(?:image|photo|picture)\s+(?:shows|depicts|contains)\s+(?:a|an)\s+([^.,]+)"
            ]
            
            for pattern in scene_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    structured_data["scene_type"] = match.group(1).strip()
                    break
            
            # Extract people count
            people_patterns = [
                r"(\d+)\s+(?:person|people|individuals)",
                r"(?:person|individual)"
            ]
            
            for pattern in people_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    try:
                        if match.group(1):
                            structured_data["people_count"] = int(match.group(1))
                        else:
                            structured_data["people_count"] = 1
                    except (IndexError, ValueError):
                        structured_data["people_count"] = 1
                    break
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error extracting structured information: {str(e)}")
            return {"raw_description": response_text, "parsing_error": str(e)}
