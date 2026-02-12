from .base_agent import BaseAgent
from backend.ai_models import LLMConfig
from typing import Dict, Any, List, Optional
import json
import structlog
import asyncio
from datetime import datetime
import re

# Configure logger
logger = structlog.get_logger(__name__)

class CreativeAgent(BaseAgent):
    """Agent for generating creative content using LLM providers.
    
    This agent generates various types of creative content including marketing copy,
    stories, poems, scripts, product descriptions, and other creative text forms.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default creative content types if not specified in config
        self.creative_types = config.get("creative_types", [
            "marketing", "storytelling", "poetry", "scriptwriting", 
            "product_description", "social_media", "blog_post"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"
    
    def process(self, creative_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a creative request to generate content.
        
        Args:
            creative_data: Dictionary containing content type, topic, guidelines, 
                          target audience, and other parameters.
            
        Returns:
            Dictionary with generated content and metadata.
        """
        # Extract request parameters
        content_type = creative_data.get("content_type", "general")
        topic = creative_data.get("topic", "")
        guidelines = creative_data.get("guidelines", "")
        audience = creative_data.get("audience", "general audience")
        tone = creative_data.get("tone", "professional")
        length = creative_data.get("length", "medium")
        
        if not topic:
            return {"error": "No topic or content prompt provided"}
            
        # Use the configured model to process the creative request
        prompt = self._create_creative_prompt(content_type, topic, guidelines, audience, tone, length)
        
        try:
            # Get creative content based on prompt
            content_result = self._call_model(
                prompt=prompt, 
                model=self.model_config.get("model")
            )
            
            # Parse the content based on its type
            parsed_content = self._parse_creative_content(content_result, content_type)
            
            # Create comprehensive response
            result = {
                "content": parsed_content,
                "raw_content": content_result,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "content_type": content_type,
                "metadata": {
                    "topic": topic,
                    "audience": audience,
                    "tone": tone,
                    "length": length
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing creative request: {str(e)}")
            return {"error": f"Creative content generation failed: {str(e)}"}
    
    async def process_async(self, creative_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a creative request asynchronously with optimized processing.
        
        Args:
            creative_data: Dictionary containing content type, topic, guidelines, 
                          target audience, and other parameters.
            
        Returns:
            Dictionary with generated content and metadata.
        """
        # Extract request parameters
        content_type = creative_data.get("content_type", "general")
        topic = creative_data.get("topic", "")
        guidelines = creative_data.get("guidelines", "")
        audience = creative_data.get("audience", "general audience")
        tone = creative_data.get("tone", "professional")
        length = creative_data.get("length", "medium")
        
        if not topic:
            return {"error": "No topic or content prompt provided"}
            
        try:
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
            
            # Create system and user prompts
            system_prompt, user_prompt = self._create_async_prompts(
                content_type, topic, guidelines, audience, tone, length
            )
            
            # Format messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM via provider-agnostic interface
            config = LLMConfig(
                model=self.model_config.get("model"),
                temperature=0.8,  # Slightly higher temperature for creativity
            )
            response = await self.llm_client.generate(messages, config=config)
            
            # Parse the content based on its type
            parsed_content = self._parse_creative_content(response.content, content_type)
            
            # Create comprehensive response
            result = {
                "content": parsed_content,
                "raw_content": response.content,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "content_type": content_type,
                "metadata": {
                    "topic": topic,
                    "audience": audience,
                    "tone": tone,
                    "length": length
                }
            }
            
            # Add citations if available
            if response.citations:
                result["citations"] = response.citations
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async creative processing: {str(e)}")
            return {"error": f"Async creative content generation failed: {str(e)}"}
    
    def _create_creative_prompt(self, content_type: str, topic: str, 
                               guidelines: str, audience: str,
                               tone: str, length: str) -> str:
        """Create prompt for generating creative content."""
        # Base prompt template
        prompt = f"Create {content_type} content about {topic}. "
        
        # Add length specification
        if length == "short":
            prompt += "Keep it concise and brief. "
        elif length == "medium":
            prompt += "Use a moderate length. "
        elif length == "long":
            prompt += "Make it comprehensive and detailed. "
        
        # Add tone specification
        prompt += f"Use a {tone} tone. "
        
        # Add audience specification
        prompt += f"Target audience: {audience}. "
        
        # Add specific guidelines if provided
        if guidelines:
            prompt += f"\n\nAdditional guidelines: {guidelines}"
        
        # Add content-specific instructions
        if content_type == "marketing":
            prompt += "\n\nFocus on compelling benefits and clear calls to action. Highlight value propositions."
        elif content_type == "storytelling":
            prompt += "\n\nCreate an engaging narrative with character development, setting, and plot."
        elif content_type == "poetry":
            prompt += "\n\nUse rich imagery, metaphor, and appropriate poetic devices."
        elif content_type == "scriptwriting":
            prompt += "\n\nFormat as a script with dialogue, scene descriptions, and character actions."
        elif content_type == "product_description":
            prompt += "\n\nHighlight features, benefits, use cases, and specifications. Be informative yet persuasive."
        elif content_type == "social_media":
            prompt += "\n\nKeep it engaging, shareable, and optimized for the platform. Include relevant hashtags."
        elif content_type == "blog_post":
            prompt += "\n\nStructure with introduction, body paragraphs, and conclusion. Use headings and maintain a clear narrative."
        
        # Request structured output
        prompt += "\n\nProvide the content in a well-structured format appropriate for the content type."
        
        return prompt
    
    def _create_async_prompts(self, content_type: str, topic: str, 
                            guidelines: str, audience: str,
                            tone: str, length: str) -> tuple:
        """Create system and user prompts for async processing."""
        # Create appropriate system prompt based on content type
        if content_type == "marketing":
            system_prompt = (
                "You are an expert marketing copywriter who creates compelling, persuasive content "
                "that drives engagement and conversions. Your copy is benefit-focused, audience-aware, "
                "and includes strong calls to action. You adapt your style based on the brand voice, target audience, "
                "and marketing objectives."
            )
        elif content_type == "storytelling":
            system_prompt = (
                "You are a skilled storyteller who creates engaging narratives with well-developed characters, "
                "vivid settings, and compelling plots. You understand story structure, pacing, and how to evoke "
                "emotion in readers. You adapt your style based on the genre, audience, and narrative goals."
            )
        elif content_type == "poetry":
            system_prompt = (
                "You are an accomplished poet who creates evocative, imagery-rich poetry that resonates with readers. "
                "You understand various poetic forms, devices, and traditions. You can adapt your style based on the "
                "requested tone, theme, and poetic structure."
            )
        elif content_type == "scriptwriting":
            system_prompt = (
                "You are a professional scriptwriter who creates engaging scripts for various media formats. "
                "You understand proper script formatting, dialogue writing, scene structure, and visual storytelling. "
                "You adapt your style based on the medium, genre, and target audience."
            )
        elif content_type == "product_description":
            system_prompt = (
                "You are an expert product copywriter who creates compelling, informative product descriptions "
                "that highlight features, benefits, and use cases. Your descriptions are both persuasive and factual, "
                "addressing customer needs and pain points. You adapt your style based on the product type, audience, and platform."
            )
        elif content_type == "social_media":
            system_prompt = (
                "You are a social media content specialist who creates engaging, shareable content optimized for "
                "various platforms. Your content is concise, attention-grabbing, and drives engagement. You understand "
                "platform-specific best practices and audience behaviors. You adapt your style based on the platform, "
                "brand voice, and content goals."
            )
        elif content_type == "blog_post":
            system_prompt = (
                "You are an expert blog writer who creates informative, engaging blog content that provides value to readers. "
                "Your posts are well-structured with clear headings, engaging introductions, and satisfying conclusions. "
                "You understand SEO best practices and how to maintain reader interest throughout longer content pieces."
            )
        else:  # general creative content
            system_prompt = (
                "You are a versatile creative content specialist who produces high-quality written content across "
                "various formats and styles. You adapt your approach based on the specific content requirements, "
                "audience needs, and communication objectives."
            )
        
        # Create user prompt
        user_prompt = f"Create {content_type} content about: {topic}\n\n"
        
        if guidelines:
            user_prompt += f"Guidelines: {guidelines}\n\n"
            
        user_prompt += f"Audience: {audience}\n"
        user_prompt += f"Tone: {tone}\n"
        user_prompt += f"Length: {length}\n"
        
        return system_prompt, user_prompt
        
    def _parse_creative_content(self, content_text: str, content_type: str) -> Dict[str, Any]:
        """Parse the creative content into a structured format based on content type."""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(content_text)
            except json.JSONDecodeError:
                # If not JSON, use content-type specific parsing
                pass
                
            result = {
                "full_content": content_text,
                "sections": []
            }
            
            # Different parsing approaches based on content type
            if content_type == "marketing":
                # Parse marketing content
                result = self._parse_marketing_content(content_text)
            elif content_type == "storytelling":
                # Parse storytelling content
                result = self._parse_story_content(content_text)
            elif content_type == "poetry":
                # Parse poetry content
                result = self._parse_poetry_content(content_text)
            elif content_type == "scriptwriting":
                # Parse script content
                result = self._parse_script_content(content_text)
            elif content_type == "product_description":
                # Parse product description content
                result = self._parse_product_description(content_text)
            elif content_type == "social_media":
                # Parse social media content
                result = self._parse_social_media_content(content_text)
            elif content_type == "blog_post":
                # Parse blog post content
                result = self._parse_blog_post_content(content_text)
            else:
                # Generic content parsing
                sections = content_text.split("\n\n")
                result["sections"] = [section.strip() for section in sections if section.strip()]
                
                # Try to extract a title if present
                if sections and ":" in sections[0]:
                    possible_title = sections[0].strip().split(":")[0]
                    if len(possible_title) < 100:  # Reasonable title length
                        result["title"] = possible_title
                
            return result
            
        except Exception as e:
            logger.error(f"Error parsing creative content: {str(e)}")
            return {"full_content": content_text, "parsing_error": str(e)}
    
    def _parse_marketing_content(self, content_text: str) -> Dict[str, Any]:
        """Parse marketing content into structured format."""
        result = {
            "full_content": content_text,
            "headline": "",
            "subheadline": "",
            "body": [],
            "call_to_action": "",
            "tagline": ""
        }
        
        # Extract headline
        headline_match = re.search(r"(?:Title|Headline|Subject):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if headline_match:
            result["headline"] = headline_match.group(1).strip()
        
        # Extract subheadline
        subhead_match = re.search(r"(?:Subhead|Subtitle|Subheadline):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if subhead_match:
            result["subheadline"] = subhead_match.group(1).strip()
        
        # Extract body content (all paragraphs that aren't other elements)
        paragraphs = [p.strip() for p in content_text.split("\n\n") if p.strip()]
        body_paragraphs = []
        for p in paragraphs:
            if not (p.lower().startswith("headline:") or p.lower().startswith("title:") or
                   p.lower().startswith("subhead:") or p.lower().startswith("subtitle:") or
                   p.lower().startswith("call to action:") or p.lower().startswith("cta:") or
                   p.lower().startswith("tagline:")):
                body_paragraphs.append(p)
        
        result["body"] = body_paragraphs
        
        # Extract call to action
        cta_match = re.search(r"(?:Call to Action|CTA):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if cta_match:
            result["call_to_action"] = cta_match.group(1).strip()
        
        # Extract tagline
        tagline_match = re.search(r"(?:Tagline|Slogan):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if tagline_match:
            result["tagline"] = tagline_match.group(1).strip()
        
        return result
    
    def _parse_story_content(self, content_text: str) -> Dict[str, Any]:
        """Parse story content into structured format."""
        result = {
            "full_content": content_text,
            "title": "",
            "introduction": "",
            "body": [],
            "conclusion": ""
        }
        
        # Extract title
        title_match = re.search(r"(?:Title|Story Title|Name):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1).strip()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content_text.split("\n\n") if p.strip()]
        
        # First paragraph after title is typically introduction
        intro_index = 0
        if result["title"] and paragraphs and paragraphs[0].lower().startswith("title:"):
            intro_index = 1
        
        if len(paragraphs) > intro_index:
            result["introduction"] = paragraphs[intro_index]
            
        # Last paragraph is typically conclusion
        if len(paragraphs) > intro_index + 1:
            result["conclusion"] = paragraphs[-1]
            
        # Middle paragraphs are the body
        if len(paragraphs) > intro_index + 2:
            result["body"] = paragraphs[intro_index+1:-1]
        
        return result
    
    def _parse_poetry_content(self, content_text: str) -> Dict[str, Any]:
        """Parse poetry content into structured format."""
        result = {
            "full_content": content_text,
            "title": "",
            "stanzas": []
        }
        
        # Extract title
        title_match = re.search(r"(?:Title|Poem Title|Name):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1).strip()
            # Remove title line from content for stanza parsing
            content_text = re.sub(r"(?:Title|Poem Title|Name):\s*.*?\n", "", content_text, flags=re.IGNORECASE)
        
        # Split into stanzas (groups of lines separated by blank lines)
        stanzas = content_text.split("\n\n")
        result["stanzas"] = [stanza.strip() for stanza in stanzas if stanza.strip()]
        
        return result
    
    def _parse_script_content(self, content_text: str) -> Dict[str, Any]:
        """Parse script content into structured format."""
        result = {
            "full_content": content_text,
            "title": "",
            "scene_heading": "",
            "dialogue": [],
            "action": []
        }
        
        # Extract title
        title_match = re.search(r"(?:Title|Script Title|Name):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1).strip()
        
        # Extract scene heading
        heading_match = re.search(r"(?:INT\.|EXT\.|INT/EXT\.)\s*(.*?)(?:\n|$)", content_text)
        if heading_match:
            result["scene_heading"] = heading_match.group(0).strip()
        
        # Extract dialogue (CHARACTER: dialogue format)
        dialogue_matches = re.finditer(r"([A-Z][A-Z\s]+)(?:\s*\([^)]*\))?:\s*(.*?)(?:\n\n|\n(?=[A-Z][A-Z\s]+:)|\Z)", 
                                      content_text, re.MULTILINE)
        for match in dialogue_matches:
            character = match.group(1).strip()
            dialogue = match.group(2).strip()
            result["dialogue"].append({"character": character, "text": dialogue})
        
        # Extract action blocks (paragraphs that aren't dialogue)
        lines = content_text.split("\n")
        action_block = ""
        for line in lines:
            if not re.match(r"[A-Z][A-Z\s]+:", line) and line.strip():
                action_block += line + "\n"
            elif action_block:
                action_block = action_block.strip()
                if action_block:
                    result["action"].append(action_block)
                action_block = ""
        
        if action_block.strip():
            result["action"].append(action_block.strip())
        
        return result
    
    def _parse_product_description(self, content_text: str) -> Dict[str, Any]:
        """Parse product description content into structured format."""
        result = {
            "full_content": content_text,
            "product_name": "",
            "tagline": "",
            "overview": "",
            "features": [],
            "benefits": [],
            "specifications": []
        }
        
        # Extract product name
        name_match = re.search(r"(?:Product Name|Name|Title):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if name_match:
            result["product_name"] = name_match.group(1).strip()
        
        # Extract tagline
        tagline_match = re.search(r"(?:Tagline|Slogan):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if tagline_match:
            result["tagline"] = tagline_match.group(1).strip()
        
        # Extract overview
        overview_match = re.search(r"(?:Overview|Introduction|Summary):(.*?)(?:\n\n|\n(?=[A-Z])|$)", 
                                  content_text, re.IGNORECASE | re.DOTALL)
        if overview_match:
            result["overview"] = overview_match.group(1).strip()
        
        # Extract features
        features_match = re.search(r"(?:Features|Key Features|Main Features):(.*?)(?:\n\n|\n(?=[A-Z])|$)", 
                                  content_text, re.IGNORECASE | re.DOTALL)
        if features_match:
            features_text = features_match.group(1).strip()
            features = re.findall(r"[-•*]\s*(.*?)(?:\n[-•*]|\Z)", features_text + "\n", re.DOTALL)
            result["features"] = [feature.strip() for feature in features if feature.strip()]
        
        # Extract benefits
        benefits_match = re.search(r"(?:Benefits|Advantages):(.*?)(?:\n\n|\n(?=[A-Z])|$)", 
                                 content_text, re.IGNORECASE | re.DOTALL)
        if benefits_match:
            benefits_text = benefits_match.group(1).strip()
            benefits = re.findall(r"[-•*]\s*(.*?)(?:\n[-•*]|\Z)", benefits_text + "\n", re.DOTALL)
            result["benefits"] = [benefit.strip() for benefit in benefits if benefit.strip()]
        
        # Extract specifications
        specs_match = re.search(r"(?:Specifications|Specs|Technical Details):(.*?)(?:\n\n|\n(?=[A-Z])|$)", 
                              content_text, re.IGNORECASE | re.DOTALL)
        if specs_match:
            specs_text = specs_match.group(1).strip()
            specs = re.findall(r"[-•*]\s*(.*?)(?:\n[-•*]|\Z)", specs_text + "\n", re.DOTALL)
            result["specifications"] = [spec.strip() for spec in specs if spec.strip()]
        
        return result
    
    def _parse_social_media_content(self, content_text: str) -> Dict[str, Any]:
        """Parse social media content into structured format."""
        result = {
            "full_content": content_text,
            "posts": [],
            "hashtags": []
        }
        
        # Extract posts (separated by line breaks or post indicators)
        post_matches = re.finditer(r"(?:Post \d+:|Option \d+:)?\s*(.*?)(?:\n\n|\n(?=Post \d+:|Option \d+:)|\Z)", 
                                 content_text, re.DOTALL)
        for match in post_matches:
            post_text = match.group(1).strip()
            if post_text and len(post_text) > 5:
                result["posts"].append(post_text)
        
        # If no posts found with the pattern, just use the whole text as one post
        if not result["posts"]:
            result["posts"].append(content_text)
        
        # Extract hashtags from all posts
        for post in result["posts"]:
            hashtag_matches = re.findall(r"#(\w+)", post)
            for tag in hashtag_matches:
                if tag not in result["hashtags"]:
                    result["hashtags"].append(tag)
        
        return result
    
    def _parse_blog_post_content(self, content_text: str) -> Dict[str, Any]:
        """Parse blog post content into structured format."""
        result = {
            "full_content": content_text,
            "title": "",
            "introduction": "",
            "headings": [],
            "sections": [],
            "conclusion": ""
        }
        
        # Extract title
        title_match = re.search(r"(?:Title|Blog Title|Headline):\s*(.*?)(?:\n|$)", content_text, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1).strip()
        
        # Extract headings
        heading_matches = re.findall(r"(?:^|\n)(?:#+\s+|)(?!Title:)([A-Z][\w\s:]+)(?:\n|$)", content_text)
        result["headings"] = [h.strip() for h in heading_matches if h.strip() and len(h.strip()) < 100]
        
        # Extract introduction (first paragraph after title)
        content_lines = content_text.split("\n\n")
        for i, block in enumerate(content_lines):
            if i > 0 or (i == 0 and not block.lower().startswith("title:")):
                if block.strip() and not block.startswith("#") and not any(h in block for h in result["headings"]):
                    result["introduction"] = block.strip()
                    break
        
        # Extract conclusion (typically last section or after "Conclusion" heading)
        conclusion_index = -1
        for i, block in enumerate(content_lines):
            if "conclusion" in block.lower() or "summary" in block.lower():
                conclusion_index = i
                break
        
        if conclusion_index > 0 and conclusion_index < len(content_lines) - 1:
            result["conclusion"] = content_lines[conclusion_index + 1].strip()
        elif len(content_lines) > 2:
            result["conclusion"] = content_lines[-1].strip()
        
        # Extract sections (content between headings)
        current_heading = ""
        current_content = []
        
        for line in content_text.split("\n"):
            # Check if this line is a heading
            is_heading = False
            for heading in result["headings"]:
                if heading in line:
                    # Save previous section
                    if current_heading and current_content:
                        result["sections"].append({
                            "heading": current_heading,
                            "content": "\n".join(current_content).strip()
                        })
                    
                    # Start new section
                    current_heading = heading
                    current_content = []
                    is_heading = True
                    break
            
            if not is_heading and current_heading:
                current_content.append(line)
        
        # Add the last section
        if current_heading and current_content:
            result["sections"].append({
                "heading": current_heading,
                "content": "\n".join(current_content).strip()
            })
        
        return result

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the creative agent asynchronously.
        
        Args:
            query: The query string for creative content generation
            context: Optional context dictionary containing content parameters and preferences
            
        Returns:
            Dictionary containing generated creative content
        """
        try:
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
            
            # Get content parameters from context
            content_type = context.get("content_type", "general") if context else "general"
            topic = context.get("topic", "") if context else ""
            guidelines = context.get("guidelines", []) if context else []
            audience = context.get("audience", "general") if context else "general"
            tone = context.get("tone", "neutral") if context else "neutral"
            length = context.get("length", "medium") if context else "medium"
            
            # Create system and user prompts
            system_prompt, user_prompt = self._create_async_prompts(
                content_type, topic or query, guidelines, audience, tone, length
            )
            
            # Format messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM via provider-agnostic interface
            config = LLMConfig(
                model=self.model_config.get("model"),
                temperature=0.8,  # Slightly higher temperature for creativity
            )
            response = await self.llm_client.generate(messages, config=config)
            
            # Parse the content based on its type
            parsed_content = self._parse_creative_content(response.content, content_type)
            
            # Create comprehensive response
            result = {
                "content": parsed_content,
                "raw_content": response.content,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "content_type": content_type,
                "metadata": {
                    "topic": topic or query,
                    "audience": audience,
                    "tone": tone,
                    "length": length
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying creative agent: {str(e)}")
            return {"error": f"Creative content generation failed: {str(e)}"}
