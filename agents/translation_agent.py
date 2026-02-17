from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional, Union
import json
import structlog
import asyncio
from datetime import datetime
from enum import Enum

# Configure logger
logger = structlog.get_logger(__name__)

class TranslationMode(Enum):
    """Enum for different translation modes."""
    DOCUMENT = "document"  # Full document translation
    CONVERSATION = "conversation"  # Real-time conversation translation
    CULTURAL = "cultural"  # Cultural adaptation
    TECHNICAL = "technical"  # Technical content translation
    LITERARY = "literary"  # Literary content translation

class TranslationAgent(BaseAgent):
    """Agent for multilingual translation and cultural adaptation using LLM providers.
    
    This agent provides comprehensive translation services including:
    - Document translation with format preservation
    - Real-time conversation translation
    - Cultural context adaptation
    - Technical and literary content translation
    - Idiom and nuance handling
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default agent capabilities if not specified in config
        self.capabilities = config.get("capabilities", [
            "document_translation",
            "conversation_translation",
            "cultural_adaptation",
            "technical_translation",
            "literary_translation",
            "format_preservation",
            "idiom_handling"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"  # Use Perplexity's Sonar model
        
        # Set translation parameters
        self.preserve_formatting = config.get("preserve_formatting", True)
        self.include_cultural_context = config.get("include_cultural_context", True)
        self.handle_idioms = config.get("handle_idioms", True)
        
        # Supported languages (can be expanded)
        self.supported_languages = config.get("supported_languages", [
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"
        ])
        
        # Translation modes and their specific settings
        self.translation_modes = {
            TranslationMode.DOCUMENT: {
                "preserve_formatting": True,
                "include_cultural_context": True,
                "handle_idioms": True
            },
            TranslationMode.CONVERSATION: {
                "preserve_formatting": False,
                "include_cultural_context": True,
                "handle_idioms": True
            },
            TranslationMode.CULTURAL: {
                "preserve_formatting": True,
                "include_cultural_context": True,
                "handle_idioms": True
            },
            TranslationMode.TECHNICAL: {
                "preserve_formatting": True,
                "include_cultural_context": False,
                "handle_idioms": False
            },
            TranslationMode.LITERARY: {
                "preserve_formatting": True,
                "include_cultural_context": True,
                "handle_idioms": True
            }
        }

    async def process_async(self, translation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process translation request asynchronously.
        
        Args:
            translation_data: Dictionary containing:
                - content: Text to translate
                - source_language: Source language code
                - target_language: Target language code
                - mode: Translation mode (document, conversation, cultural, etc.)
                - context: Additional context for translation
                - preserve_formatting: Whether to preserve formatting
                - include_cultural_context: Whether to include cultural context
                - handle_idioms: Whether to handle idioms specially
        
        Returns:
            Dictionary with translation results and metadata
        """
        try:
            # Extract translation parameters
            content = translation_data.get("content", "")
            source_lang = translation_data.get("source_language", "auto")
            target_lang = translation_data.get("target_language")
            mode = TranslationMode(translation_data.get("mode", "document"))
            context = translation_data.get("context", {})
            
            if not content or not target_lang:
                return {"error": "Missing required translation parameters"}
            
            # Validate languages
            if source_lang != "auto" and source_lang not in self.supported_languages:
                return {"error": f"Unsupported source language: {source_lang}"}
            if target_lang not in self.supported_languages:
                return {"error": f"Unsupported target language: {target_lang}"}
            
            # Get mode-specific settings
            mode_settings = self.translation_modes.get(mode, self.translation_modes[TranslationMode.DOCUMENT])
            
            # Create system prompt based on mode and settings
            system_prompt = self._create_system_prompt(
                mode=mode,
                source_lang=source_lang,
                target_lang=target_lang,
                settings=mode_settings,
                context=context
            )
            
            # Create user prompt
            user_prompt = self._create_user_prompt(
                content=content,
                mode=mode,
                context=context
            )
            
            # Call model asynchronously
            response = await self._call_model_async(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.model_config.get("model")
            )
            
            # Parse and process the response
            result = self._parse_translation_response(
                response=response,
                mode=mode,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            # Add metadata
            result.update({
                "metadata": {
                    "mode": mode.value,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "preserve_formatting": mode_settings["preserve_formatting"],
                    "include_cultural_context": mode_settings["include_cultural_context"],
                    "handle_idioms": mode_settings["handle_idioms"],
                    "processed_at": datetime.utcnow().isoformat(),
                    "model_used": self.model_config.get("model")
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing translation request: {str(e)}")
            return {"error": f"Translation processing failed: {str(e)}"}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the translation agent asynchronously."""
        context = context or {}
        return await self.process_async({
            "content": query,
            "target_language": context.get("target_language", "en"),
            "source_language": context.get("source_language", "auto"),
            "mode": TranslationMode.CONVERSATION.value,
            "context": context
        })
    
    def _create_system_prompt(
        self,
        mode: TranslationMode,
        source_lang: str,
        target_lang: str,
        settings: Dict[str, bool],
        context: Dict[str, Any]
    ) -> str:
        """Create system prompt for translation based on mode and settings."""
        base_prompt = f"You are a professional translation AI assistant. "
        base_prompt += f"Translate from {source_lang} to {target_lang}. "
        
        if mode == TranslationMode.DOCUMENT:
            base_prompt += "Focus on accurate document translation while preserving formatting and structure. "
        elif mode == TranslationMode.CONVERSATION:
            base_prompt += "Provide natural, conversational translations suitable for real-time communication. "
        elif mode == TranslationMode.CULTURAL:
            base_prompt += "Adapt content culturally while maintaining the original meaning and intent. "
        elif mode == TranslationMode.TECHNICAL:
            base_prompt += "Ensure precise technical terminology translation with domain-specific accuracy. "
        elif mode == TranslationMode.LITERARY:
            base_prompt += "Preserve literary style, tone, and artistic elements in the translation. "
        
        if settings["include_cultural_context"]:
            base_prompt += "Include relevant cultural context and explanations where necessary. "
        
        if settings["handle_idioms"]:
            base_prompt += "Pay special attention to idioms and cultural expressions, providing appropriate equivalents. "
        
        if context:
            base_prompt += f"\nAdditional context: {json.dumps(context)}"
        
        return base_prompt
    
    def _create_user_prompt(self, content: str, mode: TranslationMode, context: Dict[str, Any]) -> str:
        """Create user prompt for translation request."""
        prompt = f"Please translate the following content:\n\n{content}"
        
        if mode == TranslationMode.DOCUMENT:
            prompt += "\n\nPlease preserve all formatting, including paragraphs, lists, and special characters."
        elif mode == TranslationMode.CONVERSATION:
            prompt += "\n\nTranslate in a natural, conversational style suitable for real-time communication."
        elif mode == TranslationMode.CULTURAL:
            prompt += "\n\nAdapt the content culturally while maintaining the original meaning."
        elif mode == TranslationMode.TECHNICAL:
            prompt += "\n\nEnsure accurate translation of technical terms and maintain domain-specific terminology."
        elif mode == TranslationMode.LITERARY:
            prompt += "\n\nPreserve the literary style, tone, and artistic elements of the original text."
        
        return prompt
    
    def _parse_translation_response(
        self,
        response: str,
        mode: TranslationMode,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """Parse and process the translation response."""
        try:
            # Basic response structure
            result = {
                "translated_text": response,
                "mode": mode.value,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            # Add mode-specific processing
            if mode == TranslationMode.CULTURAL:
                result["cultural_notes"] = self._extract_cultural_notes(response)
            elif mode == TranslationMode.TECHNICAL:
                result["technical_terms"] = self._extract_technical_terms(response)
            elif mode == TranslationMode.LITERARY:
                result["style_analysis"] = self._analyze_literary_style(response)
            
            # Add idiom handling if enabled
            if self.handle_idioms:
                result["idiom_handling"] = self._extract_idiom_translations(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing translation response: {str(e)}")
            return {"error": f"Failed to parse translation response: {str(e)}"}
    
    def _extract_cultural_notes(self, response: str) -> List[Dict[str, str]]:
        """Extract cultural context notes from translation."""
        # Implementation would identify and extract cultural notes
        return []
    
    def _extract_technical_terms(self, response: str) -> List[Dict[str, str]]:
        """Extract technical terms and their translations."""
        # Implementation would identify technical terms
        return []
    
    def _analyze_literary_style(self, response: str) -> Dict[str, Any]:
        """Analyze literary style elements in translation."""
        # Implementation would analyze style elements
        return {}
    
    def _extract_idiom_translations(self, response: str) -> List[Dict[str, str]]:
        """Extract idiom translations and explanations."""
        # Implementation would identify and explain idioms
        return []
    
    async def translate_document(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
        preserve_formatting: bool = True
    ) -> Dict[str, Any]:
        """Translate a document with format preservation."""
        return await self.process_async({
            "content": content,
            "source_language": source_lang,
            "target_language": target_lang,
            "mode": TranslationMode.DOCUMENT.value,
            "preserve_formatting": preserve_formatting
        })
    
    async def translate_conversation(
        self,
        content: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """Translate conversation content for real-time communication."""
        return await self.process_async({
            "content": content,
            "source_language": source_lang,
            "target_language": target_lang,
            "mode": TranslationMode.CONVERSATION.value
        })
    
    async def adapt_culturally(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt content culturally while maintaining meaning."""
        return await self.process_async({
            "content": content,
            "source_language": source_lang,
            "target_language": target_lang,
            "mode": TranslationMode.CULTURAL.value,
            "context": context
        })
    
    async def translate_technical(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
        domain: str
    ) -> Dict[str, Any]:
        """Translate technical content with domain-specific accuracy."""
        return await self.process_async({
            "content": content,
            "source_language": source_lang,
            "target_language": target_lang,
            "mode": TranslationMode.TECHNICAL.value,
            "context": {"domain": domain}
        })
    
    async def translate_literary(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
        style_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Translate literary content preserving style and artistic elements."""
        return await self.process_async({
            "content": content,
            "source_language": source_lang,
            "target_language": target_lang,
            "mode": TranslationMode.LITERARY.value,
            "context": style_context
        }) 