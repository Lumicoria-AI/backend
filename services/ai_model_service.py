from typing import Any, Dict, List, Optional
import structlog

from backend.ai_models import get_llm_client, LLMClient, LLMConfig
from backend.core.config import settings as app_settings

logger = structlog.get_logger()


class AIModeService:
    def __init__(self):
        # Initialize the default LLM client via the provider-agnostic abstraction
        try:
            self.llm_client: Optional[LLMClient] = get_llm_client()
        except Exception as e:
            logger.warning(f"Default LLM client not initialized: {e}")
            self.llm_client = None

    async def process_text(
        self,
        model_name: str,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process text input using the LLM abstraction layer.
        
        The model_name parameter now maps to a provider name (e.g., 'perplexity', 'gemini')
        or the default provider is used.
        """
        await logger.info("Processing text with AI model", model_name=model_name)

        # Resolve which provider to use
        provider = self._resolve_provider(model_name)
        
        try:
            client = get_llm_client(provider=provider)
        except Exception as e:
            raise ValueError(f"Failed to get LLM client for '{provider}': {e}")

        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build config from settings
        config = LLMConfig()
        if settings:
            if "temperature" in settings:
                config.temperature = settings["temperature"]
            if "max_tokens" in settings:
                config.max_tokens = settings["max_tokens"]
            if "model" in settings:
                config.model = settings["model"]
            if "system_prompt" in settings:
                messages.insert(0, {"role": "system", "content": settings["system_prompt"]})

        response = await client.generate(messages, config=config)
        return {"content": response.content, "model": response.model, "id": response.response_id}

    def _resolve_provider(self, model_name: str) -> Optional[str]:
        """Map a model_name string to a provider name."""
        name = model_name.lower()
        if name in ("perplexity", "sonar") or "sonar" in name:
            return "perplexity"
        if name in ("gemini",) or "gemini" in name:
            return "gemini"
        if name in ("openai",) or "gpt" in name or name.startswith(("o1", "o3")):
            return "openai"
        if name in ("anthropic",) or "claude" in name:
            return "anthropic"
        if name in ("mistral",) or "codestral" in name or "pixtral" in name or "mixtral" in name:
            return "mistral"
        # Fall back to default provider
        return None

    async def process_document(
        self,
        model_name: str,
        document_content: bytes, # Raw document bytes
        file_type: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document content using a specified AI model (e.g., for extraction, analysis).
        """
        await logger.info("Processing document with AI model", model_name=model_name, file_type=file_type)

        # TODO: Route call to the appropriate document processing/OCR AI model client
        # Example routing:
        # if model_name == "OCR": # Example model name for OCR
        #     if not self.ocr_client:
        #          raise ValueError("OCR client is not initialized.")
        #     result = await self.ocr_client.process_document(document_content, file_type, settings)
        #     return result
        # elif model_name == "DocumentAnalysis": # Another example
        #      if not self.another_document_client:
        #           raise ValueError("Document analysis client is not initialized.")
        #      result = await self.another_document_client.process_document(document_content, file_type, settings)
        #      return result
        # else:
        await logger.warning("Unsupported document model", model_name=model_name)
        raise ValueError(f"Unsupported document model: {model_name}")

    async def process_image(
        self,
        model_name: str,
        image_content: bytes, # Raw image bytes
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process image content using a specified AI model (e.g., for analysis, object detection).
        Used in Live Interaction for camera feed.
        """
        await logger.info("Processing image with AI model", model_name=model_name)

        # TODO: Route call to the appropriate image analysis AI model client
        # Example routing:
        # if model_name == "ImageAnalysis": # Example model name
        #     if not self.image_analysis_client:
        #          raise ValueError("Image analysis client is not initialized.")
        #     result = await self.image_analysis_client.process_image(image_content, settings)
        #     return result
        # elif model_name == "VisionOCR": # Another example
        #      if not self.vision_ocr_client:
        #           raise ValueError("Vision OCR client is not initialized.")
        #      result = await self.vision_ocr_client.process_image(image_content, settings)
        #      return result
        # else:
        await logger.warning("Unsupported image model", model_name=model_name)
        raise ValueError(f"Unsupported image model: {model_name}")

    async def process_audio(
        self,
        model_name: str,
        audio_content: bytes, # Raw audio bytes
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process audio content using a specified AI model (e.g., for speech-to-text).
        Used in Live Interaction for voice input.
        """
        await logger.info("Processing audio with AI model", model_name=model_name)

        # TODO: Route call to the appropriate speech-to-text AI model client
        # Example routing:
        # if model_name == "STT": # Example model name
        #     if not self.stt_client:
        #          raise ValueError("STT client is not initialized.")
        #     result = await self.stt_client.process_audio(audio_content, settings)
        #     return result
        # elif model_name == "TranscriptionModel": # Another example
        #      if not self.transcription_client:
        #           raise ValueError("Transcription client is not initialized.")
        #      result = await self.transcription_client.process_audio(audio_content, settings)
        #      return result
        # else:
        await logger.warning("Unsupported audio model", model_name=model_name)
        raise ValueError(f"Unsupported audio model: {model_name}")

    async def process_sketch(
        self,
        model_name: str,
        sketch_data: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process sketch data using a specified AI model (e.g., for recognition).
        Used in Live Interaction for sketch input.
        """
        await logger.info("Processing sketch with AI model", model_name=model_name)

        # TODO: Route call to the appropriate sketch recognition AI model client
        # Example routing:
        # if model_name == "SketchRecognition": # Example model name
        #     if not self.sketch_recognition_client:
        #          raise ValueError("Sketch recognition client is not initialized.")
        #     result = await self.sketch_recognition_client.process_sketch(sketch_data, settings)
        #     return result
        # elif model_name == "DiagramAnalysis": # Another example
        #      if not self.diagram_analysis_client:
        #           raise ValueError("Diagram analysis client is not initialized.")
        #      result = await self.diagram_analysis_client.process_sketch(sketch_data, settings)
        #      return result
        # else:
        await logger.warning("Unsupported sketch model", model_name=model_name)
        raise ValueError(f"Unsupported sketch model: {model_name}")

# Create a singleton instance
ai_model_service = AIModeService() 