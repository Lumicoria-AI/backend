"""
Speech-to-Text service using Faster-Whisper.

Provides a singleton STT service that wraps faster-whisper for:
- Real-time chunk transcription (WebSocket streaming)
- Full file transcription (uploaded audio files)

All inference runs via asyncio.to_thread() to avoid blocking the event loop.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Audio file extensions we can transcribe
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm", ".opus", ".aac"}


class STTService:
    """Singleton service wrapping faster-whisper for speech-to-text."""

    def __init__(self):
        self._model = None
        self._model_size = settings.STT_MODEL_SIZE
        self._device = settings.STT_DEVICE
        self._compute_type = settings.STT_COMPUTE_TYPE
        self._language = settings.STT_LANGUAGE
        self._chunk_duration = settings.STT_CHUNK_DURATION

    def _get_model(self):
        """Lazy-load the faster-whisper model (thread-safe via GIL)."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                logger.info(
                    "stt_loading_model",
                    model_size=self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                logger.info("stt_model_loaded", model_size=self._model_size)
            except ImportError:
                logger.error(
                    "stt_import_error",
                    message="faster-whisper not installed. Run: pip install faster-whisper>=1.0.0",
                )
                raise
            except Exception as e:
                logger.error("stt_model_load_failed", error=str(e))
                raise
        return self._model

    def preload(self):
        """Eagerly load the model at startup (called from lifespan)."""
        self._get_model()

    def _transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous transcription of an audio file.

        Args:
            audio_path: Path to the audio file (any format ffmpeg supports).
            language: ISO 639-1 language code, or None for auto-detect.

        Returns:
            Dict with 'text' (full transcript), 'segments' (list of segment dicts),
            'language' (detected language), 'duration' (total audio duration).
        """
        model = self._get_model()
        lang = language or self._language

        kwargs = {}
        if lang:
            kwargs["language"] = lang

        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            **kwargs,
        )

        segments = []
        full_text_parts = []
        for segment in segments_iter:
            seg_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }
            segments.append(seg_data)
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }

    async def transcribe_chunk(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe a chunk of audio bytes (from WebSocket streaming).

        Writes bytes to a temp file, runs faster-whisper in a thread,
        then cleans up.

        Args:
            audio_bytes: Raw audio data (WebM/Opus, WAV, etc.).
            language: ISO 639-1 language code, or None for auto-detect.

        Returns:
            Dict with 'text', 'segments', 'language', 'duration'.
        """
        tmp_path = None
        try:
            # Write audio bytes to temp file (faster-whisper needs a file path)
            with tempfile.NamedTemporaryFile(
                suffix=".webm", delete=False
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Run blocking transcription in thread pool
            result = await asyncio.to_thread(
                self._transcribe_sync, tmp_path, language
            )
            return result

        except Exception as e:
            logger.error("stt_chunk_transcribe_error", error=str(e))
            return {"text": "", "segments": [], "error": str(e)}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe a full audio file (for uploaded recordings).

        Args:
            file_path: Path to the audio file on disk.
            language: ISO 639-1 language code, or None for auto-detect.

        Returns:
            Dict with 'text', 'segments', 'language', 'language_probability', 'duration'.
        """
        try:
            result = await asyncio.to_thread(
                self._transcribe_sync, file_path, language
            )
            logger.info(
                "stt_file_transcribed",
                file=file_path,
                duration=result.get("duration"),
                language=result.get("language"),
                segments=len(result.get("segments", [])),
            )
            return result

        except Exception as e:
            logger.error("stt_file_transcribe_error", file=file_path, error=str(e))
            return {"text": "", "segments": [], "error": str(e)}

    @property
    def chunk_duration(self) -> float:
        return self._chunk_duration

    @staticmethod
    def is_audio_file(filename: str) -> bool:
        """Check if a filename has an audio extension we can transcribe."""
        ext = Path(filename).suffix.lower()
        return ext in AUDIO_EXTENSIONS


# Module-level singleton
stt_service = STTService()
