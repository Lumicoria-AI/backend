"""
Lumicoria Huddle — Text-to-Speech (Virtual Agent Voice).

Used by the "virtual agent participant" feature: when an attached AI
agent emits a response in a live huddle, the host can opt to have it
spoken aloud into the call. The TTS audio is mixed with the host's
microphone client-side, so other participants hear the agent's voice
without needing a self-hosted Jitsi audio bridge or Jibri.

Provider order (env-driven):
  1. OpenAI TTS (tts-1 / tts-1-hd, mp3) — highest quality
  2. Google Cloud TTS (text-to-speech) — only if google-cloud-texttospeech installed
  3. Gemini TTS (Vertex AI) — only if GEMINI_API_KEY set
  4. None → frontend falls back to browser SpeechSynthesis (lower quality)

The endpoint returns audio bytes (audio/mpeg or audio/wav). Frontend
plays via Web Audio + MediaStreamDestination to inject into mic.
"""

from __future__ import annotations

from typing import Optional, Tuple

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Voice catalog — user-facing names; provider mapping per backend.
VOICE_CATALOG = [
    {"id": "warm",    "label": "Warm",         "openai": "alloy",    "lang": "en"},
    {"id": "calm",    "label": "Calm",         "openai": "echo",     "lang": "en"},
    {"id": "bright",  "label": "Bright",       "openai": "fable",    "lang": "en"},
    {"id": "deep",    "label": "Deep",         "openai": "onyx",     "lang": "en"},
    {"id": "soft",    "label": "Soft",         "openai": "nova",     "lang": "en"},
    {"id": "neutral", "label": "Neutral",      "openai": "shimmer",  "lang": "en"},
]


def _voice_meta(voice_id: str) -> dict:
    for v in VOICE_CATALOG:
        if v["id"] == voice_id:
            return v
    return VOICE_CATALOG[0]


async def synthesize(text: str, *, voice: str = "warm", quality: str = "standard") -> Tuple[bytes, str]:
    """Synthesize speech. Returns (audio_bytes, mime_type).

    Raises `RuntimeError` when no provider is configured."""
    if not text or not text.strip():
        return b"", "audio/mpeg"
    text = text.strip()[:4096]  # cap by OpenAI tts-1 max

    # ── Provider 1: OpenAI TTS ─────────────────────────────────────
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    if api_key:
        try:
            import httpx
            meta = _voice_meta(voice)
            model = "tts-1-hd" if quality == "hd" else "tts-1"
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "input": text,
                        "voice": meta["openai"],
                        "response_format": "mp3",
                    },
                )
                if r.status_code == 200:
                    return r.content, "audio/mpeg"
                logger.warning("openai_tts_failed", status=r.status_code, body=r.text[:200])
        except Exception as e:
            logger.warning("openai_tts_exception", error=str(e))

    # ── Provider 2: Google Cloud TTS ───────────────────────────────
    try:
        from google.cloud import texttospeech  # type: ignore
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_meta = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice_meta, audio_config=audio_config,
        )
        return response.audio_content, "audio/mpeg"
    except Exception:
        pass

    raise RuntimeError("No TTS provider configured. Set OPENAI_API_KEY or install google-cloud-texttospeech.")
