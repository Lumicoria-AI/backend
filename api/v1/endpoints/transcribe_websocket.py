"""
WebSocket endpoint for real-time speech-to-text transcription.

Protocol:
  Client -> Server:
    - JSON {"type": "start_transcription", "language": "en"}  (begin session)
    - Binary frames                                            (audio chunks)
    - JSON {"type": "stop_transcription"}                      (end session)
    - JSON {"type": "pong"}                                    (heartbeat)

  Server -> Client:
    - JSON {"type": "connected"}                               (ack)
    - JSON {"type": "final_transcript", "text": "...", ...}    (chunk result)
    - JSON {"type": "transcription_stopped", "full_transcript": "..."}
    - JSON {"type": "error", "message": "..."}
    - JSON {"type": "ping"}                                    (heartbeat)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
import json
import asyncio
import structlog

from backend.api.v1.endpoints.websocket import authenticate_websocket
from backend.services.stt_service import stt_service

logger = structlog.get_logger(__name__)

router = APIRouter()

# Minimum buffer size before transcribing (rough estimate).
# 16kHz mono 16-bit PCM = 32000 bytes/sec. WebM/Opus is much smaller,
# but faster-whisper handles the decoding. We use a time-based threshold
# tracked via chunk count instead.
MIN_BUFFER_BYTES = 8_000  # ~1 second of compressed audio minimum


@router.websocket("/transcribe/{user_id}")
async def websocket_transcribe(
    websocket: WebSocket,
    user_id: str,
    token: Optional[str] = Query(None),
):
    """
    Real-time speech-to-text via WebSocket.

    Connect: ws://host/api/v1/ws/transcribe/{user_id}?token=JWT

    Send binary audio frames (WebM/Opus from MediaRecorder).
    Receive JSON transcription results every ~4 seconds of audio.
    """
    # Authenticate
    authenticated_user_id = await authenticate_websocket(websocket, token)
    if token and authenticated_user_id != user_id:
        logger.warning(
            "transcribe_ws_unauthorized",
            requested_user_id=user_id,
            authenticated_user_id=authenticated_user_id,
        )
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("transcribe_ws_connected", user_id=user_id)

    await websocket.send_json({
        "type": "connected",
        "user_id": user_id,
        "message": "Transcription WebSocket ready",
    })

    # Per-connection state
    #
    # WebM/Opus from MediaRecorder is a single continuous stream — only the
    # first chunk contains the EBML header.  We must keep a *cumulative*
    # buffer so every temp file we write is a valid WebM from byte 0.
    # After transcribing the full cumulative buffer we diff against the
    # previous transcript to extract only the newly-spoken text.
    cumulative_audio = bytearray()
    previous_transcript = ""       # what we already sent to the client
    full_transcript_parts: list[str] = []
    chunk_index = 0
    is_transcribing = False
    language: Optional[str] = None
    slices_since_last_flush = 0
    target_slices = int(stt_service.chunk_duration)  # e.g. 4

    # Heartbeat task
    heartbeat_task = asyncio.create_task(_send_heartbeat(websocket, user_id))

    try:
        while True:
            message = await websocket.receive()

            # Binary frame = audio data
            if "bytes" in message and message["bytes"]:
                if not is_transcribing:
                    continue

                cumulative_audio.extend(message["bytes"])
                slices_since_last_flush += 1

                # Flush when we have enough new audio
                if slices_since_last_flush >= target_slices and len(cumulative_audio) >= MIN_BUFFER_BYTES:
                    slices_since_last_flush = 0
                    current_chunk = chunk_index
                    chunk_index += 1

                    # Transcribe the ENTIRE cumulative buffer (valid WebM)
                    result = await stt_service.transcribe_chunk(
                        bytes(cumulative_audio), language=language
                    )

                    full_text = result.get("text", "").strip()
                    # Extract only the new portion
                    if full_text and len(full_text) > len(previous_transcript):
                        new_text = full_text[len(previous_transcript):].strip()
                        if new_text:
                            previous_transcript = full_text
                            full_transcript_parts.append(new_text)
                            await websocket.send_json({
                                "type": "final_transcript",
                                "chunk_index": current_chunk,
                                "text": new_text,
                                "language": result.get("language", language),
                                "duration": result.get("duration", 0),
                            })

            # Text frame = control message
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON",
                    })
                    continue

                msg_type = data.get("type", "")

                if msg_type == "start_transcription":
                    is_transcribing = True
                    language = data.get("language", stt_service._language)
                    cumulative_audio.clear()
                    previous_transcript = ""
                    full_transcript_parts.clear()
                    chunk_index = 0
                    slices_since_last_flush = 0
                    logger.info(
                        "transcribe_started",
                        user_id=user_id,
                        language=language,
                    )
                    await websocket.send_json({
                        "type": "transcription_started",
                        "language": language,
                    })

                elif msg_type == "stop_transcription":
                    is_transcribing = False

                    # Final flush of remaining audio
                    if len(cumulative_audio) >= MIN_BUFFER_BYTES:
                        result = await stt_service.transcribe_chunk(
                            bytes(cumulative_audio), language=language
                        )
                        full_text = result.get("text", "").strip()
                        if full_text and len(full_text) > len(previous_transcript):
                            new_text = full_text[len(previous_transcript):].strip()
                            if new_text:
                                full_transcript_parts.append(new_text)
                                await websocket.send_json({
                                    "type": "final_transcript",
                                    "chunk_index": chunk_index,
                                    "text": new_text,
                                    "language": result.get("language", language),
                                    "duration": result.get("duration", 0),
                                })

                    cumulative_audio.clear()
                    slices_since_last_flush = 0

                    full_text = " ".join(full_transcript_parts)
                    logger.info(
                        "transcribe_stopped",
                        user_id=user_id,
                        total_chunks=chunk_index,
                        transcript_length=len(full_text),
                    )
                    await websocket.send_json({
                        "type": "transcription_stopped",
                        "full_transcript": full_text,
                        "total_chunks": chunk_index,
                    })

                elif msg_type == "pong":
                    pass  # heartbeat ack

                else:
                    logger.debug(
                        "transcribe_ws_unknown_message",
                        user_id=user_id,
                        msg_type=msg_type,
                    )

    except WebSocketDisconnect:
        logger.info("transcribe_ws_disconnect", user_id=user_id)
    except RuntimeError as e:
        # These two RuntimeError messages happen during the normal close
        # race when the client drops mid-frame. They are not bugs.
        msg = str(e)
        if (
            'Cannot call "receive"' in msg
            or 'Cannot call "send"' in msg
            or "WebSocket is not connected" in msg
        ):
            logger.info("transcribe_ws_close_race", user_id=user_id)
        else:
            logger.error("transcribe_ws_error", user_id=user_id, error=msg)
    except Exception as e:
        logger.error("transcribe_ws_error", user_id=user_id, error=str(e))
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


async def _send_heartbeat(websocket: WebSocket, user_id: str):
    """Send periodic pings to keep connection alive."""
    while True:
        try:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
        except Exception:
            break
