#!/bin/bash
# Lumicoria Huddle — Jibri finalize hook.
#
# Jibri invokes this script after each recording finishes. We:
#   1. Read the room name + file path from Jibri's args.
#   2. Upload the .mp4 to MinIO under huddle-recordings/<huddle_id>/jibri-final.mp4
#      (room name format: "lumi-<huddle_id_short>" — see backend/db/postgres_models.py:_huddle_room_name).
#   3. Sign an HMAC over the payload using LUMICORIA_JIBRI_SECRET.
#   4. POST to https://api.lumicoria.ai/api/v1/huddles/jibri/webhook.
#
# Required env vars (set by docker-compose.yml):
#   LUMICORIA_BACKEND_URL       — https://api.lumicoria.ai
#   LUMICORIA_JIBRI_SECRET      — shared HMAC secret
#   LUMICORIA_MINIO_BUCKET      — bucket name (default: huddle-recordings)
#   MINIO_ENDPOINT              — http://minio:9000
#   MINIO_ACCESS_KEY            — MinIO root user
#   MINIO_SECRET_KEY            — MinIO root password
#
# Args (passed by Jibri):
#   $1 — recording directory containing the .mp4 + metadata.json

set -eu

REC_DIR="$1"

if [ -z "${REC_DIR}" ] || [ ! -d "${REC_DIR}" ]; then
  echo "finalize_recording: invalid REC_DIR='${REC_DIR}'" >&2
  exit 1
fi

# Find the .mp4 file
MP4_PATH="$(find "${REC_DIR}" -maxdepth 1 -name '*.mp4' | head -n 1)"
if [ -z "${MP4_PATH}" ]; then
  echo "finalize_recording: no .mp4 in '${REC_DIR}'" >&2
  exit 1
fi

# Parse room name from metadata.json (Jibri writes this).
META_PATH="${REC_DIR}/metadata.json"
ROOM_NAME="$(jq -r '.meeting_url // empty' "${META_PATH}" 2>/dev/null || true)"
ROOM_NAME="${ROOM_NAME##*/}"  # strip URL prefix
ROOM_NAME="${ROOM_NAME%%@*}"  # strip @domain suffix
ROOM_NAME="${ROOM_NAME%%\?*}" # strip query string

# Derive huddle_id from our naming convention: lumi-<huddle_id_short>
# Our backend persists the full HuddleSQL.id (uuid); the room_name keeps
# only the first 16 hex chars. We pass the room_name; the backend looks
# up the matching HuddleSQL row by room_name.
HUDDLE_ID="${ROOM_NAME}"

OBJECT_KEY="${LUMICORIA_MINIO_BUCKET}/${HUDDLE_ID}/jibri-final.mp4"
SIZE_BYTES=$(stat -c '%s' "${MP4_PATH}" 2>/dev/null || wc -c < "${MP4_PATH}")

# Upload to MinIO via mc CLI (alias 'lumi' configured at jibri build time).
if command -v mc >/dev/null 2>&1; then
  mc alias set lumi "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" --api S3v4 >/dev/null 2>&1 || true
  mc cp "${MP4_PATH}" "lumi/${OBJECT_KEY}" >/dev/null 2>&1
fi

# Sign HMAC and notify backend.
HMAC_INPUT="${HUDDLE_ID}.${OBJECT_KEY}"
SIGNATURE="$(printf '%s' "${HMAC_INPUT}" | openssl dgst -sha256 -hmac "${LUMICORIA_JIBRI_SECRET}" -r | awk '{print $1}')"

PAYLOAD="$(jq -nc \
  --arg huddle_id "${HUDDLE_ID}" \
  --arg room_name "${ROOM_NAME}" \
  --arg object_key "${OBJECT_KEY}" \
  --arg signature "${SIGNATURE}" \
  --argjson size_bytes "${SIZE_BYTES}" \
  '{
    huddle_id: $huddle_id,
    room_name: $room_name,
    object_key: $object_key,
    mime: "video/mp4",
    size_bytes: $size_bytes,
    signature: $signature
  }'
)"

curl -sS -X POST \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  "${LUMICORIA_BACKEND_URL}/api/v1/huddles/jibri/webhook" \
  || echo "finalize_recording: webhook POST failed" >&2
