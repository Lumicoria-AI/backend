# Lumicoria Meet — self-hosted Jitsi (production)

Branded, recordable, moderatable video meetings at **meet.lumicoria.ai**.
Six containers: `web`, `prosody`, `jicofo`, `jvb`, `jibri`, `coturn`.
JWT-only auth, MinIO recording uploads, signed Jibri webhooks, per-org
branding plumbed through `_enrich_with_jitsi`.

```
[Browser]  ──HTTPS──>  [nginx (host)]  ──>  [jitsi-web :8000]
                                              │ XMPP/BOSH
                                              ▼
                                          [jitsi-prosody :5280]   (JWT auth)
                                          ┌──┬──┬──────────┐
                                          ▼  ▼  ▼          ▼
                                       jicofo jvb jibri    coturn (host net)
                                                  │
                                                  │ MinIO + HMAC POST
                                                  ▼
                                  backend  /api/v1/huddles/jibri/webhook
```

---

## Prerequisites

- **VM**: 4 vCPU / 16 GB RAM minimum. Ubuntu 22.04 LTS recommended.
- **Public IPv4** with the following ports open:
  - `80/tcp`, `443/tcp` (HTTP/HTTPS via nginx)
  - `10000/udp` (JVB media)
  - `3478/tcp+udp`, `5349/tcp` (Coturn STUN/TURN)
  - `49152-65535/udp` (Coturn relay range)
- **DNS**: an A record for `meet.lumicoria.ai` → VM public IP.
- **`snd_aloop` kernel module** loaded for Jibri:
  ```bash
  echo "snd_aloop" | sudo tee -a /etc/modules
  sudo modprobe snd_aloop
  lsmod | grep snd_aloop   # confirm
  ```
- **Docker + Compose v2** installed on the VM.
- **MinIO** running and reachable from the Jibri container (typically the
  same VM with the backend's MinIO).

---

## First-time setup

```bash
cd ~/lumicoria/backend/docker/jitsi
cp env.example .env
```

### 1. Generate component secrets

```bash
for var in JICOFO_AUTH_PASSWORD JICOFO_COMPONENT_SECRET \
           JVB_AUTH_PASSWORD JIBRI_XMPP_PASSWORD JIBRI_RECORDER_PASSWORD; do
  val=$(openssl rand -hex 24)
  sed -i "s|^${var}=<random>|${var}=${val}|" .env
done
```

### 2. Generate the JWT signing secret

**The same value goes in three places**: this `.env`, the backend's
`.env` (`JITSI_APP_SECRET`), and as `LUMICORIA_JIBRI_SECRET` so the
Jibri finalize hook can sign its webhook to the backend.

```bash
JWT_SECRET=$(openssl rand -hex 32)
sed -i "s|^JWT_APP_SECRET=<paste-secret-here>|JWT_APP_SECRET=${JWT_SECRET}|" .env
sed -i "s|^LUMICORIA_JIBRI_SECRET=<paste-secret-here>|LUMICORIA_JIBRI_SECRET=${JWT_SECRET}|" .env

# Backend .env:
nano ~/lumicoria/backend/.env
#   JITSI_DOMAIN=meet.lumicoria.ai
#   JITSI_APP_ID=lumicoria
#   JITSI_APP_SECRET=<paste the same JWT_SECRET above>
#   JITSI_JIBRI_ALLOWED_CIDR=<vm-ip>/32   # webhook IP allowlist
```

### 3. Generate the TURN shared secret

Same value goes in the docker/jitsi `.env` and as the `TURNCREDENTIALS_SECRET`.

```bash
TURN_SECRET=$(openssl rand -hex 32)
sed -i "s|^TURNCREDENTIALS_SECRET=<paste-secret-here>|TURNCREDENTIALS_SECRET=${TURN_SECRET}|" .env
```

### 4. Set the VM public IP

```bash
PUBLIC_IP=$(curl -s ifconfig.io)
sed -i "s|^DOCKER_HOST_ADDRESS=.*|DOCKER_HOST_ADDRESS=${PUBLIC_IP}|" .env
```

### 5. Boot the stack

```bash
docker compose --env-file .env pull
docker compose --env-file .env up -d
docker compose ps   # all 6 should report (healthy) within 60s
```

---

## Reverse proxy (nginx on the host)

```nginx
# /etc/nginx/sites-available/meet.lumicoria.ai
server {
    listen 80;
    server_name meet.lumicoria.ai;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name meet.lumicoria.ai;

    ssl_certificate     /etc/letsencrypt/live/meet.lumicoria.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/meet.lumicoria.ai/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # ── Security headers ──────────────────────────────────────────────
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "camera=(self), microphone=(self), display-capture=(self)" always;
    # CSP — allow the Jitsi external_api.js + our MinIO bucket + Lumicoria origin.
    add_header Content-Security-Policy
        "default-src 'self';
         script-src 'self' 'unsafe-inline' 'unsafe-eval';
         style-src 'self' 'unsafe-inline';
         img-src 'self' data: https://*.lumicoria.ai https://minio.lumicoria.ai;
         connect-src 'self' wss://meet.lumicoria.ai;
         frame-ancestors 'self' https://*.lumicoria.ai;
         media-src 'self' blob: https://minio.lumicoria.ai;"
        always;

    # WebSocket upgrade (required for Jitsi XMPP-over-WebSocket).
    location /xmpp-websocket {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600s;
    }

    location /colibri-ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        proxy_http_version 1.1;
    }
}
```

Get the cert via certbot (`certbot --nginx -d meet.lumicoria.ai`) before
first boot. Coturn shares the same cert via the volume mount.

Then `sudo nginx -t && sudo systemctl reload nginx`.

---

## Prometheus scrape

JVB exposes colibri stats at `:8080/colibri/stats` (host port `127.0.0.1:8080`).
Add to your Prometheus config:

```yaml
scrape_configs:
  - job_name: 'jvb'
    static_configs:
      - targets: ['127.0.0.1:8080']
    metrics_path: '/colibri/stats'
```

JVB metrics you care about:

- `bit_rate_download_mbps` / `bit_rate_upload_mbps` — bandwidth per bridge
- `conferences` — current active rooms
- `participants` — current active users
- `octo_conferences` — distributed conferences (relevant in multi-bridge mode)

---

## Sanity checks

```bash
# All 6 healthy?
docker compose ps

# Web reachable?
curl -I https://meet.lumicoria.ai

# JWT required (no token → 401 or login redirect)?
curl -I https://meet.lumicoria.ai/test-room

# TURN reachable?
nc -zvu meet.lumicoria.ai 3478
```

Open Lumicoria, start a Huddle. In the browser DevTools network panel,
confirm:
- The Jitsi iframe loads `https://meet.lumicoria.ai/...`
- `external_api.js` is fetched from `meet.lumicoria.ai`
- Zero references to `meet.jit.si` in the page source

---

## Recording flow

1. Host clicks **Record** in `RoomControlsPanel` (host-only).
2. Jibri launches a headless Chrome, joins the room, records the SFU stream.
3. On meeting end, Jibri saves `/config/recordings/<room>/recording.mp4`.
4. The mounted `finalize_recording.sh` uploads the MP4 to MinIO at
   `huddle-recordings/<huddle_id>/jibri-final.mp4`, signs an HMAC,
   POSTs to `https://api.lumicoria.ai/api/v1/huddles/jibri/webhook`.
5. Backend:
   - Verifies signature
   - Checks the source IP against `JITSI_JIBRI_ALLOWED_CIDR`
   - Drops the request if the `(huddle_id, object_key)` pair was seen
     in the last 24 h (Redis-backed replay protection)
   - Stamps `recording_object_key` + `recording_expires_at` on HuddleSQL
   - Fires `huddle.recording_ready` webhook
   - Emails the host with a `/huddles/<id>/recording` playback link
6. Frontend's `HuddleRecording` page renders the signed playback URL.

---

## Branding

Per-org branding (logo, colors, app name, watermark link, welcome message)
is set via the Lumicoria UI at `/settings/meeting-branding` and persisted
on `OrgBrandingSQL.meeting_*` columns. The backend's `_enrich_with_jitsi`
caches lookups in Redis with a 5-min TTL and includes the block in every
huddle response. `JitsiEmbed` applies the branding via
`interfaceConfigOverwrite` (which sets `DEFAULT_LOGO_URL`, `APP_NAME`,
`SHOW_JITSI_WATERMARK=false`, etc.) + a `<style>` tag for CSS custom
properties.

No system-level Jitsi config needs to change — branding is per-room and
applied client-side.

---

## Operational notes

- **Image pinning**: `JITSI_IMAGE_VERSION` in `.env` controls all five
  Jitsi images. To upgrade: bump the tag, `docker compose pull`,
  `docker compose up -d --force-recreate`. Tags:
  https://github.com/jitsi/docker-jitsi-meet/releases
- **JVB UDP 10000** must be reachable from the public internet. Without
  it, participants can't connect even when TURN works.
- **Jibri runs as `--privileged`** because it needs to manage X server +
  ALSA devices. Run on the same VM as JVB unless you're operating a
  separate "media node" tier.
- **Coturn uses `network_mode: host`** so it can advertise the VM's real
  public IP for ICE. This is required.
- **Two-secret rotation** (zero-downtime): rotate `JWT_APP_SECRET` first
  in `backend/.env`, force-recreate the backend (new tokens signed with
  the new key), then rotate in `docker/jitsi/.env` (prosody starts
  accepting them). Allow ~5 min overlap so in-flight calls don't drop.
- **Recording retention**: per-huddle `recording_retention_days` (default
  30). A nightly Celery task purges expired recordings — verify it's
  registered in `tasks/celery_app.py:beat_schedule`.
- **Disk space**: Jibri recordings can be large (~25 MB / min). Mount
  `${CONFIG}/jibri/recordings` to a persistent disk you can grow.
- **Logs**:
  - `docker compose logs -f web prosody jicofo` for signaling issues
  - `docker compose logs -f jvb` for media bridge problems
  - `docker compose logs -f jibri` for recording failures
  - `docker compose logs -f coturn` for TURN connectivity issues
