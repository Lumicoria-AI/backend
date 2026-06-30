# Lumicoria Meet — self-hosted Jitsi (production)

Branded, recordable, moderatable video meetings at **meet.lumicoria.ai**.
Six containers (`jitsi-web`, `jitsi-prosody`, `jitsi-jicofo`, `jitsi-jvb`,
`jitsi-jibri`, `coturn`) live in the **root `docker/docker-compose.yml`**
alongside the rest of the platform under a `meet` profile. They share
the same network, the same project-root `.env`, and the same operator
commands as every other service in this repo.

```
[Browser]  ──HTTPS──>  [nginx (host)]  ──>  [jitsi-web :80]
                                              │ XMPP/BOSH
                                              ▼
                                        [jitsi-prosody]   (JWT auth)
                                          ┌──┬──┬──────────┐
                                          ▼  ▼  ▼          ▼
                                       jicofo jvb jibri    coturn (host net)
                                                  │
                                                  │ MinIO + HMAC POST
                                                  ▼
                                  backend  /api/v1/huddles/jibri/webhook
```

## Configuration model

There is **one** `.env` for the whole product, at the project root
(`/opt/lumicoria/.env` on the VM, `~/Documents/LUMICORIA AI/.env`
locally). The backend reads it via pydantic-settings; the Meet stack
reads it via the `env_file: ../.env` directive on every service.

The Jitsi-related variables in `.env` are documented in `env.example`
in this directory — **that file is read-only documentation. Real values
live in the project-root `.env`.** The repo already contains generated
dev secrets there.

One detail: `docker/.env` in this repo is a **symlink to `../.env`**,
not a separate file. Docker Compose looks for `.env` in the same
directory as the compose file for `${VAR}` interpolation; the symlink
makes that resolve to the global `.env` without duplicating values.
After a fresh `git clone` on the VM, recreate it once:
```bash
cd /opt/lumicoria/docker && ln -sf ../.env .env
```

## Boot commands

Meet services are part of the default compose set — `docker compose up -d`
boots them alongside backend/celery/databases/etc. No special flag.

```bash
cd /opt/lumicoria/docker

# Everything — platform + meet
docker compose up -d

# Just the meet services (when the platform is already running)
docker compose up -d jitsi-web jitsi-prosody jitsi-jicofo jitsi-jvb jitsi-jibri coturn

# Logs / restart / down — same shape as any other service
docker compose logs -f jitsi-jibri
docker compose restart jitsi-jvb
docker compose down       # stops everything; keeps volumes
docker compose down -v    # nukes volumes too
```

## Prerequisites

- **VM**: 4 vCPU / 16 GB RAM minimum (recording is CPU-heavy). Ubuntu 22.04 LTS recommended.
- **Public IPv4** with the following ports open:
  - `80/tcp`, `443/tcp` (HTTP/HTTPS via nginx)
  - `10000/udp` (JVB media)
  - `3478/tcp+udp`, `5349/tcp` (Coturn STUN/TURN)
  - `49152-65535/udp` (Coturn relay range)
- **DNS**: an A record for `meet.lumicoria.ai` → VM public IP.
- **`snd_aloop` kernel module** for Jibri (Linux only):
  ```bash
  echo "snd_aloop" | sudo tee -a /etc/modules
  sudo modprobe snd_aloop
  lsmod | grep snd_aloop   # confirm
  ```

## First-time secret rotation on a fresh VM

The repo ships generated dev secrets in the project-root `.env`. On the
VM, regenerate them so production doesn't share secrets with dev. Run
once after `git clone`:

```bash
cd /opt/lumicoria

# JWT signing secret (one value, shared by THREE env vars).
JWT_SECRET=$(openssl rand -hex 32)
sed -i "s|^JITSI_APP_SECRET=.*|JITSI_APP_SECRET=${JWT_SECRET}|" .env
sed -i "s|^JWT_APP_SECRET=.*|JWT_APP_SECRET=${JWT_SECRET}|" .env
sed -i "s|^LUMICORIA_JIBRI_SECRET=.*|LUMICORIA_JIBRI_SECRET=${JWT_SECRET}|" .env

# Component secrets (each must be unique).
sed -i "s|^JICOFO_AUTH_PASSWORD=.*|JICOFO_AUTH_PASSWORD=$(openssl rand -hex 24)|" .env
sed -i "s|^JICOFO_COMPONENT_SECRET=.*|JICOFO_COMPONENT_SECRET=$(openssl rand -hex 24)|" .env
sed -i "s|^JVB_AUTH_PASSWORD=.*|JVB_AUTH_PASSWORD=$(openssl rand -hex 24)|" .env
sed -i "s|^JIBRI_XMPP_PASSWORD=.*|JIBRI_XMPP_PASSWORD=$(openssl rand -hex 24)|" .env
sed -i "s|^JIBRI_RECORDER_PASSWORD=.*|JIBRI_RECORDER_PASSWORD=$(openssl rand -hex 24)|" .env

# TURN shared secret.
sed -i "s|^TURNCREDENTIALS_SECRET=.*|TURNCREDENTIALS_SECRET=$(openssl rand -hex 32)|" .env

# VM public IP — JVB advertises, Coturn uses for ICE, webhook allowlist locks to it.
PUBLIC_IP=$(curl -s ifconfig.io)
sed -i "s|^DOCKER_HOST_ADDRESS=.*|DOCKER_HOST_ADDRESS=${PUBLIC_IP}|" .env
sed -i "s|^JITSI_JIBRI_ALLOWED_CIDR=.*|JITSI_JIBRI_ALLOWED_CIDR=${PUBLIC_IP}/32|" .env

# Restart backend so it picks up the new JITSI_APP_SECRET, then bring Meet up.
sudo systemctl restart lumicoria-backend
cd docker && docker compose --profile meet up -d
```

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
        proxy_pass http://127.0.0.1:8800;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600s;
    }

    location /colibri-ws {
        proxy_pass http://127.0.0.1:8800;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        proxy_pass http://127.0.0.1:8800;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        proxy_http_version 1.1;
    }
}
```

`HTTP_PORT` defaults to `8800` so nginx proxies to it on `127.0.0.1`.
Then `sudo nginx -t && sudo systemctl reload nginx`.

## Prometheus scrape

JVB exposes colibri stats at the host loopback `127.0.0.1:8090/colibri/stats`.
Add to your Prometheus config:

```yaml
scrape_configs:
  - job_name: 'jvb'
    static_configs:
      - targets: ['127.0.0.1:8090']
    metrics_path: '/colibri/stats'
```

## Sanity checks

```bash
# All 6 healthy?
docker compose --profile meet ps

# Web reachable through nginx?
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

## Recording flow

1. Host clicks **Record** in `RoomControlsPanel` (host-only).
2. Jibri launches a headless Chrome, joins the room, records the SFU stream.
3. On meeting end, Jibri saves `/config/recordings/<room>/recording.mp4`
   (persistent volume `jitsi_jibri_recordings`).
4. The mounted `finalize_recording.sh` uploads the MP4 to MinIO at
   `huddle-recordings/<huddle_id>/jibri-final.mp4` (reaches MinIO via
   the docker network — `http://minio:9000`), signs an HMAC, POSTs to
   the backend's `/api/v1/huddles/jibri/webhook`.
5. Backend stamps `recording_object_key` + `recording_expires_at` on
   HuddleSQL, fires `huddle.recording_ready`, emails the host with a
   playback link.
6. Frontend's `HuddleRecording` page renders the signed playback URL.

## Branding

Per-org branding (logo, colors, app name, watermark link, welcome
message) lives in `OrgBrandingSQL.meeting_*` columns and is uploaded by
admins at `/settings/meeting-branding` in the Lumicoria UI. The backend
includes the branding block in every huddle response via
`_enrich_with_jitsi` (cached in Redis 5 min). `JitsiEmbed` applies it to
`interfaceConfigOverwrite` (so `DEFAULT_LOGO_URL`, `APP_NAME`,
`SHOW_JITSI_WATERMARK=false`, etc.) plus a `<style>` tag for CSS custom
properties.

No system-level Jitsi config needs to change — branding is per-room and
applied client-side.

## Operational notes

- **Image pinning**: `JITSI_IMAGE_VERSION` in `.env` controls all five
  Jitsi images. Upgrade: bump the tag, `docker compose --profile meet pull`,
  `docker compose --profile meet up -d --force-recreate`. Tags:
  https://github.com/jitsi/docker-jitsi-meet/releases
- **JVB UDP 10000** must be reachable from the public internet. Without
  it, participants can't connect even when TURN works.
- **Jibri runs as `--privileged`** because it needs to manage X server +
  ALSA devices. Run on the same VM as JVB unless you're operating a
  separate "media node" tier.
- **Coturn uses `network_mode: host`** so it can advertise the VM's real
  public IP for ICE. This is required — bridge networking breaks STUN.
- **Two-secret rotation** (zero-downtime): rotate `JITSI_APP_SECRET`
  in `.env`, force-recreate the backend, then force-recreate
  `jitsi-prosody` so it picks up the new `JWT_APP_SECRET`. Allow ~5 min
  overlap so in-flight calls don't drop.
- **Recording retention**: per-huddle `recording_retention_days`
  (default 30). A daily Celery task purges expired recordings.
- **Disk space**: Jibri recordings can be ~25 MB / minute. Mount the
  `jitsi_jibri_recordings` volume to a persistent disk you can grow:
  in production, edit the volume definition in `docker/docker-compose.yml`
  to bind-mount a local path on a separate disk.
- **Logs**:
  - `docker compose --profile meet logs -f jitsi-web jitsi-prosody jitsi-jicofo`
  - `docker compose --profile meet logs -f jitsi-jvb` (media bridge)
  - `docker compose --profile meet logs -f jitsi-jibri` (recording)
  - `docker compose --profile meet logs -f coturn`
