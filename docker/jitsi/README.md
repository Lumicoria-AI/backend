# Self-hosted Jitsi for Lumicoria Huddle

Branded video meetings at **meet.lumicoria.ai** with JWT auth and Jibri
server-side recording, hot-swappable with the public `meet.jit.si` we
ship by default.

## Architecture

```
[Browser]  ──HTTPS──>  [Cloudflare]  ──HTTP──>  [nginx]  ──>  [jitsi-web :80]
                                                                  │ XMPP/BOSH
                                                                  ▼
                                                            [jitsi-prosody :5280]
                                                            ┌──┬──┬──────────┐
                                                            ▼  ▼  ▼          ▼
                                                         jicofo jvb jibri  ...
                                                                    │
                                                                    │ MinIO + HMAC
                                                                    ▼
                                                       backend /huddles/jibri/webhook
```

## First-time setup

1. Provision the VM (or use the existing one — 4 vCPU / 16 GB recommended).
2. Add a Cloudflare DNS record:
   - **A** `meet` → your VM public IP (proxied / orange cloud)
3. From the VM:
   ```bash
   cd ~/lumicoria/backend/docker/jitsi
   cp env.example .env
   ```
4. Generate the inter-component secrets — Jitsi ships a helper, or pipe `openssl`:
   ```bash
   for var in JICOFO_AUTH_PASSWORD JICOFO_COMPONENT_SECRET \
              JVB_AUTH_PASSWORD JIBRI_XMPP_PASSWORD JIBRI_RECORDER_PASSWORD; do
     val=$(openssl rand -hex 24)
     sed -i "s|^${var}=<random>|${var}=${val}|" .env
   done
   ```
5. Generate the JWT signing secret. **Same value goes in both files** —
   the main backend's `.env` AND this one — so prosody can verify the
   tokens our backend signs.
   ```bash
   JWT_SECRET=$(openssl rand -hex 32)
   sed -i "s|^JWT_APP_SECRET=<paste-secret-here>|JWT_APP_SECRET=${JWT_SECRET}|" .env
   sed -i "s|^LUMICORIA_JIBRI_SECRET=<paste-secret-here>|LUMICORIA_JIBRI_SECRET=${JWT_SECRET}|" .env

   # Now copy the same secret into the main backend's .env:
   cd ~/lumicoria/backend
   nano .env
   #   JITSI_DOMAIN=meet.lumicoria.ai
   #   JITSI_APP_ID=lumicoria
   #   JITSI_APP_SECRET=<paste the same value as JWT_SECRET above>
   docker compose -f docker/docker-compose.yml up -d --force-recreate backend
   ```
6. Boot the Jitsi stack:
   ```bash
   cd ~/lumicoria/backend/docker/jitsi
   docker compose up -d
   ```
7. Wire nginx (running on the host) to reverse-proxy `meet.lumicoria.ai`:
   ```nginx
   server {
     listen 80;
     server_name meet.lumicoria.ai;
     location / {
       proxy_pass http://127.0.0.1:8000;
       proxy_set_header Host $host;
       proxy_set_header X-Forwarded-For $remote_addr;
       proxy_set_header X-Forwarded-Proto https;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
     }
   }
   ```
8. Reload nginx + flip Cloudflare proxy status to **Proxied** for `meet.lumicoria.ai`.
9. Test from your Mac:
   ```bash
   curl -I https://meet.lumicoria.ai
   ```
   Expect a 200 + Jitsi headers.

## Sanity checks

- `docker compose -f docker/jitsi/docker-compose.yml ps` → 5 services Up.
- Visit `https://meet.lumicoria.ai/anything` → you should get a JWT-required prompt.
- Start a Huddle from Lumicoria UI — the iframe should embed `meet.lumicoria.ai` instead of `meet.jit.si`.

## Recording flow

1. Host clicks **Record** in the room.
2. Jibri launches a Chrome instance, joins as a hidden participant, records the SFU stream.
3. On meeting end, Jibri saves the .mp4 to `/config/recordings/<room>/`.
4. The container's `finalize_recording.sh` (in this directory) uploads the file to MinIO and POSTs a signed payload to `https://api.lumicoria.ai/api/v1/huddles/jibri/webhook`.
5. The backend stamps `HuddleSQL.recording_url` + fires `huddle.recording_ready` webhook.
6. Frontend picks up the new recording on next `GET /huddles/{id}/recording`.

## Operational notes

- **JVB UDP 10000** must be reachable from the public internet. Open it in the GCP firewall.
- **Jibri runs as `--privileged`** because it needs to manage X server + audio devices. Run it on the same VM as JVB unless you have a separate "media node" tier.
- **Two-secret rotation**: rotate `JWT_APP_SECRET` first in `backend/.env`, force-recreate backend (new tokens signed with the new key), then rotate in `docker/jitsi/.env` (prosody starts accepting them). Allow ~5 min overlap for in-flight calls.
- **CMK BYOK**: orgs with `cmk_enabled` apply Fernet envelope encryption to the recording before upload. Jibri's MP4 is encrypted by the backend on the read path, not by Jibri itself. See `services/cmk_service.py`.
