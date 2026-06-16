# Lumicoria AI — Backend

> The FastAPI service powering **Lumicoria AI**, an AI-native workforce platform built on the Google Cloud agent stack. This README documents the production codebase as it ships today: **~1,053 endpoints across 24 routers**, multi-tenant by construction, with 21 specialised agents collaborating through a shared orchestrator.

[![Routes](https://img.shields.io/badge/routes-1%2C053-6C4AB0)](#api-surface)
[![Tests](https://img.shields.io/badge/tests-192%20passing-10B981)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](#prerequisites)
[![Built on](https://img.shields.io/badge/built%20on-Google%20Cloud-4285F4)](#built-on-google-cloud)

---

## Contents

- [What this service does](#what-this-service-does)
- [Built on Google Cloud](#built-on-google-cloud)
- [Architecture at a glance](#architecture-at-a-glance)
- [API surface](#api-surface)
- [The 21 platform agents](#the-21-platform-agents)
- [Multi-agent orchestration](#multi-agent-orchestration)
- [Repository layout](#repository-layout)
- [Local development](#local-development)
- [Environment variables](#environment-variables)
- [Background workers](#background-workers)
- [Realtime + WebSockets](#realtime--websockets)
- [Authentication](#authentication)
- [Billing](#billing)
- [Storage + RAG](#storage--rag)
- [Observability](#observability)
- [Testing](#testing)
- [Deploying to production](#deploying-to-production)
- [License](#license)

---

## What this service does

Lumicoria is a multi-tenant AI workforce: every customer organisation has its own teams, projects, members, agents, tasks, documents, knowledge base, and billing. The backend exposes that fabric over a single coherent REST + WebSocket surface and runs **21 specialised platform agents** plus a custom-agent studio.

Capability surface:

- **Workspace**: organisations, teams, projects, members, invites, audit log, custom roles, tags, announcements, dashboards, search.
- **Agents**: catalogue, lineage, runs, cost tracking, batches, presets, schedules, handoffs, shared knowledge bases, feedback loops.
- **Collaboration**: tasks (board / list / calendar / gantt / timeline), comments + reviews, real-time chat channels, automations engine, notification rules, reminders, calendar.
- **Enterprise**: SAML 2.0 SSO, SCIM 2.0 provisioning, API tokens, outbound webhooks, domain claims, data residency, session policy, just-in-time access, compliance docs.
- **Billing**: org-scoped Stripe subscriptions, per-seat pricing ($39 Team / $79 Business / $129 Enterprise floor), credits ledger, promo codes, contracts, quotes, BYOK.
- **AI surfaces**: chat (multi-provider routing), document RAG, vision, meeting transcription + summarisation, research, legal, customer service portal, fact-checking, social media, data analysis, learning coach, workspace ergonomics, focus flow, knowledge graph, ethics-bias, creative content, translation.

---

## Built on Google Cloud

| Layer                 | Service                                                           |
|-----------------------|-------------------------------------------------------------------|
| Reasoning             | **Gemini 2.5 Pro** (grounded reasoning + structured output)       |
| Intent / routing      | **Gemini 2.5 Flash** (classification + fast paths)                |
| Retrieval grounding   | **Vertex AI Search** (managed retrieval over customer corpora)    |
| Agent state           | **Agent Runtime sessions + Memory Bank**                          |
| Multi-agent transport | **Agent-to-Agent (A2A) protocol**                                 |
| Tool gateway          | **Model Context Protocol (MCP)** with per-tenant scopes + audit   |
| Agent framework       | **Agent Development Kit (ADK)** + LangChain Google components     |
| Real-time speech      | **WebSockets + `faster-whisper`** for streaming STT               |
| Auth                  | **Firebase Authentication** + SAML 2.0 + SCIM 2.0                 |
| Compute (production)  | **Cloud Run** (containers from `backend/Dockerfile`)              |
| Tracing               | **Cloud Trace** (OpenTelemetry tagged by org/team/project/agent)  |
| Analytics             | **BigQuery** + **Looker Studio**                                  |
| Object storage        | **Cloud Storage** (with **MinIO** for local dev)                  |
| Database              | **MongoDB Atlas** (primary), Postgres + Cassandra (optional)      |
| Vector stores         | **Weaviate** (default), **Qdrant**, **Chroma** (pluggable)        |
| Model fallback        | **Google Gemini ↔ Mistral** auto-failover                         |

Defensibility compounds at three layers: the operator-facing Workspace above the Google Cloud stack, multi-agent collaboration over A2A, and optimisation patterns harvested from real workloads that flow back into the shared orchestrator on every release.

---

## Architecture at a glance

```
                    ┌──────────────────────────────────────────┐
                    │       React frontend (Lumicoria-frontend)│
                    └─────────────┬────────────────────────────┘
                                  │ HTTPS + WebSocket
                                  ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    FastAPI (this repo, backend/)                │
   │                  ~1,053 routes · 24 routers                      │
   │   ┌──────────────────────────────────────────────────────────┐   │
   │   │  Shared Orchestrator  (backend/agents/, services/)       │   │
   │   │  • Permissions resolver  • Plan caps  • Event bus        │   │
   │   │  • Realtime broker (Redis pub/sub)  • Activity logger    │   │
   │   │  • Audit hooks  • Idempotency  • Rate limiting           │   │
   │   └────────────┬────────────────────────────────┬────────────┘   │
   │                │                                │                │
   │     21 Platform Agents              Customer Custom Agents       │
   │     (router.py registry)             (Agent Studio + ADK)        │
   └────────────────┬────────────────────────────────┬────────────────┘
                    │ A2A handoffs    MCP tool calls │
        ┌───────────┴────────────┐      ┌────────────┴───────────┐
        ▼                        ▼      ▼                        ▼
   Vertex AI Search        Gemini 2.5             MCP Gateway        Cloud Trace
   (grounded retrieval)    (Pro + Flash)          (typed tools)      (per-step spans)

   ┌──────────────────────────────────────────────────────────────────┐
   │  Persistence                                                     │
   │   MongoDB (primary)  ·  Postgres (workflows)  ·  Redis (cache)   │
   │   MinIO / GCS (blobs)  ·  Weaviate (vectors)  ·  Cassandra (TS)  │
   └──────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────┐
   │  Background workers (Celery)                                     │
   │   webhook delivery  ·  agent metrics materialiser  ·  automations│
   │   task reminders  ·  wellbeing digests  ·  webhook retry         │
   └──────────────────────────────────────────────────────────────────┘
```

---

## API surface

The router inventory below reflects the live OpenAPI document mounted at `/api/v1/openapi.json`.

| Group                | Routers                                              | Notes |
|----------------------|------------------------------------------------------|-------|
| Auth + users         | `auth`, `users`                                      | Firebase + JWT, email/password + Google + SAML SSO |
| Workspace shell      | `workspace`, `organizations`, `organizations_extended`, `teams`, `teams_extended`, `projects_v2`, `projects_v2_extended` | Multi-tenant org → team → project hierarchy |
| Members + invites    | `invites`, `invites_extended`                        | Bulk CSV / Google Workspace import, shareable links |
| Tasks                | `tasks`, `tasks_extended`, `tasks_v2_extras`         | Comments, dependencies, watchers, board/list/calendar/gantt/timeline views, cross-project imports |
| Comments             | `comments`, `comments_extras`                        | Threads, mentions, reviews (request/approve/reject), shares, watches |
| Agents (collab)      | `agents`, `agents_v2`, `agents_v2_extras`, `agent_studio` | Platform agents + custom agents (CRUD, publish, fork, share), batches, presets, schedules, handoffs |
| Workflow execution   | `workflow_execution`                                 | Visual workflow runs |
| Analytics            | `analytics`, `analytics_v2`, `analytics_v2_extras`   | Org/team/project/user/agent/cost — overview/throughput/cycle-time/forecast/cohorts/funnel/timeline |
| Activity + audit     | `activity`                                           | Filterable feed, per-user audit export to CSV |
| Notifications        | `notifications`, `notifications_extras`, `notification_rules` | Devices, topics, broadcast, rules, digests |
| Email                | `emails`                                             | Templates + preview, sent log, sending domains + DKIM, deliverability |
| Chat (channels)      | `chat`, `lumicoria_chat`                             | Project channels, mentions, threads, agent invocation |
| Realtime             | `websocket`, `transcribe_websocket`                  | Presence, typing, notifications, streaming STT |
| Documents + RAG      | `documents`, `rag`, `upload`                         | Vector ingest, multi-store backend, citations |
| Vision               | `vision`                                             | Image analysis, OCR, visual Q&A |
| Live interaction     | `live_interaction`                                   | Multimodal sessions (Live Studio) |
| Reminders            | `reminders`                                          | Per-user + per-resource, recurring, snooze, bulk |
| Automations          | `automations`                                        | Trigger → conditions → actions, event catalogue, runs, test-run |
| Calendar             | `calendar`                                           | Internal + Google Calendar sync, events from tasks |
| Integrations         | `integrations`, `integrations_v2`                    | 15 providers, scoped (org/team/project), OAuth start/callback, outbox |
| Org billing          | `org_billing`, `org_billing_extended`                | Stripe per-seat checkout/portal/webhook, credits, contracts, BYOK |
| Personal billing     | `billing`                                            | Individual plans, invoices, credits balance |
| Enterprise           | `enterprise`, `scim`                                 | SSO config, SAML ACS, SCIM 2.0, domains, JIT, residency, compliance |
| Search               | `search`                                             | Federated query, suggest, saved + recent |
| Media                | `media`                                              | Avatars + covers per scope, library, signed-URL, resize/crop |
| Onboarding           | `onboarding`                                         | Checklist, industry templates, provisioning |
| Permissions          | `permissions`                                        | Capability probe `/permissions/me` |
| Wellbeing            | `wellbeing`                                          | Mood prompts, focus sessions, coach state, weekly digest |
| Vertical agents      | `meeting`, `meeting_fact_checker`, `research`, `research_mentor`, `student`, `creative`, `customer_service` (+ templates / tickets / branding / articles / public portal), `translation`, `data_analysis`, `social_media`, `legal_document`, `learning_coach`, `knowledge_graph`, `ethics_bias`, `focus_flow`, `workspace_ergonomics` | 21 platform agents, full CRUD + analytics surface each |
| Blog                 | `blog`                                               | Public posts + AI generate + comments + analytics |
| Security             | `security`                                           | Login activity, active sessions, password change, revoke-all |
| Operations           | `ops`                                                | Deep health, queue depth, db/cache stats, per-org status |
| Device tokens        | `device_tokens`                                      | FCM push registration |

Browse interactively at `http://localhost:8000/docs` (Swagger) or `/redoc`.

---

## The 21 platform agents

Every customer on every plan gets the full roster for free; custom agents (Agent Studio) are project-bound.

1. Document Intelligence  2. Meeting Assistant  3. Meeting Fact-Checker  4. Vision  5. Research  6. Research Mentor  7. Student  8. Creative  9. Customer Service  10. Translation  11. Data Analysis  12. Social Media  13. Legal Document  14. Learning Coach  15. RAG (knowledge query)  16. Knowledge Graph  17. Ethics & Bias  18. Workspace Ergonomics  19. Focus Flow  20. Wellbeing  21. Live Studio (multimodal)

Each has CRUD, an analytics surface, full lineage (`agent_runs`, child runs, cost rollups), and emits structured A2A handoffs.

---

## Multi-agent orchestration

The shared orchestrator lives in `backend/agents/` and `backend/services/`. Highlights:

- **Structured output** — Gemini 2.5 returns typed objects validated at the boundary; cross-agent contracts are enforceable (`backend/agents/base_agent.py`).
- **Per-run cost ceilings** — every run records token in/out + USD cost; hard caps via `services/billing/plan_caps.py`.
- **Tool gateway (MCP)** — `services/mcp_gateway.py` mediates every external call with per-tenant scopes, audit, and a prompt-injection guard at the boundary.
- **A2A handoffs** — `agents/router.py` + `/agents-v2/handoffs/*` endpoints; the receiving agent gets a typed payload, not raw text.
- **Human-in-the-loop** — high-risk tools fire approval hooks; the `/comments-v2/reviews/*` flow surfaces them; any agent has a kill switch (`/agents-v2/runs/{id}/cancel`).
- **Evaluation harness** — `tests/` includes per-agent fixtures and a multi-agent grading layer for cross-agent scenarios.
- **Observability** — every step writes a span; `/ops/router-summary`, `/ops/queue-depth`, and the OpenTelemetry tags (`organization_id`, `team_id`, `project_id`, `agent_key`) feed Cloud Trace.

---

## Repository layout

```
backend/
├── main.py                      # FastAPI bootstrap, middleware, lifespan
├── api/v1/
│   ├── api.py                   # Router aggregator (24 routers)
│   ├── deps.py                  # JWT / Firebase auth + permission deps
│   └── endpoints/               # ~75 router files
├── agents/                      # 21 platform agents + shared orchestrator
│   ├── base_agent.py
│   ├── router.py                # AGENT_REGISTRY
│   └── <agent_key>.py
├── ai_models/                   # Provider clients (Gemini, Mistral, fallback)
├── core/
│   ├── config.py                # pydantic-settings (env + secrets)
│   └── security.py              # JWT, password hashing, Firebase admin
├── db/
│   ├── mongodb/
│   │   ├── mongodb.py           # async client + collection helpers
│   │   ├── base_repository.py
│   │   ├── repositories/        # one repository per collection
│   │   └── serializers.py       # ObjectId → str helpers
│   ├── postgres/                # SQLAlchemy + alembic (workflows + tasks dual-write)
│   ├── cassandra/               # Time-series telemetry (activity, wellbeing)
│   └── scoping.py               # @require_org tenant-scoping decorator
├── models/                      # pydantic v2 models (request/response + Mongo)
├── services/                    # Shared business logic
│   ├── permissions.py           # 44-action resolver
│   ├── plan_caps.py             # Quota pre-flight + 402 with upsell
│   ├── event_bus.py             # In-process + Redis pub/sub
│   ├── realtime.py              # Redis-backed WS fan-out
│   ├── presence_service.py      # /ws/presence broker
│   ├── notification_service.py
│   ├── notification_engine.py   # Rules + quiet-hours + dedup
│   ├── automation_engine.py     # Trigger → conditions → actions
│   ├── activity_logger.py
│   ├── email_service.py         # SendGrid + Resend failover
│   ├── push_notification_service.py
│   ├── storage_service.py       # Dual-write MinIO + R2/GCS
│   ├── context_service.py       # Vector retrieval w/ org+team+project filters
│   ├── saml_verifier.py         # SAML 2.0 ACS
│   └── mcp_gateway.py
├── tasks/
│   ├── celery_app.py            # 6 named beats
│   ├── document_tasks.py        # RAG ingest
│   ├── webhook_dispatcher.py    # Egress worker w/ retries
│   ├── agent_metrics_tasks.py   # Materialised metrics rebuild
│   └── automation_runner.py     # Scheduled + retry
├── load/                        # Locust smoke test
└── tests/                       # 192 passing (permissions matrix, plan caps,
                                 # tenant isolation, SAML, SCIM, billing)
```

---

## Local development

### Prerequisites

- **Python 3.11+** (3.14 also tested)
- **Docker Desktop** (easiest path)
- A **Firebase service-account JSON** (`firebase-credentials.json`) placed at `backend/firebase-credentials.json`
- A **Gemini API key** from <https://aistudio.google.com/apikey>

### Option A — single Docker command (recommended)

```bash
cp .env.example .env             # fill in GEMINI_API_KEY, Firebase, etc.
cd docker
docker compose up -d --build
```

Everything (backend, worker, beat, frontend, Mongo, Redis, MinIO, Weaviate, Postgres, Prometheus, Grafana) starts in one shot. Visit <http://localhost:8000/docs> and <http://localhost:8080/>.

### Option B — three terminals (closer to how the orchestrator team works)

Run dependency services in Docker but the API + worker on your host so reloads are instant.

```bash
# Terminal 1 — infra
cd docker
docker compose up -d mongodb redis weaviate minio postgres

# Terminal 2 — backend API
python -m venv lumicoria_ai_venv
source lumicoria_ai_venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env             # then edit
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3 — Celery worker + beat (one command via honcho, or two terminals)
celery -A backend.tasks.celery_app worker --loglevel=info --concurrency=4
# in a fourth terminal:
celery -A backend.tasks.celery_app beat --loglevel=info
```

Frontend (separately):
```bash
cd Lumicoria-frontend
npm install
npm run dev          # http://localhost:8080
```

### Useful commands

```bash
# Type-check
cd Lumicoria-frontend && npx tsc --noEmit

# Run backend tests
lumicoria_ai_venv/bin/python -m pytest backend/tests/ -q

# Boot inspection — total route count
lumicoria_ai_venv/bin/python -c "
from backend.api.v1.api import api_router
print(len(api_router.routes), 'routes')
"

# Reset all data (DESTRUCTIVE)
cd docker && docker compose down -v

# Migrate legacy projects → projects v2
python scripts/migrate_projects_to_v2.py --dry-run
python scripts/migrate_projects_to_v2.py
```

---

## Environment variables

`.env.example` documents every value with comments. The required minimum:

| Key                              | What it's for                              |
|----------------------------------|--------------------------------------------|
| `SECRET_KEY`                     | JWT signing — `openssl rand -hex 32`       |
| `ENVIRONMENT`                    | `development` / `production`               |
| `GEMINI_API_KEY`                 | Gemini 2.5 Pro + Flash                     |
| `FIREBASE_CREDENTIALS_PATH`      | Path to Firebase service-account JSON      |
| `MONGODB_URL`                    | `mongodb://localhost:27017` or Atlas URI   |
| `REDIS_HOST` / `REDIS_PORT`      | Realtime broker + cache                    |
| `MINIO_*`                        | Object storage (S3-compatible)             |
| `BACKEND_CORS_ORIGINS`           | JSON array of allowed origins              |
| `VITE_*`                         | Frontend build-time config                 |

Optional surfaces light up automatically when their keys are present:
- `STRIPE_*` → billing pages start to function
- `SENDGRID_API_KEY` / `RESEND_API_KEY` → email sends
- `VITE_FIREBASE_VAPID_KEY` → web push
- `CASSANDRA_ENABLED=true` → time-series telemetry dual-write
- `POSTGRES_ENABLED=true` → workflow + dual-write

---

## Background workers

`backend/tasks/celery_app.py` registers **six named beats**:

| Beat                          | Cadence | What it does                                            |
|-------------------------------|---------|---------------------------------------------------------|
| `webhooks.deliver_due`        | 60 s    | Drains pending `webhook_deliveries` with exp. back-off  |
| `agent_metrics.materialise`   | 10 min  | Rebuilds `agent_metrics` from `agent_runs`              |
| `automations.tick_scheduled`  | 60 s    | Cron-trigger ticker for automations                     |
| `automations.retry_errored`   | 5 min   | Exponential retry for errored automation runs           |
| `wellbeing.daily_digest`      | 1 day   | Per-org wellbeing weekly digest                         |
| `tasks.due_reminders`         | 5 min   | Sends task-due reminders via the notification engine    |

Scale workers with `docker compose up -d --scale celery-worker=4`. **Never scale `celery-beat`** — exactly one instance.

---

## Realtime + WebSockets

| Endpoint                              | Purpose                                |
|---------------------------------------|----------------------------------------|
| `WS /api/v1/ws/notifications/{user}`  | Per-user notification fan-out          |
| `WS /api/v1/ws/presence`              | Org-scoped presence + typing per room  |
| `WS /api/v1/ws/transcribe/{user}`     | Streaming speech-to-text (faster-whisper) |
| `WS /api/v1/chat/ws/{channel_id}`     | Channel chat (mentions, agent slash commands) |

Cross-worker fan-out goes through Redis pub/sub (`services/realtime.py`). Topics: `org:{id}`, `team:{id}`, `project:{id}`, `user:{id}`.

---

## Authentication

- **JWT** is the canonical bearer token used by every endpoint (see `core/security.py`).
- **Firebase Authentication** handles email/password, Google OAuth, and the password reset flow. The `auth/google` endpoint exchanges a Firebase ID token for a Lumicoria JWT.
- **SAML 2.0** for enterprise SSO. SP metadata: `GET /api/v1/enterprise/sso/metadata.xml`. ACS: `POST /api/v1/enterprise/sso/saml/acs`. Per-org IdP config at `enterprise/{org_id}/sso`.
- **SCIM 2.0** for enterprise provisioning. Token-auth via `scim_tokens`. Endpoints at `/api/v1/scim/v2/*` (ServiceProviderConfig, ResourceTypes, Schemas, Users CRUD + PATCH, Groups).

---

## Billing

- **User-scoped subscriptions** (Free / Starter / Professional) live in `subscriptions` keyed by `user_id`.
- **Org-scoped subscriptions** (Team / Business / Enterprise) live in `org_subscriptions` and drive per-seat pricing ($39 / $79 / $129 floor + 15% annual). Stripe handles checkout, customer portal, and webhooks (`/api/v1/org-billing/webhook`).
- **Plan caps** are enforced at member-add, invite-accept, agent-run, project-create, and document-upload. Cap violations return a structured `402 Payment Required` with an `upgrade_suggested` payload.
- **Credits ledger** (`/api/v1/org-billing/{org}/credits/*`) is a prepaid balance used for premium agent runs.

---

## Storage + RAG

- **Object storage**: `services/storage_service.py` dual-writes to **MinIO** (primary, S3-compatible) and **Cloudflare R2** / **GCS** (best-effort secondary). Local dev uses the MinIO container; production typically points the primary at GCS.
- **Vector store**: Weaviate by default; flip `VECTOR_STORE_TYPE` to `qdrant` or `chroma` to use those. **Vertex AI Search** is configured as the grounded retrieval layer in production.
- **RAG**: `services/context_service.py` retrieves with org/team/project filters; **Gemini 2.5** is the answerer with citations.
- **PDF / Word ingest** uses **PyMuPDF**; ingest jobs run on Celery (`tasks/document_tasks.py`) and emit per-doc progress events over WS.

---

## Observability

- Prometheus metrics at `GET /metrics` (Docker compose ships Prometheus on `:9090` + Grafana on `:3001` with provisioned dashboards in `docker/grafana/`).
- OpenTelemetry tracing via `backend/main.py`; spans are tagged `organization_id`, `team_id`, `project_id`, `agent_key` — these flow through to **Cloud Trace** in production.
- Structured JSON logs via `structlog` — every request emits an `http_request` event with `path`, `status_code`, and timing.
- Activity audit in the `activity_logs` collection (180+ event types). Export to CSV via `/api/v1/activity/me/audit/export`. SIEM forwarders (Splunk / Datadog HEC) configured via `enterprise/{org_id}/audit/siem`.

---

## Testing

```bash
lumicoria_ai_venv/bin/python -m pytest backend/tests/ -q
```

192 passing. Coverage focuses on:

- **Permissions matrix** — `test_permissions.py` (every org × team × project × plan × action combination)
- **Tenant isolation** — `test_tenant_isolation.py` + `test_per_endpoint_isolation.py` (dynamic walk that asserts every org-scoped repo method either carries `@require_org` or references `organization_id` in-body)
- **Plan caps** — `test_plan_caps.py` (seat enforcement, custom-agent quota, 402 payloads)
- **SAML** — `test_saml_verifier.py` (Okta + Azure AD response shapes)
- **SCIM** — `test_scim_filter_parser.py` (Okta + Azure canonical filters)
- **Billing** — `test_billing.py` + integration suites (Stripe webhook idempotency)

A Locust smoke load test ships at `backend/load/locustfile.py` — see its README.

---

## Deploying to production

Three paths, in increasing order of operational sophistication.

1. **Single VPS (recommended for demos + early customers)** — Google Cloud Compute Engine, Ubuntu 22.04, e2-standard-4. Docker Compose runs every service. **See [`DEPLOY_GCP_VPS.md`](../DEPLOY_GCP_VPS.md) at the repo root for the step-by-step guide.**
2. **Cloud Run + MongoDB Atlas + Memorystore Redis** — the production target. The `Dockerfile` in this repo is the exact image Cloud Run runs. Push to Artifact Registry, attach Cloud SQL for Postgres if you enable workflow dual-write, MongoDB Atlas (M10+) for primary, Memorystore for Redis. Celery worker + beat each get their own Cloud Run Job. Vertex AI Search becomes the grounded retrieval layer.
3. **GKE** — for customers with on-prem connectivity or VPC-SC requirements. Manifests live in `k8s/`.

In every path, the backend image is identical. Environment variables and the Firebase service-account JSON are injected at deploy time.

---

## License

Proprietary — © 2026 Lumicoria AI. All rights reserved. See [LICENSE](../LICENSE) for the formal terms.
