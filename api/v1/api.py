from fastapi import APIRouter
from backend.api.v1.endpoints import (
    auth,
    users,
    agents,
    documents,
    live_interaction,
    tasks,
    activity,
    integrations,
    notifications,
    vision,
    student,
    creative,
    research,
    projects,
    meeting,
    wellbeing,
    customer_service,
    customer_service_tickets,
    customer_service_templates,
    customer_service_branding,
    customer_service_articles,
    customer_service_public,
    translation,
    data_analysis,
    social_media,
    legal_document,
    learning_coach,
    rag,
    lumicoria_chat,
    agent_studio,
    workflow_execution,
    onboarding,
    websocket,
    transcribe_websocket,
    device_tokens,
    billing,
    security,
    upload,
    blog,
    calendar,  # Phase 2: Lumicoria-native Calendar
    invites,   # Phase 5: invite module
    organizations,  # Phase 8: Organizations REST API
    analytics,      # Phase 9: Dashboard analytics
    teams,          # Phase A1: Workspace > Teams REST API
    projects_v2,    # Phase A2: org-scoped Projects v2
    org_billing,    # Phase A3: per-seat org billing
    comments,       # Phase A4: cross-resource comments
    tasks_extended, # Phase A4: tasks bulk + watchers + dependencies + saved-views
    agents_v2,      # Phase B: agents collaboration surface (metrics, lineage, schedules, handoffs)
    analytics_v2,   # Phase D: per-level analytics + audit export
    reminders,           # Phase C: cross-resource reminders
    automations,         # Phase C: rules engine REST surface
    notification_rules,  # Phase C: per-user notification prefs + quiet hours
    enterprise,          # Phase E: API tokens + webhooks + SSO + domains + session policy
    scim,                # Phase E: SCIM 2.0 user/group provisioning
    chat as chat_module,         # Phase C: chat channels + WS + slash command
    analytics_v2_extras,         # Phase D: analytics depth (retention/funnel/cohorts/exports)
    search as search_module,     # Phase A: federated search
    media as media_module,       # Phase A: avatars/covers + library + signed URLs
    ops as ops_module,           # Phase A: health/queue/db/cache/per-org status
    agents_v2_extras,            # Phase B: custom agents + batches + presets + KB + feedback
    notifications_extras,        # Phase C: rules + devices + topics + broadcast + subs
    emails,                      # Phase C: templates + branding + sending domains + DKIM
    comments_extras,             # Phase C: reviews + shares + counts + watch
    integrations_v2,             # Phase A: 15 provider connectors per scope
    workspace,                   # Phase A: workspace navigation surface
    organizations_extended,      # Phase A: org profile/branding/limits/admins/tags/announcements
    teams_extended,              # Phase A: team CSV import + integrations + reminders + analytics
    projects_v2_extended,        # Phase A: project task views + templates + KB + analytics + sharing
    invites_extended,            # Phase A: bulk + CSV + Google Workspace + shareable links
    tasks_v2_extras,             # Phase A: subtasks + history + templates + snooze + imports
    org_billing_extended,        # Phase A: credits + promos + contracts + BYOK + forecasts
)
from backend.api.routers.research_mentor import router as research_mentor_router
from backend.api.routers.ethics_bias_router import router as ethics_bias_router
from backend.api.routers.knowledge_graph_router import router as knowledge_graph_router
from backend.api.routers.workspace_ergonomics_router import router as workspace_ergonomics_router
from backend.api.routers.focus_flow_router import router as focus_flow_router
from backend.api.routers.meeting_fact_checker_router import router as meeting_fact_checker_router

api_router = APIRouter()

# Authentication endpoints
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# User management endpoints
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Agent endpoints
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])

# Document endpoints
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])

# Wellbeing endpoints
api_router.include_router(wellbeing.router, prefix="/wellbeing", tags=["wellbeing"])

# Live Interaction endpoints
api_router.include_router(live_interaction.router, prefix="/live", tags=["live interaction"])

# Task endpoints
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])

# Calendar endpoints (Lumicoria-native; Google mirror is opt-in)
api_router.include_router(calendar.router, prefix="/calendar", tags=["calendar"])

# Invite endpoints (Phase 5 — invite-to-task / invite-to-org collaboration)
api_router.include_router(invites.router, prefix="/invites", tags=["invites"])

# Organization endpoints (Phase 8 — settings + members + invites)
api_router.include_router(organizations.router, prefix="/organizations", tags=["organizations"])

# Teams (Phase A1 — Workspace > Teams)
api_router.include_router(
    teams.router,
    prefix="/organizations/{org_id}/teams",
    tags=["teams"],
)

# Projects v2 (Phase A2 — org-scoped projects with normalised tasks)
api_router.include_router(
    projects_v2.router,
    prefix="/organizations/{org_id}/projects",
    tags=["projects-v2"],
)

# Org-scoped per-seat billing (Phase A3)
api_router.include_router(
    org_billing.router,
    prefix="/org-billing",
    tags=["org-billing"],
)

# Cross-resource comments (Phase A4 / C)
api_router.include_router(
    comments.router,
    prefix="/comments",
    tags=["comments"],
)

# Tasks extended surface (Phase A4)
api_router.include_router(
    tasks_extended.router,
    prefix="/tasks-extended",
    tags=["tasks-extended"],
)

# Agents v2 collaboration surface (Phase B)
api_router.include_router(
    agents_v2.router,
    prefix="/agents-v2",
    tags=["agents-v2"],
)

# Analytics v2 — per-level dashboards (Phase D)
api_router.include_router(
    analytics_v2.router,
    prefix="/analytics-v2",
    tags=["analytics-v2"],
)

# Reminders (Phase C)
api_router.include_router(
    reminders.router,
    prefix="/reminders",
    tags=["reminders"],
)

# Automations / rules engine (Phase C)
api_router.include_router(
    automations.router,
    prefix="/automations",
    tags=["automations"],
)

# Notification rules + preferences (Phase C)
api_router.include_router(
    notification_rules.router,
    prefix="/notification-rules",
    tags=["notification-rules"],
)

# Enterprise governance (Phase E)
api_router.include_router(
    enterprise.router,
    prefix="/enterprise",
    tags=["enterprise"],
)

# SCIM 2.0 (Phase E)
api_router.include_router(
    scim.router,
    prefix="/scim/v2",
    tags=["scim"],
)

# Dashboard analytics (Phase 9 — single payload for the Dashboard surface)
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])

# Activity endpoints
api_router.include_router(activity.router, prefix="/activity", tags=["activity"])

# Permissions probe (frontend capability gating)
from backend.api.v1.endpoints import permissions as _permissions  # noqa: E402
api_router.include_router(_permissions.router, prefix="/permissions", tags=["permissions"])

# Integrations endpoints
api_router.include_router(integrations.router, prefix="/integrations", tags=["integrations"])

# Notifications endpoints
api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["notifications"]
)

# WebSocket endpoints for real-time notifications
api_router.include_router(
    websocket.router,
    prefix="/ws",
    tags=["websocket"]
)

# WebSocket endpoints for real-time speech-to-text transcription
api_router.include_router(
    transcribe_websocket.router,
    prefix="/ws",
    tags=["websocket", "transcription"]
)

# Device token endpoints for push notifications
api_router.include_router(
    device_tokens.router,
    prefix="/device-tokens",
    tags=["device-tokens"]
)

# Vision endpoints
api_router.include_router(
    vision.router,
    prefix="/vision",
    tags=["vision"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Onboarding endpoints
api_router.include_router(
    onboarding.router,
    prefix="/onboarding",
    tags=["onboarding"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Meeting endpoints
api_router.include_router(
    meeting.router,
    prefix="/meeting",
    tags=["meeting"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# RAG endpoints
api_router.include_router(
    rag.router,
    prefix="/rag",
    tags=["retrieval augmented generation"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Project management endpoints
api_router.include_router(
    projects.router,
    prefix="/projects",
    tags=["projects"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Research endpoints
api_router.include_router(
    research.router,
    prefix="/research",
    tags=["research"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Student endpoints
api_router.include_router(
    student.router,
    prefix="/student",
    tags=["student"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Creative endpoints
api_router.include_router(
    creative.router,
    prefix="/creative",
    tags=["creative"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Customer Service endpoints
api_router.include_router(
    customer_service.router,
    prefix="/customer-service",
    tags=["customer service"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Customer Service: Tickets (operator-facing, auth required)
api_router.include_router(
    customer_service_tickets.router,
    prefix="/customer-service/tickets",
    tags=["customer service - tickets"],
)

# Customer Service: Templates (operator-facing, auth required)
api_router.include_router(
    customer_service_templates.router,
    prefix="/customer-service/templates",
    tags=["customer service - templates"],
)

# Customer Service: Branding admin (auth required)
api_router.include_router(
    customer_service_branding.router,
    prefix="/customer-service/branding",
    tags=["customer service - branding"],
)

# Customer Service: Help-center article admin (auth required)
api_router.include_router(
    customer_service_articles.router,
    prefix="/customer-service/articles",
    tags=["customer service - articles"],
)

# Customer Service: Public portal endpoints (no auth, rate-limited)
api_router.include_router(
    customer_service_public.router,
    prefix="/public",
    tags=["customer service - public"],
)

# Translation endpoints
api_router.include_router(
    translation.router,
    prefix="/translation",
    tags=["translation"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Data Analysis endpoints
api_router.include_router(
    data_analysis.router,
    prefix="/data-analysis",
    tags=["data analysis"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Social Media endpoints
api_router.include_router(
    social_media.router,
    prefix="/social-media",
    tags=["social media"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Legal Document endpoints
api_router.include_router(
    legal_document.router,
    prefix="/legal",
    tags=["legal documents"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Learning Coach endpoints
api_router.include_router(
    learning_coach.router,
    prefix="/learning-coach",
    tags=["learning coach"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Lumicoria Chat endpoints - the main assistant interface
api_router.include_router(
    lumicoria_chat.router,
    prefix="/chat",
    tags=["chat"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Agent Studio endpoints
api_router.include_router(
    agent_studio.router,
    prefix="/agent-studio",
    tags=["agent studio"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Workflow Execution endpoints
api_router.include_router(
    workflow_execution.router,
    prefix="/workflows",
    tags=["workflow execution"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Security endpoints
api_router.include_router(
    security.router,
    prefix="/security",
    tags=["security"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        500: {"description": "Internal Server Error"},
    }
)

# Billing & Subscription endpoints
api_router.include_router(
    billing.router,
    prefix="/billing",
    tags=["billing"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        402: {"description": "Payment Required"},
        403: {"description": "Forbidden"},
        429: {"description": "Usage Limit Exceeded"},
        500: {"description": "Internal Server Error"},
        502: {"description": "Payment Gateway Error"},
    }
)

# Research Mentor endpoints
api_router.include_router(
    research_mentor_router,
    prefix="/research-mentor",
    tags=["research mentor"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Ethics & Bias endpoints
api_router.include_router(
    ethics_bias_router,
    prefix="/ethics-bias",
    tags=["ethics bias"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Knowledge Graph endpoints
api_router.include_router(
    knowledge_graph_router,
    prefix="/knowledge-graph",
    tags=["knowledge graph"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Workspace Ergonomics endpoints
api_router.include_router(
    workspace_ergonomics_router,
    prefix="/workspace-ergonomics",
    tags=["workspace ergonomics"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Focus Flow endpoints
api_router.include_router(
    focus_flow_router,
    prefix="/focus-flow",
    tags=["focus flow"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Upload endpoints
api_router.include_router(
    upload.router,
    prefix="/upload",
    tags=["upload"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        413: {"description": "File Too Large"},
        500: {"description": "Internal Server Error"}
    }
)

# Blog endpoints
api_router.include_router(
    blog.router,
    prefix="/blog",
    tags=["blog"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# Meeting Fact Checker endpoints
api_router.include_router(
    meeting_fact_checker_router,
    prefix="/meeting-fact-checker",
    tags=["meeting fact checker"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# API Documentation
"""
API Endpoints for Frontend Integration

Authentication Endpoints (/api/v1/auth):
- POST /login - Login with email and password
  Request Body: { email: string, password: string }
  Response: { access_token: string, token_type: string, user: User }

- POST /signup - Create new user account
  Request Body: { email: string, password: string, full_name: string }
  Response: { access_token: string, token_type: string, user: User }

- POST /google - Sign in with Google
  Request Body: { id_token: string }
  Response: { access_token: string, token_type: string, user: User, refresh_token: string }

- POST /refresh - Refresh access token
  Headers: Authorization: Bearer <refresh_token>
  Response: { access_token: string, token_type: string, user: User }

- POST /logout - Logout user
  Headers: Authorization: Bearer <access_token>
  Response: { message: string }

- GET /me - Get current user info
  Headers: Authorization: Bearer <access_token>
  Response: User object

User Management Endpoints (/api/v1/users):
- GET /me - Get current user profile
  Headers: Authorization: Bearer <access_token>
  Response: User object

- PUT /me - Update current user
  Headers: Authorization: Bearer <access_token>
  Request Body: { full_name?: string, job_title?: string, company?: string, timezone?: string, preferred_language?: string }
  Response: Updated User object

- GET /me/profile - Get user profile
  Headers: Authorization: Bearer <access_token>
  Response: UserProfile object

- PUT /me/profile - Update user profile
  Headers: Authorization: Bearer <access_token>
  Request Body: { job_title?: string, company?: string, timezone?: string, preferred_language?: string }
  Response: Updated UserProfile object

- GET /me/settings - Get user settings
  Headers: Authorization: Bearer <access_token>
  Response: UserSettings object

- PUT /me/settings - Update user settings
  Headers: Authorization: Bearer <access_token>
  Request Body: {
    email_notifications?: boolean,
    push_notifications?: boolean,
    task_reminders?: boolean,
    break_reminders?: boolean,
    work_hours_start?: string,
    work_hours_end?: string,
    break_interval_minutes?: number,
    break_duration_minutes?: number,
    preferred_ai_model?: string
  }
  Response: Updated UserSettings object

- POST /me/avatar - Upload user avatar
  Headers: 
    - Authorization: Bearer <access_token>
    - Content-Type: multipart/form-data
  Request Body: file: File (image)
  Response: Updated User object with avatar_url

Common Response Types:
User: {
  id: string
  email: string
  full_name: string
  avatar_url?: string
  is_active: boolean
  created_at: string
  updated_at?: string
}

UserProfile: {
  id: string
  user_id: string
  job_title?: string
  company?: string
  timezone: string
  preferred_language: string
  created_at: string
  updated_at?: string
}

UserSettings: {
  id: string
  user_id: string
  email_notifications: boolean
  push_notifications: boolean
  task_reminders: boolean
  break_reminders: boolean
  work_hours_start: string
  work_hours_end: string
  break_interval_minutes: number
  break_duration_minutes: number
  preferred_ai_model: string
  created_at: string
  updated_at?: string
}

Error Response:
{
  "detail": string | object[]
}
"""

# ────────────────────────────────────────────────────────────────────
# Phase A — Workspace + extended surfaces
# ────────────────────────────────────────────────────────────────────

api_router.include_router(
    workspace.router,
    prefix="/workspaces",
    tags=["workspace"],
)

api_router.include_router(
    chat_module.router,
    prefix="/chat-v2",
    tags=["chat-v2"],
)

api_router.include_router(
    analytics_v2_extras.router,
    prefix="/analytics-v2",
    tags=["analytics-v2-extras"],
)

api_router.include_router(
    search_module.router,
    prefix="/search",
    tags=["search"],
)

api_router.include_router(
    media_module.router,
    prefix="/media",
    tags=["media"],
)

api_router.include_router(
    ops_module.router,
    prefix="/ops",
    tags=["ops"],
)

api_router.include_router(
    agents_v2_extras.router,
    prefix="/agents-v2",
    tags=["agents-v2-extras"],
)

api_router.include_router(
    notifications_extras.router,
    prefix="/notifications-v2",
    tags=["notifications-v2"],
)

api_router.include_router(
    emails.router,
    prefix="/emails",
    tags=["emails"],
)

api_router.include_router(
    comments_extras.router,
    prefix="/comments-v2",
    tags=["comments-v2"],
)

api_router.include_router(
    integrations_v2.router,
    prefix="/integrations-v2",
    tags=["integrations-v2"],
)

api_router.include_router(
    organizations_extended.router,
    prefix="/organizations",
    tags=["organizations-extended"],
)

api_router.include_router(
    teams_extended.router,
    prefix="/organizations/{org_id}/teams",
    tags=["teams-extended"],
)

api_router.include_router(
    projects_v2_extended.router,
    prefix="/organizations/{org_id}/projects",
    tags=["projects-v2-extended"],
)

api_router.include_router(
    invites_extended.router,
    prefix="/invites",
    tags=["invites-extended"],
)

api_router.include_router(
    tasks_v2_extras.router,
    prefix="/tasks-v2",
    tags=["tasks-v2-extras"],
)

api_router.include_router(
    org_billing_extended.router,
    prefix="/org-billing",
    tags=["org-billing-extended"],
)

# Add response codes to all routers
for route in api_router.routes:
    if hasattr(route, "responses"):
        route.responses.update({
            200: {"description": "Success"},
            400: {"description": "Bad Request"},
            401: {"description": "Unauthorized"},
            403: {"description": "Forbidden"},
            404: {"description": "Not Found"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"}
        }) 