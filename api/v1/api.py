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
    translation,
    data_analysis,
    social_media,
    legal_document,
    learning_coach,
    rag,
    lumicoria_chat,
    agent_studio,
    workflow_execution,
    onboarding
)

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

# Activity endpoints
api_router.include_router(activity.router, prefix="/activity", tags=["activity"])

# Integrations endpoints
api_router.include_router(integrations.router, prefix="/integrations", tags=["integrations"])

# Notifications endpoints
api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["notifications"]
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