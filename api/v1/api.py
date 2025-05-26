from fastapi import APIRouter

from api.v1.endpoints import auth, users, agents, documents, wellbeing, live_interaction, tasks, activity, integrations

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
api_router.include_router(live_interaction.router, prefix="/live-interaction", tags=["live interaction"])

# Task endpoints
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])

# Activity endpoints
api_router.include_router(activity.router, prefix="/activity", tags=["activity"])

# Integrations endpoints
api_router.include_router(integrations.router, prefix="/integrations", tags=["integrations"])

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