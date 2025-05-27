from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
from .routers import (
    auth_router,
    users_router,
    agents_router,
    focus_flow_router,
    meeting_fact_checker_router,
    slack_router
)
from ..agents.agent_service import agent_service
from ..services.integration_service import integration_service

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Lumicoria.ai API",
    description="""
    Lumicoria.ai API provides intelligent agents and integrations to enhance productivity and well-being:
    
    - Focus & Flow Guardian: Monitors focus states, tracks distractions, and provides personalized productivity recommendations
    - Meeting Fact Checker: Verifies claims and statements during meetings in real-time
    - Document Agent: Processes and analyzes documents
    - Wellbeing Agent: Monitors and supports user well-being
    - Meeting Agent: Manages and optimizes meetings
    - Customer Service Agent: Handles customer interactions
    - Studio Agent: Manages creative workflows
    
    Integrations:
    - Notion: Project and task management
    - Google Workspace: Calendar, documents, and email
    - Slack: Team communication and collaboration
    """,
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(agents_router.router)
app.include_router(focus_flow_router.router)
app.include_router(meeting_fact_checker_router.router)
app.include_router(slack_router.router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        # Initialize agent service
        await agent_service.initialize()
        logger.info("Agent service initialized")
        
        # Initialize integration service
        integration_service._initialize_integrations()
        logger.info("Integration service initialized")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/")
async def root():
    """Get API information."""
    return {
        "name": "Lumicoria.ai API",
        "version": "1.0.0",
        "description": "Intelligent agents and integrations for enhanced productivity and well-being",
        "endpoints": {
            "auth": "/auth",
            "users": "/users",
            "agents": "/agents",
            "focus-flow": "/focus-flow",
            "meeting-fact-checker": "/meeting-fact-checker",
            "slack": "/slack"
        }
    }

@app.get("/health")
async def health_check():
    """Check API health."""
    try:
        # Check agent service
        agent_status = "initialized" if agent_service.is_initialized else "not initialized"
        
        # Check integration service
        integrations = integration_service.get_available_integrations()
        integration_status = {
            name: "available" if info["available"] else "not available"
            for name, info in integrations.items()
        }
        
        return {
            "status": "healthy",
            "agent_service": agent_status,
            "integrations": integration_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 