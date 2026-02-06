from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Body
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from backend.services.live_interaction_service import live_interaction_service
from backend.models.user import User

router = APIRouter()

class LiveInteractionSession(BaseModel):
    id: str
    user_id: str
    organization_id: str
    start_time: datetime
    end_time: Optional[datetime]
    interaction_mode: str # e.g., 'document', 'voice', 'sketch'
    active_agents: List[str] # List of agent IDs involved
    status: str # e.g., 'active', 'completed', 'error'
    metadata: Optional[Dict[str, Any]]

class LiveInteractionData(BaseModel):
    session_id: str
    data_type: str # e.g., 'image', 'audio', 'sketch_data'
    timestamp: datetime
    content: str # Base64 encoded data or text content
    metadata: Optional[Dict[str, Any]]

class LiveSessionCreate(BaseModel):
    interaction_mode: str
    active_agent_ids: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

@router.post("/sessions", response_model=LiveInteractionSession)
async def start_live_session(
    session_data: LiveSessionCreate = Body(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Start a new live interaction session.
    """    # Here we would create a new session in the database
    # For now, we'll return a dummy response
    session_id = "dummy_session_id_" + str(datetime.now().timestamp())
    return LiveInteractionSession(
        id=session_id,
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        start_time=datetime.now(),
        interaction_mode=session_data.interaction_mode,
        active_agents=session_data.active_agent_ids,
        status="active",
        metadata=session_data.metadata
    )

@router.post("/sessions/{session_id}/data")
async def send_live_data(
    session_id: str,
    data_in: LiveInteractionData,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Send live interaction data (image chunk, audio snippet, sketch data).
    """
    # Here we would process the incoming data, potentially trigger agent execution
    # and store relevant parts
    # print(f"Received data for session {session_id}: {data_in.data_type}")

    # Delegate processing to the LiveInteractionService
    processed_result = await live_interaction_service.process_live_data(
        session_id=session_id,
        organization_id=current_user.organization_id,
        user_id=current_user.id,
        data_type=data_in.data_type,
        content=data_in.content,
        metadata=data_in.metadata
    )

    return processed_result

@router.post("/sessions/{session_id}/end")
async def end_live_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    End a live interaction session.
    """
    # Here we would update the session status to 'completed' or 'ended'
    # and perform any final processing
    print(f"Ending session {session_id}")

    # Dummy response
    return {"status": "session ended", "session_id": session_id}

@router.get("/sessions/{session_id}", response_model=LiveInteractionSession)
async def get_live_session_status(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get the status of a live interaction session.
    """
    # Here we would retrieve the session details from the database
    # For now, returning a dummy active session
    return LiveInteractionSession(
        id=session_id,
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        start_time=datetime.now(),
        interaction_mode="dummy_mode",
        active_agents=[],
        status="active"
    ) 