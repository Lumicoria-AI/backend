from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field
import structlog
import asyncio
from datetime import datetime
import uuid

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.agents.studio_service import StudioService
from backend.agents.orchestration import WorkflowOrchestrator, WorkflowStatus
from backend.agents.security import AgentSecurityContext, AgentPermission

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()

# Initialize the StudioService and orchestrator - in production this should be handled via dependency injection
def get_studio_service() -> StudioService:
    # This is simplified for example purposes
    from agents.factory import AgentFactory
    from agents.security import AgentSecurityManager
    
    agent_factory = AgentFactory()
    security_manager = AgentSecurityManager(jwt_secret="temp-secret-for-dev")  # In production, load from environment
    return StudioService(agent_factory, security_manager)

def get_orchestrator() -> WorkflowOrchestrator:
    return WorkflowOrchestrator()

# Active workflow executions, keyed by execution_id
active_executions: Dict[str, WorkflowOrchestrator] = {}
_execution_lock = asyncio.Lock()

# Pydantic models for request/response
class ExecuteWorkflowRequest(BaseModel):
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the workflow")

class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

class ExecutionStatusResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    node_statuses: Dict[str, Dict[str, Any]]

@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: ExecuteWorkflowRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service),
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
) -> Any:
    """
    Execute a workflow with the provided input data.
    """
    try:
        workflow_id = request.workflow_id
        
        # Check if the workflow exists
        if workflow_id not in studio_service._workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
        workflow = studio_service._workflows[workflow_id]
        
        # Set up security context
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.EXECUTE]
        )
        
        # Check if user has permission to access this workflow
        if (not workflow.is_public and 
            workflow.created_by != security_context.user_id and 
            AgentPermission.ADMIN not in security_context.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to execute this workflow"
            )
        
        # Load the workflow into the orchestrator
        await orchestrator.load_workflow(workflow)
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Store in active executions
        async with _execution_lock:
            active_executions[execution_id] = orchestrator
        
        # Execute the workflow in a background task to avoid blocking
        asyncio.create_task(
            execute_workflow_background(
                execution_id=execution_id,
                orchestrator=orchestrator,
                input_data=request.input_data
            )
        )
        
        # Return immediate response
        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error executing workflow", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute workflow: {str(e)}"
        )

async def execute_workflow_background(
    execution_id: str, 
    orchestrator: WorkflowOrchestrator,
    input_data: Dict[str, Any]
) -> None:
    """Background task to execute a workflow"""
    try:
        # Execute the workflow
        results = await orchestrator.execute_workflow(input_data)
        
        # Store results in orchestrator's execution status
        logger.info(
            "workflow_execution_completed_background",
            execution_id=execution_id,
            workflow_id=orchestrator.workflow_id
        )
        
    except Exception as e:
        logger.error(
            "workflow_execution_failed_background",
            execution_id=execution_id,
            workflow_id=orchestrator.workflow_id,
            error=str(e)
        )
        
        # Remove orchestrator from active executions after a delay
        asyncio.create_task(remove_execution_after_delay(execution_id, delay_seconds=3600))  # Remove after 1 hour

async def remove_execution_after_delay(execution_id: str, delay_seconds: int) -> None:
    """Remove an execution from active_executions after a delay"""
    await asyncio.sleep(delay_seconds)
    
    async with _execution_lock:
        if execution_id in active_executions:
            del active_executions[execution_id]
            
            logger.info(
                "removed_expired_execution",
                execution_id=execution_id
            )

@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_execution_results(
    execution_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get the results of a workflow execution.
    """
    try:
        async with _execution_lock:
            if execution_id not in active_executions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Execution not found: {execution_id}"
                )
                
            orchestrator = active_executions[execution_id]
        
        # Get execution status
        execution_status = orchestrator.get_execution_status()
        
        # Build response
        response = WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id=orchestrator.workflow_id,
            status=execution_status["status"],
            started_at=execution_status["start_time"] or datetime.utcnow(),
            completed_at=execution_status["end_time"],
            results=None  # Will be filled if completed
        )
        
        # If workflow completed, include results
        if execution_status["status"] == WorkflowStatus.COMPLETED:
            # Collect results from output nodes
            results = {}
            for node_id, node_status in execution_status["nodes"].items():
                if node_status.get("has_result") and "is_output_node" in node_status and node_status["is_output_node"]:
                    # In a real implementation, you'd get the actual results from the node
                    # Here we're just including placeholder data
                    orchestrator_node = orchestrator.nodes.get(node_id)
                    if orchestrator_node and orchestrator_node.result:
                        results[node_id] = orchestrator_node.result.data
            
            response.results = results
            
            # Schedule removal after a delay (e.g., 1 hour)
            asyncio.create_task(remove_execution_after_delay(execution_id, delay_seconds=3600))
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting execution results", error=str(e), execution_id=execution_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution results: {str(e)}"
        )

@router.get("/executions/{execution_id}/status", response_model=ExecutionStatusResponse)
async def get_execution_status(
    execution_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get the detailed status of a workflow execution.
    """
    try:
        async with _execution_lock:
            if execution_id not in active_executions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Execution not found: {execution_id}"
                )
                
            orchestrator = active_executions[execution_id]
        
        # Get execution status
        execution_status = orchestrator.get_execution_status()
        
        # Build response
        return ExecutionStatusResponse(
            execution_id=execution_id,
            workflow_id=orchestrator.workflow_id,
            status=execution_status["status"],
            started_at=execution_status["start_time"] or datetime.utcnow(),
            completed_at=execution_status["end_time"],
            node_statuses=execution_status["nodes"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting execution status", error=str(e), execution_id=execution_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution status: {str(e)}"
        )

@router.delete("/executions/{execution_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_execution(
    execution_id: str,
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Cancel a running workflow execution.
    """
    try:
        async with _execution_lock:
            if execution_id not in active_executions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Execution not found: {execution_id}"
                )
                
            orchestrator = active_executions[execution_id]
        
        # Cancel the execution
        await orchestrator.cancel_execution()
        
        # Schedule removal after a delay (e.g., 15 minutes)
        asyncio.create_task(remove_execution_after_delay(execution_id, delay_seconds=900))
        
        logger.info(
            "execution_canceled",
            execution_id=execution_id,
            workflow_id=orchestrator.workflow_id,
            user_id=str(current_user.id)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error canceling execution", error=str(e), execution_id=execution_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}"
        )
