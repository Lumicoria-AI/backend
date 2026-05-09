from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.agents.agent_service import AgentService
from backend.agents.data_analysis_agent import DataAnalysisAgent, AnalysisMode
from backend.services.activity_logger import log_activity

router = APIRouter()

# Request and Response Models
class AnalysisRequest(BaseModel):
    """Base model for data analysis requests."""
    data: str = Field(..., description="The data to analyze (CSV or JSON string)")
    mode: str = Field("exploratory", description="Analysis mode (exploratory, statistical, visualization, anomaly, trend, report)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis-specific parameters")
    model: Optional[str] = Field(None, description="AI model to use (defaults to sonar-large-online)")

class AnalysisResponse(BaseModel):
    """Base model for analysis responses."""
    results: Dict[str, Any]
    metadata: Dict[str, Any]

class ExploratoryAnalysisRequest(AnalysisRequest):
    """Model for exploratory analysis requests."""
    include_correlations: bool = Field(True, description="Whether to include correlation analysis")
    include_distributions: bool = Field(True, description="Whether to include distribution analysis")

class StatisticalAnalysisRequest(AnalysisRequest):
    """Model for statistical analysis requests."""
    hypothesis_testing: bool = Field(True, description="Whether to perform hypothesis testing")
    regression_analysis: bool = Field(False, description="Whether to perform regression analysis")
    confidence_level: float = Field(0.95, description="Confidence level for statistical tests")

class VisualizationRequest(AnalysisRequest):
    """Model for visualization requests."""
    chart_types: List[str] = Field(["line", "bar", "scatter", "histogram", "box"], description="Types of charts to generate")
    interactive: bool = Field(True, description="Whether to generate interactive visualizations")
    export_format: str = Field("png", description="Export format for visualizations")

class AnomalyDetectionRequest(AnalysisRequest):
    """Model for anomaly detection requests."""
    detection_methods: List[str] = Field(["zscore", "iqr"], description="Methods to use for anomaly detection")
    threshold: float = Field(3.0, description="Threshold for anomaly detection (in standard deviations)")
    include_visualization: bool = Field(True, description="Whether to include anomaly visualizations")

class TrendAnalysisRequest(AnalysisRequest):
    """Model for trend analysis requests."""
    forecasting_methods: List[str] = Field(["linear", "exponential"], description="Methods to use for trend forecasting")
    seasonality_detection: bool = Field(True, description="Whether to detect seasonality")
    forecast_periods: int = Field(30, description="Number of periods to forecast")

class ReportGenerationRequest(AnalysisRequest):
    """Model for report generation requests."""
    sections: List[str] = Field(["summary", "findings", "recommendations"], description="Sections to include in the report")
    include_visualizations: bool = Field(True, description="Whether to include visualizations in the report")
    export_format: str = Field("pdf", description="Export format for the report")

# Helper function to get agent service
def get_agent_service() -> AgentService:
    """Get or create an instance of AgentService."""
    config = {
        "model": "sonar-large-online",
        "model_config": {
            "model": "sonar-large-online",
            "temperature": 0.7,
            "max_tokens": 2048
        }
    }
    return AgentService(config)

@router.post("/analyze", response_model=AnalysisResponse)
async def process_analysis(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Process a data analysis request using the Data Analysis Agent.
    
    This endpoint handles various types of analysis tasks including:
    - Exploratory data analysis
    - Statistical analysis
    - Data visualization
    - Anomaly detection
    - Trend analysis
    - Report generation
    """
    try:
        # Permission gate.  Use getattr so personal accounts (org_id=None)
        # pass the early-return inside check_permission rather than
        # tripping a 403.
        permission_org = getattr(current_user, "organization_id", None)
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=permission_org,
            resource_type="AGENT",
            resource_id="data_analysis",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the data analysis agent"
            )

        # Build the agent config from the platform's configured provider
        # so we don't hardcode Perplexity here.
        from backend.core.config import settings as _settings
        provider = (_settings.DEFAULT_LLM_PROVIDER or "gemini").lower()
        default_model = {
            "gemini": getattr(_settings, "GEMINI_MODEL", None) or "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5-20251001",
            "mistral": "mistral-small-latest",
            "perplexity": "sonar",
        }.get(provider, "sonar")
        chosen_model = request.model or default_model

        agent_config = {
            "provider": provider,
            "model": chosen_model,
            # BaseAgent reads from `agent_model_config`, NOT `model_config`.
            "agent_model_config": {
                "model": chosen_model,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        }

        analysis_agent = DataAnalysisAgent(agent_config)

        # Process the request
        result = await analysis_agent.process_async({
            "data": request.data,
            "mode": request.mode,
            "context": request.context or {},
            "parameters": request.parameters or {}
        })

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=permission_org,
            activity_type="data_analysis.analyzed",
            details={"mode": request.mode, "data_preview": request.data[:100]},
            related_resource_type="AGENT",
            agent_name="Data Analysis Agent",
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing analysis request: {str(e)}"
        )

@router.post("/analyze/exploratory", response_model=AnalysisResponse)
async def perform_exploratory_analysis(
    request: ExploratoryAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Perform exploratory data analysis.
    
    This endpoint provides:
    - Summary statistics
    - Correlation analysis
    - Distribution analysis
    - Key insights
    """
    try:
        # Set request type for exploratory analysis
        request.mode = AnalysisMode.EXPLORATORY.value
        
        # Add exploratory analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "include_correlations": request.include_correlations,
            "include_distributions": request.include_distributions
        })
        
        # Process using the main endpoint
        return await process_analysis(
            AnalysisRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing exploratory analysis: {str(e)}"
        )

@router.post("/analyze/statistical", response_model=AnalysisResponse)
async def perform_statistical_analysis(
    request: StatisticalAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Perform statistical analysis.
    
    This endpoint provides:
    - Hypothesis testing
    - Regression analysis
    - Confidence intervals
    - Statistical insights
    """
    try:
        # Set request type for statistical analysis
        request.mode = AnalysisMode.STATISTICAL.value
        
        # Add statistical analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "hypothesis_testing": request.hypothesis_testing,
            "regression_analysis": request.regression_analysis,
            "confidence_level": request.confidence_level
        })
        
        # Process using the main endpoint
        return await process_analysis(
            AnalysisRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing statistical analysis: {str(e)}"
        )

@router.post("/analyze/visualize", response_model=AnalysisResponse)
async def generate_visualizations(
    request: VisualizationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate data visualizations.
    
    This endpoint provides:
    - Multiple chart types
    - Interactive visualizations
    - Export options
    - Visualization descriptions
    """
    try:
        # Set request type for visualization
        request.mode = AnalysisMode.VISUALIZATION.value
        
        # Add visualization parameters
        parameters = request.parameters or {}
        parameters.update({
            "chart_types": request.chart_types,
            "interactive": request.interactive,
            "export_format": request.export_format
        })
        
        # Process using the main endpoint
        return await process_analysis(
            AnalysisRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating visualizations: {str(e)}"
        )

@router.post("/analyze/anomalies", response_model=AnalysisResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Detect anomalies and outliers.
    
    This endpoint provides:
    - Multiple detection methods
    - Anomaly visualization
    - Threshold configuration
    - Anomaly insights
    """
    try:
        # Set request type for anomaly detection
        request.mode = AnalysisMode.ANOMALY.value
        
        # Add anomaly detection parameters
        parameters = request.parameters or {}
        parameters.update({
            "detection_methods": request.detection_methods,
            "threshold": request.threshold,
            "include_visualization": request.include_visualization
        })
        
        # Process using the main endpoint
        return await process_analysis(
            AnalysisRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting anomalies: {str(e)}"
        )

@router.post("/analyze/trends", response_model=AnalysisResponse)
async def analyze_trends(
    request: TrendAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Analyze trends and patterns.
    
    This endpoint provides:
    - Trend detection
    - Forecasting
    - Seasonality analysis
    - Trend insights
    """
    try:
        # Set request type for trend analysis
        request.mode = AnalysisMode.TREND.value
        
        # Add trend analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "forecasting_methods": request.forecasting_methods,
            "seasonality_detection": request.seasonality_detection,
            "forecast_periods": request.forecast_periods
        })
        
        # Process using the main endpoint
        return await process_analysis(
            AnalysisRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing trends: {str(e)}"
        )

@router.post("/analyze/report", response_model=AnalysisResponse)
async def generate_report(
    request: ReportGenerationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate a comprehensive analysis report.
    
    This endpoint provides:
    - Customizable report sections
    - Integrated visualizations
    - Multiple export formats
    - Actionable insights
    """
    try:
        # Set request type for report generation
        request.mode = AnalysisMode.REPORT.value
        
        # Add report generation parameters
        parameters = request.parameters or {}
        parameters.update({
            "sections": request.sections,
            "include_visualizations": request.include_visualizations,
            "export_format": request.export_format
        })
        
        # Process using the main endpoint
        return await process_analysis(
            AnalysisRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating report: {str(e)}"
        )

# ─── Production grade upload + persisted runs ──────────────────────────


_ALLOWED_UPLOAD_MIME = {
    "text/csv",
    "application/csv",
    "text/plain",
    "application/json",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_VALID_MODES = {"exploratory", "statistical", "visualization", "anomaly", "trend", "report"}
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


async def _require_data_analysis_perm(current_user: User) -> str:
    """Returns the tenant scope id; raises 403 on missing permission."""
    permission_org = getattr(current_user, "organization_id", None)
    has = await permission_repository.check_permission(
        user_id=str(current_user.id),
        organization_id=permission_org,
        resource_type="AGENT",
        resource_id="data_analysis",
        permission_type="EXECUTE",
    )
    if not has:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to use the data analysis agent",
        )
    return permission_org or str(current_user.id)


@router.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    mode: str = Query("exploratory", description="Analysis mode"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Upload a CSV / XLSX / JSON file, persist it to object storage,
    create a run row, and run the analysis pipeline synchronously.

    Returns the full run dict with summary stats, preview rows,
    visualizations, insights, and AI summary.
    """
    org_id = await _require_data_analysis_perm(current_user)
    user_id = str(current_user.id)

    if mode not in _VALID_MODES:
        mode = "exploratory"

    # Validate MIME up front so we don't waste cycles on garbage uploads.
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in _ALLOWED_UPLOAD_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: CSV, XLSX, JSON.",
        )

    # Read with a hard cap.  Reading content_length up front is unreliable
    # behind some proxies, so we enforce on the actual buffer.
    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max upload size is 100 MB.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file.",
        )

    # Persist the original file to object storage.
    import os
    from backend.services.storage_service import storage_service
    from backend.services.data_analysis import runs as runs_svc
    from backend.services.data_analysis import pipeline as pipeline_svc

    original_filename = file.filename or "upload"
    file_ext = os.path.splitext(original_filename)[1] or ".dat"

    # Create the run row first so we have an id for the s3_key.
    run = await runs_svc.create_run(
        organization_id=org_id,
        user_id=user_id,
        mode=mode,
        s3_key="placeholder",  # patched below
        filename=original_filename,
        original_filename=original_filename,
        content_type=content_type or None,
        size_bytes=len(contents),
        status="pending",
    )
    run_id = run["id"]
    s3_key = f"data-analysis/{user_id}/{run_id}{file_ext}"

    try:
        await storage_service.upload_file(contents, s3_key, content_type or "application/octet-stream")
    except Exception as e:
        await runs_svc.update_run_results(
            org_id, run_id, status="error", error_message=f"Storage upload failed: {e}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage upload failed: {e}",
        )

    # Patch the run with the real s3_key.
    await runs_svc.update_run_results(org_id, run_id)
    # SQLAlchemy update_run_results expects fields by name; do a small
    # dedicated update for the s3_key path.
    from sqlalchemy import update as _sa_update
    from backend.db.postgres import get_async_sessionmaker
    from backend.db.postgres_models import DataAnalysisRunSQL
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            _sa_update(DataAnalysisRunSQL)
            .where(DataAnalysisRunSQL.id == run_id, DataAnalysisRunSQL.organization_id == org_id)
            .values(s3_key=s3_key)
        )
        await session.commit()

    # Run the pipeline (synchronously for now — Celery offload is a
    # follow on for files > 50 MB).
    try:
        run_dict = await pipeline_svc.run_pipeline(
            run_id=run_id,
            organization_id=org_id,
            user_id=user_id,
            s3_key=s3_key,
            content_type=content_type or "text/csv",
            filename=original_filename,
            mode=mode,
        )
    except Exception as e:
        # The pipeline already marked status=error; surface a 500 with a
        # short message but keep the row so the operator can inspect.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {e}",
        )

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="data_analysis.run_created",
        details={
            "run_id": run_id,
            "mode": mode,
            "filename": original_filename,
            "size_bytes": len(contents),
        },
        related_resource_type="AGENT",
        agent_name="Data Analysis Agent",
    )
    return run_dict


@router.get("/runs")
async def list_runs(
    status: Optional[str] = Query(None),
    mode: Optional[str] = Query(None),
    time_range: Optional[str] = Query(None, pattern="^(1d|7d|30d|90d|1y)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """List the caller's analysis runs, scoped by organization."""
    org_id = await _require_data_analysis_perm(current_user)
    from backend.services.data_analysis import runs as runs_svc
    return await runs_svc.list_runs(
        org_id, status=status, mode=mode, time_range=time_range,
        limit=limit, offset=offset,
    )


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    org_id = await _require_data_analysis_perm(current_user)
    from backend.services.data_analysis import runs as runs_svc
    run = await runs_svc.get_run(org_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(
    run_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    org_id = await _require_data_analysis_perm(current_user)
    user_id = str(current_user.id)
    from backend.services.data_analysis import runs as runs_svc
    ok = await runs_svc.soft_delete_run(org_id, run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found")
    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="data_analysis.run_deleted",
        details={"run_id": run_id},
        related_resource_type="AGENT",
        agent_name="Data Analysis Agent",
    )


class _AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


@router.post("/runs/{run_id}/ask")
async def ask_about_run(
    run_id: str,
    payload: _AskRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Ask a natural language question about a stored run."""
    org_id = await _require_data_analysis_perm(current_user)
    user_id = str(current_user.id)

    from backend.services.data_analysis import nlq
    try:
        turn = await nlq.ask(
            organization_id=org_id,
            run_id=run_id,
            question=payload.question,
        )
    except ValueError as e:
        if str(e) == "run_not_found":
            raise HTTPException(status_code=404, detail="Run not found")
        raise HTTPException(status_code=400, detail=str(e))

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="data_analysis.run_ask",
        details={"run_id": run_id, "question_preview": payload.question[:120]},
        related_resource_type="AGENT",
        agent_name="Data Analysis Agent",
    )
    return turn


class _RegenerateRequest(BaseModel):
    mode: str = Field(..., min_length=1, max_length=32)


@router.post("/runs/{run_id}/regenerate")
async def regenerate_run(
    run_id: str,
    payload: _RegenerateRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Re run analysis on a stored file in a different mode without
    re uploading.  Re uses the s3_key already on the row."""
    org_id = await _require_data_analysis_perm(current_user)
    user_id = str(current_user.id)

    new_mode = payload.mode.lower().strip()
    if new_mode not in _VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {payload.mode}")

    from backend.services.data_analysis import runs as runs_svc
    from backend.services.data_analysis import pipeline as pipeline_svc

    run = await runs_svc.get_run(org_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        updated = await pipeline_svc.run_pipeline(
            run_id=run_id,
            organization_id=org_id,
            user_id=user_id,
            s3_key=run["s3_key"],
            content_type=run.get("content_type") or "text/csv",
            filename=run.get("original_filename") or run.get("filename") or "upload",
            mode=new_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regenerate failed: {e}")

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="data_analysis.run_regenerated",
        details={"run_id": run_id, "mode": new_mode},
        related_resource_type="AGENT",
        agent_name="Data Analysis Agent",
    )
    return updated


@router.get("/analytics", response_model=Dict[str, Any])
async def get_data_analysis_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Real aggregations over `data_analysis_runs` rows for the caller's
    organization within the chosen time range.  Same response shape as
    the previous mock so existing clients do not need to change.
    """
    permission_org = getattr(current_user, "organization_id", None)
    user_id = str(current_user.id)
    scope_id = permission_org or user_id

    from backend.services.data_analysis import analytics as analytics_svc
    return await analytics_svc.get_analytics(scope_id, time_range=time_range)
