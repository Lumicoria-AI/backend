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
        # Check permissions
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            resource_type="AGENT",
            resource_id="data_analysis",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the data analysis agent"
            )

        # Create data analysis agent
        agent_config = {
            "model": request.model or "sonar-large-online",
            "model_config": {
                "model": request.model or "sonar-large-online",
                "temperature": 0.7,
                "max_tokens": 2048
            }
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
            organization_id=current_user.organization_id,
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

@router.post("/upload", response_model=AnalysisResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    mode: str = Query("exploratory", description="Analysis mode"),
    context: Optional[Dict[str, Any]] = Body(None),
    parameters: Optional[Dict[str, Any]] = Body(None),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Upload a file and analyze its contents.
    
    This endpoint supports:
    - CSV and JSON file uploads
    - Automatic format detection
    - Multiple analysis modes
    - Custom parameters
    """
    try:
        # Read file contents
        contents = await file.read()
        data = contents.decode()
        
        # Process using the main endpoint
        return await process_analysis(
            AnalysisRequest(
                data=data,
                mode=mode,
                context=context,
                parameters=parameters
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing uploaded file: {str(e)}"
        )

@router.get("/analytics", response_model=Dict[str, Any])
async def get_data_analysis_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get data analysis analytics for the specified time range.
    
    This endpoint provides analytics about:
    - Analysis volume by mode
    - Processing times
    - Error rates
    - User engagement
    """
    # This would typically fetch from a database
    analytics = {
        "time_range": time_range,
        "total_analyses": 3000,
        "average_processing_time": 1.2,  # seconds
        "mode_usage": {
            "exploratory": 1200,
            "statistical": 800,
            "visualization": 500,
            "anomaly": 300,
            "trend": 150,
            "report": 50
        },
        "file_types": {
            "csv": 2000,
            "json": 800,
            "excel": 200
        },
        "quality_metrics": {
            "average_confidence": 0.92,
            "error_rate": 0.03,
            "user_satisfaction": 0.89
        }
    }
    
    return analytics 