from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional, Union
import json
import structlog
import asyncio
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
from io import StringIO

# Configure logger
logger = structlog.get_logger(__name__)

class AnalysisMode(Enum):
    """Enum for different data analysis modes."""
    EXPLORATORY = "exploratory"  # Basic data exploration and summary
    STATISTICAL = "statistical"  # Statistical analysis and hypothesis testing
    VISUALIZATION = "visualization"  # Data visualization and chart generation
    ANOMALY = "anomaly"  # Anomaly detection and outlier analysis
    TREND = "trend"  # Trend analysis and forecasting
    REPORT = "report"  # Report generation with insights

class DataAnalysisAgent(BaseAgent):
    """Agent for data analysis and insights generation using LLM providers.
    
    This agent provides comprehensive data analysis services including:
    - Data exploration and summary statistics
    - Statistical analysis and hypothesis testing
    - Data visualization and chart generation
    - Anomaly detection and outlier analysis
    - Trend analysis and forecasting
    - Automated report generation
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default agent capabilities if not specified in config
        self.capabilities = config.get("capabilities", [
            "data_exploration",
            "statistical_analysis",
            "visualization",
            "anomaly_detection",
            "trend_analysis",
            "report_generation"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"  # Use Perplexity's Sonar model
        
        # Set analysis parameters
        self.default_sample_size = config.get("default_sample_size", 1000)
        self.confidence_level = config.get("confidence_level", 0.95)
        self.anomaly_threshold = config.get("anomaly_threshold", 3.0)  # Standard deviations
        
        # Analysis modes and their specific settings
        self.analysis_modes = {
            AnalysisMode.EXPLORATORY: {
                "summary_stats": True,
                "correlation_analysis": True,
                "distribution_analysis": True
            },
            AnalysisMode.STATISTICAL: {
                "hypothesis_testing": True,
                "regression_analysis": True,
                "confidence_intervals": True
            },
            AnalysisMode.VISUALIZATION: {
                "chart_types": ["line", "bar", "scatter", "histogram", "box"],
                "interactive": True,
                "export_formats": ["png", "svg", "html"]
            },
            AnalysisMode.ANOMALY: {
                "detection_methods": ["zscore", "iqr", "isolation_forest"],
                "visualization": True,
                "explanation": True
            },
            AnalysisMode.TREND: {
                "forecasting_methods": ["linear", "exponential", "seasonal"],
                "seasonality_detection": True,
                "confidence_intervals": True
            },
            AnalysisMode.REPORT: {
                "sections": ["summary", "findings", "recommendations"],
                "visualization": True,
                "export_formats": ["pdf", "html", "markdown"]
            }
        }

    async def process_async(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data analysis request asynchronously.
        
        Args:
            analysis_data: Dictionary containing:
                - data: Input data (CSV, JSON, or DataFrame)
                - mode: Analysis mode (exploratory, statistical, etc.)
                - context: Additional context for analysis
                - parameters: Analysis-specific parameters
        
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            # Extract analysis parameters
            data = analysis_data.get("data")
            mode = AnalysisMode(analysis_data.get("mode", "exploratory"))
            context = analysis_data.get("context", {})
            parameters = analysis_data.get("parameters", {})
            
            if not data:
                return {"error": "No data provided for analysis"}
            
            # Convert input data to DataFrame if needed
            df = self._prepare_data(data)
            if isinstance(df, dict) and "error" in df:
                return df
            
            # Get mode-specific settings
            mode_settings = self.analysis_modes.get(mode, self.analysis_modes[AnalysisMode.EXPLORATORY])
            
            # Create system prompt based on mode and settings
            system_prompt = self._create_system_prompt(
                mode=mode,
                settings=mode_settings,
                context=context
            )
            
            # Perform analysis based on mode
            if mode == AnalysisMode.EXPLORATORY:
                result = await self._perform_exploratory_analysis(df, parameters)
            elif mode == AnalysisMode.STATISTICAL:
                result = await self._perform_statistical_analysis(df, parameters)
            elif mode == AnalysisMode.VISUALIZATION:
                result = await self._generate_visualizations(df, parameters)
            elif mode == AnalysisMode.ANOMALY:
                result = await self._detect_anomalies(df, parameters)
            elif mode == AnalysisMode.TREND:
                result = await self._analyze_trends(df, parameters)
            elif mode == AnalysisMode.REPORT:
                result = await self._generate_report(df, parameters)
            else:
                return {"error": f"Unsupported analysis mode: {mode}"}
            
            # Add metadata
            result.update({
                "metadata": {
                    "mode": mode.value,
                    "data_shape": df.shape,
                    "columns": df.columns.tolist(),
                    "processed_at": datetime.utcnow().isoformat(),
                    "model_used": self.model_config.get("model")
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing analysis request: {str(e)}")
            return {"error": f"Analysis processing failed: {str(e)}"}

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the data analysis agent asynchronously."""
        return await self.process_async({
            "mode": AnalysisMode.EXPLORATORY.value,
            "data": query,
            "context": context or {}
        })
    
    def _prepare_data(self, data: Union[str, Dict, pd.DataFrame]) -> Union[pd.DataFrame, Dict[str, str]]:
        """Convert input data to pandas DataFrame."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, str):
                # Try to parse as CSV
                try:
                    return pd.read_csv(StringIO(data))
                except:
                    # Try to parse as JSON
                    try:
                        return pd.read_json(StringIO(data))
                    except:
                        return {"error": "Could not parse input data as CSV or JSON"}
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            else:
                return {"error": "Unsupported data format"}
        except Exception as e:
            return {"error": f"Error preparing data: {str(e)}"}
    
    def _create_system_prompt(
        self,
        mode: AnalysisMode,
        settings: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Create system prompt for analysis based on mode and settings."""
        base_prompt = "You are a professional data analysis AI assistant. "
        
        if mode == AnalysisMode.EXPLORATORY:
            base_prompt += "Perform exploratory data analysis, including summary statistics, correlations, and distributions. "
        elif mode == AnalysisMode.STATISTICAL:
            base_prompt += "Conduct statistical analysis, including hypothesis testing and regression analysis. "
        elif mode == AnalysisMode.VISUALIZATION:
            base_prompt += "Generate appropriate visualizations to represent the data effectively. "
        elif mode == AnalysisMode.ANOMALY:
            base_prompt += "Detect anomalies and outliers in the dataset using multiple methods. "
        elif mode == AnalysisMode.TREND:
            base_prompt += "Analyze trends and patterns in the data, including forecasting where appropriate. "
        elif mode == AnalysisMode.REPORT:
            base_prompt += "Generate a comprehensive analysis report with insights and recommendations. "
        
        if context:
            base_prompt += f"\nAdditional context: {json.dumps(context)}"
        
        return base_prompt
    
    async def _perform_exploratory_analysis(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform exploratory data analysis."""
        try:
            # Calculate summary statistics
            summary_stats = df.describe().to_dict()
            
            # Calculate correlations for numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlations = df[numeric_cols].corr().to_dict() if len(numeric_cols) > 0 else {}
            
            # Analyze distributions
            distributions = {}
            for col in numeric_cols:
                distributions[col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "skew": df[col].skew(),
                    "kurtosis": df[col].kurtosis()
                }
            
            # Generate insights using the model
            insights_prompt = f"""
            Analyze the following data summary and provide key insights:
            Summary Statistics: {json.dumps(summary_stats)}
            Correlations: {json.dumps(correlations)}
            Distributions: {json.dumps(distributions)}
            """
            
            insights = await self._call_model_async(
                prompt=insights_prompt,
                system_prompt=self._create_system_prompt(
                    AnalysisMode.EXPLORATORY,
                    self.analysis_modes[AnalysisMode.EXPLORATORY],
                    {}
                )
            )
            
            return {
                "summary_statistics": summary_stats,
                "correlations": correlations,
                "distributions": distributions,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error in exploratory analysis: {str(e)}")
            return {"error": f"Exploratory analysis failed: {str(e)}"}
    
    async def _perform_statistical_analysis(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical analysis."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            results = {}
            
            # Perform statistical tests based on parameters
            if parameters.get("hypothesis_testing"):
                for col in numeric_cols:
                    # Example: One-sample t-test against mean
                    test_stat, p_value = stats.ttest_1samp(df[col].dropna(), df[col].mean())
                    results[f"{col}_t_test"] = {
                        "test_statistic": test_stat,
                        "p_value": p_value,
                        "significant": p_value < (1 - self.confidence_level)
                    }
            
            # Generate statistical insights
            insights_prompt = f"""
            Analyze the following statistical test results and provide insights:
            {json.dumps(results)}
            """
            
            insights = await self._call_model_async(
                prompt=insights_prompt,
                system_prompt=self._create_system_prompt(
                    AnalysisMode.STATISTICAL,
                    self.analysis_modes[AnalysisMode.STATISTICAL],
                    {}
                )
            )
            
            return {
                "statistical_tests": results,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    async def _generate_visualizations(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate data visualizations."""
        try:
            # Generate visualization specifications
            viz_specs = self._create_visualization_specs(df, parameters)
            
            # Generate visualization descriptions using the model
            viz_prompt = f"""
            Create visualization specifications for the following data:
            Data Summary: {df.describe().to_dict()}
            Visualization Requirements: {json.dumps(parameters)}
            """
            
            viz_descriptions = await self._call_model_async(
                prompt=viz_prompt,
                system_prompt=self._create_system_prompt(
                    AnalysisMode.VISUALIZATION,
                    self.analysis_modes[AnalysisMode.VISUALIZATION],
                    {}
                )
            )
            
            return {
                "visualization_specs": viz_specs,
                "descriptions": viz_descriptions
            }
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {"error": f"Visualization generation failed: {str(e)}"}
    
    async def _detect_anomalies(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect anomalies and outliers in the data."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            anomalies = {}
            
            for col in numeric_cols:
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                z_score_anomalies = df[z_scores > self.anomaly_threshold]
                
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                iqr_anomalies = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                
                anomalies[col] = {
                    "z_score_anomalies": z_score_anomalies[col].tolist(),
                    "iqr_anomalies": iqr_anomalies[col].tolist()
                }
            
            # Generate anomaly insights
            insights_prompt = f"""
            Analyze the following anomaly detection results and provide insights:
            {json.dumps(anomalies)}
            """
            
            insights = await self._call_model_async(
                prompt=insights_prompt,
                system_prompt=self._create_system_prompt(
                    AnalysisMode.ANOMALY,
                    self.analysis_modes[AnalysisMode.ANOMALY],
                    {}
                )
            )
            
            return {
                "anomalies": anomalies,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    async def _analyze_trends(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze trends and patterns in the data."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            trends = {}
            
            for col in numeric_cols:
                # Calculate moving averages
                if len(df) > 1:
                    trends[col] = {
                        "moving_average_7": df[col].rolling(window=7).mean().tolist(),
                        "moving_average_30": df[col].rolling(window=30).mean().tolist(),
                        "trend_direction": "increasing" if df[col].diff().mean() > 0 else "decreasing"
                    }
            
            # Generate trend insights
            insights_prompt = f"""
            Analyze the following trend analysis results and provide insights:
            {json.dumps(trends)}
            """
            
            insights = await self._call_model_async(
                prompt=insights_prompt,
                system_prompt=self._create_system_prompt(
                    AnalysisMode.TREND,
                    self.analysis_modes[AnalysisMode.TREND],
                    {}
                )
            )
            
            return {
                "trends": trends,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {"error": f"Trend analysis failed: {str(e)}"}
    
    async def _generate_report(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        try:
            # Gather all analysis results
            exploratory = await self._perform_exploratory_analysis(df, {})
            statistical = await self._perform_statistical_analysis(df, {})
            anomalies = await self._detect_anomalies(df, {})
            trends = await self._analyze_trends(df, {})
            
            # Generate report using the model
            report_prompt = f"""
            Generate a comprehensive data analysis report based on the following results:
            Exploratory Analysis: {json.dumps(exploratory)}
            Statistical Analysis: {json.dumps(statistical)}
            Anomaly Detection: {json.dumps(anomalies)}
            Trend Analysis: {json.dumps(trends)}
            """
            
            report = await self._call_model_async(
                prompt=report_prompt,
                system_prompt=self._create_system_prompt(
                    AnalysisMode.REPORT,
                    self.analysis_modes[AnalysisMode.REPORT],
                    {}
                )
            )
            
            return {
                "report": report,
                "sections": {
                    "exploratory_analysis": exploratory,
                    "statistical_analysis": statistical,
                    "anomaly_detection": anomalies,
                    "trend_analysis": trends
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {"error": f"Report generation failed: {str(e)}"}
    
    def _create_visualization_specs(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create specifications for data visualizations."""
        specs = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Generate specs for each numeric column
        for col in numeric_cols:
            specs[col] = {
                "histogram": {
                    "type": "histogram",
                    "x": col,
                    "title": f"Distribution of {col}",
                    "bins": 30
                },
                "box_plot": {
                    "type": "box",
                    "y": col,
                    "title": f"Box Plot of {col}"
                }
            }
        
        # Generate correlation heatmap spec if multiple numeric columns
        if len(numeric_cols) > 1:
            specs["correlation_heatmap"] = {
                "type": "heatmap",
                "data": df[numeric_cols].corr().to_dict(),
                "title": "Correlation Heatmap"
            }
        
        return specs 