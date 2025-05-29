"""
Wellbeing models for Lumicoria.ai
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

class WellbeingCategory(str, Enum):
    """Categories of wellbeing tracking."""
    PHYSICAL = "physical"
    MENTAL = "mental"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    PROFESSIONAL = "professional"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"
    SPIRITUAL = "spiritual"

class WellbeingMetricType(str, Enum):
    """Types of wellbeing metrics."""
    MOOD = "mood"
    ENERGY = "energy"
    STRESS = "stress"
    SLEEP = "sleep"
    EXERCISE = "exercise"
    NUTRITION = "nutrition"
    SOCIAL_INTERACTION = "social_interaction"
    WORK_LIFE_BALANCE = "work_life_balance"
    MINDFULNESS = "mindfulness"
    PRODUCTIVITY = "productivity"
    CUSTOM = "custom"

class BreakType(str, Enum):
    """Types of breaks that can be recommended."""
    MICRO_BREAK = "micro_break"  # 5-10 minutes
    SHORT_BREAK = "short_break"  # 15-30 minutes
    LUNCH_BREAK = "lunch_break"  # 30-60 minutes
    LONG_BREAK = "long_break"    # 1-2 hours
    REST_DAY = "rest_day"        # Full day

class ActivityType(str, Enum):
    """Types of wellbeing activities."""
    PHYSICAL = "physical"        # Exercise, stretching
    MENTAL = "mental"           # Meditation, reading
    SOCIAL = "social"           # Social interaction
    CREATIVE = "creative"       # Art, music, writing
    RELAXATION = "relaxation"   # Rest, sleep
    LEARNING = "learning"       # Educational activities
    MINDFULNESS = "mindfulness" # Mindfulness practices
    CUSTOM = "custom"           # Custom activities

class WellbeingMetricBase(BaseModel):
    """Base model for wellbeing metrics."""
    metric_type: WellbeingMetricType = Field(..., description="Type of metric being tracked")
    value: float = Field(..., description="Numeric value of the metric (0-10 scale)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    source: str = Field(..., description="Source of the metric (e.g., 'user_input', 'device', 'ai')")

class WellbeingMetricCreate(WellbeingMetricBase):
    """Model for creating a new wellbeing metric."""
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Timestamp of the metric")

class WellbeingMetric(WellbeingMetricBase):
    """Complete wellbeing metric model including database fields."""
    id: str = Field(..., description="Unique identifier for the metric")
    user_id: str = Field(..., description="ID of the user")
    organization_id: str = Field(..., description="ID of the organization")
    timestamp: datetime = Field(..., description="Timestamp when the metric was recorded")
    created_at: datetime = Field(..., description="Timestamp when the metric was created")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the metric was last updated")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # For ORM mode
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WellbeingGoalBase(BaseModel):
    """Base model for wellbeing goals."""
    goal_type: WellbeingMetricType = Field(..., description="Type of goal being tracked")
    target_value: float = Field(..., description="Target value for the goal")
    start_date: datetime = Field(..., description="Start date of the goal")
    end_date: datetime = Field(..., description="End date of the goal")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class WellbeingGoalCreate(WellbeingGoalBase):
    """Model for creating a new wellbeing goal."""
    pass

class WellbeingGoal(WellbeingGoalBase):
    """Complete wellbeing goal model including database fields."""
    id: str = Field(..., description="Unique identifier for the goal")
    user_id: str = Field(..., description="ID of the user")
    organization_id: str = Field(..., description="ID of the organization")
    current_value: float = Field(..., description="Current value of the goal")
    status: str = Field(default="active", description="Status of the goal")
    progress: float = Field(..., description="Progress towards the goal (0-100%)")
    created_at: datetime = Field(..., description="Timestamp when the goal was created")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the goal was last updated")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # For ORM mode
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WellbeingRecommendation(BaseModel):
    """Model for wellbeing recommendations."""
    id: str = Field(..., description="Unique identifier for the recommendation")
    user_id: str = Field(..., description="ID of the user")
    organization_id: str = Field(..., description="ID of the organization")
    category: WellbeingCategory = Field(..., description="Category of wellbeing")
    title: str = Field(..., description="Title of the recommendation")
    description: str = Field(..., description="Description of the recommendation")
    action_items: List[str] = Field(..., description="List of action items")
    priority: str = Field(..., description="Priority level of the recommendation")
    metrics: Dict[str, Any] = Field(..., description="Relevant metrics and data")
    created_at: datetime = Field(..., description="Timestamp when the recommendation was created")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the recommendation was last updated")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # For ORM mode
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WellbeingEntryBase(BaseModel):
    """Base model for wellbeing tracking entries."""
    user_id: str = Field(..., description="ID of the user")
    category: WellbeingCategory = Field(..., description="Category of wellbeing")
    metric_type: WellbeingMetricType = Field(..., description="Type of metric being tracked")
    value: float = Field(..., description="Numeric value of the metric (0-10 scale)")
    notes: Optional[str] = Field(None, description="Additional notes or context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class WellbeingEntryCreate(WellbeingEntryBase):
    """Model for creating a new wellbeing entry."""
    organization_id: str = Field(..., description="ID of the organization")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Timestamp of the entry")

class WellbeingEntryUpdate(BaseModel):
    """Model for updating an existing wellbeing entry."""
    value: Optional[float] = Field(None, description="Updated metric value")
    notes: Optional[str] = Field(None, description="Updated notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")

class WellbeingEntry(WellbeingEntryBase):
    """Complete wellbeing entry model including database fields."""
    id: str = Field(..., description="Unique identifier for the entry")
    organization_id: str = Field(..., description="ID of the organization")
    timestamp: datetime = Field(..., description="Timestamp when the entry was created")
    created_at: datetime = Field(..., description="Timestamp when the entry was created")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the entry was last updated")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # For ORM mode
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WellbeingInsight(BaseModel):
    """Model for wellbeing insights and recommendations."""
    id: str = Field(..., description="Unique identifier for the insight")
    user_id: str = Field(..., description="ID of the user")
    organization_id: str = Field(..., description="ID of the organization")
    category: WellbeingCategory = Field(..., description="Category of wellbeing")
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Title of the insight")
    description: str = Field(..., description="Description of the insight")
    recommendations: List[str] = Field(..., description="List of recommendations")
    metrics: Dict[str, Any] = Field(..., description="Relevant metrics and data")
    created_at: datetime = Field(..., description="Timestamp when the insight was created")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the insight was last updated")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # For ORM mode
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WellbeingModel(BaseModel):
    """Base model for wellbeing data."""
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    @classmethod
    def model_serializer(cls, obj: Any) -> Any:
        """Custom serializer for wellbeing models."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

class WellbeingRecord(WellbeingModel):
    """Model for wellbeing records."""
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mood_score: int = Field(ge=1, le=10)
    energy_level: int = Field(ge=1, le=10)
    stress_level: int = Field(ge=1, le=10)
    sleep_hours: float = Field(ge=0, le=24)
    notes: Optional[str] = None
    activities: List[str] = []
    tags: List[str] = []

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "timestamp": "2024-01-01T00:00:00Z",
                "mood_score": 8,
                "energy_level": 7,
                "stress_level": 3,
                "sleep_hours": 7.5,
                "notes": "Feeling productive today",
                "activities": ["exercise", "meditation"],
                "tags": ["productive", "energetic"]
            }
        }
    )

class WellbeingStats(WellbeingModel):
    """Model for wellbeing statistics."""
    user_id: str
    period_start: datetime
    period_end: datetime
    average_mood: float
    average_energy: float
    average_stress: float
    average_sleep: float
    total_records: int
    mood_trend: List[float]
    energy_trend: List[float]
    stress_trend: List[float]
    sleep_trend: List[float]

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "period_start": "2024-01-01T00:00:00Z",
                "period_end": "2024-01-07T23:59:59Z",
                "average_mood": 7.5,
                "average_energy": 6.8,
                "average_stress": 4.2,
                "average_sleep": 7.2,
                "total_records": 7,
                "mood_trend": [7, 8, 7, 8, 7, 8, 7],
                "energy_trend": [6, 7, 6, 7, 6, 7, 6],
                "stress_trend": [4, 3, 4, 3, 4, 3, 4],
                "sleep_trend": [7, 7.5, 7, 7.5, 7, 7.5, 7]
            }
        }
    )

class WellbeingGoal(WellbeingModel):
    """Model for wellbeing goals."""
    user_id: str
    goal_type: str
    target_value: float
    start_date: datetime
    end_date: datetime
    current_value: float
    progress: float
    status: str
    notes: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "goal_type": "sleep",
                "target_value": 8.0,
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "current_value": 7.5,
                "progress": 0.75,
                "status": "in_progress",
                "notes": "Getting closer to target"
            }
        }
    )

class WellbeingRecommendation(WellbeingModel):
    """Model for wellbeing recommendations."""
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    category: str
    title: str
    description: str
    priority: int = Field(ge=1, le=5)
    status: str = "pending"
    action_items: List[str] = []
    resources: List[Dict[str, str]] = []

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "user_id": "user123",
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "sleep",
                "title": "Improve Sleep Quality",
                "description": "Based on your recent sleep patterns, try these tips to improve sleep quality",
                "priority": 3,
                "status": "pending",
                "action_items": [
                    "Set consistent bedtime",
                    "Limit screen time before bed",
                    "Create a relaxing bedtime routine"
                ],
                "resources": [
                    {
                        "title": "Sleep Hygiene Guide",
                        "url": "https://example.com/sleep-guide"
                    }
                ]
            }
        }
    ) 