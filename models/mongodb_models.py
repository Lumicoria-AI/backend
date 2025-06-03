from typing import Optional, List, Dict, Any, Union, Annotated
from pydantic import BaseModel, Field, GetJsonSchemaHandler, ConfigDict
from pydantic.json_schema import JsonSchemaValue
from enum import Enum
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        
    @classmethod
    def validate(cls, v, info):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {"type": "string"}

class MongoModel(BaseModel):
    """Base model for MongoDB documents."""
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }
    )

    @classmethod
    def model_serializer(cls, obj: Any) -> Any:
        """Custom serializer for MongoDB models."""
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

class MongoDocument(MongoModel):
    """Base model for MongoDB documents with ObjectId."""
    id: Optional[ObjectId] = Field(None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }
    )

    @classmethod
    def model_serializer(cls, obj: Any) -> Any:
        """Custom serializer for MongoDB documents."""
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

class MongoBaseModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    model_config = {
        'protected_namespaces': (),  # Disable protected namespace warnings
        'populate_by_name': True,
        'arbitrary_types_allowed': True,
        'json_encoders': {ObjectId: str}
    }

# Enums
class AgentType(str, Enum):
    DOCUMENT = "document"
    WELLBEING = "wellbeing"
    VISION = "vision"
    MEETING = "meeting"
    CREATIVE = "creative"
    STUDENT = "student"
    RESEARCH = "research"
    LEARNING_COACH = "learning_coach"
    CUSTOM = "custom"

class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    DEFERRED = "deferred"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPLOYED = "deployed"
    DRAFT = "draft"
    ARCHIVED = "archived"
    ERROR = "error"

class AgentCapability(str, Enum):
    DOCUMENT_PROCESSING = "document_processing"
    OCR = "ocr"
    TEXT_ANALYSIS = "text_analysis"
    TASK_GENERATION = "task_generation"
    CALENDAR_INTEGRATION = "calendar_integration"
    WELLBEING_MONITORING = "wellbeing_monitoring"
    VISION_ANALYSIS = "vision_analysis"
    VOICE_PROCESSING = "voice_processing"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    DATA_EXTRACTION = "data_extraction"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CITATION_MANAGEMENT = "citation_management"

class ComponentType(str, Enum):
    INPUT = "input"
    PROCESSOR = "processor"
    OUTPUT = "output"
    INTEGRATION = "integration"

class ComponentCategory(str, Enum):
    DOCUMENT = "document"
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    WELLBEING = "wellbeing"
    PRODUCTIVITY = "productivity"
    RESEARCH = "research"
    COMMUNICATION = "communication"

class NotificationType(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    TASK_REMINDER = "task_reminder"
    SYSTEM_UPDATE = "system_update"
    AGENT_ALERT = "agent_alert"

class NotificationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class WellbeingCategory(str, Enum):
    PHYSICAL = "physical"
    MENTAL = "mental"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    PROFESSIONAL = "professional"
    ENVIRONMENTAL = "environmental"

class WellbeingStatus(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

# Agent Models
class AgentConfig(BaseModel):
    model: str = "perplexity-sonar"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    cache_enabled: bool = True
    cache_ttl: int = 3600
    custom_settings: Dict[str, Any] = {}

class AgentState(BaseModel):
    status: AgentStatus
    last_active: Optional[datetime] = None
    current_task: Optional[str] = None
    memory_usage: int = 0
    error_count: int = 0
    execution_count: int = 0

class AgentModel(BaseModel):
    provider: str
    name: str  # Changed from model_name to avoid namespace conflict
    version: str = "latest"
    configuration: Dict[str, Any] = {}

    model_config = {
        'protected_namespaces': ()  # Disable protected namespace warnings
    }

# Component Models
class ComponentDefinition(MongoBaseModel):
    name: str
    description: Optional[str] = None
    component_type: ComponentType
    category: ComponentCategory
    version: str = "1.0.0"
    icon: Optional[str] = None
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    config_schema: Dict[str, Any] = {}
    implementation: Optional[str] = None
    dependencies: List[str] = []
    tags: List[str] = []
    is_public: bool = False
    organization_id: Optional[PyObjectId] = None
    created_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = 0
    rating: float = 0.0
    rating_count: int = 0

class ComponentInstance(BaseModel):
    id: str
    component_id: str
    name: str
    config: Dict[str, Any] = {}
    position: Dict[str, float] = {"x": 0, "y": 0}
    connections: List[Dict[str, Any]] = []

class AgentWorkflowNode(BaseModel):
    id: str
    component_instance_id: str
    position: Dict[str, float]
    size: Dict[str, float] = {"width": 200, "height": 100}

class AgentWorkflowConnection(BaseModel):
    id: str
    source_node_id: str
    target_node_id: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    label: Optional[str] = None

class AgentWorkflow(MongoBaseModel):
    name: str
    description: Optional[str] = None
    components: List[ComponentInstance] = []
    nodes: List[AgentWorkflowNode] = []
    connections: List[AgentWorkflowConnection] = []
    organization_id: PyObjectId
    created_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    is_public: bool = False
    tags: List[str] = []
    status: AgentStatus = Field(default=AgentStatus.DRAFT)

# Agent Models
class Agent(MongoBaseModel):
    name: str
    description: Optional[str] = None
    agent_type: AgentType
    capabilities: List[AgentCapability] = []
    configuration: Dict[str, Any] = {}
    workflow_id: Optional[PyObjectId] = None
    agent_model_config: AgentConfig = Field(default=AgentConfig())
    state: AgentState = Field(default=AgentState(status=AgentStatus.INACTIVE))
    organization_id: PyObjectId
    created_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    is_public: bool = False
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    usage_statistics: Dict[str, Any] = {}

    model_config = {
        'protected_namespaces': (),  # Disable protected namespace warnings
        'populate_by_name': True,
        'arbitrary_types_allowed': True,
        'json_encoders': {ObjectId: str}
    }

# Create/Update Models
class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    agent_type: AgentType
    capabilities: List[AgentCapability] = []
    configuration: Dict[str, Any] = {}
    agent_model_config: Optional[AgentConfig] = None
    is_public: bool = False
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[AgentCapability]] = None
    configuration: Optional[Dict[str, Any]] = None
    agent_model_config: Optional[AgentConfig] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ComponentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    component_type: ComponentType
    category: ComponentCategory
    version: str = "1.0.0"
    icon: Optional[str] = None
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    config_schema: Dict[str, Any] = {}
    implementation: Optional[str] = None
    dependencies: List[str] = []
    tags: List[str] = []
    is_public: bool = False

class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    components: List[ComponentInstance] = []
    nodes: List[AgentWorkflowNode] = []
    connections: List[AgentWorkflowConnection] = []
    is_public: bool = False
    tags: List[str] = []

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    components: Optional[List[ComponentInstance]] = None
    nodes: Optional[List[AgentWorkflowNode]] = None
    connections: Optional[List[AgentWorkflowConnection]] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None

# Component Types for the Studio
class AgentComponent(MongoBaseModel):
    name: str
    description: Optional[str] = None
    component_type: ComponentType
    category: ComponentCategory
    configuration: Dict[str, Any] = {}
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    organization_id: Optional[PyObjectId] = None
    created_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    is_public: bool = False
    tags: List[str] = []
    dependencies: List[Dict[str, Any]] = []
    usage_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    last_used: Optional[datetime] = None

# Re-export common types
AgentComponentType = ComponentType

# Permission Enums
class ResourceType(str, Enum):
    AGENT = "agent"
    COMPONENT = "component"
    DOCUMENT = "document"
    WORKFLOW = "workflow"
    ORGANIZATION = "organization"
    USER = "user"
    CONVERSATION = "conversation"
    TASK = "task"
    INTEGRATION = "integration"

class PermissionType(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    SHARE = "share"
    DELETE = "delete"
    CREATE = "create"
    VIEW = "view"
    EDIT = "edit"
    MANAGE = "manage"

# Permission Models
class Permission(MongoBaseModel):
    organization_id: PyObjectId
    user_id: Optional[PyObjectId] = None
    role_id: Optional[PyObjectId] = None
    resource_type: ResourceType
    resource_id: PyObjectId
    permission_type: PermissionType
    granted_by: PyObjectId
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class RolePermission(BaseModel):
    role_id: PyObjectId
    resource_type: ResourceType
    permission_type: PermissionType
    metadata: Dict[str, Any] = {}

# Activity Log
class ActivityLogEntry(MongoBaseModel):
    user_id: Optional[PyObjectId] = None
    organization_id: PyObjectId
    activity_type: str
    resource_type: Optional[str] = None
    resource_id: Optional[PyObjectId] = None
    details: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    severity: str = "info"
    agent_id: Optional[PyObjectId] = None

# Task Models
class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None
    assigned_to: Optional[PyObjectId] = None
    organization_id: Optional[PyObjectId] = None
    project_id: Optional[PyObjectId] = None
    parent_task_id: Optional[PyObjectId] = None
    agent_id: Optional[PyObjectId] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    
class Task(MongoBaseModel):
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None
    assigned_to: Optional[PyObjectId] = None
    created_by: Optional[PyObjectId] = None
    organization_id: Optional[PyObjectId] = None
    project_id: Optional[PyObjectId] = None
    parent_task_id: Optional[PyObjectId] = None
    agent_id: Optional[PyObjectId] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    progress: int = 0
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    due_date: Optional[datetime] = None
    assigned_to: Optional[PyObjectId] = None
    project_id: Optional[PyObjectId] = None
    parent_task_id: Optional[PyObjectId] = None
    agent_id: Optional[PyObjectId] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    progress: Optional[int] = None

# User Model
class User(MongoBaseModel):
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    profile_picture: Optional[str] = None
    organization_id: Optional[PyObjectId] = None
    settings: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

# Notification Models
class Notification(MongoBaseModel):
    user_id: PyObjectId
    title: str
    message: str
    notification_type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.MEDIUM
    is_read: bool = False
    is_dismissed: bool = False
    action_url: Optional[str] = None
    action_label: Optional[str] = None
    metadata: Dict[str, Any] = {}
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    read_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None

class NotificationCreate(BaseModel):
    user_id: PyObjectId
    title: str
    message: str
    notification_type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.MEDIUM
    action_url: Optional[str] = None
    action_label: Optional[str] = None
    metadata: Dict[str, Any] = {}
    expires_at: Optional[datetime] = None

class NotificationUpdate(BaseModel):
    title: Optional[str] = None
    message: Optional[str] = None
    notification_type: Optional[NotificationType] = None
    priority: Optional[NotificationPriority] = None
    is_read: Optional[bool] = None
    is_dismissed: Optional[bool] = None
    action_url: Optional[str] = None
    action_label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None

# Wellbeing Models
class WellbeingMetric(BaseModel):
    """Wellbeing metric with value and metadata."""
    category: WellbeingCategory
    name: str
    value: float
    unit: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    confidence: Optional[float] = None

class WellbeingRecommendation(BaseModel):
    """Recommendation for improving wellbeing."""
    category: WellbeingCategory
    title: str
    description: str
    priority: str = "medium"
    difficulty: str = "medium"
    estimated_impact: float = 0.0
    citations: Optional[List[str]] = None
    suggested_duration: Optional[str] = None

class WellbeingGoal(BaseModel):
    """Wellbeing improvement goal."""
    category: WellbeingCategory
    title: str
    description: str
    target_value: Optional[float] = None
    target_date: Optional[datetime] = None
    status: str = "active"
    progress: float = 0.0
    metrics: List[str] = []
    recommendations: List[str] = []

class WellbeingCreate(BaseModel):
    """Create model for wellbeing data."""
    user_id: str
    metrics: List[WellbeingMetric]
    mood_score: Optional[float] = None
    stress_level: Optional[float] = None
    energy_level: Optional[float] = None
    notes: Optional[str] = None
    tags: List[str] = []
    recommendations: List[WellbeingRecommendation] = []
    goals: List[WellbeingGoal] = []

class WellbeingData(BaseModel):
    """Model for wellbeing tracking and monitoring."""
    id: str = Field(default_factory=lambda: str(ObjectId()))
    user_id: str
    metrics: List[WellbeingMetric]
    mood_score: Optional[float] = None
    stress_level: Optional[float] = None
    energy_level: Optional[float] = None
    overall_status: WellbeingStatus = WellbeingStatus.GOOD
    notes: Optional[str] = None
    tags: List[str] = []
    recommendations: List[WellbeingRecommendation] = []
    goals: List[WellbeingGoal] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
