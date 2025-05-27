from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class MongoBaseModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

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
    model_name: str
    version: str = "latest"
    configuration: Dict[str, Any] = {}

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
    status: AgentStatus = AgentStatus.DRAFT

# Agent Models
class Agent(MongoBaseModel):
    name: str
    description: Optional[str] = None
    agent_type: AgentType
    capabilities: List[AgentCapability] = []
    configuration: Dict[str, Any] = {}
    workflow_id: Optional[PyObjectId] = None
    model_config: AgentConfig = Field(default_factory=AgentConfig)
    state: AgentState = Field(default_factory=lambda: AgentState(status=AgentStatus.INACTIVE))
    organization_id: PyObjectId
    created_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    is_public: bool = False
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    usage_statistics: Dict[str, Any] = {}

# Create/Update Models
class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    agent_type: AgentType
    capabilities: List[AgentCapability] = []
    configuration: Dict[str, Any] = {}
    model_config: Optional[AgentConfig] = None
    is_public: bool = False
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[AgentCapability]] = None
    configuration: Optional[Dict[str, Any]] = None
    model_config: Optional[AgentConfig] = None
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
