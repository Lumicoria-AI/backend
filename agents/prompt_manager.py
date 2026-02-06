"""
Prompt Manager for Lumicoria AI Platform

Manages storage and retrieval of AI prompts for various agent types.
Supports customization and version control of prompts.
"""

from typing import Dict, Any, List, Optional, Union
import json
import os
from datetime import datetime
import structlog
from pydantic import BaseModel, Field

# Configure logger
logger = structlog.get_logger(__name__)

class PromptTemplate(BaseModel):
    """A template for agent prompts with customization options."""
    name: str
    description: str
    template: str
    version: str = "1.0.0"
    agent_type: str
    variables: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Union[str, datetime] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[Union[str, datetime]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PromptManager:
    """
    Manager for AI agent prompts.
    Handles storage, retrieval, and customization of prompts.
    """
    
    def __init__(self, prompts_dir: str = None):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory to store prompt templates
        """
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", 
            "data", 
            "prompts"
        )
        
        # Ensure prompts directory exists
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Default prompts for each agent type
        self.default_prompts: Dict[str, PromptTemplate] = {}
        
        # Load default prompts
        self._load_default_prompts()
    
    def _load_default_prompts(self):
        """Load default prompts for all agent types."""
        # Document Agent prompt
        self.default_prompts["document"] = PromptTemplate(
            name="Document Analysis Agent",
            description="Analyzes documents to extract key information, tasks, dates, and action items",
            template="""You are an AI document analysis specialist. Your task is to carefully analyze the provided document and extract the following information:
1. Key dates and deadlines mentioned in the document
2. Action items and tasks that need to be completed
3. Important people, organizations, and entities mentioned
4. Financial figures and monetary amounts
5. Main topics and themes of the document

For each item you extract, include the exact text from the document and its context.
If the document contains tables, properly interpret and extract their data.
Format your response in a clear, structured manner with appropriate headings for each section.

Document to analyze: {document_text}

Additional context: {context}""",
            agent_type="document",
            variables=["document_text", "context"],
            examples=[
                {
                    "document_text": "Meeting Minutes: Project Alpha\nDate: June 15, 2025\n\nAttendees: John Smith, Alice Johnson, Bob Wilson\n\nAction Items:\n1. John to complete budget report by June 30\n2. Alice to schedule client meeting for next week\n3. Bob to update project timeline by Friday",
                    "context": "Project Alpha is a high-priority software development project",
                    "output": "# Document Analysis\n\n## Key Dates and Deadlines\n- Meeting Date: June 15, 2025\n- Budget report deadline: June 30, 2025\n- Project timeline update: Friday\n- Client meeting: Next week\n\n## Action Items\n1. John Smith to complete budget report by June 30\n2. Alice Johnson to schedule client meeting for next week\n3. Bob Wilson to update project timeline by Friday\n\n## People and Organizations\n- John Smith\n- Alice Johnson\n- Bob Wilson\n- Client (unnamed)\n\n## Project Information\n- Project Name: Alpha\n- Meeting type: Project meeting with minutes"
                }
            ]
        )
        
        # Wellbeing Agent prompt
        self.default_prompts["wellbeing"] = PromptTemplate(
            name="Wellbeing Coach Agent",
            description="Provides personalized wellbeing advice and break recommendations",
            template="""You are a compassionate wellbeing coach focused on helping knowledge workers maintain balance and health. 
Provide personalized suggestions based on the user's data. Include recommendations for:

1. Breaks and rest periods
2. Physical activity and posture
3. Mental health and stress management
4. Focus techniques and productivity
5. Overall wellbeing improvement

Be encouraging and thoughtful in your tone. Consider the user's current activity patterns, metrics, and work context.
Format your response with clear sections and actionable advice.

User data: {user_data}

Additional context: {context}""",
            agent_type="wellbeing",
            variables=["user_data", "context"],
            examples=[
                {
                    "user_data": "Screen time today: 6.5 hours\nLast break: 3 hours ago\nCurrent stress level: High\nRecent activities: coding, virtual meetings\nPosture: Slouching",
                    "context": "Working on an urgent project deadline tomorrow",
                    "output": "# Wellbeing Recommendations\n\n## Break Recommendations\n- Take a 5-minute break now - you haven't had one in 3 hours\n- Use the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds\n- Schedule a 15-minute proper break after your next meeting\n\n## Physical Health\n- Your posture is slouching - adjust your chair height and monitor position\n- Stand up and stretch your shoulders and neck right now\n- Consider a short walk outside to reset your body and mind\n\n## Stress Management\n- Try a 2-minute breathing exercise: 4-count inhale, 4-count exhale\n- Write down your 3 most important tasks to regain focus\n- Break your project into smaller milestones to reduce overwhelm"
                }
            ]
        )
        
        # Meeting Agent prompt
        self.default_prompts["meeting"] = PromptTemplate(
            name="Meeting Agent",
            description="Processes meeting transcripts to extract summaries and action items",
            template="""You are a meeting assistant specialized in analyzing meeting transcripts.
Review the meeting transcript and provide:

1. A concise summary of the key discussion points
2. All action items with assigned persons and deadlines
3. Important decisions made during the meeting
4. Follow-up items and next steps

Format your response clearly with appropriate sections and bullet points.

Meeting transcript: {meeting_text}

Additional context: {context}""",
            agent_type="meeting",
            variables=["meeting_text", "context"],
            examples=[]
        )
        
        # Vision Agent prompt
        self.default_prompts["vision"] = PromptTemplate(
            name="Vision Agent",
            description="Analyzes visual information from camera or images",
            template="""You are a computer vision assistant that analyzes visual information from camera feeds or images.
Carefully examine the visual content and provide:

1. A clear description of what you see
2. Identification of any text content visible in the image
3. Recognition of people, objects, and environmental elements
4. Any potential action items based on the visual information

Be precise and thorough in your analysis.

Image description: {image_description}

Additional context: {context}""",
            agent_type="vision",
            variables=["image_description", "context"],
            examples=[]
        )
        
        # Creative Agent prompt
        self.default_prompts["creative"] = PromptTemplate(
            name="Creative Agent",
            description="Generates creative content based on user inputs and requirements",
            template="""You are a creative assistant specialized in generating high-quality creative content.
Based on the user's request, create:

{creative_request}

Your output should be original, engaging, and tailored to the specific requirements.
Pay close attention to the tone, style, and purpose requested.

Additional context: {context}""",
            agent_type="creative",
            variables=["creative_request", "context"],
            examples=[]
        )
        
        # Student Agent prompt
        self.default_prompts["student"] = PromptTemplate(
            name="Student Agent",
            description="Assists with organizing study materials, assignments, and learning",
            template="""You are an AI study assistant specialized in helping students organize their academic work.
Based on the provided information, help with:

1. Organizing study materials and resources
2. Planning study schedules and prioritizing assignments
3. Tracking deadlines and important academic dates
4. Suggesting effective study techniques and focus methods
5. Providing learning support and clarification on topics

Student data: {student_data}

Additional context: {context}""",
            agent_type="student",
            variables=["student_data", "context"],
            examples=[]
        )
        
        # Save default prompts to disk
        for agent_type, prompt in self.default_prompts.items():
            self._save_prompt(prompt, is_default=True)
    
    def _get_prompt_path(self, agent_type: str, is_default: bool = False) -> str:
        """Get the file path for a prompt template."""
        if is_default:
            return os.path.join(self.prompts_dir, f"{agent_type}_default.json")
        return os.path.join(self.prompts_dir, f"{agent_type}.json")
    
    def _save_prompt(self, prompt: PromptTemplate, is_default: bool = False) -> None:
        """Save a prompt template to disk."""
        path = self._get_prompt_path(prompt.agent_type, is_default)
        
        try:
            with open(path, "w") as f:
                f.write(prompt.json(indent=2))
            logger.info(f"Saved prompt template for {prompt.agent_type}", path=path)
        except Exception as e:
            logger.error(f"Error saving prompt template: {e}", agent_type=prompt.agent_type)
    
    def _load_prompt(self, agent_type: str, use_default: bool = False) -> Optional[PromptTemplate]:
        """Load a prompt template from disk."""
        path = self._get_prompt_path(agent_type, use_default)
        
        if not os.path.exists(path):
            if use_default:
                # If default doesn't exist on disk but we have it in memory
                if agent_type in self.default_prompts:
                    return self.default_prompts[agent_type]
            return None
            
        try:
            with open(path, "r") as f:
                prompt_data = json.load(f)
                return PromptTemplate(**prompt_data)
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}", agent_type=agent_type, path=path)
            return None
    
    def get_prompt(self, agent_type: str, use_default: bool = False) -> Optional[PromptTemplate]:
        """
        Get a prompt template for the specified agent type.
        
        Args:
            agent_type: Type of agent (e.g., "document", "wellbeing")
            use_default: Whether to use the default prompt template
            
        Returns:
            Prompt template if found, None otherwise
        """
        if use_default and agent_type in self.default_prompts:
            return self.default_prompts[agent_type]
        
        # Try loading from disk
        return self._load_prompt(agent_type, use_default)
    
    def create_or_update_prompt(self, prompt: PromptTemplate) -> bool:
        """
        Create or update a prompt template.
        
        Args:
            prompt: Prompt template to create or update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set updated timestamp
            prompt.updated_at = datetime.utcnow().isoformat()
            
            # Save to disk
            self._save_prompt(prompt)
            return True
        except Exception as e:
            logger.error(f"Error creating/updating prompt: {e}", agent_type=prompt.agent_type)
            return False
    
    def fill_prompt_template(self, agent_type: str, variables: Dict[str, Any], use_default: bool = False) -> str:
        """
        Fill a prompt template with variables.
        
        Args:
            agent_type: Type of agent
            variables: Dictionary of variable values
            use_default: Whether to use the default prompt template
            
        Returns:
            Filled prompt template string
        """
        prompt_template = self.get_prompt(agent_type, use_default)
        
        if not prompt_template:
            # If no template found, return a basic prompt
            return f"You are an AI assistant specialized in {agent_type} tasks. " + \
                   f"Please help with the following: {variables.get('task', '')}"
        
        # Start with the template
        filled_prompt = prompt_template.template
        
        # Fill in variables
        for var_name, var_value in variables.items():
            placeholder = "{" + var_name + "}"
            if placeholder in filled_prompt:
                filled_prompt = filled_prompt.replace(placeholder, str(var_value))
        
        # Fill any remaining variables with empty strings
        import re
        filled_prompt = re.sub(r'\{[a-zA-Z0-9_]+\}', '', filled_prompt)
        
        return filled_prompt
    
    def get_agent_types(self) -> List[str]:
        """
        Get a list of all available agent types.
        
        Returns:
            List of agent types
        """
        return list(self.default_prompts.keys())
    
    def reset_prompt(self, agent_type: str) -> bool:
        """
        Reset a prompt to its default version.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            True if successful, False otherwise
        """
        if agent_type not in self.default_prompts:
            return False
            
        try:
            # Copy default prompt
            default_prompt = self.default_prompts[agent_type]
            
            # Save as current prompt
            self._save_prompt(default_prompt)
            return True
        except Exception as e:
            logger.error(f"Error resetting prompt: {e}", agent_type=agent_type)
            return False

# Singleton instance
prompt_manager = PromptManager()
