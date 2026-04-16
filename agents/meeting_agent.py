from .base_agent import BaseAgent
from backend.ai_models import LLMConfig
from typing import Dict, Any, List, Optional
import json
import structlog
import asyncio
from datetime import datetime, timedelta
import re

# Configure logger
logger = structlog.get_logger(__name__)

class MeetingAgent(BaseAgent):
    """Agent for processing meeting transcripts and notes using LLM providers.
    
    This agent extracts key information, action items, decisions, and insights from 
    meeting content and generates structured summaries.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default extraction targets if not specified in config
        self.extraction_targets = config.get("extraction_targets", [
            "action_items", "decisions", "key_points", "follow_ups",
            "questions", "concerns", "deadlines"
        ])

        # Default meeting types if not specified in config
        self.meeting_types = config.get("meeting_types", [
            "status_update", "planning", "brainstorming", "decision_making",
            "problem_solving", "review", "team_building", "client"
        ])
            
    def process(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process meeting transcript to extract key information and generate summary.
        
        Args:
            meeting_data: Dictionary containing meeting transcript, metadata, and context.
            
        Returns:
            Dictionary with meeting summary, action items, decisions, and insights.
        """
        # Extract meeting content and metadata
        transcript = meeting_data.get("transcript", "")
        meeting_metadata = meeting_data.get("metadata", {})
        meeting_context = meeting_data.get("context", {})
        meeting_type = meeting_metadata.get("type", "general")
        participants = meeting_metadata.get("participants", [])
        
        if not transcript:
            return {"error": "No meeting transcript provided"}
            
        # Use the configured model to process the meeting transcript
        prompt = self._create_meeting_prompt(transcript, meeting_type, participants, meeting_context)
        
        try:
            # Process transcript with the prompt
            meeting_response = self._call_model(
                prompt=prompt, 
                model=self.model_config.get("model")
            )
            
            # Parse the response into structured data
            parsed_result = self._parse_meeting_response(meeting_response, meeting_type)
            
            # Create comprehensive response
            result = {
                "meeting_id": meeting_metadata.get("id", ""),
                "summary": parsed_result.get("summary", ""),
                "action_items": parsed_result.get("action_items", []),
                "decisions": parsed_result.get("decisions", []),
                "key_points": parsed_result.get("key_points", []),
                "follow_ups": parsed_result.get("follow_ups", []),
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing meeting transcript: {str(e)}")
            return {"error": f"Meeting processing failed: {str(e)}"}
    
    async def process_async(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process meeting transcript asynchronously with optimized processing.
        
        Args:
            meeting_data: Dictionary containing meeting transcript, metadata, and context.
            
        Returns:
            Dictionary with meeting summary, action items, decisions, and insights.
        """
        # Extract meeting content and metadata
        transcript = meeting_data.get("transcript", "")
        meeting_metadata = meeting_data.get("metadata", {})
        meeting_context = meeting_data.get("context", {})
        meeting_type = meeting_metadata.get("type", "general")
        participants = meeting_metadata.get("participants", [])
        
        if not transcript:
            return {"error": "No meeting transcript provided"}
            
        try:
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
                
            # Create system and user prompts
            system_prompt, user_prompt = self._create_async_prompts(
                transcript, meeting_type, participants, meeting_context
            )
            
            # Format messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM via provider-agnostic interface
            config = LLMConfig(
                model=self.model_config.get("model"),
                temperature=0.3,  # Lower temperature for more accurate extraction
            )
            response = await self.llm_client.generate(messages, config=config)
            
            # Parse the response into structured data
            parsed_result = self._parse_meeting_response(response.content, meeting_type)
            
            # Create comprehensive response
            result = {
                "meeting_id": meeting_metadata.get("id", ""),
                "summary": parsed_result.get("summary", ""),
                "action_items": parsed_result.get("action_items", []),
                "decisions": parsed_result.get("decisions", []),
                "key_points": parsed_result.get("key_points", []),
                "follow_ups": parsed_result.get("follow_ups", []),
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "raw_response": response.content
            }
            
            # Add citations if available (though rarely needed for meeting summaries)
            if response.citations:
                result["citations"] = response.citations
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async meeting processing: {str(e)}")
            return {"error": f"Async meeting processing failed: {str(e)}"}
    
    def _create_meeting_prompt(self, transcript: str, meeting_type: str, 
                           participants: List[str], context: Dict[str, Any]) -> str:
        """Create a specialized prompt for meeting transcript processing."""
        # Base prompt template
        prompt = f"Analyze the following {meeting_type} meeting transcript and extract key information.\n\n"
        
        # Add context if provided
        if context:
            if "project" in context:
                prompt += f"Project context: {context['project']}\n"
            if "previous_meeting" in context:
                prompt += f"Previous meeting summary: {context['previous_meeting']}\n"
            if "goals" in context:
                goals_str = ", ".join(context['goals']) if isinstance(context['goals'], list) else context['goals']
                prompt += f"Meeting goals: {goals_str}\n"
                
        # Add participants if provided
        if participants:
            participants_str = ", ".join(participants)
            prompt += f"\nParticipants: {participants_str}\n"
        
        # Add meeting-type specific instructions
        if meeting_type == "status_update":
            prompt += "\nFocus on progress updates, blockers, and next steps for each project or workstream."
        elif meeting_type == "planning":
            prompt += "\nFocus on goals, strategies, timelines, resource allocation, and risk identification."
        elif meeting_type == "brainstorming":
            prompt += "\nFocus on ideas generated, concept evaluation, and next steps for idea development."
        elif meeting_type == "decision_making":
            prompt += "\nFocus on options presented, criteria discussed, decisions made, and rationales."
        elif meeting_type == "problem_solving":
            prompt += "\nFocus on problems discussed, root causes identified, solutions proposed, and implementation plans."
        elif meeting_type == "review":
            prompt += "\nFocus on accomplishments, challenges, lessons learned, and improvement areas."
        elif meeting_type == "client":
            prompt += "\nFocus on client needs, feedback, concerns, agreements, and relationship development."
        
        # Add extraction targets
        prompt += "\n\nPlease extract and structure the following information:"
        prompt += "\n1. A concise summary of the meeting (3-5 sentences)"
        prompt += "\n2. Key discussion points"
        prompt += "\n3. Decisions made"
        prompt += "\n4. Action items with assignees and deadlines (if mentioned)"
        prompt += "\n5. Follow-up items"
        prompt += "\n6. Questions raised that need answers"
        prompt += "\n7. Any concerns or issues highlighted"
        
        # Add formatting instructions
        prompt += "\n\nFormat the output in JSON with the following structure:"
        prompt += """\n{
    "summary": "concise meeting summary",
    "key_points": ["point 1", "point 2", ...],
    "decisions": ["decision 1", "decision 2", ...],
    "action_items": [
        {"task": "task description", "assignee": "person name", "deadline": "date or timeframe"},
        ...
    ],
    "follow_ups": ["item 1", "item 2", ...],
    "questions": ["question 1", "question 2", ...],
    "concerns": ["concern 1", "concern 2", ...]
}"""
        
        # Add the transcript
        prompt += f"\n\nMeeting Transcript:\n{transcript}"
        
        return prompt
    
    def _create_async_prompts(self, transcript: str, meeting_type: str, 
                            participants: List[str], context: Dict[str, Any]) -> tuple:
        """Create system and user prompts for async processing."""
        # Create appropriate system prompt based on meeting type
        if meeting_type == "status_update":
            system_prompt = (
                "You are an expert meeting analyst specialized in status update meetings. "
                "You extract progress updates, achievements, blockers, and planned next steps for each project or workstream. "
                "Your summaries highlight what's on track, what's at risk, and what needs attention, organizing information "
                "by project or team member to provide a clear status snapshot."
            )
        elif meeting_type == "planning":
            system_prompt = (
                "You are an expert meeting analyst specialized in planning meetings. "
                "You extract goals, strategies, timelines, resource allocations, dependencies, and risk identification. "
                "Your summaries capture both the big picture and tactical details, highlighting decision points, "
                "accountability assignments, and critical path items."
            )
        elif meeting_type == "brainstorming":
            system_prompt = (
                "You are an expert meeting analyst specialized in brainstorming sessions. "
                "You extract and categorize ideas generated, concept evaluations, connections between ideas, "
                "and next steps for idea development. Your summaries preserve the creative energy while "
                "bringing structure to the output, highlighting novel concepts and potential directions."
            )
        elif meeting_type == "decision_making":
            system_prompt = (
                "You are an expert meeting analyst specialized in decision-making meetings. "
                "You extract options presented, evaluation criteria, pros and cons discussed, decisions made, "
                "and their rationales. Your summaries provide a clear record of the decision process, "
                "highlighting the path from alternatives to final decisions."
            )
        elif meeting_type == "problem_solving":
            system_prompt = (
                "You are an expert meeting analyst specialized in problem-solving meetings. "
                "You extract problems discussed, root causes identified, solutions proposed, implementation plans, "
                "and risk mitigations. Your summaries provide a clear progression from problem definition "
                "to selected solution, highlighting decision points and action steps."
            )
        elif meeting_type == "review":
            system_prompt = (
                "You are an expert meeting analyst specialized in review meetings. "
                "You extract accomplishments, challenges, metrics, lessons learned, feedback, and improvement areas. "
                "Your summaries highlight both successes and opportunities, providing balanced retrospective insights "
                "and forward-looking recommendations."
            )
        elif meeting_type == "client":
            system_prompt = (
                "You are an expert meeting analyst specialized in client meetings. "
                "You extract client needs, feedback, concerns, agreements, relationship dynamics, and next steps. "
                "Your summaries capture both the stated requirements and unstated implications, highlighting "
                "commitments made, value delivered, and opportunities for strengthening the relationship."
            )
        else:  # general meeting
            system_prompt = (
                "You are an expert meeting analyst who extracts and organizes key information from meeting transcripts. "
                "You identify main topics discussed, decisions made, action items assigned, questions raised, "
                "and follow-up tasks. Your meeting summaries are concise, well-structured, and highlight the "
                "most important outcomes and next steps."
            )
        
        # Create user prompt
        user_prompt = f"Please analyze this {meeting_type} meeting transcript and extract key information.\n\n"
        
        # Add context if provided
        if context:
            user_prompt += "Context:\n"
            if "project" in context:
                user_prompt += f"- Project: {context['project']}\n"
            if "previous_meeting" in context:
                user_prompt += f"- Previous meeting summary: {context['previous_meeting']}\n"
            if "goals" in context:
                goals_str = ", ".join(context['goals']) if isinstance(context['goals'], list) else context['goals']
                user_prompt += f"- Meeting goals: {goals_str}\n"
            user_prompt += "\n"
            
        # Add participants if provided
        if participants:
            participants_str = ", ".join(participants)
            user_prompt += f"Participants: {participants_str}\n\n"
        
        # Add output format requirement
        user_prompt += """Please format your response as a JSON object with the following structure:
{
    "summary": "concise meeting summary (3-5 sentences)",
    "key_points": ["point 1", "point 2", ...],
    "decisions": ["decision 1", "decision 2", ...],
    "action_items": [
        {"task": "task description", "assignee": "person name", "deadline": "date or timeframe"},
        ...
    ],
    "follow_ups": ["item 1", "item 2", ...],
    "questions": ["question 1", "question 2", ...],
    "concerns": ["concern 1", "concern 2", ...]
}
"""
        
        # Add the transcript
        user_prompt += f"\nMeeting Transcript:\n{transcript}"
        
        return system_prompt, user_prompt
    
    def _parse_meeting_response(self, response_text: str, meeting_type: str) -> Dict[str, Any]:
        """Parse the meeting response into structured data.
        
        Args:
            response_text: Raw text from the model's meeting analysis
            meeting_type: Type of meeting for specialized parsing
            
        Returns:
            Structured dictionary with meeting information
        """
        try:
            # First, try to parse as JSON (since we asked for JSON format)
            # LLMs often wrap JSON in markdown code fences — strip them
            try:
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    # Remove opening fence (```json or ```)
                    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                    # Remove closing fence
                    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
                parsed = json.loads(cleaned)
                # Ensure action_items have the right shape
                if "action_items" in parsed and isinstance(parsed["action_items"], list):
                    for i, item in enumerate(parsed["action_items"]):
                        if isinstance(item, str):
                            parsed["action_items"][i] = {"task": item, "assignee": "Unassigned", "deadline": "No deadline specified"}
                        elif isinstance(item, dict):
                            item.setdefault("assignee", "Unassigned")
                            item.setdefault("deadline", "No deadline specified")
                            item.setdefault("task", "")
                return parsed
            except (json.JSONDecodeError, ValueError):
                # If not JSON, use regex-based parsing
                pass
            
            # Initialize structured result
            result = {
                "summary": "",
                "key_points": [],
                "decisions": [],
                "action_items": [],
                "follow_ups": [],
                "questions": [],
                "concerns": []
            }
            
            # Extract summary
            summary_match = re.search(r"(?:Summary|Meeting Summary|Brief Summary):(.*?)(?:\n\n|\n#|\Z)", 
                                     response_text, re.IGNORECASE | re.DOTALL)
            if summary_match:
                result["summary"] = summary_match.group(1).strip()
            
            # Extract key points/discussion points
            key_points_match = re.search(r"(?:Key Points|Discussion Points|Main Points|Key Discussion Points):(.*?)(?:\n\n|\n#|\Z)", 
                                        response_text, re.IGNORECASE | re.DOTALL)
            if key_points_match:
                points_text = key_points_match.group(1).strip()
                points = re.findall(r"[-•*\d+]\s*(.*?)(?:\n[-•*\d+]|\Z)", points_text + "\n", re.DOTALL)
                result["key_points"] = [point.strip() for point in points if point.strip()]
            
            # Extract decisions
            decisions_match = re.search(r"(?:Decisions|Decisions Made|Decisions Taken|Agreed Upon):(.*?)(?:\n\n|\n#|\Z)", 
                                       response_text, re.IGNORECASE | re.DOTALL)
            if decisions_match:
                decisions_text = decisions_match.group(1).strip()
                decisions = re.findall(r"[-•*\d+]\s*(.*?)(?:\n[-•*\d+]|\Z)", decisions_text + "\n", re.DOTALL)
                result["decisions"] = [decision.strip() for decision in decisions if decision.strip()]
            
            # Extract action items with regex pattern for structured data
            action_items_match = re.search(r"(?:Action Items|Action Points|Tasks|To-Do|Next Steps):(.*?)(?:\n\n|\n#|\Z)", 
                                          response_text, re.IGNORECASE | re.DOTALL)
            if action_items_match:
                actions_text = action_items_match.group(1).strip()
                
                # Try to extract structured action items (task, assignee, deadline)
                action_pattern = r"[-•*\d+]\s*(.*?)(?:(?:--|:|\bby\b|\bassigned to\b|\bresponsible\b|\bowner\b))\s*([^,\n]*)(?:(?:,|--|:|\bby\b|\bdue\b|\bdeadline\b|\btimeframe\b))\s*([^,\n]*)(?:\n[-•*\d+]|\Z)"
                actions = re.finditer(action_pattern, actions_text + "\n", re.IGNORECASE | re.DOTALL)
                
                for action in actions:
                    task = action.group(1).strip()
                    assignee = action.group(2).strip()
                    deadline = action.group(3).strip()
                    
                    if task:  # Ensure task is not empty
                        result["action_items"].append({
                            "task": task,
                            "assignee": assignee if assignee else "Unassigned",
                            "deadline": deadline if deadline else "No deadline specified"
                        })
                
                # If structured extraction failed, fall back to simple list extraction
                if not result["action_items"]:
                    actions = re.findall(r"[-•*\d+]\s*(.*?)(?:\n[-•*\d+]|\Z)", actions_text + "\n", re.DOTALL)
                    for action in actions:
                        if action.strip():
                            # Try to extract assignee and deadline from unstructured text
                            assignee_match = re.search(r"(?:assigned to|responsible|owner)[\s:]+([^,\.]+)", action, re.IGNORECASE)
                            deadline_match = re.search(r"(?:by|due|deadline|timeframe)[\s:]+([^,\.]+)", action, re.IGNORECASE)
                            
                            assignee = assignee_match.group(1).strip() if assignee_match else "Unassigned"
                            deadline = deadline_match.group(1).strip() if deadline_match else "No deadline specified"
                            
                            result["action_items"].append({
                                "task": action.strip(),
                                "assignee": assignee,
                                "deadline": deadline
                            })
            
            # Extract follow-up items
            follow_ups_match = re.search(r"(?:Follow-ups|Follow Up Items|Follow Up Actions|Follow-up Required):(.*?)(?:\n\n|\n#|\Z)", 
                                        response_text, re.IGNORECASE | re.DOTALL)
            if follow_ups_match:
                follow_ups_text = follow_ups_match.group(1).strip()
                follow_ups = re.findall(r"[-•*\d+]\s*(.*?)(?:\n[-•*\d+]|\Z)", follow_ups_text + "\n", re.DOTALL)
                result["follow_ups"] = [item.strip() for item in follow_ups if item.strip()]
            
            # Extract questions
            questions_match = re.search(r"(?:Questions|Open Questions|Questions Raised|Questions to Answer):(.*?)(?:\n\n|\n#|\Z)", 
                                       response_text, re.IGNORECASE | re.DOTALL)
            if questions_match:
                questions_text = questions_match.group(1).strip()
                questions = re.findall(r"[-•*\d+]\s*(.*?)(?:\n[-•*\d+]|\Z)", questions_text + "\n", re.DOTALL)
                result["questions"] = [q.strip() for q in questions if q.strip()]
            
            # Extract concerns
            concerns_match = re.search(r"(?:Concerns|Issues|Risks|Challenges|Concerns Raised):(.*?)(?:\n\n|\n#|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if concerns_match:
                concerns_text = concerns_match.group(1).strip()
                concerns = re.findall(r"[-•*\d+]\s*(.*?)(?:\n[-•*\d+]|\Z)", concerns_text + "\n", re.DOTALL)
                result["concerns"] = [c.strip() for c in concerns if c.strip()]
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing meeting response: {str(e)}")
            return {
                "summary": "Error parsing meeting response.",
                "key_points": [],
                "decisions": [],
                "action_items": [],
                "follow_ups": [],
                "raw_response": response_text,
                "parsing_error": str(e)
            }
    
    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the meeting agent asynchronously.
        
        Args:
            query: The query string about a meeting
            context: Optional context dictionary containing meeting data and metadata
            
        Returns:
            Dictionary containing meeting analysis results
        """
        try:
            if not context or not context.get("transcript"):
                return {"error": "No meeting transcript provided in context"}
            
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
            
            # Get meeting data from context
            transcript = context.get("transcript", "")
            meeting_type = context.get("meeting_type", "general")
            participants = context.get("participants", [])
            meeting_context = context.get("meeting_context", {})
            
            # Create system and user prompts
            system_prompt, user_prompt = self._create_async_prompts(
                transcript, meeting_type, participants, meeting_context
            )
            
            # Add the specific query to the user prompt
            user_prompt = f"{user_prompt}\n\nSpecific query: {query}"
            
            # Format messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM via provider-agnostic interface
            config = LLMConfig(
                model=self.model_config.get("model"),
                temperature=0.3,  # Lower temperature for more accurate extraction
            )
            response = await self.llm_client.generate(messages, config=config)
            
            # Parse the response into structured data
            parsed_result = self._parse_meeting_response(response.content, meeting_type)
            
            # Create comprehensive response
            result = {
                "meeting_id": context.get("meeting_id", ""),
                "query_response": parsed_result.get("summary", ""),
                "action_items": parsed_result.get("action_items", []),
                "decisions": parsed_result.get("decisions", []),
                "key_points": parsed_result.get("key_points", []),
                "follow_ups": parsed_result.get("follow_ups", []),
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "raw_response": response.content
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying meeting agent: {str(e)}")
            return {"error": f"Meeting analysis failed: {str(e)}"}
