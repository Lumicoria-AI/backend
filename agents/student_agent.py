from .base_agent import BaseAgent
from backend.ai_models import LLMConfig
from typing import Dict, Any, List, Optional, Union
import json
import structlog
import asyncio
from datetime import datetime, timedelta
import re

# Configure logger
logger = structlog.get_logger(__name__)

class StudentAgent(BaseAgent):
    """Agent for student learning and academic task management using LLM providers.
    
    This agent helps students organize study materials, track assignments, 
    generate study plans, provide research assistance, and offer learning strategies.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default agent capabilities if not specified in config
        self.capabilities = config.get("capabilities", [
            "assignment_tracking", "study_planning", "concept_explanation",
            "research_assistance", "motivation_support", "exam_preparation"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"  # Use Perplexity's Sonar model

    def process(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process student data to provide academic assistance.
        
        Args:
            student_data: Dictionary containing student context, assignments,
                         study materials, and specific requests.
            
        Returns:
            Dictionary with academic assistance, study plans, and guidance.
        """
        # Extract student context and request type
        student_context = student_data.get("context", {})
        request_type = student_data.get("request_type", "general_assistance")
        content = student_data.get("content", "")
        
        if not content:
            return {"error": "No student content provided"}
            
        try:
            # Select appropriate prompt based on request type
            if request_type == "assignment_help":
                prompt = self._create_assignment_prompt(content, student_context)
            elif request_type == "study_plan":
                prompt = self._create_study_plan_prompt(content, student_context)
            elif request_type == "concept_explanation":
                prompt = self._create_explanation_prompt(content, student_context)
            elif request_type == "research":
                prompt = self._create_research_prompt(content, student_context)
            else:
                # General assistance prompt
                prompt = f"As an academic assistant, help with the following student request. "
                prompt += f"Consider the student's context and provide organized, actionable guidance. "
                prompt += f"\n\nStudent context: {json.dumps(student_context)}\n\nRequest: {content}"
            
            # Use the configured model to process the request
            model_response = self._call_model(
                prompt=prompt, 
                model=self.model_config.get("model")
            )
            
            # Parse response based on request type
            if request_type == "assignment_help":
                parsed_result = self._parse_assignment_help(model_response)
            elif request_type == "study_plan":
                parsed_result = self._parse_study_plan(model_response)
            elif request_type == "concept_explanation":
                parsed_result = self._parse_explanation(model_response)
            elif request_type == "research":
                parsed_result = self._parse_research(model_response)
            else:
                parsed_result = self._parse_general_assistance(model_response)
            
            # Create comprehensive response
            result = {
                "response": parsed_result,
                "raw_response": model_response,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "request_type": request_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing student request: {str(e)}")
            return {"error": f"Student assistance processing failed: {str(e)}"}
    
    async def process_async(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process student data asynchronously with optimized processing.
        
        Args:
            student_data: Dictionary containing student context, assignments, 
                         study materials, and specific requests.
            
        Returns:
            Dictionary with academic assistance, study plans, and guidance.
        """
        # Extract student context and request type
        student_context = student_data.get("context", {})
        request_type = student_data.get("request_type", "general_assistance")
        content = student_data.get("content", "")
        
        if not content:
            return {"error": "No student content provided"}
        
        try:
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
            
            # Select appropriate system prompt and user prompt based on request type
            system_prompt, user_prompt = self._create_async_prompts(request_type, content, student_context)
            
            # Format messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM via provider-agnostic interface
            config = LLMConfig(
                model=self.model_config.get("model"),
                temperature=0.7,
            )
            response = await self.llm_client.generate(messages, config=config)
            
            # Parse response based on request type
            if request_type == "assignment_help":
                parsed_result = self._parse_assignment_help(response.content)
            elif request_type == "study_plan":
                parsed_result = self._parse_study_plan(response.content)
            elif request_type == "concept_explanation":
                parsed_result = self._parse_explanation(response.content)
            elif request_type == "research":
                # For research, we want to include citations
                parsed_result = self._parse_research(response.content)
                
                # For research-specific tasks, we might want to add more context using a specialized call
                if "topic" in content and len(content) < 500:
                    focus_areas = student_context.get("interests", [])
                    focus_str = f"\n\nFocus areas: {', '.join(focus_areas)}" if focus_areas else ""
                    research_messages = [
                        {"role": "system", "content": "You are an expert academic researcher. Provide comprehensive, well-cited research findings."},
                        {"role": "user", "content": f"Conduct detailed academic research on:\n\n{content}{focus_str}"}
                    ]
                    research_config = LLMConfig(
                        model=self.model_config.get("model"),
                        temperature=0.4,
                    )
                    research_result = await self.llm_client.generate(
                        research_messages, config=research_config
                    )
                    
                    # Add the additional research context
                    parsed_result["extended_research"] = research_result.content
                    
                    # Add any citations
                    if research_result.citations:
                        parsed_result["citations"] = research_result.citations
            else:
                parsed_result = self._parse_general_assistance(response.content)
            
            # Create comprehensive response
            result = {
                "response": parsed_result,
                "raw_response": response.content,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "request_type": request_type
            }
            
            # Add citations if available
            if response.citations:
                result["citations"] = response.citations
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async student assistance processing: {str(e)}")
            return {"error": f"Async student assistance processing failed: {str(e)}"}
    
    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the student agent asynchronously.
        
        Args:
            query: The query string for academic assistance
            context: Optional context dictionary containing student data and preferences
            
        Returns:
            Dictionary containing academic assistance and guidance
        """
        try:
            # Ensure LLM client is initialized
            if not self.llm_client:
                self.initialize_models()
                
            if not self.llm_client:
                return {"error": "LLM client not initialized"}
            
            # Get student context from context parameter
            student_context = context.get("context", {}) if context else {}
            request_type = context.get("request_type", "general_assistance") if context else "general_assistance"
            
            # Create system and user prompts
            system_prompt, user_prompt = self._create_async_prompts(request_type, query, student_context)
            
            # Format messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM via provider-agnostic interface
            config = LLMConfig(
                model=self.model_config.get("model"),
                temperature=0.7,
            )
            response = await self.llm_client.generate(messages, config=config)
            
            # Parse response based on request type
            if request_type == "assignment_help":
                parsed_result = self._parse_assignment_help(response.content)
            elif request_type == "study_plan":
                parsed_result = self._parse_study_plan(response.content)
            elif request_type == "concept_explanation":
                parsed_result = self._parse_explanation(response.content)
            elif request_type == "research":
                # For research, we want to include citations
                parsed_result = self._parse_research(response.content)
                
                # For research-specific tasks, add more context using a research call
                if "topic" in query and len(query) < 500:
                    focus_areas = student_context.get("interests", [])
                    focus_str = f"\n\nFocus areas: {', '.join(focus_areas)}" if focus_areas else ""
                    research_messages = [
                        {"role": "system", "content": "You are an expert academic researcher. Provide comprehensive, well-cited research findings."},
                        {"role": "user", "content": f"Conduct detailed academic research on:\n\n{query}{focus_str}"}
                    ]
                    research_config = LLMConfig(
                        model=self.model_config.get("model"),
                        temperature=0.4,
                    )
                    research_result = await self.llm_client.generate(
                        research_messages, config=research_config
                    )
                    
                    # Add the additional research context
                    parsed_result["extended_research"] = research_result.content
                    
                    # Add any citations
                    if research_result.citations:
                        parsed_result["citations"] = research_result.citations
            else:
                parsed_result = self._parse_general_assistance(response.content)
            
            # Create comprehensive response
            result = {
                "response": parsed_result,
                "raw_response": response.content,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "request_type": request_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying student agent: {str(e)}")
            return {"error": f"Academic assistance failed: {str(e)}"}
    
    def _create_assignment_prompt(self, assignment: str, context: Dict[str, Any]) -> str:
        """Create prompt for assignment help."""
        subject = context.get("subject", "this subject")
        level = context.get("level", "college")
        deadline = context.get("deadline", "")
        deadline_str = f" The deadline is {deadline}." if deadline else ""
        
        prompt = (
            f"As an academic assistant for {level} students, help with the following assignment for {subject}. "
            f"Provide a structured approach to completing this assignment, including suggested resources, "
            f"methodology, and key points to address.{deadline_str}\n\n"
            f"Assignment: {assignment}\n\n"
            f"Your response should include:\n"
            f"1. A breakdown of the assignment into manageable steps\n"
            f"2. Key concepts that should be addressed\n"
            f"3. Suggested resources or references\n"
            f"4. A rough outline or approach"
        )
        
        return prompt
    
    def _create_study_plan_prompt(self, content: str, context: Dict[str, Any]) -> str:
        """Create prompt for generating study plans."""
        subjects = context.get("subjects", [])
        subjects_str = ", ".join(subjects) if subjects else "the mentioned subjects"
        time_available = context.get("time_available", "")
        time_str = f" The student has {time_available} available for studying." if time_available else ""
        learning_style = context.get("learning_style", "")
        style_str = f" The student's learning style is {learning_style}." if learning_style else ""
        
        prompt = (
            f"Create a comprehensive study plan for {subjects_str} based on the following request.{time_str}{style_str} "
            f"The plan should be structured, realistic, and include specific techniques for effective learning.\n\n"
            f"Request: {content}\n\n"
            f"Your response should include:\n"
            f"1. A daily/weekly schedule with time allocations\n"
            f"2. Specific study techniques for each subject\n"
            f"3. Resource recommendations\n"
            f"4. Progress tracking methods\n"
            f"5. Break and recovery strategies to maintain well-being"
        )
        
        return prompt
    
    def _create_explanation_prompt(self, concept: str, context: Dict[str, Any]) -> str:
        """Create prompt for concept explanation."""
        level = context.get("level", "college")
        background = context.get("background", "")
        background_str = f" The student has the following background: {background}." if background else ""
        
        prompt = (
            f"Explain the following concept clearly for a {level} level student.{background_str} "
            f"Use analogies, examples, and visual descriptions where helpful. Break down complex ideas "
            f"into manageable components. Cite reliable sources for definitions and key points.\n\n"
            f"Concept to explain: {concept}\n\n"
            f"Your explanation should:\n"
            f"1. Start with a clear definition\n"
            f"2. Provide practical examples\n"
            f"3. Relate it to familiar concepts\n"
            f"4. Address common misconceptions\n"
            f"5. Include relevant formulas or frameworks if applicable"
        )
        
        return prompt
    
    def _create_research_prompt(self, topic: str, context: Dict[str, Any]) -> str:
        """Create prompt for research assistance."""
        depth = context.get("depth", "detailed")
        focus = context.get("focus", [])
        focus_str = f" with special focus on {', '.join(focus)}" if focus else ""
        
        prompt = (
            f"Perform {depth} academic research on the following topic{focus_str}. "
            f"Provide accurate information with proper citations to reliable sources. "
            f"Include multiple perspectives and approaches where relevant.\n\n"
            f"Research topic: {topic}\n\n"
            f"Your research should include:\n"
            f"1. Key findings and concepts\n"
            f"2. Major theories or frameworks\n"
            f"3. Recent developments in this area\n"
            f"4. Contrasting viewpoints\n"
            f"5. Practical applications or implications\n"
            f"6. Reliable sources and citations"
        )
        
        return prompt
    
    def _create_async_prompts(self, request_type: str, content: str, context: Dict[str, Any]) -> tuple:
        """Create system and user prompts for async processing."""
        # Create appropriate system prompt based on request type
        if request_type == "assignment_help":
            system_prompt = (
                "You are an expert academic assistant specializing in helping students with assignments. "
                "Provide clear, structured guidance that helps students understand and approach their assignments "
                "effectively. Offer methodologies, resources, and key points without doing the assignment for them. "
                "Your goal is to enhance their learning while developing their independent academic skills."
            )
            
            subject = context.get("subject", "this subject")
            level = context.get("level", "college")
            deadline = context.get("deadline", "")
            deadline_str = f" The deadline is {deadline}." if deadline else ""
            
            user_prompt = (
                f"I need help with this {level} level {subject} assignment.{deadline_str}\n\n"
                f"Assignment: {content}"
            )
            
        elif request_type == "study_plan":
            system_prompt = (
                "You are a specialized study planning assistant with expertise in educational psychology "
                "and effective learning techniques. Create personalized study plans that are realistic, "
                "structured, and tailored to the student's context. Incorporate evidence-based learning "
                "strategies, spaced repetition, active recall, and well-being considerations."
            )
            
            subjects = context.get("subjects", [])
            subjects_str = ", ".join(subjects) if subjects else "my subjects"
            time_available = context.get("time_available", "")
            time_str = f" I have {time_available} available for studying." if time_available else ""
            learning_style = context.get("learning_style", "")
            style_str = f" My learning style is {learning_style}." if learning_style else ""
            
            user_prompt = (
                f"I need a study plan for {subjects_str}.{time_str}{style_str}\n\n"
                f"Request: {content}"
            )
            
        elif request_type == "concept_explanation":
            system_prompt = (
                "You are an expert educator specializing in explaining complex concepts clearly. "
                "Break down difficult ideas into understandable components using analogies, examples, "
                "and visual descriptions. Tailor your explanations to the student's level and background. "
                "Address common misconceptions and provide citations for key definitions and principles."
            )
            
            level = context.get("level", "college")
            background = context.get("background", "")
            background_str = f" My background is: {background}." if background else ""
            
            user_prompt = (
                f"Please explain this concept at a {level} level.{background_str}\n\n"
                f"Concept: {content}"
            )
            
        elif request_type == "research":
            system_prompt = (
                "You are a thorough academic researcher with access to a wide range of scholarly sources. "
                "Conduct comprehensive research on topics, providing accurate information with proper citations. "
                "Present multiple perspectives, recent developments, and practical applications. "
                "Your responses should be balanced, nuanced, and well-supported by evidence."
            )
            
            depth = context.get("depth", "detailed")
            focus = context.get("focus", [])
            focus_str = f" with special focus on {', '.join(focus)}" if focus else ""
            
            user_prompt = (
                f"I need {depth} academic research on this topic{focus_str}.\n\n"
                f"Research topic: {content}"
            )
            
        else:  # general_assistance
            system_prompt = (
                "You are a comprehensive academic assistant helping students succeed in their educational journey. "
                "Provide clear, actionable guidance tailored to the student's specific needs. Incorporate evidence-based "
                "learning strategies, practical advice, and empathetic support. Your goal is to enhance their learning experience "
                "while developing their independent academic skills and well-being."
            )
            
            user_prompt = f"I need academic assistance with the following:\n\n{content}"
        
        return system_prompt, user_prompt
        
    def _parse_assignment_help(self, response_text: str) -> Dict[str, Any]:
        """Parse the assignment help response into structured format."""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, use section-based parsing
                pass
                
            result = {
                "steps": [],
                "key_concepts": [],
                "resources": [],
                "outline": []
            }
            
            # Extract steps
            steps_match = re.search(r"(?:Steps|Breakdown|Steps to complete|Approach):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                   response_text, re.IGNORECASE | re.DOTALL)
            if steps_match:
                steps_text = steps_match.group(1).strip()
                steps = re.findall(r"\d+\.\s*(.*?)(?:\n\d+\.|\Z)", steps_text + "\n", re.DOTALL)
                result["steps"] = [step.strip() for step in steps if step.strip()]
                
            # Extract key concepts
            concepts_match = re.search(r"(?:Key Concepts|Important Concepts|Concepts to Address|Key Points):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if concepts_match:
                concepts_text = concepts_match.group(1).strip()
                concepts = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", concepts_text + "\n", re.DOTALL)
                result["key_concepts"] = [concept.strip() for concept in concepts if concept.strip()]
                
            # Extract resources
            resources_match = re.search(r"(?:Resources|References|Suggested Resources|Helpful Resources):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                       response_text, re.IGNORECASE | re.DOTALL)
            if resources_match:
                resources_text = resources_match.group(1).strip()
                resources = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", resources_text + "\n", re.DOTALL)
                result["resources"] = [resource.strip() for resource in resources if resource.strip()]
                
            # Extract outline
            outline_match = re.search(r"(?:Outline|Structure|Framework|Organization):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                     response_text, re.IGNORECASE | re.DOTALL)
            if outline_match:
                outline_text = outline_match.group(1).strip()
                outline = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", outline_text + "\n", re.DOTALL)
                result["outline"] = [item.strip() for item in outline if item.strip()]
            
            return result
        except Exception as e:
            logger.error(f"Error parsing assignment help: {str(e)}")
            return {"error": str(e), "raw_text": response_text}
    
    def _parse_study_plan(self, response_text: str) -> Dict[str, Any]:
        """Parse the study plan response into structured format."""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, use section-based parsing
                pass
                
            result = {
                "schedule": [],
                "techniques": [],
                "resources": [],
                "tracking_methods": [],
                "breaks_and_recovery": []
            }
            
            # Extract schedule
            schedule_match = re.search(r"(?:Schedule|Study Schedule|Time Plan|Weekly Plan|Daily Plan):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if schedule_match:
                schedule_text = schedule_match.group(1).strip()
                schedule = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", schedule_text + "\n", re.DOTALL)
                result["schedule"] = [item.strip() for item in schedule if item.strip()]
            
            # Extract techniques
            techniques_match = re.search(r"(?:Techniques|Study Techniques|Learning Methods|Study Methods|Study Strategies):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                        response_text, re.IGNORECASE | re.DOTALL)
            if techniques_match:
                techniques_text = techniques_match.group(1).strip()
                techniques = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", techniques_text + "\n", re.DOTALL)
                result["techniques"] = [item.strip() for item in techniques if item.strip()]
            
            # Extract resources
            resources_match = re.search(r"(?:Resources|Recommended Resources|Study Materials|Materials):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                       response_text, re.IGNORECASE | re.DOTALL)
            if resources_match:
                resources_text = resources_match.group(1).strip()
                resources = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", resources_text + "\n", re.DOTALL)
                result["resources"] = [item.strip() for item in resources if item.strip()]
            
            # Extract tracking methods
            tracking_match = re.search(r"(?:Tracking|Progress Tracking|Monitoring Progress|Progress Methods):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if tracking_match:
                tracking_text = tracking_match.group(1).strip()
                tracking = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", tracking_text + "\n", re.DOTALL)
                result["tracking_methods"] = [item.strip() for item in tracking if item.strip()]
            
            # Extract break strategies
            breaks_match = re.search(r"(?:Breaks|Break Strategies|Recovery|Rest Periods|Well-being):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                    response_text, re.IGNORECASE | re.DOTALL)
            if breaks_match:
                breaks_text = breaks_match.group(1).strip()
                breaks = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", breaks_text + "\n", re.DOTALL)
                result["breaks_and_recovery"] = [item.strip() for item in breaks if item.strip()]
            
            return result
        except Exception as e:
            logger.error(f"Error parsing study plan: {str(e)}")
            return {"error": str(e), "raw_text": response_text}
    
    def _parse_explanation(self, response_text: str) -> Dict[str, Any]:
        """Parse the concept explanation response into structured format."""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, use section-based parsing
                pass
                
            result = {
                "definition": "",
                "examples": [],
                "related_concepts": [],
                "misconceptions": [],
                "formulas": []
            }
            
            # Extract definition
            definition_match = re.search(r"(?:Definition|What is it\?|Concept Definition|Introduction):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                        response_text, re.IGNORECASE | re.DOTALL)
            if definition_match:
                result["definition"] = definition_match.group(1).strip()
            
            # Extract examples
            examples_match = re.search(r"(?:Examples|Example|Practical Examples|Real-world Examples):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if examples_match:
                examples_text = examples_match.group(1).strip()
                examples = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", examples_text + "\n", re.DOTALL)
                result["examples"] = [item.strip() for item in examples if item.strip()]
            
            # Extract related concepts
            related_match = re.search(r"(?:Related Concepts|Similar Concepts|Connections|Related Topics):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                     response_text, re.IGNORECASE | re.DOTALL)
            if related_match:
                related_text = related_match.group(1).strip()
                related = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", related_text + "\n", re.DOTALL)
                result["related_concepts"] = [item.strip() for item in related if item.strip()]
            
            # Extract misconceptions
            misconceptions_match = re.search(r"(?:Misconceptions|Common Misconceptions|Mistakes|Common Errors):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                           response_text, re.IGNORECASE | re.DOTALL)
            if misconceptions_match:
                misconceptions_text = misconceptions_match.group(1).strip()
                misconceptions = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", misconceptions_text + "\n", re.DOTALL)
                result["misconceptions"] = [item.strip() for item in misconceptions if item.strip()]
            
            # Extract formulas
            formulas_match = re.search(r"(?:Formulas|Equations|Frameworks|Key Formulas):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if formulas_match:
                formulas_text = formulas_match.group(1).strip()
                formulas = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", formulas_text + "\n", re.DOTALL)
                result["formulas"] = [item.strip() for item in formulas if item.strip()]
            
            return result
        except Exception as e:
            logger.error(f"Error parsing explanation: {str(e)}")
            return {"error": str(e), "raw_text": response_text}
    
    def _parse_research(self, response_text: str) -> Dict[str, Any]:
        """Parse the research response into structured format."""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, use section-based parsing
                pass
                
            result = {
                "key_findings": [],
                "theories": [],
                "developments": [],
                "viewpoints": [],
                "applications": [],
                "sources": []
            }
            
            # Extract key findings
            findings_match = re.search(r"(?:Key Findings|Main Findings|Important Findings|Findings):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if findings_match:
                findings_text = findings_match.group(1).strip()
                findings = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", findings_text + "\n", re.DOTALL)
                result["key_findings"] = [item.strip() for item in findings if item.strip()]
            
            # Extract theories
            theories_match = re.search(r"(?:Theories|Frameworks|Major Theories|Theoretical Frameworks):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                      response_text, re.IGNORECASE | re.DOTALL)
            if theories_match:
                theories_text = theories_match.group(1).strip()
                theories = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", theories_text + "\n", re.DOTALL)
                result["theories"] = [item.strip() for item in theories if item.strip()]
            
            # Extract recent developments
            developments_match = re.search(r"(?:Recent Developments|Current Research|Recent Advances|New Developments):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                          response_text, re.IGNORECASE | re.DOTALL)
            if developments_match:
                developments_text = developments_match.group(1).strip()
                developments = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", developments_text + "\n", re.DOTALL)
                result["developments"] = [item.strip() for item in developments if item.strip()]
            
            # Extract contrasting viewpoints
            viewpoints_match = re.search(r"(?:Contrasting Viewpoints|Different Perspectives|Opposing Views|Debates):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                        response_text, re.IGNORECASE | re.DOTALL)
            if viewpoints_match:
                viewpoints_text = viewpoints_match.group(1).strip()
                viewpoints = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", viewpoints_text + "\n", re.DOTALL)
                result["viewpoints"] = [item.strip() for item in viewpoints if item.strip()]
            
            # Extract practical applications
            applications_match = re.search(r"(?:Practical Applications|Applications|Implications|Real-world Applications):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                          response_text, re.IGNORECASE | re.DOTALL)
            if applications_match:
                applications_text = applications_match.group(1).strip()
                applications = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", applications_text + "\n", re.DOTALL)
                result["applications"] = [item.strip() for item in applications if item.strip()]
            
            # Extract sources
            sources_match = re.search(r"(?:Sources|References|Citations|Works Cited|Bibliography):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                     response_text, re.IGNORECASE | re.DOTALL)
            if sources_match:
                sources_text = sources_match.group(1).strip()
                sources = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", sources_text + "\n", re.DOTALL)
                result["sources"] = [item.strip() for item in sources if item.strip()]
            
            return result
        except Exception as e:
            logger.error(f"Error parsing research: {str(e)}")
            return {"error": str(e), "raw_text": response_text}
    
    def _parse_general_assistance(self, response_text: str) -> Dict[str, Any]:
        """Parse general assistance response to extract structured information."""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, return a simple structured format
                pass
                
            # For general assistance, we'll just do basic section extraction
            result = {
                "main_points": [],
                "recommendations": [],
                "resources": []
            }
            
            # Extract main points - look for bullet points or numbered lists
            main_points = re.findall(r"[•\-\d+][.)]?\s+(.*?)(?:\n[•\-\d+][.)]?|\Z)", response_text, re.DOTALL)
            if main_points:
                result["main_points"] = [point.strip() for point in main_points if point.strip()]
            
            # Extract recommendations if they exist
            recommendations_match = re.search(r"(?:Recommend|Suggestion|Advice|Tips?|Strategies?)s?:(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                            response_text, re.IGNORECASE | re.DOTALL)
            if recommendations_match:
                recommendations_text = recommendations_match.group(1).strip()
                recommendations = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", recommendations_text + "\n", re.DOTALL)
                result["recommendations"] = [item.strip() for item in recommendations if item.strip()]
            
            # Extract resources if they exist
            resources_match = re.search(r"(?:Resources?|References?|Materials?|Tools?):(.*?)(?:\n\n|\n[A-Z]|\Z)", 
                                       response_text, re.IGNORECASE | re.DOTALL)
            if resources_match:
                resources_text = resources_match.group(1).strip()
                resources = re.findall(r"[•\-\d*]\s*(.*?)(?:\n[•\-\d*]|\Z)", resources_text + "\n", re.DOTALL)
                result["resources"] = [item.strip() for item in resources if item.strip()]
            
            # If we couldn't extract structured information, set the full text as content
            if not any(result.values()):
                result["content"] = response_text
            
            return result
        except Exception as e:
            logger.error(f"Error parsing general assistance: {str(e)}")
            return {"content": response_text}