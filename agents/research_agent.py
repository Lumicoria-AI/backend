from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional
import json
import structlog
import asyncio
from datetime import datetime
import re

# Configure logger
logger = structlog.get_logger(__name__)

class ResearchAgent(BaseAgent):
    """Agent for conducting deep research using Perplexity AI.
    
    This agent performs comprehensive research across various topics,
    leverages citations, compares information sources, and provides
    evidence-based responses with academic rigor.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default research capabilities if not specified in config
        self.research_capabilities = config.get("research_capabilities", [
            "topic_research", "literature_review", "fact_checking", 
            "source_comparison", "question_answering", "citation_analysis"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"
        
        # Set research depth
        self.research_depth = config.get("research_depth", "comprehensive")
        
        # Set citation requirements
        self.require_citations = config.get("require_citations", True)
    
    def process(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research request to generate findings.
        
        Args:
            research_data: Dictionary containing research query, context,
                          depth, and other parameters.
            
        Returns:
            Dictionary with research findings, citations, and metadata.
        """
        # Extract request parameters
        query = research_data.get("query", "")
        context = research_data.get("context", {})
        research_type = research_data.get("research_type", "general")
        depth = research_data.get("depth", self.research_depth)
        
        if not query:
            return {"error": "No research query provided"}
            
        # Use the configured model to process the research request
        prompt = self._create_research_prompt(query, context, research_type, depth)
        
        try:
            # Get research findings based on prompt
            result = self._call_model(
                prompt=prompt, 
                model=self.model_config.get("model")
            )
            
            # Parse the findings
            parsed_findings = self._parse_research_findings(result, research_type)
            
            # Create comprehensive response
            response = {
                "findings": parsed_findings,
                "raw_response": result,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "research_type": research_type,
                "query": query
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing research request: {str(e)}")
            return {"error": f"Research processing failed: {str(e)}"}
    
    async def process_async(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research request asynchronously with optimized processing.
        
        Args:
            research_data: Dictionary containing research query, context,
                          depth, and other parameters.
            
        Returns:
            Dictionary with research findings, citations, and metadata.
        """
        # Extract request parameters
        query = research_data.get("query", "")
        context = research_data.get("context", {})
        research_type = research_data.get("research_type", "general")
        depth = research_data.get("depth", self.research_depth)
        focus_areas = research_data.get("focus_areas", [])
        
        if not query:
            return {"error": "No research query provided"}
            
        try:
            # Ensure Perplexity client is initialized
            if not self.perplexity_client:
                self.initialize_models()
                
            if not self.perplexity_client:
                return {"error": "Perplexity client not initialized"}
            
            # For full research, we might want to run multiple queries and merge results
            if research_type == "comprehensive" and depth == "deep":
                try:
                    # Determine research sub-questions
                    system_prompt = (
                        "You are an expert research planner. Your task is to break down a main research "
                        "question into 3-5 targeted sub-questions that will help provide a comprehensive "
                        "answer to the main question. Focus on different aspects, perspectives, or components "
                        "of the main topic."
                    )
                    
                    user_prompt = f"Main research question: {query}\n\nPlease provide 3-5 focused sub-questions that would help thoroughly research this topic."
                    
                    # Format messages for planning query
                    planning_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # Get research plan
                    planning_response = await self.perplexity_client.chat_completion(
                        messages=planning_messages,
                        model=self.model_config.get("model"),
                        temperature=0.3  # Lower temperature for more focused planning
                    )
                    
                    # Extract sub-questions
                    sub_questions = self._extract_sub_questions(planning_response.content)
                    
                    # Run parallel research on main question and each sub-question
                    research_tasks = []
                    
                    # Main question research
                    main_research_task = self.perplexity_client.academic_research(
                        query=query,
                        depth=depth,
                        focus_areas=focus_areas
                    )
                    research_tasks.append(main_research_task)
                    
                    # Sub-questions research
                    for sub_q in sub_questions[:3]:  # Limit to 3 sub-questions
                        sub_research_task = self.perplexity_client.academic_research(
                            query=sub_q,
                            depth="focused",  # Use focused depth for sub-questions
                            focus_areas=focus_areas
                        )
                        research_tasks.append(sub_research_task)
                    
                    # Wait for all research tasks to complete
                    research_responses = await asyncio.gather(*research_tasks)
                    
                    # Synthesize findings
                    synthesis_system_prompt = (
                        "You are an expert research synthesizer. Your task is to combine and integrate "
                        "findings from multiple research queries into a cohesive, comprehensive research report. "
                        "Organize the information logically, eliminate redundancies, highlight key findings, "
                        "identify patterns or contradictions, and ensure proper attribution to sources."
                    )
                    
                    synthesis_user_prompt = (
                        f"Main research question: {query}\n\n"
                        "Please synthesize the following research findings into a comprehensive report:\n\n"
                    )
                    
                    for i, resp in enumerate(research_responses):
                        q = query if i == 0 else sub_questions[i-1]
                        synthesis_user_prompt += f"Research on: {q}\n\nFindings: {resp.content}\n\n"
                        
                    synthesis_messages = [
                        {"role": "system", "content": synthesis_system_prompt},
                        {"role": "user", "content": synthesis_user_prompt}
                    ]
                    
                    synthesis_response = await self.perplexity_client.chat_completion(
                        messages=synthesis_messages,
                        model=self.model_config.get("model"),
                        temperature=0.4
                    )
                    
                    # Collect all citations from all responses
                    all_citations = []
                    for resp in research_responses:
                        if hasattr(resp, "citations") and resp.citations:
                            all_citations.extend(resp.citations)
                    
                    # Parse the synthesized findings
                    parsed_findings = self._parse_research_findings(synthesis_response.content, "comprehensive")
                    
                    # Create comprehensive response with all citations
                    response = {
                        "findings": parsed_findings,
                        "raw_response": synthesis_response.content,
                        "processed_at": datetime.utcnow().isoformat(),
                        "model_used": self.model_config.get("model"),
                        "research_type": research_type,
                        "query": query,
                        "sub_questions": sub_questions,
                        "citations": [
                            {
                                "text": citation.text,
                                "url": citation.metadata.url,
                                "title": citation.metadata.title
                            }
                            for citation in all_citations
                        ] if all_citations else []
                    }
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Error in comprehensive research: {str(e)}")
                    # Fall back to standard research if comprehensive approach fails
                    pass
            
            # Standard research approach
            system_prompt, user_prompt = self._create_async_prompts(query, context, research_type, depth, focus_areas)
            
            # Format messages for Perplexity API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call Perplexity API for research
            response = await self.perplexity_client.chat_completion(
                messages=messages,
                model=self.model_config.get("model"),
                temperature=0.5
            )
            
            # Parse the findings
            parsed_findings = self._parse_research_findings(response.content, research_type)
            
            # Create comprehensive response
            result = {
                "findings": parsed_findings,
                "raw_response": response.content,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "research_type": research_type,
                "query": query
            }
            
            # Add citations if available
            if hasattr(response, "citations") and response.citations:
                result["citations"] = [
                    {
                        "text": citation.text,
                        "url": citation.metadata.url,
                        "title": citation.metadata.title
                    }
                    for citation in response.citations
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async research processing: {str(e)}")
            return {"error": f"Async research processing failed: {str(e)}"}
    
    def _create_research_prompt(self, query: str, context: Dict[str, Any], 
                               research_type: str, depth: str) -> str:
        """Create prompt for conducting research."""
        # Base prompt template
        prompt = f"Conduct {depth} research on the following query: {query}\n\n"
        
        # Add context if provided
        if context:
            context_str = json.dumps(context)
            prompt += f"Consider the following context: {context_str}\n\n"
        
        # Add research type specific instructions
        if research_type == "topic_research":
            prompt += (
                "Provide a comprehensive overview of this topic, including key concepts, "
                "important facts, relevant theories, current developments, and different perspectives. "
                "Include proper citations for all information sources."
            )
        elif research_type == "literature_review":
            prompt += (
                "Conduct a thorough literature review on this topic. Identify key publications, "
                "prominent researchers, major theories, research trends, methodologies used, "
                "significant findings, and gaps in the literature. "
                "Organize by themes or chronologically as appropriate. "
                "Include proper citations for all referenced works."
            )
        elif research_type == "fact_checking":
            prompt += (
                "Fact-check the following query by researching reliable sources. "
                "Determine the accuracy of the statement(s), provide evidence supporting or refuting "
                "the claim(s), note any missing context or nuances, and assess overall veracity. "
                "Include proper citations for all sources used in verification."
            )
        elif research_type == "source_comparison":
            prompt += (
                "Compare and contrast information from multiple sources on this topic. "
                "Identify areas of consensus, conflicting information, different perspectives, "
                "and assess the credibility and potential biases of each source. "
                "Present a balanced synthesis of the information. "
                "Include proper citations for all sources compared."
            )
        elif research_type == "question_answering":
            prompt += (
                "Research this question thoroughly and provide a comprehensive, evidence-based answer. "
                "Address all aspects of the question, consider different perspectives, "
                "acknowledge any limitations or uncertainties in the available information, "
                "and provide proper citations for all sources used."
            )
        else:  # general research
            prompt += (
                "Conduct thorough research on this topic. Provide key findings, relevant context, "
                "different perspectives, and evidence-based conclusions. "
                "Include proper citations for all information sources."
            )
        
        # Add depth-specific instructions
        if depth == "deep" or depth == "comprehensive":
            prompt += (
                "\n\nThis should be deep, comprehensive research using multiple high-quality sources. "
                "Go beyond surface-level information to provide nuanced insights and analysis."
            )
        elif depth == "focused":
            prompt += (
                "\n\nThis should be focused research that addresses the specific question/topic "
                "with precision and clarity, using reliable sources."
            )
        
        # Add output format instructions
        prompt += (
            "\n\nFormat the response with clear sections including:\n"
            "1. Executive Summary/Key Findings\n"
            "2. Main Research Content (organized by relevant themes or aspects)\n"
            "3. Different Perspectives/Debates (if applicable)\n"
            "4. Limitations of Current Knowledge\n"
            "5. Citations/Sources"
        )
        
        return prompt
    
    def _create_async_prompts(self, query: str, context: Dict[str, Any], 
                            research_type: str, depth: str, focus_areas: List[str]) -> tuple:
        """Create system and user prompts for async processing."""
        # Create appropriate system prompt based on research type
        if research_type == "topic_research":
            system_prompt = (
                "You are an expert research specialist who conducts comprehensive investigations into topics. "
                "You thoroughly explore subjects from multiple angles, drawing on diverse, high-quality sources. "
                "Your research is well-organized, balanced, and properly cited, providing readers with "
                "a complete understanding of the topic including key concepts, historical context, "
                "current developments, and various perspectives."
            )
        elif research_type == "literature_review":
            system_prompt = (
                "You are an academic literature review specialist who synthesizes scholarly research on specific topics. "
                "You identify key publications, prominent researchers, theoretical frameworks, methodological approaches, "
                "research trends, significant findings, and gaps in knowledge. Your reviews are systematic, critically "
                "evaluate the quality of evidence, and properly cite all scholarly sources."
            )
        elif research_type == "fact_checking":
            system_prompt = (
                "You are an expert fact-checker who verifies claims using reliable sources. "
                "You investigate statements methodically, trace information to primary sources whenever possible, "
                "assess the accuracy and context of claims, and provide a clear evaluation with supporting evidence. "
                "Your assessments are balanced, nuanced, and properly sourced."
            )
        elif research_type == "source_comparison":
            system_prompt = (
                "You are a source comparison specialist who analyzes how different information sources cover a topic. "
                "You compare and contrast content, identify consensus and contradictions, evaluate source credibility "
                "and potential biases, and synthesize information to provide a balanced understanding. "
                "Your analysis is thorough, fair, and properly attributes all sources."
            )
        elif research_type == "question_answering":
            system_prompt = (
                "You are an expert research analyst who provides comprehensive, evidence-based answers to complex questions. "
                "You investigate questions thoroughly using reliable sources, address all aspects of the query, "
                "consider different perspectives, acknowledge limitations in available information, "
                "and clearly cite all sources used to formulate your response."
            )
        elif research_type == "citation_analysis":
            system_prompt = (
                "You are a citation analysis specialist who examines the influence and relationships between academic works. "
                "You identify key papers, researchers, and institutions in a field, trace the development of ideas "
                "through citation patterns, and evaluate the impact of specific works on knowledge development. "
                "Your analysis is data-driven, precise, and properly references all sources."
            )
        else:  # general research
            system_prompt = (
                "You are a comprehensive research specialist who conducts thorough investigations across diverse topics. "
                "You systematically explore subjects using multiple reliable sources, organize findings logically, "
                "present balanced perspectives, draw evidence-based conclusions, and provide proper citations "
                "for all information presented."
            )
        
        # Create user prompt
        user_prompt = f"Research query: {query}\n\n"
        
        # Add context if provided
        if context:
            context_str = json.dumps(context)
            user_prompt += f"Context: {context_str}\n\n"
        
        # Add depth specification
        user_prompt += f"Research depth: {depth}\n"
        
        # Add focus areas if specified
        if focus_areas:
            focus_areas_str = ", ".join(focus_areas)
            user_prompt += f"Focus especially on: {focus_areas_str}\n"
        
        # Add output format requirements
        user_prompt += (
            "\nPlease provide your findings in a well-structured format including:\n"
            "1. Executive Summary/Key Findings\n"
            "2. Main Research Content (organized by relevant themes)\n"
            "3. Different Perspectives/Debates\n"
            "4. Limitations of Current Knowledge\n"
            "5. Citations/Sources"
        )
        
        return system_prompt, user_prompt
    
    def _extract_sub_questions(self, planning_content: str) -> List[str]:
        """Extract sub-questions from planning response."""
        # Pattern matching approach
        questions = []
        
        # Try to match numbered questions first (1. Question)
        numbered_pattern = r"^\s*\d+\.?\s*(.+?)\s*$"
        matches = re.findall(numbered_pattern, planning_content, re.MULTILINE)
        if matches:
            for match in matches:
                # Check if it looks like a question
                if match.strip() and ("?" in match or re.match(r"^(how|what|why|when|where|who|which)", match.lower())):
                    questions.append(match.strip())
        
        # If no numbered questions found, try bullet points
        if not questions:
            bullet_pattern = r"^\s*[-•*]\s*(.+?)\s*$"
            matches = re.findall(bullet_pattern, planning_content, re.MULTILINE)
            if matches:
                for match in matches:
                    if match.strip() and ("?" in match or re.match(r"^(how|what|why|when|where|who|which)", match.lower())):
                        questions.append(match.strip())
        
        # If still no questions found, try to split by lines and look for question marks
        if not questions:
            for line in planning_content.split("\n"):
                line = line.strip()
                if "?" in line:
                    # Extract just the question part
                    question_part = line.split("?")[0] + "?"
                    questions.append(question_part)
        
        # If we have too many questions, limit to most relevant ones
        if len(questions) > 5:
            questions = questions[:5]
        
        # If we have no questions, create generic sub-questions
        if not questions:
            questions = [
                "What are the key concepts and definitions related to this topic?",
                "What are the historical developments and current state of research on this topic?",
                "What are the different perspectives or debates surrounding this topic?"
            ]
        
        return questions
    
    def _parse_research_findings(self, findings_text: str, research_type: str) -> Dict[str, Any]:
        """Parse the research findings into a structured format."""
        try:
            # Try to parse as JSON first
            try:
                return json.loads(findings_text)
            except json.JSONDecodeError:
                # If not JSON, use section-based parsing
                pass
                
            result = {
                "executive_summary": "",
                "key_findings": [],
                "main_content": {},
                "perspectives": [],
                "limitations": [],
                "sources": []
            }
            
            # Extract executive summary
            summary_match = re.search(r"(?:Executive Summary|Summary|Key Findings|Overview):(.*?)(?:\n\n|\n#|\Z)", 
                                     findings_text, re.IGNORECASE | re.DOTALL)
            if summary_match:
                result["executive_summary"] = summary_match.group(1).strip()
            
            # Extract key findings as bullet points
            findings_match = re.search(r"(?:Key Findings|Main Findings|Key Points|Primary Discoveries):(.*?)(?:\n\n|\n#|\Z)", 
                                      findings_text, re.IGNORECASE | re.DOTALL)
            if findings_match:
                findings_text = findings_match.group(1).strip()
                findings = re.findall(r"[-•*]\s*(.*?)(?:\n[-•*]|\Z)", findings_text + "\n", re.DOTALL)
                result["key_findings"] = [finding.strip() for finding in findings if finding.strip()]
            
            # Extract main content sections
            main_content = {}
            section_pattern = r"(?:^|\n)(?:#{1,3}\s+|)([A-Z][^\n]+)(?:\n|:)(.*?)(?=(?:^|\n)(?:#{1,3}\s+|)[A-Z][^\n]+(?:\n|:)|\Z)"
            sections = re.finditer(section_pattern, findings_text, re.DOTALL)
            
            for section in sections:
                section_title = section.group(1).strip()
                section_content = section.group(2).strip()
                
                # Skip if this is one of our predefined sections
                if any(keyword in section_title.lower() for keyword in ["summary", "key finding", "perspective", 
                                                                        "limitation", "source", "citation", 
                                                                        "reference", "executive"]):
                    continue
                
                main_content[section_title] = section_content
            
            result["main_content"] = main_content
            
            # Extract different perspectives/debates
            perspectives_match = re.search(r"(?:Different Perspectives|Perspectives|Debates|Contrasting Views|Viewpoints):(.*?)(?:\n\n|\n#|\Z)", 
                                          findings_text, re.IGNORECASE | re.DOTALL)
            if perspectives_match:
                perspectives_text = perspectives_match.group(1).strip()
                perspectives = re.findall(r"[-•*]\s*(.*?)(?:\n[-•*]|\Z)", perspectives_text + "\n", re.DOTALL)
                result["perspectives"] = [p.strip() for p in perspectives if p.strip()]
            
            # Extract limitations
            limitations_match = re.search(r"(?:Limitations|Gaps|Challenges|Limitations of Current Knowledge):(.*?)(?:\n\n|\n#|\Z)", 
                                         findings_text, re.IGNORECASE | re.DOTALL)
            if limitations_match:
                limitations_text = limitations_match.group(1).strip()
                limitations = re.findall(r"[-•*]\s*(.*?)(?:\n[-•*]|\Z)", limitations_text + "\n", re.DOTALL)
                result["limitations"] = [l.strip() for l in limitations if l.strip()]
            
            # Extract sources/citations
            sources_match = re.search(r"(?:Citations|Sources|References|Bibliography):(.*?)(?:\n\n|\n#|\Z)", 
                                     findings_text, re.IGNORECASE | re.DOTALL)
            if sources_match:
                sources_text = sources_match.group(1).strip()
                # Try to extract structured citations
                sources = re.findall(r"[-•*\d+\.]\s*(.*?)(?:\n[-•*\d+\.]|\Z)", sources_text + "\n", re.DOTALL)
                result["sources"] = [s.strip() for s in sources if s.strip()]
                
                # If no structured citations found, just include the whole text
                if not result["sources"]:
                    result["sources"] = [sources_text]
            
            # If main_content is empty, try a different approach based on research type
            if not result["main_content"]:
                if research_type == "topic_research":
                    # Extract sections based on common topic research headers
                    headers = ["Background", "Key Concepts", "Historical Context", "Current Developments", 
                               "Applications", "Significance", "Future Directions"]
                elif research_type == "literature_review":
                    # Extract sections based on common literature review headers
                    headers = ["Theoretical Framework", "Methodology", "Key Studies", "Research Trends", 
                               "Gaps in Literature", "Methodological Approaches"]
                elif research_type == "fact_checking":
                    # Extract sections based on common fact-checking headers
                    headers = ["Claim Analysis", "Evidence", "Context", "Verification", "Assessment"]
                else:
                    # Generic headers
                    headers = ["Background", "Analysis", "Discussion", "Implications", "Conclusion"]
                
                # Try to extract each section
                for header in headers:
                    pattern = r"(?:^|\n)(?:#{1,3}\s+|)(" + header + r"[^\n]*?)(?:\n|:)(.*?)(?=(?:^|\n)(?:#{1,3}\s+|)[A-Z][^\n]+(?:\n|:)|\Z)"
                    section_match = re.search(pattern, findings_text, re.IGNORECASE | re.DOTALL)
                    if section_match:
                        section_title = section_match.group(1).strip()
                        section_content = section_match.group(2).strip()
                        result["main_content"][section_title] = section_content
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing research findings: {str(e)}")
            return {"error": str(e), "raw_text": findings_text}
