from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime
import json
import re

from backend.agents.base_agent import BaseAgent
from backend.db.mongodb.models.document import Document, DocumentStatus
from backend.db.mongodb.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

class ResearchMode(str, Enum):
    """Enum for different research mentoring modes."""
    PROBLEM_ANALYSIS = "problem_analysis"
    RESEARCH_PLANNING = "research_planning"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_DEVELOPMENT = "hypothesis_development"
    METHODOLOGY_GUIDANCE = "methodology_guidance"
    CRITICAL_EVALUATION = "critical_evaluation"
    SYNTHESIS = "synthesis"

class ResearchMentorAgent(BaseAgent):
    """Agent specialized in guiding users through complex reasoning tasks and research processes."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Research Mentor Agent with specific capabilities."""
        # Ensure agent_model_config exists for BaseAgent provider resolution
        if "agent_model_config" not in config:
            config["agent_model_config"] = config.get("model_config", {})
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "problem_analysis": True,
            "research_planning": True,
            "literature_review": True,
            "hypothesis_development": True,
            "methodology_guidance": True,
            "critical_evaluation": True,
            "synthesis": True
        }
        
        # Configure research modes and parameters
        self.modes = {
            ResearchMode.PROBLEM_ANALYSIS: {
                "description": "Analyze and break down complex problems into manageable components",
                "parameters": {
                    "analysis_depth": "detailed",
                    "include_examples": True,
                    "require_citations": True,
                    "suggest_methods": True
                }
            },
            ResearchMode.RESEARCH_PLANNING: {
                "description": "Create structured research plans and methodologies",
                "parameters": {
                    "planning_scope": "comprehensive",
                    "include_timeline": True,
                    "resource_requirements": True,
                    "risk_assessment": True
                }
            },
            ResearchMode.LITERATURE_REVIEW: {
                "description": "Guide through literature review process with critical analysis",
                "parameters": {
                    "review_depth": "comprehensive",
                    "citation_requirements": True,
                    "source_quality": "peer_reviewed",
                    "synthesis_required": True
                }
            },
            ResearchMode.HYPOTHESIS_DEVELOPMENT: {
                "description": "Assist in developing and refining research hypotheses",
                "parameters": {
                    "hypothesis_type": "testable",
                    "require_justification": True,
                    "include_alternatives": True,
                    "feasibility_check": True
                }
            },
            ResearchMode.METHODOLOGY_GUIDANCE: {
                "description": "Provide guidance on research methodology and methods",
                "parameters": {
                    "methodology_type": "mixed",
                    "include_validation": True,
                    "ethical_considerations": True,
                    "practical_limitations": True
                }
            },
            ResearchMode.CRITICAL_EVALUATION: {
                "description": "Guide critical evaluation of research and evidence",
                "parameters": {
                    "evaluation_criteria": "comprehensive",
                    "bias_assessment": True,
                    "strength_analysis": True,
                    "limitation_identification": True
                }
            },
            ResearchMode.SYNTHESIS: {
                "description": "Assist in synthesizing research findings and insights",
                "parameters": {
                    "synthesis_type": "integrative",
                    "include_implications": True,
                    "future_directions": True,
                    "practical_applications": True
                }
            }
        }
        
        # Set default model configuration
        self.model_config.update({
            "temperature": 0.3,  # Lower temperature for more focused and precise responses
            "max_tokens": 8192,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })

        self.document_repository = None
        self.system_prompt = """You are a research mentor AI assistant. Your role is to help users with their research tasks by:
1. Analyzing documents and extracting key insights
2. Providing guidance on research methodology
3. Suggesting relevant sources and references
4. Helping organize and structure research findings
5. Answering questions about research topics

Always maintain a professional and academic tone. When analyzing documents, focus on:
- Main arguments and key points
- Methodology and research approach
- Evidence and supporting data
- Conclusions and implications
- Areas for further research

If you're unsure about something, acknowledge the limitations and suggest how to verify the information."""

    async def _get_repository(self):
        if not self.document_repository:
            from backend.db.mongodb.repositories.document_repository import get_document_repository
            self.document_repository = await get_document_repository()
        return self.document_repository

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research document asynchronously."""
        try:
            document_id = data.get("document_id")
            if not document_id:
                raise ValueError("document_id is required")

            # Get document from repository
            repo = await self._get_repository()
            document = await repo.get_by_id(document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Update document status to processing
            await repo.update_status(
                document_id=document_id,
                status=DocumentStatus.PROCESSING
            )

            # Extract text content from document
            # This is a placeholder - implement actual text extraction based on document type
            text_content = "Sample document content"  # Replace with actual extraction

            # Analyze document using AI model
            analysis_prompt = f"""Please analyze the following research document and provide:
1. A summary of the main points
2. Key findings and insights
3. Methodology used
4. Strengths and limitations
5. Recommendations for further research

Document content:
{text_content}"""

            analysis = await self._call_model_async(
                prompt=analysis_prompt,
                system_prompt=self.system_prompt
            )

            # Update document with analysis results
            await repo.update_extraction(
                document_id=document_id,
                extraction_result={"analysis": analysis},
                extraction_status="completed"
            )

            return {
                "status": "success",
                "document_id": document_id,
                "analysis": analysis
            }

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            if document_id and self.document_repository:
                await self.document_repository.update_status(
                    document_id=document_id,
                    status=DocumentStatus.FAILED,
                    extraction_error=str(e)
                )
            raise

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the research mentor about a topic or document."""
        try:
            # If context includes a document_id, fetch the document
            document_content = ""
            if context and context.get("document_id"):
                repo = await self._get_repository()
                document = await repo.get_by_id(context["document_id"])
                if document and document.extraction_result:
                    document_content = f"\nRelevant document content:\n{document.extraction_result.get('analysis', '')}"

            # Construct the prompt
            prompt = f"""Please help with the following research query:
{query}
{document_content}

Provide a detailed and well-structured response that:
1. Directly addresses the query
2. Cites relevant sources or evidence
3. Suggests additional research directions if applicable
4. Maintains academic rigor and objectivity"""

            # Get response from AI model
            response = await self._call_model_async(
                prompt=prompt,
                system_prompt=self.system_prompt
            )

            return {
                "status": "success",
                "query": query,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research mentoring request asynchronously."""
        try:
            mode = request.get("mode", ResearchMode.PROBLEM_ANALYSIS.value)
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Validate mode
            if mode not in [m.value for m in ResearchMode]:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Process based on mode
            if mode == ResearchMode.PROBLEM_ANALYSIS.value:
                result = await self._analyze_problem(data, context, parameters)
            elif mode == ResearchMode.RESEARCH_PLANNING.value:
                result = await self._plan_research(data, context, parameters)
            elif mode == ResearchMode.LITERATURE_REVIEW.value:
                result = await self._review_literature(data, context, parameters)
            elif mode == ResearchMode.HYPOTHESIS_DEVELOPMENT.value:
                result = await self._develop_hypothesis(data, context, parameters)
            elif mode == ResearchMode.METHODOLOGY_GUIDANCE.value:
                result = await self._guide_methodology(data, context, parameters)
            elif mode == ResearchMode.CRITICAL_EVALUATION.value:
                result = await self._evaluate_critically(data, context, parameters)
            elif mode == ResearchMode.SYNTHESIS.value:
                result = await self._synthesize_findings(data, context, parameters)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            return {
                "results": result,
                "metadata": {
                    "mode": mode,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": self.get_model_name(),
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing research mentoring request: {str(e)}")
            return {"error": str(e)}

    async def _process_with_model(
        self,
        system_prompt: str,
        user_content: str,
        parameters: Dict[str, Any],
    ) -> str:
        """Call the LLM with a system prompt and user content, return raw text response."""
        response_text = await self._call_model_async(
            prompt=user_content,
            system_prompt=system_prompt,
            temperature=parameters.get("temperature", 0.3),
            max_tokens=parameters.get("max_tokens", 8192),
        )
        return response_text

    async def _analyze_problem(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze and break down complex problems into manageable components."""
        try:
            # Prepare system prompt for problem analysis
            system_prompt = self._create_system_prompt(
                ResearchMode.PROBLEM_ANALYSIS,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "problem": data.get("problem", ""),
                    "context": data.get("context", {}),
                    "constraints": data.get("constraints", {}),
                    "objectives": data.get("objectives", [])
                }),
                parameters
            )
            
            # Parse and structure the response
            analysis = self._parse_problem_analysis(response)
            
            return {
                "analysis": analysis,
                "metadata": {
                    "complexity_level": self._calculate_complexity_level(analysis),
                    "component_count": self._count_components(analysis),
                    "citation_count": self._count_citations(analysis)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing problem: {str(e)}")
            raise

    async def _plan_research(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create structured research plans and methodologies."""
        try:
            # Prepare system prompt for research planning
            system_prompt = self._create_system_prompt(
                ResearchMode.RESEARCH_PLANNING,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "research_question": data.get("research_question", ""),
                    "objectives": data.get("objectives", []),
                    "constraints": data.get("constraints", {}),
                    "timeline": data.get("timeline", {})
                }),
                parameters
            )
            
            # Parse and structure the response
            plan = self._parse_research_plan(response)
            
            return {
                "plan": plan,
                "metadata": {
                    "estimated_duration": self._calculate_duration(plan),
                    "resource_requirements": self._identify_resources(plan),
                    "risk_level": self._assess_risks(plan)
                }
            }
            
        except Exception as e:
            logger.error(f"Error planning research: {str(e)}")
            raise

    async def _review_literature(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Guide through literature review process with critical analysis."""
        try:
            # Prepare system prompt for literature review
            system_prompt = self._create_system_prompt(
                ResearchMode.LITERATURE_REVIEW,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "topic": data.get("topic", ""),
                    "scope": data.get("scope", {}),
                    "sources": data.get("sources", []),
                    "focus_areas": data.get("focus_areas", [])
                }),
                parameters
            )
            
            # Parse and structure the response
            review = self._parse_literature_review(response)
            
            return {
                "review": review,
                "metadata": {
                    "source_count": self._count_sources(review),
                    "coverage_score": self._calculate_coverage(review),
                    "quality_score": self._assess_quality(review)
                }
            }
            
        except Exception as e:
            logger.error(f"Error reviewing literature: {str(e)}")
            raise

    async def _develop_hypothesis(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assist in developing and refining research hypotheses."""
        try:
            # Prepare system prompt for hypothesis development
            system_prompt = self._create_system_prompt(
                ResearchMode.HYPOTHESIS_DEVELOPMENT,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "research_question": data.get("research_question", ""),
                    "background": data.get("background", {}),
                    "variables": data.get("variables", []),
                    "constraints": data.get("constraints", {})
                }),
                parameters
            )
            
            # Parse and structure the response
            hypothesis = self._parse_hypothesis(response)
            
            return {
                "hypothesis": hypothesis,
                "metadata": {
                    "testability_score": self._assess_testability(hypothesis),
                    "novelty_score": self._assess_novelty(hypothesis),
                    "feasibility_score": self._assess_feasibility(hypothesis)
                }
            }
            
        except Exception as e:
            logger.error(f"Error developing hypothesis: {str(e)}")
            raise

    async def _guide_methodology(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide guidance on research methodology and methods."""
        try:
            # Prepare system prompt for methodology guidance
            system_prompt = self._create_system_prompt(
                ResearchMode.METHODOLOGY_GUIDANCE,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "research_type": data.get("research_type", ""),
                    "objectives": data.get("objectives", []),
                    "constraints": data.get("constraints", {}),
                    "resources": data.get("resources", {})
                }),
                parameters
            )
            
            # Parse and structure the response
            methodology = self._parse_methodology(response)
            
            return {
                "methodology": methodology,
                "metadata": {
                    "validity_score": self._assess_validity(methodology),
                    "reliability_score": self._assess_reliability(methodology),
                    "practicality_score": self._assess_practicality(methodology)
                }
            }
            
        except Exception as e:
            logger.error(f"Error guiding methodology: {str(e)}")
            raise

    async def _evaluate_critically(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Guide critical evaluation of research and evidence."""
        try:
            # Prepare system prompt for critical evaluation
            system_prompt = self._create_system_prompt(
                ResearchMode.CRITICAL_EVALUATION,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "research": data.get("research", {}),
                    "criteria": data.get("criteria", []),
                    "context": data.get("context", {}),
                    "focus_areas": data.get("focus_areas", [])
                }),
                parameters
            )
            
            # Parse and structure the response
            evaluation = self._parse_evaluation(response)
            
            return {
                "evaluation": evaluation,
                "metadata": {
                    "strength_score": self._calculate_strength(evaluation),
                    "limitation_count": self._count_limitations(evaluation),
                    "bias_assessment": self._assess_bias(evaluation)
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluating critically: {str(e)}")
            raise

    async def _synthesize_findings(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assist in synthesizing research findings and insights."""
        try:
            # Prepare system prompt for synthesis
            system_prompt = self._create_system_prompt(
                ResearchMode.SYNTHESIS,
                context,
                parameters
            )
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "findings": data.get("findings", []),
                    "context": data.get("context", {}),
                    "objectives": data.get("objectives", []),
                    "constraints": data.get("constraints", {})
                }),
                parameters
            )
            
            # Parse and structure the response
            synthesis = self._parse_synthesis(response)
            
            return {
                "synthesis": synthesis,
                "metadata": {
                    "coherence_score": self._assess_coherence(synthesis),
                    "insight_count": self._count_insights(synthesis),
                    "implication_count": self._count_implications(synthesis)
                }
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing findings: {str(e)}")
            raise

    def _create_system_prompt(
        self,
        mode: ResearchMode,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create a system prompt based on the research mode and parameters."""
        base_prompt = f"""You are a specialized research mentor AI assistant. Your task is to {self.modes[mode]['description']}.
        
        Context:
        - Research Level: {context.get('research_level', 'advanced')}
        - Field: {context.get('field', 'general')}
        - User Experience: {context.get('user_experience', 'intermediate')}
        
        Parameters:
        {self._format_parameters(parameters)}
        
        Please provide comprehensive, well-reasoned, and evidence-based guidance.
        Focus on teaching critical thinking and research methodology, not just providing answers.
        Use markdown formatting with headers, bullet points, and bold text for clarity.
        Always include relevant citations and explain your reasoning process.
        """
        
        # Add mode-specific instructions
        if mode == ResearchMode.PROBLEM_ANALYSIS:
            base_prompt += """
            For problem analysis:
            1. Break down the problem into key components
            2. Identify underlying assumptions and constraints
            3. Suggest relevant research methods
            4. Provide examples of similar problems
            5. Include citations for key concepts
            """
        elif mode == ResearchMode.RESEARCH_PLANNING:
            base_prompt += """
            For research planning:
            1. Create a structured research plan
            2. Define clear objectives and milestones
            3. Identify required resources
            4. Assess potential risks
            5. Include timeline estimates
            """
        elif mode == ResearchMode.LITERATURE_REVIEW:
            base_prompt += """
            For literature review:
            1. Guide source selection and evaluation
            2. Help identify key themes and gaps
            3. Assist in critical analysis
            4. Guide synthesis of findings
            5. Ensure proper citation
            """
        elif mode == ResearchMode.HYPOTHESIS_DEVELOPMENT:
            base_prompt += """
            For hypothesis development:
            1. Guide hypothesis formulation
            2. Ensure testability
            3. Consider alternatives
            4. Assess feasibility
            5. Link to existing research
            """
        elif mode == ResearchMode.METHODOLOGY_GUIDANCE:
            base_prompt += """
            For methodology guidance:
            1. Recommend appropriate methods
            2. Address validity and reliability
            3. Consider ethical implications
            4. Assess practical limitations
            5. Guide implementation
            """
        elif mode == ResearchMode.CRITICAL_EVALUATION:
            base_prompt += """
            For critical evaluation:
            1. Guide systematic assessment
            2. Identify strengths and limitations
            3. Assess potential biases
            4. Evaluate evidence quality
            5. Consider alternative interpretations
            """
        elif mode == ResearchMode.SYNTHESIS:
            base_prompt += """
            For synthesis:
            1. Integrate key findings
            2. Identify patterns and themes
            3. Draw meaningful conclusions
            4. Suggest future directions
            5. Highlight practical implications
            """
        
        return base_prompt

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for the system prompt."""
        return "\n".join([f"- {k}: {v}" for k, v in parameters.items()])

    # Helper methods for parsing and analysis
    def _parse_problem_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured problem analysis."""
        return {"content": response}

    def _parse_research_plan(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into a structured research plan."""
        return {"content": response}

    def _parse_literature_review(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured literature review."""
        return {"content": response}

    def _parse_hypothesis(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured hypothesis."""
        return {"content": response}

    def _parse_methodology(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured methodology."""
        return {"content": response}

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured evaluation."""
        return {"content": response}

    def _parse_synthesis(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured synthesis."""
        return {"content": response}

    # ── Text extraction helper ─────────────────────────────────────────

    def _get_text(self, data: Dict[str, Any]) -> str:
        """Extract the raw text content from a parsed response dict."""
        return data.get("content", "") if isinstance(data, dict) else str(data)

    # ── Problem Analysis helpers ─────────────────────────────────────

    def _calculate_complexity_level(self, analysis: Dict[str, Any]) -> str:
        """Calculate complexity based on number of sections, headers, and word count."""
        text = self._get_text(analysis)
        word_count = len(text.split())
        headers = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        bullet_points = len(re.findall(r"^[\s]*[-*]\s", text, re.MULTILINE))
        total_elements = headers + bullet_points

        if word_count > 2000 or total_elements > 20:
            return "high"
        elif word_count > 800 or total_elements > 10:
            return "medium"
        return "low"

    def _count_components(self, analysis: Dict[str, Any]) -> int:
        """Count components by counting top-level headers and numbered list items."""
        text = self._get_text(analysis)
        headers = re.findall(r"^#{1,3}\s.+", text, re.MULTILINE)
        numbered = re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE)
        return max(len(headers), len(numbered))

    def _count_citations(self, content: Dict[str, Any]) -> int:
        """Count citations — looks for [n], (Author, Year), URLs, and 'et al.' patterns."""
        text = self._get_text(content)
        bracket_refs = re.findall(r"\[\d+\]", text)
        author_year = re.findall(r"\([A-Z][a-z]+(?:\s(?:et\sal\.?|&\s[A-Z][a-z]+))?,?\s*\d{4}\)", text)
        urls = re.findall(r"https?://[^\s\)]+", text)
        et_al = re.findall(r"\bet\s+al\.?\b", text, re.IGNORECASE)
        return len(bracket_refs) + len(author_year) + len(urls) + len(et_al)

    # ── Research Planning helpers ────────────────────────────────────

    def _calculate_duration(self, plan: Dict[str, Any]) -> str:
        """Estimate duration from time-related keywords in the response."""
        text = self._get_text(plan).lower()
        duration_matches = re.findall(
            r"(\d+)\s*(weeks?|months?|days?|years?|hours?|semesters?)",
            text,
        )
        if not duration_matches:
            word_count = len(text.split())
            if word_count > 2000:
                return "6-12 months (estimated from scope)"
            elif word_count > 1000:
                return "3-6 months (estimated from scope)"
            return "1-3 months (estimated from scope)"

        # Return the largest time reference found
        max_val, max_unit = 0, ""
        for val, unit in duration_matches:
            num = int(val)
            if "year" in unit:
                num *= 365
            elif "semester" in unit:
                num *= 180
            elif "month" in unit:
                num *= 30
            elif "week" in unit:
                num *= 7
            if num > max_val:
                max_val = num
                max_unit = f"{val} {unit}"
        return max_unit

    def _identify_resources(self, plan: Dict[str, Any]) -> List[str]:
        """Identify resources mentioned in the plan text."""
        text = self._get_text(plan).lower()
        resource_keywords = {
            "survey": "Survey tools",
            "questionnaire": "Questionnaire platform",
            "interview": "Interview setup",
            "database": "Database access",
            "software": "Software tools",
            "statistical": "Statistical software",
            "spss": "SPSS",
            "python": "Python",
            "r studio": "R Studio",
            "laboratory": "Lab facilities",
            "lab ": "Lab facilities",
            "funding": "Research funding",
            "grant": "Research grant",
            "ethics approval": "Ethics approval",
            "irb": "IRB approval",
            "participants": "Research participants",
            "sample": "Data sample",
            "dataset": "Dataset",
            "computing": "Computing resources",
            "gpu": "GPU compute",
            "cloud": "Cloud infrastructure",
            "api": "API access",
            "library": "Library access",
            "peer review": "Peer reviewers",
        }
        found = []
        for keyword, label in resource_keywords.items():
            if keyword in text and label not in found:
                found.append(label)
        return found

    def _assess_risks(self, plan: Dict[str, Any]) -> str:
        """Assess risk level from risk-related language in the response."""
        text = self._get_text(plan).lower()
        high_risk = ["significant risk", "major risk", "critical", "high risk", "fail", "impossible",
                      "ethical concern", "legal issue", "privacy violation"]
        medium_risk = ["moderate risk", "some risk", "challenge", "limitation", "bias", "constraint",
                       "difficult", "uncertain", "potential issue"]
        low_risk = ["low risk", "minimal risk", "straightforward", "well-established", "proven"]

        high_count = sum(1 for term in high_risk if term in text)
        medium_count = sum(1 for term in medium_risk if term in text)
        low_count = sum(1 for term in low_risk if term in text)

        if high_count >= 2:
            return "high"
        elif high_count >= 1 or medium_count >= 3:
            return "medium-high"
        elif medium_count >= 1:
            return "medium"
        elif low_count >= 1:
            return "low"
        return "not assessed"

    # ── Literature Review helpers ────────────────────────────────────

    def _count_sources(self, review: Dict[str, Any]) -> int:
        """Count distinct sources referenced in the literature review."""
        text = self._get_text(review)
        bracket_refs = set(re.findall(r"\[(\d+)\]", text))
        author_refs = re.findall(r"\([A-Z][a-z]+(?:\s(?:et\sal\.?|&\s[A-Z][a-z]+))?,?\s*\d{4}\)", text)
        urls = re.findall(r"https?://[^\s\)]+", text)
        return len(bracket_refs) + len(author_refs) + len(urls)

    def _calculate_coverage(self, review: Dict[str, Any]) -> float:
        """Score coverage 0-1 based on structural breadth of the review."""
        text = self._get_text(review)
        word_count = len(text.split())
        headers = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        sources = self._count_sources(review)
        paragraphs = len([p for p in text.split("\n\n") if len(p.strip()) > 50])

        score = 0.0
        score += min(word_count / 3000, 0.3)    # Up to 0.3 for length
        score += min(headers / 8, 0.25)          # Up to 0.25 for sections
        score += min(sources / 10, 0.25)         # Up to 0.25 for sources
        score += min(paragraphs / 10, 0.2)       # Up to 0.2 for depth
        return round(min(score, 1.0), 2)

    def _assess_quality(self, review: Dict[str, Any]) -> float:
        """Score quality 0-1 based on analytical depth indicators."""
        text = self._get_text(review).lower()
        quality_indicators = [
            "however", "in contrast", "on the other hand", "critically",
            "limitation", "strength", "weakness", "gap in", "further research",
            "meta-analysis", "systematic review", "peer-reviewed",
            "evidence suggests", "findings indicate", "data shows",
            "significant", "correlation", "causation", "methodology",
        ]
        found = sum(1 for ind in quality_indicators if ind in text)
        sources = self._count_sources(review)
        citations = self._count_citations(review)

        score = 0.0
        score += min(found / 10, 0.4)          # Up to 0.4 for analytical language
        score += min(sources / 8, 0.3)          # Up to 0.3 for sources
        score += min(citations / 6, 0.3)        # Up to 0.3 for citations
        return round(min(score, 1.0), 2)

    # ── Hypothesis Development helpers ───────────────────────────────

    def _assess_testability(self, hypothesis: Dict[str, Any]) -> float:
        """Score testability 0-1 based on presence of measurable/testable language."""
        text = self._get_text(hypothesis).lower()
        testable_indicators = [
            "measurable", "observable", "quantif", "variable", "dependent",
            "independent", "control", "experiment", "test", "predict",
            "falsifiable", "hypothesis", "null hypothesis", "alternative hypothesis",
            "operationalize", "metric", "indicator", "sample size", "statistical",
            "p-value", "significance level", "confidence interval",
        ]
        found = sum(1 for ind in testable_indicators if ind in text)
        has_if_then = bool(re.search(r"\bif\b.+\bthen\b", text))
        has_variables = bool(re.search(r"\b(independent|dependent|control)\s+variable", text))

        score = min(found / 10, 0.5)
        if has_if_then:
            score += 0.25
        if has_variables:
            score += 0.25
        return round(min(score, 1.0), 2)

    def _assess_novelty(self, hypothesis: Dict[str, Any]) -> float:
        """Score novelty 0-1 based on indicators of originality."""
        text = self._get_text(hypothesis).lower()
        novelty_indicators = [
            "novel", "new approach", "unexplored", "first", "original",
            "innovative", "unique", "not been studied", "gap", "emerging",
            "cutting-edge", "frontier", "paradigm shift", "reconceptualize",
            "understudied", "overlooked", "no prior research", "pioneering",
        ]
        diminishing = [
            "well-established", "well-known", "widely studied", "common",
            "traditional", "conventional", "standard", "typical",
        ]
        novel_count = sum(1 for ind in novelty_indicators if ind in text)
        diminish_count = sum(1 for ind in diminishing if ind in text)

        score = min(novel_count / 6, 0.8)
        score -= min(diminish_count / 4, 0.3)
        return round(max(min(score, 1.0), 0.1), 2)

    def _assess_feasibility(self, hypothesis: Dict[str, Any]) -> float:
        """Score feasibility 0-1 based on practical viability indicators."""
        text = self._get_text(hypothesis).lower()
        feasible_indicators = [
            "feasible", "practical", "achievable", "realistic", "available",
            "accessible", "within scope", "straightforward", "established method",
            "existing data", "replicable", "cost-effective",
        ]
        infeasible_indicators = [
            "infeasible", "impractical", "impossible", "too expensive",
            "too complex", "beyond scope", "unavailable", "ethical barrier",
            "prohibitive", "not possible", "insufficient",
        ]
        feasible_count = sum(1 for ind in feasible_indicators if ind in text)
        infeasible_count = sum(1 for ind in infeasible_indicators if ind in text)

        score = 0.5 + min(feasible_count / 6, 0.4) - min(infeasible_count / 4, 0.4)
        return round(max(min(score, 1.0), 0.1), 2)

    # ── Methodology Guidance helpers ─────────────────────────────────

    def _assess_validity(self, methodology: Dict[str, Any]) -> float:
        """Score validity 0-1 based on methodological rigor indicators."""
        text = self._get_text(methodology).lower()
        validity_indicators = [
            "internal validity", "external validity", "construct validity",
            "content validity", "face validity", "criterion validity",
            "triangulation", "member checking", "peer review",
            "randomiz", "control group", "blinding", "double-blind",
            "validated instrument", "reliability", "replicab",
            "representative sample", "generalizab",
        ]
        threats = [
            "threat to validity", "confound", "selection bias",
            "measurement error", "attrition", "maturation effect",
        ]
        valid_count = sum(1 for ind in validity_indicators if ind in text)
        threat_count = sum(1 for ind in threats if ind in text)

        score = min(valid_count / 8, 0.7)
        # Acknowledging threats is actually positive — shows awareness
        score += min(threat_count / 4, 0.3)
        return round(min(score, 1.0), 2)

    def _assess_reliability(self, methodology: Dict[str, Any]) -> float:
        """Score reliability 0-1 based on consistency and reproducibility indicators."""
        text = self._get_text(methodology).lower()
        reliability_indicators = [
            "reliability", "cronbach", "alpha", "test-retest",
            "inter-rater", "intra-class", "consistency", "reproducib",
            "replicab", "standardiz", "protocol", "systematic",
            "pilot test", "pre-test", "calibrat",
        ]
        found = sum(1 for ind in reliability_indicators if ind in text)

        has_protocol = bool(re.search(r"\b(step|protocol|procedure)\s*\d", text))
        has_metrics = bool(re.search(r"\b(alpha|kappa|icc|r\s*=)\b", text))

        score = min(found / 7, 0.5)
        if has_protocol:
            score += 0.25
        if has_metrics:
            score += 0.25
        return round(min(score, 1.0), 2)

    def _assess_practicality(self, methodology: Dict[str, Any]) -> float:
        """Score practicality 0-1 based on implementation feasibility."""
        text = self._get_text(methodology).lower()
        practical_indicators = [
            "practical", "feasible", "cost-effective", "time-efficient",
            "accessible", "available", "straightforward", "simple",
            "existing tool", "established", "widely used", "open source",
            "free", "low cost", "minimal equipment",
        ]
        impractical_indicators = [
            "expensive", "time-consuming", "complex", "specialized equipment",
            "difficult to implement", "requires expertise", "labor-intensive",
            "resource-intensive", "hard to access",
        ]
        practical_count = sum(1 for ind in practical_indicators if ind in text)
        impractical_count = sum(1 for ind in impractical_indicators if ind in text)

        score = 0.5 + min(practical_count / 6, 0.4) - min(impractical_count / 4, 0.3)
        return round(max(min(score, 1.0), 0.1), 2)

    # ── Critical Evaluation helpers ──────────────────────────────────

    def _calculate_strength(self, evaluation: Dict[str, Any]) -> float:
        """Score evidence strength 0-1 based on analytical depth."""
        text = self._get_text(evaluation).lower()
        strength_indicators = [
            "strong evidence", "robust", "compelling", "well-supported",
            "statistically significant", "large sample", "meta-analysis",
            "systematic review", "randomized controlled", "longitudinal",
            "replicat", "consistent findings", "converging evidence",
        ]
        weak_indicators = [
            "weak evidence", "limited", "anecdotal", "small sample",
            "correlation", "preliminary", "inconclusive", "insufficient",
            "conflicting", "mixed results",
        ]
        strong_count = sum(1 for ind in strength_indicators if ind in text)
        weak_count = sum(1 for ind in weak_indicators if ind in text)

        score = 0.5 + min(strong_count / 6, 0.4) - min(weak_count / 6, 0.3)
        return round(max(min(score, 1.0), 0.1), 2)

    def _count_limitations(self, evaluation: Dict[str, Any]) -> int:
        """Count limitations mentioned in the evaluation."""
        text = self._get_text(evaluation)
        limitation_patterns = re.findall(
            r"(?:limitation|weakness|shortcoming|drawback|caveat|concern|issue|flaw|gap|problem)",
            text,
            re.IGNORECASE,
        )
        # Also count items in a limitations section
        limitations_section = re.search(
            r"(?:limitation|weakness|concern)s?\b.*?(?=\n#{1,3}\s|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        bullet_items = 0
        if limitations_section:
            bullet_items = len(re.findall(r"^[\s]*[-*]\s", limitations_section.group(), re.MULTILINE))
            numbered_items = len(re.findall(r"^\s*\d+[\.\)]\s", limitations_section.group(), re.MULTILINE))
            bullet_items = max(bullet_items, numbered_items)

        return max(len(set(limitation_patterns)), bullet_items)

    def _assess_bias(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Identify types of bias mentioned or detected in the evaluation."""
        text = self._get_text(evaluation).lower()
        bias_types = {
            "selection_bias": ["selection bias", "sampling bias", "non-random", "convenience sample"],
            "confirmation_bias": ["confirmation bias", "cherry-pick", "selective reporting"],
            "publication_bias": ["publication bias", "file drawer", "negative results"],
            "measurement_bias": ["measurement bias", "instrument bias", "observer bias", "hawthorne"],
            "recall_bias": ["recall bias", "memory bias", "retrospective"],
            "funding_bias": ["funding bias", "conflict of interest", "industry-funded", "sponsor"],
            "cultural_bias": ["cultural bias", "western bias", "ethnocentric", "generalizability"],
            "survivorship_bias": ["survivorship bias", "survivor bias"],
            "anchoring_bias": ["anchoring bias", "anchoring effect"],
            "attrition_bias": ["attrition bias", "dropout", "loss to follow-up"],
        }
        detected = {}
        for bias_name, keywords in bias_types.items():
            if any(kw in text for kw in keywords):
                detected[bias_name] = True

        total_biases = len(detected)
        if total_biases == 0:
            risk_level = "not assessed"
        elif total_biases <= 2:
            risk_level = "low"
        elif total_biases <= 4:
            risk_level = "moderate"
        else:
            risk_level = "high"

        return {
            "biases_identified": list(detected.keys()),
            "bias_count": total_biases,
            "risk_level": risk_level,
        }

    # ── Synthesis helpers ────────────────────────────────────────────

    def _assess_coherence(self, synthesis: Dict[str, Any]) -> float:
        """Score coherence 0-1 based on structural and logical flow indicators."""
        text = self._get_text(synthesis)
        word_count = len(text.split())
        headers = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        paragraphs = len([p for p in text.split("\n\n") if len(p.strip()) > 30])

        # Transition/connective words indicate logical flow
        connectives = [
            "therefore", "consequently", "furthermore", "moreover",
            "however", "in contrast", "similarly", "in addition",
            "as a result", "in conclusion", "overall", "taken together",
            "building on", "consistent with", "in summary",
        ]
        text_lower = text.lower()
        connective_count = sum(1 for c in connectives if c in text_lower)

        score = 0.0
        score += min(headers / 6, 0.25)              # Structured sections
        score += min(paragraphs / 8, 0.25)            # Developed paragraphs
        score += min(connective_count / 8, 0.3)       # Logical flow
        score += min(word_count / 2000, 0.2)          # Sufficient depth
        return round(min(score, 1.0), 2)

    def _count_insights(self, synthesis: Dict[str, Any]) -> int:
        """Count distinct insights/findings/key points in the synthesis."""
        text = self._get_text(synthesis)
        insight_markers = re.findall(
            r"(?:key\s+(?:finding|insight|point|takeaway|observation)|insight|finding|implication|conclusion)\s*(?:\d+|:)",
            text,
            re.IGNORECASE,
        )
        # Also count numbered items and bullet points in findings/insights sections
        bullet_points = re.findall(r"^[\s]*[-*]\s.{20,}", text, re.MULTILINE)
        numbered_points = re.findall(r"^\s*\d+[\.\)]\s.{20,}", text, re.MULTILINE)

        return max(len(insight_markers), len(bullet_points), len(numbered_points))

    def _count_implications(self, synthesis: Dict[str, Any]) -> int:
        """Count implications, recommendations, or future directions mentioned."""
        text = self._get_text(synthesis).lower()
        implication_patterns = re.findall(
            r"(?:implication|recommendation|future\s+(?:research|direction|work|study)|"
            r"suggest(?:ion|s\b)|practical\s+application|policy\s+implication|"
            r"next\s+step|action\s+item|should\s+(?:be|consider)|"
            r"further\s+(?:investigation|study|research|exploration))",
            text,
        )
        # Count bullet points in implications/recommendations sections
        impl_section = re.search(
            r"(?:implication|recommendation|future|next\s+step)s?\b.*?(?=\n#{1,3}\s|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        bullet_count = 0
        if impl_section:
            bullet_count = len(re.findall(r"^[\s]*[-*]\s", impl_section.group(), re.MULTILINE))
            numbered = len(re.findall(r"^\s*\d+[\.\)]\s", impl_section.group(), re.MULTILINE))
            bullet_count = max(bullet_count, numbered)

        return max(len(set(implication_patterns)), bullet_count)