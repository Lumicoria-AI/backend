from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime
import json

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
            "max_tokens": 4096,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })

        self.document_repository = DocumentRepository()
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

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research document asynchronously."""
        try:
            document_id = data.get("document_id")
            if not document_id:
                raise ValueError("document_id is required")

            # Get document from repository
            document = await self.document_repository.get_by_id(document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Update document status to processing
            await self.document_repository.update_status(
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
            await self.document_repository.update_extraction(
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
            if document_id:
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
                document = await self.document_repository.get_by_id(context["document_id"])
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
                    "model": self.model_config.get("model", "sonar-large-online"),
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing research mentoring request: {str(e)}")
            return {"error": str(e)}

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
        # Implementation would parse the response into a structured format
        return {}

    def _parse_research_plan(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into a structured research plan."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_literature_review(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured literature review."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_hypothesis(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured hypothesis."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_methodology(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured methodology."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured evaluation."""
        # Implementation would parse the response into a structured format
        return {}

    def _parse_synthesis(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured synthesis."""
        # Implementation would parse the response into a structured format
        return {}

    # Analysis and calculation methods
    def _calculate_complexity_level(self, analysis: Dict[str, Any]) -> str:
        """Calculate the complexity level of a problem analysis."""
        # Implementation would calculate complexity level
        return ""

    def _count_components(self, analysis: Dict[str, Any]) -> int:
        """Count the number of components in a problem analysis."""
        # Implementation would count components
        return 0

    def _count_citations(self, content: Dict[str, Any]) -> int:
        """Count the number of citations in content."""
        # Implementation would count citations
        return 0

    def _calculate_duration(self, plan: Dict[str, Any]) -> str:
        """Calculate estimated duration for a research plan."""
        # Implementation would calculate duration
        return ""

    def _identify_resources(self, plan: Dict[str, Any]) -> List[str]:
        """Identify required resources for a research plan."""
        # Implementation would identify resources
        return []

    def _assess_risks(self, plan: Dict[str, Any]) -> str:
        """Assess the risk level of a research plan."""
        # Implementation would assess risks
        return ""

    def _count_sources(self, review: Dict[str, Any]) -> int:
        """Count the number of sources in a literature review."""
        # Implementation would count sources
        return 0

    def _calculate_coverage(self, review: Dict[str, Any]) -> float:
        """Calculate the coverage score of a literature review."""
        # Implementation would calculate coverage
        return 0.0

    def _assess_quality(self, review: Dict[str, Any]) -> float:
        """Assess the quality score of a literature review."""
        # Implementation would assess quality
        return 0.0

    def _assess_testability(self, hypothesis: Dict[str, Any]) -> float:
        """Assess the testability of a hypothesis."""
        # Implementation would assess testability
        return 0.0

    def _assess_novelty(self, hypothesis: Dict[str, Any]) -> float:
        """Assess the novelty of a hypothesis."""
        # Implementation would assess novelty
        return 0.0

    def _assess_feasibility(self, hypothesis: Dict[str, Any]) -> float:
        """Assess the feasibility of a hypothesis."""
        # Implementation would assess feasibility
        return 0.0

    def _assess_validity(self, methodology: Dict[str, Any]) -> float:
        """Assess the validity of a methodology."""
        # Implementation would assess validity
        return 0.0

    def _assess_reliability(self, methodology: Dict[str, Any]) -> float:
        """Assess the reliability of a methodology."""
        # Implementation would assess reliability
        return 0.0

    def _assess_practicality(self, methodology: Dict[str, Any]) -> float:
        """Assess the practicality of a methodology."""
        # Implementation would assess practicality
        return 0.0

    def _calculate_strength(self, evaluation: Dict[str, Any]) -> float:
        """Calculate the strength score of an evaluation."""
        # Implementation would calculate strength
        return 0.0

    def _count_limitations(self, evaluation: Dict[str, Any]) -> int:
        """Count the number of limitations in an evaluation."""
        # Implementation would count limitations
        return 0

    def _assess_bias(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential biases in an evaluation."""
        # Implementation would assess bias
        return {}

    def _assess_coherence(self, synthesis: Dict[str, Any]) -> float:
        """Assess the coherence of a synthesis."""
        # Implementation would assess coherence
        return 0.0

    def _count_insights(self, synthesis: Dict[str, Any]) -> int:
        """Count the number of insights in a synthesis."""
        # Implementation would count insights
        return 0

    def _count_implications(self, synthesis: Dict[str, Any]) -> int:
        """Count the number of implications in a synthesis."""
        # Implementation would count implications
        return 0 