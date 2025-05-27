from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime

from agents.base_agent import BaseAgent
from agents.agent_service import AgentService

logger = logging.getLogger(__name__)

class LegalDocumentMode(str, Enum):
    """Enum for different legal document processing modes."""
    CLAUSE_EXTRACTION = "clause_extraction"
    RISK_ANALYSIS = "risk_analysis"
    VERSION_COMPARISON = "version_comparison"
    PLAIN_LANGUAGE = "plain_language"
    COMPLIANCE_CHECK = "compliance_check"

class LegalDocumentAgent(BaseAgent):
    """Agent specialized in processing legal documents, contracts, and regulatory content."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Legal Document Agent with specific capabilities."""
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "clause_extraction": True,
            "risk_analysis": True,
            "version_comparison": True,
            "plain_language_summary": True,
            "compliance_checking": True
        }
        
        # Configure analysis modes and parameters
        self.modes = {
            LegalDocumentMode.CLAUSE_EXTRACTION: {
                "description": "Extract key clauses, obligations, and deadlines",
                "parameters": {
                    "include_metadata": True,
                    "highlight_obligations": True,
                    "extract_dates": True
                }
            },
            LegalDocumentMode.RISK_ANALYSIS: {
                "description": "Analyze potential risks and unusual terms",
                "parameters": {
                    "risk_threshold": 0.7,
                    "include_recommendations": True,
                    "categorize_risks": True
                }
            },
            LegalDocumentMode.VERSION_COMPARISON: {
                "description": "Compare contract versions and identify changes",
                "parameters": {
                    "track_changes": True,
                    "highlight_additions": True,
                    "highlight_deletions": True,
                    "summarize_changes": True
                }
            },
            LegalDocumentMode.PLAIN_LANGUAGE: {
                "description": "Provide summaries in plain language",
                "parameters": {
                    "simplify_terms": True,
                    "include_examples": True,
                    "maintain_legal_accuracy": True
                }
            },
            LegalDocumentMode.COMPLIANCE_CHECK: {
                "description": "Check regulatory compliance",
                "parameters": {
                    "jurisdiction": "global",
                    "industry_specific": True,
                    "include_citations": True,
                    "severity_levels": ["critical", "high", "medium", "low"]
                }
            }
        }
        
        # Set default model configuration
        self.model_config.update({
            "temperature": 0.3,  # Lower temperature for more precise legal analysis
            "max_tokens": 4096,  # Higher token limit for longer documents
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        })

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a legal document analysis request asynchronously."""
        try:
            mode = request.get("mode", LegalDocumentMode.CLAUSE_EXTRACTION.value)
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Validate mode
            if mode not in [m.value for m in LegalDocumentMode]:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Process based on mode
            if mode == LegalDocumentMode.CLAUSE_EXTRACTION.value:
                result = await self._extract_clauses(data, context, parameters)
            elif mode == LegalDocumentMode.RISK_ANALYSIS.value:
                result = await self._analyze_risks(data, context, parameters)
            elif mode == LegalDocumentMode.VERSION_COMPARISON.value:
                result = await self._compare_versions(data, context, parameters)
            elif mode == LegalDocumentMode.PLAIN_LANGUAGE.value:
                result = await self._generate_plain_language(data, context, parameters)
            elif mode == LegalDocumentMode.COMPLIANCE_CHECK.value:
                result = await self._check_compliance(data, context, parameters)
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
            logger.error(f"Error processing legal document request: {str(e)}")
            return {"error": str(e)}

    async def _extract_clauses(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key clauses, obligations, and deadlines from legal documents."""
        try:
            # Prepare system prompt for clause extraction
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.CLAUSE_EXTRACTION,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response
            clauses = self._parse_clauses(response)
            
            return {
                "clauses": clauses,
                "metadata": {
                    "total_clauses": len(clauses),
                    "document_type": data.get("document_type", "unknown"),
                    "extraction_confidence": self._calculate_confidence(response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting clauses: {str(e)}")
            raise

    async def _analyze_risks(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze potential risks and unusual terms in legal documents."""
        try:
            # Prepare system prompt for risk analysis
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.RISK_ANALYSIS,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response
            risks = self._parse_risks(response)
            
            return {
                "risks": risks,
                "summary": self._generate_risk_summary(risks),
                "recommendations": self._generate_risk_recommendations(risks),
                "metadata": {
                    "risk_level": self._calculate_risk_level(risks),
                    "document_type": data.get("document_type", "unknown"),
                    "analysis_confidence": self._calculate_confidence(response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risks: {str(e)}")
            raise

    async def _compare_versions(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare different versions of legal documents and identify changes."""
        try:
            # Prepare system prompt for version comparison
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.VERSION_COMPARISON,
                context,
                parameters
            )
            
            # Process both versions
            old_version = data.get("old_version", "")
            new_version = data.get("new_version", "")
            
            response = await self._process_with_model(
                system_prompt,
                f"Old Version:\n{old_version}\n\nNew Version:\n{new_version}",
                parameters
            )
            
            # Parse and structure the response
            changes = self._parse_changes(response)
            
            return {
                "changes": changes,
                "summary": self._generate_change_summary(changes),
                "metadata": {
                    "total_changes": len(changes),
                    "document_type": data.get("document_type", "unknown"),
                    "comparison_confidence": self._calculate_confidence(response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing versions: {str(e)}")
            raise

    async def _generate_plain_language(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate plain language summaries of legal documents."""
        try:
            # Prepare system prompt for plain language summary
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.PLAIN_LANGUAGE,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response
            summary = self._parse_summary(response)
            
            return {
                "summary": summary,
                "key_points": self._extract_key_points(summary),
                "metadata": {
                    "document_type": data.get("document_type", "unknown"),
                    "summary_confidence": self._calculate_confidence(response),
                    "reading_level": self._calculate_reading_level(summary)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating plain language summary: {str(e)}")
            raise

    async def _check_compliance(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check regulatory compliance based on document content."""
        try:
            # Prepare system prompt for compliance check
            system_prompt = self._create_system_prompt(
                LegalDocumentMode.COMPLIANCE_CHECK,
                context,
                parameters
            )
            
            # Process the document
            response = await self._process_with_model(
                system_prompt,
                data.get("document", ""),
                parameters
            )
            
            # Parse and structure the response
            compliance_issues = self._parse_compliance_issues(response)
            
            return {
                "compliance_issues": compliance_issues,
                "summary": self._generate_compliance_summary(compliance_issues),
                "recommendations": self._generate_compliance_recommendations(compliance_issues),
                "metadata": {
                    "jurisdiction": parameters.get("jurisdiction", "global"),
                    "industry": data.get("industry", "unknown"),
                    "compliance_confidence": self._calculate_confidence(response),
                    "severity_levels": self._calculate_severity_levels(compliance_issues)
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking compliance: {str(e)}")
            raise

    def _create_system_prompt(
        self,
        mode: LegalDocumentMode,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create a system prompt based on the analysis mode and parameters."""
        base_prompt = f"""You are a specialized legal document analysis AI assistant. Your task is to {self.modes[mode]['description']}.
        
        Context:
        - Document Type: {context.get('document_type', 'unknown')}
        - Jurisdiction: {context.get('jurisdiction', 'global')}
        - Industry: {context.get('industry', 'general')}
        
        Parameters:
        {self._format_parameters(parameters)}
        
        Please analyze the provided legal document with high accuracy and attention to detail.
        Focus on extracting relevant information while maintaining legal precision.
        """
        
        # Add mode-specific instructions
        if mode == LegalDocumentMode.CLAUSE_EXTRACTION:
            base_prompt += """
            For clause extraction:
            1. Identify and extract all key clauses
            2. Highlight obligations and deadlines
            3. Maintain the original legal terminology
            4. Include clause references and context
            """
        elif mode == LegalDocumentMode.RISK_ANALYSIS:
            base_prompt += """
            For risk analysis:
            1. Identify potential risks and unusual terms
            2. Categorize risks by severity and type
            3. Provide clear explanations for each risk
            4. Include relevant legal precedents if applicable
            """
        elif mode == LegalDocumentMode.VERSION_COMPARISON:
            base_prompt += """
            For version comparison:
            1. Identify all changes between versions
            2. Highlight additions and deletions
            3. Explain the implications of changes
            4. Maintain a clear change log
            """
        elif mode == LegalDocumentMode.PLAIN_LANGUAGE:
            base_prompt += """
            For plain language summary:
            1. Simplify complex legal terms
            2. Maintain legal accuracy
            3. Use clear, concise language
            4. Include practical examples
            """
        elif mode == LegalDocumentMode.COMPLIANCE_CHECK:
            base_prompt += """
            For compliance check:
            1. Identify regulatory requirements
            2. Check for compliance issues
            3. Reference relevant regulations
            4. Provide actionable recommendations
            """
        
        return base_prompt

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for the system prompt."""
        return "\n".join([f"- {k}: {v}" for k, v in parameters.items()])

    def _parse_clauses(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response into structured clause data."""
        # Implementation would parse the response into a structured format
        # This is a placeholder for the actual implementation
        return []

    def _parse_risks(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response into structured risk data."""
        # Implementation would parse the response into a structured format
        # This is a placeholder for the actual implementation
        return []

    def _parse_changes(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response into structured change data."""
        # Implementation would parse the response into a structured format
        # This is a placeholder for the actual implementation
        return []

    def _parse_summary(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into a structured summary."""
        # Implementation would parse the response into a structured format
        # This is a placeholder for the actual implementation
        return {}

    def _parse_compliance_issues(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response into structured compliance issue data."""
        # Implementation would parse the response into a structured format
        # This is a placeholder for the actual implementation
        return []

    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for the analysis."""
        # Implementation would calculate confidence based on various factors
        # This is a placeholder for the actual implementation
        return 0.0

    def _calculate_risk_level(self, risks: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level based on identified risks."""
        # Implementation would calculate risk level based on risk analysis
        # This is a placeholder for the actual implementation
        return "low"

    def _calculate_reading_level(self, summary: Dict[str, Any]) -> str:
        """Calculate the reading level of the plain language summary."""
        # Implementation would calculate reading level using appropriate metrics
        # This is a placeholder for the actual implementation
        return "intermediate"

    def _calculate_severity_levels(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate the distribution of severity levels in compliance issues."""
        # Implementation would calculate severity level distribution
        # This is a placeholder for the actual implementation
        return {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def _generate_risk_summary(self, risks: List[Dict[str, Any]]) -> str:
        """Generate a summary of identified risks."""
        # Implementation would generate a risk summary
        # This is a placeholder for the actual implementation
        return ""

    def _generate_risk_recommendations(self, risks: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for addressing identified risks."""
        # Implementation would generate risk recommendations
        # This is a placeholder for the actual implementation
        return []

    def _generate_change_summary(self, changes: List[Dict[str, Any]]) -> str:
        """Generate a summary of document changes."""
        # Implementation would generate a change summary
        # This is a placeholder for the actual implementation
        return ""

    def _generate_compliance_summary(self, issues: List[Dict[str, Any]]) -> str:
        """Generate a summary of compliance issues."""
        # Implementation would generate a compliance summary
        # This is a placeholder for the actual implementation
        return ""

    def _generate_compliance_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for addressing compliance issues."""
        # Implementation would generate compliance recommendations
        # This is a placeholder for the actual implementation
        return []

    def _extract_key_points(self, summary: Dict[str, Any]) -> List[str]:
        """Extract key points from a plain language summary."""
        # Implementation would extract key points
        # This is a placeholder for the actual implementation
        return [] 