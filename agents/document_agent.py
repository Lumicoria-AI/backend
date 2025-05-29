from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional
import json
import structlog
import asyncio
from datetime import datetime, timedelta
import re

# Configure logger
logger = structlog.get_logger(__name__)

class DocumentAgent(BaseAgent):
    """Agent for processing documents using Perplexity AI.
    
    This agent extracts key information, tasks, dates, and insights from 
    documents using Perplexity's Sonar models.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default extraction targets if not specified in config
        self.extraction_targets = config.get("extraction_targets", [
            "tasks", "dates", "names", "organizations", 
            "monetary amounts", "action items", "key points"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the document with natural language questions.
        
        Args:
            query: The natural language query about the document
            context: Optional context including document content and metadata
            
        Returns:
            Dictionary containing the response and any relevant extracted information
        """
        if not context or not context.get("document_text"):
            return {"error": "No document content provided in context"}
            
        document_text = context["document_text"]
        prompt = (
            f"Using the following document content, answer this question: {query}\n\n"
            f"Document content:\n{document_text[:8000]}..."  # Limit document length
        )
        
        try:
            if self.perplexity_client:
                messages = [{"role": "user", "content": prompt}]
                response = await self.perplexity_client.chat_completion(messages)
                return {
                    "response": response.content,
                    "query": query,
                    "document_id": context.get("document_id"),
                    "confidence": response.metadata.get("confidence", 0.0) if response.metadata else 0.0
                }
            else:
                return {"error": "Perplexity client not initialized"}
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            return {"error": f"Failed to process query: {str(e)}"}

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process document data asynchronously.
        
        Args:
            data: Dictionary containing document text and metadata
            
        Returns:
            Dictionary with extracted information and analysis
        """
        try:
            # Extract document content and metadata
            document_text = data.get("text", "")
            document_metadata = data.get("metadata", {})
            
            if not document_text:
                return {"error": "No document text provided"}
            
            prompt = (
                f"Analyze the following document and extract key information including "
                f"{', '.join(self.extraction_targets)}. For each extracted item, include "
                f"the exact text from the document and the relevant context.\n\n"
                f"Document content:\n{document_text[:8000]}..."  # Limit document length
            )
            
            if self.perplexity_client:
                messages = [{"role": "user", "content": prompt}]
                response = await self.perplexity_client.chat_completion(messages)
                
                # Process and structure the response
                return {
                    "analysis": response.content,
                    "metadata": document_metadata,
                    "extraction_targets": self.extraction_targets,
                    "model_used": self.model_config.get("model", "unknown"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "confidence": response.metadata.get("confidence", 0.0) if response.metadata else 0.0
                }
            else:
                return {"error": "Perplexity client not initialized"}
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Failed to process document: {str(e)}"}
            
    def process(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document to extract key information and generate tasks.
        
        Args:
            document_data: Dictionary containing document text, metadata, and context.
            
        Returns:
            Dictionary with extracted information, tasks, and insights.
        """
        # Extract document content and metadata
        document_text = document_data.get("text", "")
        document_metadata = document_data.get("metadata", {})
        user_context = document_data.get("user_context", {})
        
        if not document_text:
            return {"error": "No document text provided"}
            
        # Use the configured model to process the document
        prompt = (
            f"Analyze the following document and extract key information including "
            f"{', '.join(self.extraction_targets)}. For each extracted item, include "
            f"the exact text from the document and the relevant context."
        )
        
        try:
            # Get basic extraction first
            extraction_result = self._call_model(
                prompt=prompt, 
                document=document_text,
                model=self.model_config.get("model")
            )
            
            # Generate tasks separately for better specialization
            tasks_prompt = (
                f"Extract all tasks, deadlines, and action items from this document. "
                f"Format them as a list where each task includes a title, priority (High/Medium/Low), "
                f"deadline (if available), and responsible party (if mentioned). "
                f"Only include actionable items that require someone to do something."
            )
            
            # This could run in parallel with extraction in a real implementation
            tasks_result = self._call_model(
                prompt=tasks_prompt,
                document=document_text,
                model=self.model_config.get("model")
            )
            
            # Parse tasks into structured format
            parsed_tasks = self._parse_tasks(tasks_result)
            
            # Create comprehensive response
            result = {
                "document_id": document_metadata.get("id", ""),
                "extracted_info": self._parse_extraction(extraction_result),
                "tasks": parsed_tasks,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Document processing failed: {str(e)}"}

    def _parse_extraction(self, extraction_text: str) -> Dict[str, Any]:
        """Parse the extraction results into a structured format.
        
        Args:
            extraction_text: Raw text from the model extraction
            
        Returns:
            Structured dictionary with categorized extractions
        """
        try:
            # Try to parse as JSON first (in case model returned JSON)
            try:
                return json.loads(extraction_text)
            except json.JSONDecodeError:
                # If not JSON, use regex-based parsing
                pass
            
            # Initialize categories
            extraction_dict = {
                "dates": [],
                "names": [],
                "organizations": [],
                "monetary_amounts": [],
                "action_items": [],
                "key_points": [],
                "raw_extraction": extraction_text
            }
            
            # Extract dates with regex
            date_patterns = [
                r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",  # MM/DD/YYYY or DD/MM/YYYY
                r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",  # 1 Jan 2024
                r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4})\b",  # January 1st, 2024
            ]
            
            for pattern in date_patterns:
                for match in re.finditer(pattern, extraction_text, re.IGNORECASE):
                    if match.group(1) not in [d["text"] for d in extraction_dict["dates"]]:
                        extraction_dict["dates"].append({"text": match.group(1)})
            
            # Use section-based parsing for the rest
            sections = {
                "Names:": "names",
                "Organizations:": "organizations",
                "Monetary Amounts:": "monetary_amounts",
                "Action Items:": "action_items",
                "Key Points:": "key_points",
                "Tasks:": "action_items"
            }
            
            lines = extraction_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line is a section header
                for header, section_name in sections.items():
                    if line.startswith(header) or line == header:
                        current_section = section_name
                        break
                
                # If we're in a section and this isn't a header, it's an item
                if current_section and not any(line.startswith(h) for h in sections.keys()):
                    # Clean up bullet points and other markers
                    item_text = re.sub(r"^[-•*]\s*", "", line)
                    if item_text and len(item_text) > 3:  # Avoid very short items
                        if item_text not in [i["text"] for i in extraction_dict[current_section]]:
                            extraction_dict[current_section].append({"text": item_text})
            
            return extraction_dict
            
        except Exception as e:
            logger.error(f"Error parsing extraction: {str(e)}")
            return {"raw_extraction": extraction_text, "parsing_error": str(e)}

    def _parse_tasks(self, tasks_text: str) -> List[Dict[str, Any]]:
        """Parse the tasks extraction into a structured list.
        
        Args:
            tasks_text: Raw text from the model's task extraction
            
        Returns:
            List of structured task dictionaries
        """
        try:
            # Try to parse as JSON first
            try:
                parsed_json = json.loads(tasks_text)
                if isinstance(parsed_json, list):
                    return parsed_json
                elif isinstance(parsed_json, dict) and "tasks" in parsed_json:
                    return parsed_json["tasks"]
            except json.JSONDecodeError:
                # If not JSON, use regex-based parsing
                pass
                
            # Parse free-text format
            tasks = []
            current_task = {}
            
            # Detect bullet points or numbered items
            tasks_pattern = r"(?:^|\n)(?:\d+[\.\)]\s*|[-•*]\s*|Task\s*\d+:\s*|)([A-Z].*?)(?:\n|$)"
            
            for match in re.finditer(tasks_pattern, tasks_text, re.MULTILINE):
                task_text = match.group(1).strip()
                
                # Skip empty or very short tasks
                if not task_text or len(task_text) < 5:
                    continue
                    
                task = {"title": task_text}
                
                # Extract priority if present
                priority_match = re.search(r"\b(High|Medium|Low)\s+priority\b", task_text, re.IGNORECASE)
                if priority_match:
                    task["priority"] = priority_match.group(1).lower()
                
                # Extract deadline if present
                deadline_patterns = [
                    r"due\s+by\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"due\s+on\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                    r"deadline:\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"by\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                ]
                
                for pattern in deadline_patterns:
                    deadline_match = re.search(pattern, task_text, re.IGNORECASE)
                    if deadline_match:
                        task["deadline"] = deadline_match.group(1)
                        break
                
                # Extract responsible party if present
                assignee_patterns = [
                    r"assigned\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                    r"responsible:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                ]
                
                for pattern in assignee_patterns:
                    assignee_match = re.search(pattern, task_text, re.IGNORECASE)
                    if assignee_match:
                        task["assignee"] = assignee_match.group(1)
                        break
                
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error parsing tasks: {str(e)}")
            return [{"title": tasks_text, "parsing_error": str(e)}]
