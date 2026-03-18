from typing import Optional, List, Dict, Any
from pymongo import ASCENDING, DESCENDING
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from datetime import datetime
from backend.db.mongodb.base_repository import BaseRepository
from backend.db.mongodb.models.document import (
    Document,
    DocumentStatus,
    DocumentType,
)
import structlog
from backend.db.mongodb.mongodb import get_mongodb

logger = structlog.get_logger()

class DocumentRepository(BaseRepository[Document]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__("documents", Document)
        self.db = db
        self._agent_service = None

    @property
    async def agent_service(self):
        """Lazy load the agent service to avoid circular imports."""
        if self._agent_service is None:
            from backend.agents.agent_service import AgentService
            self._agent_service = AgentService()
        return self._agent_service

    @classmethod
    async def create(cls) -> 'DocumentRepository':
        db = await get_mongodb()
        return cls(db)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("organization_id")
        await collection.create_index("created_by")
        await collection.create_index("status")
        await collection.create_index("document_type")
        await collection.create_index("created_at", DESCENDING)
        # Compound indexes for filtering and sorting
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("created_at", DESCENDING)
        ])
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("status", ASCENDING),
        ])
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("document_type", ASCENDING),
        ])
        # Text search index for name and content (assuming extracted text is stored/indexed)
        await collection.create_index([
            ("name", "text"),
            ("extracted_text", "text") # Assuming an 'extracted_text' field exists
        ])

    async def create_document(
        self,
        document_data: Dict[str, Any]
    ) -> Document:
        """Create a new document entry."""
        # Ensure organization_id and created_by are ObjectId if they exist
        if "organization_id" in document_data and isinstance(document_data["organization_id"], str):
            document_data["organization_id"] = ObjectId(document_data["organization_id"])
        if "created_by" in document_data and isinstance(document_data["created_by"], str):
            document_data["created_by"] = ObjectId(document_data["created_by"])

        # Set default status and timestamps if not provided
        if "status" not in document_data:
            document_data["status"] = DocumentStatus.UPLOADED.value # Or DocumentStatus.PROCESSING
        if "created_at" not in document_data:
            document_data["created_at"] = datetime.utcnow()
        if "updated_at" not in document_data:
             document_data["updated_at"] = datetime.utcnow()

        return await self.create(document_data)

    async def get_document_by_id(
        self,
        document_id: str,
        organization_id: Optional[str] = None # Optional: enforce organization ownership
    ) -> Optional[Document]:
        """Get a document by its ID."""
        filters = {"_id": ObjectId(document_id)}
        if organization_id:
             filters["organization_id"] = ObjectId(organization_id)
        return await self.find_one(filters)

    async def update_document(
        self,
        document_id: str,
        update_data: Dict[str, Any],
        organization_id: Optional[str] = None # Optional: enforce organization ownership
    ) -> Optional[Document]:
        """Update a document."""
        filters = {"_id": ObjectId(document_id)}
        if organization_id:
             filters["organization_id"] = ObjectId(organization_id)

        # Update updated_at timestamp
        if "$set" in update_data:
            update_data["$set"]["updated_at"] = datetime.utcnow()
        elif "$set" not in update_data:
            update_data["$set"] = {"updated_at": datetime.utcnow()}

        return await self.update_one(filters, update_data)

    async def delete_document(
        self,
        document_id: str,
        organization_id: Optional[str] = None # Optional: enforce organization ownership
    ) -> bool:
        """Delete a document."""
        filters = {"_id": ObjectId(document_id)}
        if organization_id:
             filters["organization_id"] = ObjectId(organization_id)
        result = await self.delete_one(filters)
        return result.deleted_count > 0

    async def get_documents_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
        sort_by: str = "created_at",
        sort_order: int = DESCENDING
    ) -> List[Document]:
        """Get all documents owned by a specific user."""
        try:
            filters = {"created_by": ObjectId(user_id)}
        except Exception:
            filters = {"created_by": user_id}
        sort_criteria = [(sort_by, sort_order)]
        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=sort_criteria,
        )

    async def get_organization_documents(
        self,
        organization_id: str,
        status: Optional[DocumentStatus] = None,
        document_type: Optional[DocumentType] = None,
        search_query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "created_at",
        sort_order: int = DESCENDING
    ) -> List[Document]:
        """Get all documents for an organization with filtering, searching, and pagination."""
        filters = {"organization_id": ObjectId(organization_id)}
        if status:
            filters["status"] = status
        if document_type:
            filters["document_type"] = document_type
        if search_query:
             filters["$text"] = {"$search": search_query}

        sort_criteria = [(sort_by, sort_order)]
        # For text search, add text score sorting if no other sort is specified
        if search_query and not sort_by:
             sort_criteria = [("score", {"$meta": "textScore"})]

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=sort_criteria,
            # Projection for text search score if needed
            projection={"score": {"$meta": "textScore"}} if search_query and not sort_by else None
        )


    async def get_document_summary(
        self,
        organization_id: str
    ) -> Dict[str, Any]:
        """Get summary statistics for documents in an organization."""
        pipeline = [
            {"$match": {"organization_id": ObjectId(organization_id)}},
            {"$group": {
                "_id": {
                    "status": "$status",
                    "type": "$document_type"
                },
                "count": {"$sum": 1}
            }},
            {"$group": {
                "_id": None,
                "total_count": {"$sum": "$count"},
                "summary_by_status": {"$push": {"status": "$_id.status", "count": "$count"}},
                "summary_by_type": {"$push": {"type": "$_id.type", "count": "$count"}}
            }},            {"$project": {
                "_id": 0,
                "total_count": 1,
                "summary_by_status": 1,
                "summary_by_type": 1
            }}
        ]
        
        results = await self.aggregate(pipeline)
        if results:
            return results[0]
        return {
            "total_count": 0,
            "summary_by_status": [],
            "summary_by_type": []
        }
        
    async def get_recent_documents_with_counts(
        self,
        organization_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get a list of recent documents for an organization with counts of extracted items and tasks."""
        pipeline = [
            {"$match": {"organization_id": ObjectId(organization_id)}},
            {"$sort": {"created_at": DESCENDING}},
            {"$limit": limit},
            {"$lookup": {
                "from": "tasks", # The collection to join with
                "localField": "_id", # Field from the input documents (documents collection)
                "foreignField": "document_id", # Field from the documents of the "from" collection (tasks collection)
                "as": "tasks" # Output array field name
            }},
            {"$addFields": {
                # Assuming extraction_result is a dict and count is number of keys
                # Adjust this based on actual structure of extraction_result
                "extracted_items_count": {"$size": {"$objectToArray": "$extraction_result"}},
                "tasks_created_count": {"$size": "$tasks"}
            }},
            {"$project": {
                "_id": 0,
                "id": {"$toString": "$_id"},
                "name": 1,
                "document_type": 1,
                "created_at": 1,
                "extracted_items_count": 1,
                "tasks_created_count": 1
            }}
        ]

        results = await self.aggregate(pipeline)
        return results
        
    async def extract_document_data(
        self,
        document_id: str,
        organization_id: str,
        extraction_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract data from a document using AI agents."""
        try:
            document = await self.get_document_by_id(document_id, organization_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Update document status to processing
            await self.update_document(
                document_id=document_id,
                update_data={"$set": {"status": DocumentStatus.PROCESSING.value}}
            )

            # Get agent service lazily
            agent_service = await self.agent_service
            document_agent = await agent_service.get_agent("document")

            # Extract content from document
            content = await self._get_document_content(document)

            # Process with document agent
            extraction_result = await document_agent.process_async({
                "document_id": document_id,
                "content": content,
                "config": extraction_config or {}
            })

            # Update document with extraction results
            await self.update_document(
                document_id=document_id,
                update_data={
                    "$set": {
                        "status": DocumentStatus.PROCESSED.value,
                        "extraction_result": extraction_result,
                        "extraction_status": "completed",
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            return extraction_result

        except Exception as e:
            logger.error(f"Error extracting document data: {str(e)}")
            # Update document status to failed
            await self.update_document(
                document_id=document_id,
                update_data={
                    "$set": {
                        "status": DocumentStatus.FAILED.value,
                        "extraction_error": str(e),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            raise
    
    async def _get_document_content(self, document: Document) -> str:
        """
        Get the content of a document.
        
        In a real implementation, this would retrieve the actual document content
        from storage (e.g., S3, file system).
        
        Args:
            document: Document model
            
        Returns:
            Document content as text
        """
        # TODO: Implement actual document content retrieval
        # This is a placeholder - in a real implementation, you would:
        # 1. Get the file from storage using document.file_url
        # 2. Extract the text content (using appropriate parser for the file type)
        # 3. Return the text content
        
        # For now, return a placeholder
        return f"This is the content of document '{document.name}' of type {document.document_type}."
        
    async def create_tasks_from_document(
        self,
        document_id: str,
        organization_id: str,
        created_by: str,
        task_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create tasks from a document using Perplexity AI.
        
        Args:
            document_id: Document ID
            organization_id: Organization ID
            created_by: User ID of task creator
            task_config: Optional task generation configuration
            
        Returns:
            List of created tasks
        """
        # Get the document
        document = await self.get_document_by_id(document_id, organization_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
            
        # Get document content
        document_content = await self._get_document_content(document)
        
        # Configure document agent
        agent_config = {
            "type": "document",
            "model_config": {
                "model": "sonar-large-online"  # Use Perplexity's Sonar model
            }
        }
        
        try:
            # Create document agent
            document_agent = DocumentAgent(agent_config)
            
            # Process document to extract tasks
            document_data = {
                "text": document_content,
                "metadata": {
                    "id": str(document.id),
                    "name": document.name,
                    "document_type": document.document_type
                },
                "user_context": task_config or {}  # Pass any task configuration as user context
            }
            
            # Extract tasks
            result = await document_agent.process_async(document_data)
            
            # Extract tasks from result
            tasks = result.get("tasks", [])
            
            # Create tasks in database
            created_tasks = []
            for task in tasks:
                # Map extracted task properties to task model
                task_data = {
                    "title": task.get("title", "Untitled Task"),
                    "description": task.get("description", ""),
                    "priority": task.get("priority", "medium"),
                    "status": "open",
                    "organization_id": ObjectId(organization_id),
                    "created_by": ObjectId(created_by),
                    "document_id": ObjectId(document_id),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                # Add deadline if present
                if "deadline" in task:
                    # In a real implementation, parse the deadline string to a datetime
                    task_data["due_date"] = task["deadline"]
                
                # Add assignee if present
                if "assignee" in task:
                    # In a real implementation, look up the user ID for this assignee name
                    # For now, just store the name
                    task_data["assignee_name"] = task["assignee"]
                
                # Create task in database
                # In a real implementation, use a task repository
                # For now, just append to our result list
                created_tasks.append(task_data)
            
            return created_tasks
            
        except Exception as e:
            logger.error(f"Error creating tasks from document: {str(e)}")
            raise
    
    async def query_document(
        self,
        document_id: str,
        organization_id: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        include_extracted_data: bool = False
    ) -> Dict[str, Any]:
        """
        Query document content using natural language.
        
        Args:
            document_id: Document ID
            organization_id: Organization ID
            query: Natural language query
            filters: Optional query filters
            include_extracted_data: Whether to include full extracted data
            
        Returns:
            Query results
        """
        # Get the document
        document = await self.get_document_by_id(document_id, organization_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")
            
        # Get document content
        document_content = await self._get_document_content(document)
        
        # Configure document agent
        agent_config = {
            "type": "document",
            "model_config": {
                "model": "sonar-large-online"  # Use Perplexity's Sonar model
            }
        }
        
        try:
            # Create document agent
            document_agent = DocumentAgent(agent_config)
            
            # Get existing extraction if available
            extraction_result = getattr(document, "extraction_result", None)
            
            # Process query
            document_data = {
                "text": document_content,
                "metadata": {
                    "id": str(document.id),
                    "name": document.name,
                    "document_type": document.document_type
                },
                "extracted_data": extraction_result
            }
            
            # Use LLM client for querying
            llm_client = document_agent.llm_client
            if not llm_client:
                document_agent.initialize_models()
                llm_client = document_agent.llm_client
            
            if not llm_client:
                raise ValueError("Failed to initialize LLM client")
            
            # Query document via provider-agnostic interface
            messages = [
                {"role": "system", "content": f"You are analyzing the following document. Answer questions based on its content.\n\nDocument:\n{document_content[:8000]}"},
                {"role": "user", "content": query}
            ]
            response = await llm_client.generate(messages)
            
            # Format response
            result = {
                "query": query,
                "response": response.content,
                "document_id": str(document.id),
                "document_name": document.name
            }
            
            # Add citations if available
            if response.citations:
                result["citations"] = response.citations
            
            # Add extracted data if requested
            if include_extracted_data and extraction_result:
                result["extracted_data"] = extraction_result
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            raise

# Create a singleton instance
document_repository: Optional[DocumentRepository] = None

async def get_document_repository() -> DocumentRepository:
    global document_repository
    if document_repository is None:
        document_repository = await DocumentRepository.create()
    return document_repository 