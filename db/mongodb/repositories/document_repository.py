from typing import Optional, List, Dict, Any
from motor.motor_asyncio import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime
from ..base_repository import BaseRepository
from ...models.mongodb_models import (
    Document,
    DocumentStatus,
    DocumentType,
)
import structlog

logger = structlog.get_logger()

class DocumentRepository(BaseRepository[Document]):
    def __init__(self):
        super().__init__("documents", Document)

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
            }},
            {"$project": {
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

# Create a singleton instance
document_repository = DocumentRepository() 