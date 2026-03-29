from typing import Optional, List, Dict, Any
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from backend.models.mongodb_models import (
    Task,
    TaskCreate,
    TaskStatus,
    TaskPriority,
    User
)
from .user_repository import user_repository
import structlog

logger = structlog.get_logger()

class TaskRepository(BaseRepository[Task]):
    def __init__(self):
        super().__init__("tasks", Task)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("created_by")
        await collection.create_index("assigned_to")
        await collection.create_index("organization_id")
        await collection.create_index("status")
        await collection.create_index("priority")
        await collection.create_index("due_date")
        await collection.create_index("created_at")
        await collection.create_index("parent_task_id")
        await collection.create_index("metadata.postgres_id")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("status", ASCENDING),
            ("due_date", ASCENDING)
        ])
        await collection.create_index([
            ("assigned_to", ASCENDING),
            ("status", ASCENDING),
            ("due_date", ASCENDING)
        ])
        # Text search index
        await collection.create_index([
            ("title", "text"),
            ("description", "text"),
            ("tags", "text")
        ])

    async def create_task(
        self,
        task_data: TaskCreate,
        creator_id: str,
        organization_id: str
    ) -> Task:
        """Create a new task."""
        task_dict = task_data.dict()
        task_dict.update({
            "created_by": ObjectId(creator_id),
            "organization_id": ObjectId(organization_id),
            "status": TaskStatus.TODO,
            "created_at": datetime.utcnow()
        })
        
        if task_dict.get("assigned_to"):
            task_dict["assigned_to"] = ObjectId(task_dict["assigned_to"])
        
        try:
            return await self.create(task_dict)
        except Exception as e:
            logger.error(
                "Failed to create task",
                error=str(e),
                creator_id=creator_id,
                organization_id=organization_id
            )
            raise

    async def create_task_with_postgres_id(
        self,
        task_data: TaskCreate,
        creator_id: str,
        organization_id: str,
        postgres_id: str
    ) -> Task:
        """Create a new task and store the linked Postgres ID."""
        task_dict = task_data.dict()
        metadata = task_dict.get("metadata") or {}
        metadata["postgres_id"] = postgres_id
        task_dict["metadata"] = metadata
        task_dict.update({
            "created_by": ObjectId(creator_id),
            "organization_id": ObjectId(organization_id),
            "status": TaskStatus.TODO if not task_dict.get("status") else task_dict.get("status"),
            "created_at": datetime.utcnow()
        })

        if task_dict.get("assigned_to"):
            task_dict["assigned_to"] = ObjectId(task_dict["assigned_to"])

        try:
            return await self.create(task_dict)
        except Exception as e:
            logger.error(
                "Failed to create task with postgres_id",
                error=str(e),
                creator_id=creator_id,
                organization_id=organization_id,
                postgres_id=postgres_id
            )
            raise

    async def get_task_by_postgres_id(
        self,
        postgres_id: str,
        organization_id: Optional[str] = None
    ) -> Optional[Task]:
        """Get a task by linked Postgres ID."""
        collection = await self.collection
        query: Dict[str, Any] = {"metadata.postgres_id": postgres_id}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        doc = await collection.find_one(query)
        return Task(**doc) if doc else None

    async def update_task_by_postgres_id(
        self,
        postgres_id: str,
        update_data: Dict[str, Any],
        organization_id: Optional[str] = None
    ) -> Optional[Task]:
        """Update a task by linked Postgres ID."""
        collection = await self.collection
        query: Dict[str, Any] = {"metadata.postgres_id": postgres_id}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        doc = await collection.find_one(query)
        if not doc:
            return None
        return await self.update(str(doc["_id"]), update_data)

    async def delete_task_by_postgres_id(
        self,
        postgres_id: str,
        organization_id: Optional[str] = None
    ) -> bool:
        """Delete a task by linked Postgres ID."""
        collection = await self.collection
        query: Dict[str, Any] = {"metadata.postgres_id": postgres_id}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        doc = await collection.find_one(query)
        if not doc:
            return False
        return await self.delete(str(doc["_id"]))

    async def get_user_tasks(
        self,
        user_id: str,
        status: Optional[TaskStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Task]:
        """Get tasks assigned to a user."""
        filters = {"assigned_to": ObjectId(user_id)}
        if status:
            filters["status"] = status
            
        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("due_date", ASCENDING), ("priority", DESCENDING)]
        )

    async def get_organization_tasks(
        self,
        organization_id: str,
        status: Optional[TaskStatus] = None,
        assigned_to: Optional[str] = None,
        document_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Task]:
        """Get tasks in an organization with optional filters."""
        filters = {"organization_id": ObjectId(organization_id)}
        if status:
            filters["status"] = status
        if assigned_to:
            try:
                filters["assigned_to"] = ObjectId(assigned_to)
            except Exception:
                filters["assigned_to"] = assigned_to
        if document_id:
            filters["metadata.document_id"] = document_id
        if agent_id:
            try:
                filters["agent_id"] = ObjectId(agent_id)
            except Exception:
                filters["agent_id"] = agent_id

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("due_date", ASCENDING), ("priority", DESCENDING)]
        )

    async def search_tasks(
        self,
        query: str,
        organization_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Task]:
        """Search tasks by title, description, or tags."""
        search_filter = {
            "$text": {"$search": query}
        }
        
        if organization_id:
            search_filter["organization_id"] = ObjectId(organization_id)
        if status:
            search_filter["status"] = status

        return await self.find_many(
            search_filter,
            skip=skip,
            limit=limit,
            sort=[("score", {"$meta": "textScore"})]
        )

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        user_id: str
    ) -> Optional[Task]:
        """Update task status and add to history."""
        update_data = {
            "status": status,
            "$push": {
                "status_history": {
                    "status": status,
                    "changed_by": ObjectId(user_id),
                    "changed_at": datetime.utcnow()
                }
            }
        }
        return await self.update(task_id, update_data)

    async def assign_task(
        self,
        task_id: str,
        user_id: str,
        assigned_by: str
    ) -> Optional[Task]:
        """Assign a task to a user."""
        update_data = {
            "assigned_to": ObjectId(user_id),
            "$push": {
                "assignment_history": {
                    "assigned_to": ObjectId(user_id),
                    "assigned_by": ObjectId(assigned_by),
                    "assigned_at": datetime.utcnow()
                }
            }
        }
        return await self.update(task_id, update_data)

    async def add_comment(
        self,
        task_id: str,
        user_id: str,
        content: str
    ) -> Optional[Task]:
        """Add a comment to a task."""
        comment = {
            "content": content,
            "user_id": ObjectId(user_id),
            "created_at": datetime.utcnow()
        }
        return await self.update(
            task_id,
            {"$push": {"comments": comment}}
        )

    async def add_subtask(
        self,
        parent_task_id: str,
        subtask_data: TaskCreate,
        creator_id: str
    ) -> Optional[Task]:
        """Add a subtask to a parent task."""
        parent_task = await self.get_by_id(parent_task_id)
        if not parent_task:
            return None

        subtask_dict = subtask_data.dict()
        subtask_dict.update({
            "created_by": ObjectId(creator_id),
            "organization_id": parent_task.organization_id,
            "parent_task_id": ObjectId(parent_task_id),
            "status": TaskStatus.TODO,
            "created_at": datetime.utcnow()
        })

        try:
            subtask = await self.create(subtask_dict)
            # Update parent task's subtasks list
            await self.update(
                parent_task_id,
                {"$push": {"subtasks": {"id": subtask.id, "title": subtask.title}}}
            )
            return subtask
        except Exception as e:
            logger.error(
                "Failed to create subtask",
                error=str(e),
                parent_task_id=parent_task_id,
                creator_id=creator_id
            )
            raise

    async def get_subtasks(
        self,
        parent_task_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Task]:
        """Get all subtasks of a parent task."""
        return await self.find_many(
            {"parent_task_id": ObjectId(parent_task_id)},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def get_task_with_subtasks(self, task_id: str) -> Dict[str, Any]:
        """Get a task with all its subtasks."""
        pipeline = [
            {"$match": {"_id": ObjectId(task_id)}},
            {"$lookup": {
                "from": "tasks",
                "localField": "_id",
                "foreignField": "parent_task_id",
                "as": "subtasks"
            }},
            {"$sort": {"subtasks.created_at": ASCENDING}}
        ]
        
        results = await self.aggregate(pipeline)
        return results[0] if results else {}

    async def get_task_stats(
        self,
        organization_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get task statistics for an organization or user."""
        match = {"organization_id": ObjectId(organization_id)}
        if user_id:
            match["assigned_to"] = ObjectId(user_id)

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1},
                "tasks": {"$push": {
                    "id": "$_id",
                    "title": "$title",
                    "due_date": "$due_date"
                }}
            }},
            {"$group": {
                "_id": None,
                "total": {"$sum": "$count"},
                "statuses": {
                    "$push": {
                        "status": "$_id",
                        "count": "$count",
                        "tasks": "$tasks"
                    }
                }
            }}
        ]
        
        results = await self.aggregate(pipeline)
        return results[0] if results else {"total": 0, "statuses": []}

    async def get_overdue_tasks(
        self,
        organization_id: str,
        user_id: Optional[str] = None
    ) -> List[Task]:
        """Get overdue tasks."""
        filters = {
            "organization_id": ObjectId(organization_id),
            "status": {"$nin": [TaskStatus.COMPLETED, TaskStatus.ARCHIVED]},
            "due_date": {"$lt": datetime.utcnow()}
        }
        
        if user_id:
            filters["assigned_to"] = ObjectId(user_id)

        return await self.find_many(
            filters,
            sort=[("due_date", ASCENDING)]
        )

    async def get_upcoming_tasks(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        days: int = 7
    ) -> List[Task]:
        """Get upcoming tasks within specified days."""
        end_date = datetime.utcnow() + timedelta(days=days)
        filters = {
            "organization_id": ObjectId(organization_id),
            "status": {"$nin": [TaskStatus.COMPLETED, TaskStatus.ARCHIVED]},
            "due_date": {
                "$gte": datetime.utcnow(),
                "$lte": end_date
            }
        }
        
        if user_id:
            filters["assigned_to"] = ObjectId(user_id)

        return await self.find_many(
            filters,
            sort=[("due_date", ASCENDING)]
        )

# Create a singleton instance
task_repository = TaskRepository() 
