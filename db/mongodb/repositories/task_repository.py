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

        # ── Phase 1 additions: invite path, agent assignment, calendar link ──
        # Each is sparse so legacy rows without the field don't bloat the index.
        await collection.create_index("assigned_to_email", sparse=True)
        await collection.create_index("assigned_to_agent", sparse=True)
        await collection.create_index("assignee_kind", sparse=True)
        await collection.create_index("invite_id", sparse=True)
        await collection.create_index("calendar_event_id", sparse=True)
        await collection.create_index("gcal_event_id", sparse=True)
        await collection.create_index("project_id", sparse=True)
        await collection.create_index("agent_proposal.status", sparse=True)

        # Compound used by the reminder cron: org + status + due_date already
        # covered above.  Add one more for "tasks that have agent proposals
        # awaiting review" — the in-app review surface in Phase 6.
        await collection.create_index(
            [
                ("organization_id", ASCENDING),
                ("agent_proposal.status", ASCENDING),
                ("updated_at", DESCENDING),
            ],
            name="agent_proposal_review_idx",
            sparse=True,
        )

        # Reminder pipeline (Phase 4): scan upcoming dues efficiently.
        await collection.create_index(
            [("due_date", ASCENDING), ("status", ASCENDING)],
            name="reminder_scan_idx",
        )

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
        """Create a new task.

        Phase 1 — also accepts the new optional fields on TaskCreate
        (assignee_kind, assigned_to_email, assigned_to_agent) and resolves
        `assignee_kind` automatically when only one of the assignment
        fields is provided.
        """
        task_dict = task_data.dict()
        task_dict.update({
            "created_by": self._coerce_oid(creator_id),
            "organization_id": self._coerce_oid(organization_id),
            "status": task_dict.get("status") or TaskStatus.TODO,
            "created_at": datetime.utcnow(),
        })

        if task_dict.get("assigned_to"):
            task_dict["assigned_to"] = self._coerce_oid(task_dict["assigned_to"])

        # Auto-derive assignee_kind from the populated assignment fields when
        # the caller didn't specify one.  This keeps existing callers working
        # while letting Phase 5/6 paths use the explicit field.
        if not task_dict.get("assignee_kind"):
            has_user = bool(task_dict.get("assigned_to"))
            has_email = bool(task_dict.get("assigned_to_email"))
            has_agent = bool(task_dict.get("assigned_to_agent"))
            if has_user and has_agent:
                task_dict["assignee_kind"] = "user_and_agent"
            elif has_agent and not has_user and not has_email:
                task_dict["assignee_kind"] = "agent"
            elif has_email and not has_user:
                task_dict["assignee_kind"] = "email_invite"
            elif has_user:
                task_dict["assignee_kind"] = "user"
            else:
                task_dict["assignee_kind"] = None

        # Normalise enum values stored as enum to strings (Motor friendlier)
        for k in ("status", "priority", "assignee_kind"):
            v = task_dict.get(k)
            if hasattr(v, "value"):
                task_dict[k] = v.value

        try:
            return await self.create(task_dict)
        except Exception as e:
            logger.error(
                "Failed to create task",
                error=str(e),
                creator_id=creator_id,
                organization_id=organization_id,
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

    # ── CRUD helpers used by the /tasks API endpoint ───────────────────────
    # These wrap the BaseRepository primitives with org-scoped lookups so the
    # API layer (backend/api/v1/endpoints/tasks.py) gets tenant isolation for
    # free.  Org IDs may arrive as ObjectId strings, UUIDs, or Firebase UIDs;
    # we coerce to ObjectId where possible and fall back to a string compare.

    @staticmethod
    def _coerce_oid(value: Optional[str]) -> Optional[Any]:
        """ObjectId if `value` looks like one, otherwise the original string."""
        if value is None:
            return None
        if isinstance(value, ObjectId):
            return value
        try:
            return ObjectId(str(value))
        except Exception:
            return str(value)

    async def get_task_by_id(
        self,
        task_id: str,
        organization_id: Optional[str] = None,
    ) -> Optional[Task]:
        """Fetch a task by id, optionally constrained to an organization."""
        oid = self._coerce_oid(task_id)
        if not isinstance(oid, ObjectId):
            return None
        collection = await self.collection
        query: Dict[str, Any] = {"_id": oid}
        if organization_id:
            query["organization_id"] = self._coerce_oid(organization_id)
        doc = await collection.find_one(query)
        return Task(**doc) if doc else None

    async def update_task(
        self,
        task_id: str,
        organization_id: str,
        update_data: Dict[str, Any],
        *,
        changed_by: Optional[str] = None,
        changed_by_name: Optional[str] = None,
    ) -> Optional[Task]:
        """Org-scoped task update.

        Auto-historizes `status` and `assigned_to` transitions.  When
        `changed_by` (a user id) is supplied, it's recorded on the history
        entry so the UI can render "Completed by Grace" without a second
        round-trip.
        """
        oid = self._coerce_oid(task_id)
        if not isinstance(oid, ObjectId):
            return None
        collection = await self.collection

        # Strip Nones — except for fields we explicitly allow nulling.
        nullable_fields = {"assigned_to", "assigned_to_email", "due_date", "calendar_event_id", "gcal_event_id", "invite_id"}
        cleaned: Dict[str, Any] = {}
        for k, v in (update_data or {}).items():
            if k in {"_id", "id", "created_by", "organization_id", "created_at"}:
                continue
            if v is None and k not in nullable_fields:
                continue
            cleaned[k] = v
        if not cleaned:
            return await self.get_task_by_id(task_id, organization_id)

        # Coerce assignee to ObjectId when possible.
        if "assigned_to" in cleaned and cleaned["assigned_to"]:
            cleaned["assigned_to"] = self._coerce_oid(cleaned["assigned_to"])

        now = datetime.utcnow()
        cleaned["updated_at"] = now

        update_doc: Dict[str, Any] = {"$set": cleaned}
        pushes: Dict[str, Any] = {}

        # Auto-historize status changes through this generic path.
        if "status" in cleaned:
            history_entry: Dict[str, Any] = {
                "status": cleaned["status"],
                "changed_at": now,
            }
            if changed_by:
                history_entry["changed_by"] = self._coerce_oid(changed_by)
            if changed_by_name:
                history_entry["changed_by_name"] = changed_by_name
            pushes["status_history"] = history_entry

        # Also historize assignment changes — useful for "who reassigned this".
        if "assigned_to" in cleaned or "assigned_to_email" in cleaned:
            assign_entry: Dict[str, Any] = {
                "assigned_to": cleaned.get("assigned_to"),
                "assigned_to_email": cleaned.get("assigned_to_email"),
                "changed_at": now,
            }
            if changed_by:
                assign_entry["changed_by"] = self._coerce_oid(changed_by)
            if changed_by_name:
                assign_entry["changed_by_name"] = changed_by_name
            pushes["assignment_history"] = assign_entry

        if pushes:
            update_doc["$push"] = pushes

        query: Dict[str, Any] = {"_id": oid}
        if organization_id:
            query["organization_id"] = self._coerce_oid(organization_id)

        result = await collection.find_one_and_update(
            query, update_doc, return_document=True
        )
        return Task(**result) if result else None

    async def delete_task(
        self,
        task_id: str,
        organization_id: str,
    ) -> bool:
        """Org-scoped task delete."""
        oid = self._coerce_oid(task_id)
        if not isinstance(oid, ObjectId):
            return False
        collection = await self.collection
        query: Dict[str, Any] = {"_id": oid}
        if organization_id:
            query["organization_id"] = self._coerce_oid(organization_id)
        result = await collection.delete_one(query)
        return result.deleted_count > 0

    # ── Phase 6: agent proposal helpers ──────────────────────────────────
    async def find_tasks_for_executor(
        self,
        *,
        limit: int = 25,
        organization_id: Optional[str] = None,
    ) -> List[Task]:
        """Return tasks whose `assigned_to_agent` is set AND whose proposal
        is either missing (never run) or in the REVISION state (human
        asked for a redo).  Used by the autonomous task executor.
        """
        collection = await self.collection
        query: Dict[str, Any] = {
            "assigned_to_agent": {"$exists": True, "$ne": None, "$nin": [""]},
            "status": {"$nin": [TaskStatus.COMPLETED.value if hasattr(TaskStatus.COMPLETED, "value") else "completed", "cancelled"]},
            "$or": [
                {"agent_proposal": {"$exists": False}},
                {"agent_proposal": None},
                {"agent_proposal.status": "revision"},
            ],
        }
        if organization_id:
            query["organization_id"] = self._coerce_oid(organization_id)

        cursor = collection.find(query).sort("created_at", ASCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        out: List[Task] = []
        for d in docs:
            try:
                out.append(Task(**d))
            except Exception:
                continue
        return out

    async def set_agent_proposal(
        self,
        task_id: str,
        organization_id: Optional[str],
        proposal: Dict[str, Any],
    ) -> Optional[Task]:
        """Write the full `agent_proposal` block atomically.  Used by the
        executor when it stores a new draft or transitions the status.
        """
        oid = self._coerce_oid(task_id)
        if not isinstance(oid, ObjectId):
            return None
        collection = await self.collection
        query: Dict[str, Any] = {"_id": oid}
        if organization_id:
            query["organization_id"] = self._coerce_oid(organization_id)

        proposal = dict(proposal or {})
        proposal.setdefault("updated_at", datetime.utcnow())
        result = await collection.find_one_and_update(
            query,
            {"$set": {"agent_proposal": proposal, "updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return Task(**result) if result else None

    async def list_proposals_awaiting_review(
        self,
        organization_id: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Task]:
        """Used by the bell + the /tasks proposals filter."""
        collection = await self.collection
        query: Dict[str, Any] = {
            "organization_id": self._coerce_oid(organization_id),
            "agent_proposal.status": "pending_review",
        }
        if user_id:
            query["$or"] = [
                {"assigned_to": self._coerce_oid(user_id)},
                {"created_by": self._coerce_oid(user_id)},
            ]
        cursor = collection.find(query).sort("agent_proposal.updated_at", DESCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        out: List[Task] = []
        for d in docs:
            try:
                out.append(Task(**d))
            except Exception:
                continue
        return out

    async def get_task_analytics(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        time_range: str = "7d",
    ) -> Dict[str, Any]:
        """Lightweight analytics for the /tasks/analytics endpoint.

        Returns counts by status, totals, completion rate, and a per-day
        completion series for the requested window (1d / 7d / 30d / 90d / 1y).
        """
        window_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 7)
        since = datetime.utcnow() - timedelta(days=window_days)

        match: Dict[str, Any] = {
            "organization_id": self._coerce_oid(organization_id),
            "created_at": {"$gte": since},
        }
        if user_id:
            match["assigned_to"] = self._coerce_oid(user_id)

        collection = await self.collection

        # Bucket by status
        status_pipeline = [
            {"$match": match},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        status_rows = await collection.aggregate(status_pipeline).to_list(length=None)
        by_status = {row["_id"] or "unknown": row["count"] for row in status_rows}
        total = sum(by_status.values())
        completed = by_status.get("completed", 0) + by_status.get(TaskStatus.COMPLETED, 0)

        # Completion series (per-day)
        completion_pipeline = [
            {"$match": {**match, "status": {"$in": ["completed", TaskStatus.COMPLETED]}}},
            {
                "$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$updated_at"}},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        completion_rows = await collection.aggregate(completion_pipeline).to_list(length=None)
        by_day = {row["_id"]: row["count"] for row in completion_rows if row["_id"]}

        return {
            "time_range": time_range,
            "since": since.isoformat() + "Z",
            "total": total,
            "completed": completed,
            "completion_rate": (completed / total) if total else 0.0,
            "by_status": by_status,
            "completions_by_day": by_day,
        }

# Create a singleton instance
task_repository = TaskRepository() 
