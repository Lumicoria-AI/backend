from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.postgres_models import TaskSQL
from backend.models.mongodb_models import TaskStatus, TaskPriority


class PostgresTaskRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_task(
        self,
        task_data: Dict[str, Any],
        creator_id: Optional[str],
        organization_id: Optional[str],
    ) -> TaskSQL:
        def _str_or_none(value: Any) -> Optional[str]:
            if value is None:
                return None
            return str(value)

        status = task_data.get("status", TaskStatus.TODO)
        if isinstance(status, str):
            try:
                status = TaskStatus(status)
            except Exception:
                status = TaskStatus.TODO

        priority = task_data.get("priority", TaskPriority.MEDIUM)
        if isinstance(priority, str):
            try:
                priority = TaskPriority(priority)
            except Exception:
                priority = TaskPriority.MEDIUM

        task = TaskSQL(
            title=task_data.get("title"),
            description=task_data.get("description"),
            status=status,
            priority=priority,
            due_date=task_data.get("due_date"),
            assigned_to=_str_or_none(task_data.get("assigned_to")),
            created_by=_str_or_none(creator_id),
            organization_id=_str_or_none(organization_id),
            project_id=_str_or_none(task_data.get("project_id")),
            parent_task_id=_str_or_none(task_data.get("parent_task_id")),
            agent_id=_str_or_none(task_data.get("agent_id")),
            tags=task_data.get("tags", []),
            metadata=task_data.get("metadata", {}),
            progress=task_data.get("progress", 0),
            completed_at=task_data.get("completed_at"),
            created_at=task_data.get("created_at") or datetime.utcnow(),
            updated_at=task_data.get("updated_at") or datetime.utcnow(),
        )
        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def get_task_by_id(self, task_id: str, organization_id: Optional[str]) -> Optional[TaskSQL]:
        stmt = select(TaskSQL).where(TaskSQL.id == task_id)
        if organization_id:
            stmt = stmt.where(TaskSQL.organization_id == organization_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_task(
        self,
        task_id: str,
        organization_id: Optional[str],
        update_data: Dict[str, Any],
    ) -> Optional[TaskSQL]:
        if "status" in update_data and isinstance(update_data["status"], str):
            try:
                update_data["status"] = TaskStatus(update_data["status"])
            except Exception:
                update_data.pop("status", None)
        if "priority" in update_data and isinstance(update_data["priority"], str):
            try:
                update_data["priority"] = TaskPriority(update_data["priority"])
            except Exception:
                update_data.pop("priority", None)
        update_data["updated_at"] = datetime.utcnow()
        stmt = (
            update(TaskSQL)
            .where(TaskSQL.id == task_id)
            .values(**update_data)
            .returning(TaskSQL)
        )
        if organization_id:
            stmt = stmt.where(TaskSQL.organization_id == organization_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete_task(self, task_id: str, organization_id: Optional[str]) -> bool:
        stmt = delete(TaskSQL).where(TaskSQL.id == task_id)
        if organization_id:
            stmt = stmt.where(TaskSQL.organization_id == organization_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def get_organization_tasks(
        self,
        organization_id: Optional[str],
        status: Optional[TaskStatus] = None,
        assigned_to: Optional[str] = None,
        document_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[TaskSQL]:
        stmt = select(TaskSQL)
        if organization_id:
            stmt = stmt.where(TaskSQL.organization_id == organization_id)
        if status:
            stmt = stmt.where(TaskSQL.status == status)
        if assigned_to:
            stmt = stmt.where(TaskSQL.assigned_to == assigned_to)
        if agent_id:
            stmt = stmt.where(TaskSQL.agent_id == agent_id)
        if document_id:
            # stored in metadata for now
            stmt = stmt.where(TaskSQL.meta["document_id"].astext == str(document_id))

        stmt = stmt.order_by(TaskSQL.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_upcoming_tasks(
        self,
        organization_id: Optional[str],
        user_id: Optional[str],
        days: int = 7,
    ) -> List[TaskSQL]:
        now = datetime.utcnow()
        future = now + timedelta(days=days)
        stmt = select(TaskSQL).where(TaskSQL.due_date != None)
        if organization_id:
            stmt = stmt.where(TaskSQL.organization_id == organization_id)
        if user_id:
            stmt = stmt.where(TaskSQL.assigned_to == user_id)
        stmt = stmt.where(TaskSQL.due_date >= now).where(TaskSQL.due_date <= future)
        stmt = stmt.order_by(TaskSQL.due_date.asc())
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_task_stats(self, organization_id: Optional[str]) -> Dict[str, Any]:
        stats = {
            "total_tasks": 0,
            "tasks_by_status": {},
            "tasks_by_priority": {},
            "overdue_tasks_count": 0,
            "upcoming_tasks_count": 0,
        }
        base_filter = []
        if organization_id:
            base_filter.append(TaskSQL.organization_id == organization_id)

        total_stmt = select(func.count()).select_from(TaskSQL)
        if base_filter:
            total_stmt = total_stmt.where(*base_filter)
        stats["total_tasks"] = (await self.session.execute(total_stmt)).scalar_one()

        status_stmt = select(TaskSQL.status, func.count()).select_from(TaskSQL).group_by(TaskSQL.status)
        if base_filter:
            status_stmt = status_stmt.where(*base_filter)
        status_rows = (await self.session.execute(status_stmt)).all()
        stats["tasks_by_status"] = {row[0]: row[1] for row in status_rows}

        priority_stmt = select(TaskSQL.priority, func.count()).select_from(TaskSQL).group_by(TaskSQL.priority)
        if base_filter:
            priority_stmt = priority_stmt.where(*base_filter)
        priority_rows = (await self.session.execute(priority_stmt)).all()
        stats["tasks_by_priority"] = {row[0]: row[1] for row in priority_rows}

        now = datetime.utcnow()
        overdue_stmt = select(func.count()).select_from(TaskSQL).where(
            TaskSQL.due_date != None,
            TaskSQL.due_date < now,
            TaskSQL.status != TaskStatus.COMPLETED,
        )
        if base_filter:
            overdue_stmt = overdue_stmt.where(*base_filter)
        stats["overdue_tasks_count"] = (await self.session.execute(overdue_stmt)).scalar_one()

        upcoming_stmt = select(func.count()).select_from(TaskSQL).where(
            TaskSQL.due_date != None,
            TaskSQL.due_date >= now,
        )
        if base_filter:
            upcoming_stmt = upcoming_stmt.where(*base_filter)
        stats["upcoming_tasks_count"] = (await self.session.execute(upcoming_stmt)).scalar_one()

        return stats

    async def get_task_analytics(
        self,
        organization_id: Optional[str],
        user_id: Optional[str],
        time_range: str,
    ) -> Dict[str, Any]:
        def _range_to_start(tr: str) -> datetime:
            now = datetime.utcnow()
            mapping = {
                "1d": timedelta(days=1),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30),
                "90d": timedelta(days=90),
                "1y": timedelta(days=365),
            }
            return now - mapping.get(tr, timedelta(days=7))

        start_date = _range_to_start(time_range)
        base_filter = [TaskSQL.created_at >= start_date]
        if organization_id:
            base_filter.append(TaskSQL.organization_id == organization_id)
        if user_id:
            base_filter.append(TaskSQL.created_by == user_id)

        total_stmt = select(func.count()).select_from(TaskSQL).where(*base_filter)
        total_tasks = (await self.session.execute(total_stmt)).scalar_one()

        completed_stmt = select(func.count()).select_from(TaskSQL).where(
            *base_filter,
            TaskSQL.status == TaskStatus.COMPLETED
        )
        completed_tasks = (await self.session.execute(completed_stmt)).scalar_one()

        open_tasks = total_tasks - completed_tasks

        timeline_stmt = select(
            func.date_trunc("day", TaskSQL.created_at).label("day"),
            func.count().label("count")
        ).where(*base_filter).group_by("day").order_by("day")
        timeline_rows = (await self.session.execute(timeline_stmt)).all()

        return {
            "time_range": time_range,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "open_tasks": open_tasks,
            "timeline": [{"date": row.day, "count": row.count} for row in timeline_rows],
        }
