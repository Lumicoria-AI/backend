from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.postgres_models import WorkflowSQL


class PostgresWorkflowRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_workflow(self, data: Dict[str, Any]) -> WorkflowSQL:
        def _str_or_none(value: Any) -> Optional[str]:
            if value is None:
                return None
            return str(value)

        workflow = WorkflowSQL(
            name=data["name"],
            description=data.get("description"),
            components=data.get("components", []),
            nodes=data.get("nodes", []),
            connections=data.get("connections", []),
            organization_id=_str_or_none(data.get("organization_id")),
            created_by=_str_or_none(data.get("created_by")),
            version=data.get("version", "1.0.0"),
            is_public=data.get("is_public", False),
            tags=data.get("tags", []),
            status=data.get("status", "draft"),
            created_at=data.get("created_at") or datetime.utcnow(),
            updated_at=data.get("updated_at") or datetime.utcnow(),
        )
        self.session.add(workflow)
        await self.session.commit()
        await self.session.refresh(workflow)
        return workflow

    async def get_workflow_by_id(self, workflow_id: str) -> Optional[WorkflowSQL]:
        stmt = select(WorkflowSQL).where(WorkflowSQL.id == workflow_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_workflows(
        self,
        organization_id: Optional[str] = None,
        created_by: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[WorkflowSQL]:
        stmt = select(WorkflowSQL)
        if organization_id:
            stmt = stmt.where(WorkflowSQL.organization_id == organization_id)
        if created_by:
            stmt = stmt.where(WorkflowSQL.created_by == created_by)
        stmt = stmt.order_by(WorkflowSQL.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_accessible_workflows(
        self,
        organization_id: Optional[str],
        user_id: Optional[str],
        is_admin: bool,
        skip: int = 0,
        limit: int = 100,
    ) -> List[WorkflowSQL]:
        stmt = select(WorkflowSQL)
        if is_admin:
            if organization_id:
                stmt = stmt.where(WorkflowSQL.organization_id == organization_id)
        else:
            filters = []
            if user_id:
                filters.append(WorkflowSQL.created_by == user_id)
            if organization_id:
                filters.append(and_(WorkflowSQL.organization_id == organization_id, WorkflowSQL.is_public == True))
            else:
                filters.append(WorkflowSQL.is_public == True)
            stmt = stmt.where(or_(*filters))

        stmt = stmt.order_by(WorkflowSQL.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Optional[WorkflowSQL]:
        updates["updated_at"] = datetime.utcnow()
        stmt = (
            update(WorkflowSQL)
            .where(WorkflowSQL.id == workflow_id)
            .values(**updates)
            .returning(WorkflowSQL)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete_workflow(self, workflow_id: str) -> bool:
        stmt = delete(WorkflowSQL).where(WorkflowSQL.id == workflow_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def get_workflow_stats(
        self,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        base_filter = []
        if organization_id:
            base_filter.append(WorkflowSQL.organization_id == organization_id)

        total_stmt = select(func.count()).select_from(WorkflowSQL)
        if base_filter:
            total_stmt = total_stmt.where(*base_filter)
        total = (await self.session.execute(total_stmt)).scalar_one()

        public_stmt = select(func.count()).select_from(WorkflowSQL).where(WorkflowSQL.is_public == True)
        if base_filter:
            public_stmt = public_stmt.where(*base_filter)
        public_count = (await self.session.execute(public_stmt)).scalar_one()

        status_stmt = select(WorkflowSQL.status, func.count()).select_from(WorkflowSQL).group_by(WorkflowSQL.status)
        if base_filter:
            status_stmt = status_stmt.where(*base_filter)
        status_rows = (await self.session.execute(status_stmt)).all()

        return {
            "total_workflows": total,
            "public_workflows": public_count,
            "private_workflows": total - public_count,
            "by_status": {row[0]: row[1] for row in status_rows},
        }
