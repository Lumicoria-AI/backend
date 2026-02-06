from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.postgres_models import AgentExecutionSQL


class PostgresAgentExecutionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_execution(
        self,
        execution_id: str,
        agent_name: str,
        agent_type: str,
        started_at: datetime,
        ended_at: datetime,
        success: bool,
        async_execution: bool = False,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionSQL:
        duration_ms = int((ended_at - started_at).total_seconds() * 1000)
        record = AgentExecutionSQL(
            id=execution_id,
            agent_name=agent_name,
            agent_type=agent_type,
            status="success" if success else "error",
            error_message=error_message,
            async_execution=async_execution,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_execution(self, execution_id: str) -> Optional[AgentExecutionSQL]:
        stmt = select(AgentExecutionSQL).where(AgentExecutionSQL.id == execution_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_execution_stats(
        self,
        organization_id: Optional[str] = None,
        time_range: str = "7d",
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
        base_filter = [AgentExecutionSQL.started_at >= start_date]
        if organization_id:
            base_filter.append(AgentExecutionSQL.organization_id == organization_id)

        total_stmt = select(func.count()).select_from(AgentExecutionSQL).where(*base_filter)
        total = (await self.session.execute(total_stmt)).scalar_one()

        success_stmt = select(func.count()).select_from(AgentExecutionSQL).where(
            *base_filter,
            AgentExecutionSQL.status == "success"
        )
        success_count = (await self.session.execute(success_stmt)).scalar_one()
        failed_count = total - success_count

        avg_stmt = select(func.avg(AgentExecutionSQL.duration_ms)).where(*base_filter)
        avg_duration = (await self.session.execute(avg_stmt)).scalar_one()

        by_type_stmt = select(
            AgentExecutionSQL.agent_type,
            func.count()
        ).where(*base_filter).group_by(AgentExecutionSQL.agent_type)
        by_type_rows = (await self.session.execute(by_type_stmt)).all()

        return {
            "time_range": time_range,
            "total_executions": total,
            "successful_executions": success_count,
            "failed_executions": failed_count,
            "avg_duration_ms": float(avg_duration) if avg_duration is not None else 0.0,
            "executions_by_agent_type": {row[0] or "unknown": row[1] for row in by_type_rows},
            "success_rate": (success_count / total) if total else 0.0,
            "error_rate": (failed_count / total) if total else 0.0,
        }
