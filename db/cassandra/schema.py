from __future__ import annotations

from cassandra.cluster import Session


def ensure_cassandra_schema(session: Session) -> None:
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS wellbeing_metrics (
            organization_id text,
            user_id text,
            metric_type text,
            timestamp timestamp,
            value double,
            source text,
            metadata text,
            PRIMARY KEY ((organization_id, user_id), metric_type, timestamp)
        ) WITH CLUSTERING ORDER BY (metric_type ASC, timestamp DESC)
        """
    )

    session.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_logs (
            organization_id text,
            user_id text,
            timestamp timestamp,
            activity_type text,
            details text,
            related_resource_type text,
            related_resource_id text,
            severity text,
            agent_id text,
            metadata text,
            PRIMARY KEY ((organization_id, user_id), timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp DESC)
        """
    )

    session.execute(
        """
        CREATE TABLE IF NOT EXISTS live_telemetry (
            organization_id text,
            session_id text,
            timestamp timestamp,
            user_id text,
            data_type text,
            payload text,
            metadata text,
            PRIMARY KEY ((organization_id, session_id), timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp DESC)
        """
    )
