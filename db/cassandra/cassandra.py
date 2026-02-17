from typing import Optional, List
import asyncio
import structlog

logger = structlog.get_logger()

# Cassandra driver may not be available on Python 3.12+ (asyncore was removed).
# Guard the import so the app can start with CASSANDRA_ENABLED=false.
try:
    from cassandra.cluster import Cluster, Session
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cqlengine.connection import register_connection, set_default_connection
    from cassandra.query import BatchStatement, SimpleStatement
    _CASSANDRA_AVAILABLE = True
except Exception:
    _CASSANDRA_AVAILABLE = False
    Cluster = None   # type: ignore
    Session = None   # type: ignore
    logger.warning("Cassandra driver not available — Cassandra features disabled")

from ...core.config import settings


class CassandraClient:
    cluster: Optional[object] = None
    session: Optional[object] = None

    @classmethod
    async def connect(cls) -> None:
        if not settings.db.CASSANDRA_ENABLED:
            logger.info("Cassandra disabled; skipping connect")
            return
        if not _CASSANDRA_AVAILABLE:
            logger.error("Cassandra is enabled but driver is not available (Python 3.12+ needs libev)")
            return
        try:
            await asyncio.to_thread(cls._connect_sync)
            logger.info("Connected to Cassandra")
        except Exception as e:
            logger.error("Failed to connect to Cassandra", error=str(e))
            raise

    @classmethod
    def _connect_sync(cls) -> None:
        auth_provider = None
        if settings.db.CASSANDRA_USERNAME and settings.db.CASSANDRA_PASSWORD:
            auth_provider = PlainTextAuthProvider(
                username=settings.db.CASSANDRA_USERNAME,
                password=settings.db.CASSANDRA_PASSWORD
            )

        cls.cluster = Cluster(
            contact_points=settings.db.CASSANDRA_HOSTS,
            port=settings.db.CASSANDRA_PORT,
            auth_provider=auth_provider,
            protocol_version=4,
            connect_timeout=settings.db.CASSANDRA_CONNECT_TIMEOUT
        )
        
        cls.session = cls.cluster.connect()
        
        # Create keyspace if it doesn't exist
        cls.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {settings.db.CASSANDRA_KEYSPACE}
            WITH replication = {{
                'class': 'SimpleStrategy',
                'replication_factor': {settings.db.CASSANDRA_REPLICATION_FACTOR}
            }}
        """)
        
        # Use the keyspace
        cls.session.set_keyspace(settings.db.CASSANDRA_KEYSPACE)
        
        # Register connection for cassandra.cqlengine
        register_connection(str(cls.session), session=cls.session)
        set_default_connection(str(cls.session))

        # Ensure tables exist
        from .schema import ensure_cassandra_schema
        ensure_cassandra_schema(cls.session)

    @classmethod
    async def disconnect(cls) -> None:
        if not settings.db.CASSANDRA_ENABLED:
            return
        await asyncio.to_thread(cls._disconnect_sync)
        logger.info("Disconnected from Cassandra")

    @classmethod
    def _disconnect_sync(cls) -> None:
        if cls.session:
            cls.session.shutdown()
        if cls.cluster:
            cls.cluster.shutdown()

    @classmethod
    async def get_session(cls):
        if not cls.session:
            await cls.connect()
        return cls.session

    @classmethod
    async def execute(cls, query: str, params: Optional[dict] = None) -> List[dict]:
        session = await cls.get_session()
        if not session:
            return []
        return await asyncio.to_thread(cls._execute_sync, query, params)

    @classmethod
    def _execute_sync(cls, query: str, params: Optional[dict] = None) -> List[dict]:
        if params:
            result = cls.session.execute(query, params)
        else:
            result = cls.session.execute(query)
        return [dict(row) for row in result]

    @classmethod
    async def execute_batch(cls, queries: List[tuple[str, Optional[dict]]]) -> None:
        session = await cls.get_session()
        if not session:
            return
        await asyncio.to_thread(cls._execute_batch_sync, queries)

    @classmethod
    def _execute_batch_sync(cls, queries: List[tuple[str, Optional[dict]]]) -> None:
        batch = BatchStatement()
        for query, params in queries:
            statement = SimpleStatement(query)
            batch.add(statement, params or {})
        cls.session.execute(batch)


async def get_cassandra() -> CassandraClient:
    return CassandraClient()
