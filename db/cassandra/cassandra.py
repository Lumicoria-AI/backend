from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.cqlengine.connection import register_connection, set_default_connection
from typing import Optional, List
from ...core.config import settings
import structlog

logger = structlog.get_logger()

class CassandraClient:
    cluster: Optional[Cluster] = None
    session: Optional[Session] = None

    @classmethod
    async def connect(cls) -> None:
        try:
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
                protocol_version=4
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
            
            logger.info("Connected to Cassandra")
        except Exception as e:
            logger.error("Failed to connect to Cassandra", error=str(e))
            raise

    @classmethod
    async def disconnect(cls) -> None:
        if cls.session:
            cls.session.shutdown()
        if cls.cluster:
            cls.cluster.shutdown()
        logger.info("Disconnected from Cassandra")

    @classmethod
    async def get_session(cls) -> Session:
        if not cls.session:
            await cls.connect()
        return cls.session

    @classmethod
    async def execute(cls, query: str, params: Optional[dict] = None) -> List[dict]:
        session = await cls.get_session()
        if params:
            result = session.execute(query, params)
        else:
            result = session.execute(query)
        return [dict(row) for row in result]

    @classmethod
    async def execute_batch(cls, queries: List[tuple[str, Optional[dict]]]) -> None:
        session = await cls.get_session()
        batch = session.prepare_batch()
        for query, params in queries:
            if params:
                batch.add(query, params)
            else:
                batch.add(query)
        await session.execute(batch)


async def get_cassandra() -> CassandraClient:
    return CassandraClient() 