"""
Legacy SQLAlchemy session module.

This file now delegates to backend/db/postgres.py to avoid duplicated
configuration and to support optional Postgres usage.
"""

from backend.db.postgres import get_db, get_sync_sessionmaker

# Do not instantiate the sessionmaker at import time to avoid
# requiring Postgres configuration in environments that don't use it.
SessionLocal = None
