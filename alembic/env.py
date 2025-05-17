import sys
import os
from logging.config import fileConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Import our settings object and our Base
from backend.core.config import settings
from backend.db.postgresql import Base

# Add the project root to sys.path
# Assuming alembic directory is directly inside the backend directory
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import all models here to ensure they are registered with Base.metadata
from backend.db.models import (
    user,
    document,
    task,
    wellbeing,
    agent,
    permissions,
    integrations,
    organization,
    agent_studio,
    conversation,
    context as context_models # Renamed to avoid conflict with alembic.context
)


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

# Use the database URL from our settings
url = settings.DATABASE_URL


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Use the url defined above
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Use the url defined above instead of engine_from_config
    connectable = context.config.attributes.get("connection", None)

    if connectable is None:
        # fall back to regular sqlalchemy engine
        import asyncio
        from sqlalchemy.ext.asyncio import create_async_engine

        connectable = create_async_engine(url)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # include_symbol = include_symbol
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
