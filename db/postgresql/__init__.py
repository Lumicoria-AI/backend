from .database import get_db, engine
# Import all models so they are registered with SQLAlchemy Base metadata
from backend.db.models.user import *
from backend.db.models.document import *
from backend.db.models.task import *
from backend.db.models.wellbeing import *
from backend.db.models.agent import *
from backend.db.models.permissions import *
from backend.db.models.integrations import *
from backend.db.models.organization import *
from backend.db.models.agent_studio import *
from backend.db.models.conversation import *
from backend.db.models.context import *

from sqlalchemy.orm import declarative_base

# This Base will be used by the models in backend.db.models
Base = declarative_base() 