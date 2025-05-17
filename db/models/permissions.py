from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Enum
from sqlalchemy.orm import relationship
import enum

from ..base import Base

class PermissionType(str, enum.Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    MANAGE = "manage"
    VIEW = "view"
    EXECUTE = "execute"

class ResourceType(str, enum.Enum):
    USER = "user"
    DOCUMENT = "document"
    TASK = "task"
    CALENDAR_EVENT = "calendar_event"
    WELLBEING_METRICS = "wellbeing_metrics"
    AGENT = "agent"
    ORGANIZATION = "organization"
    TEAM = "team"
    INTEGRATION = "integration"
    SETTING = "setting"
    PERMISSION = "permission"
    ROLE = "role"

class Permission(Base):
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False) # e.g., "create_document", "manage_users"
    description = Column(String, nullable=True)
    permission_type = Column(Enum(PermissionType), nullable=False)
    resource_type = Column(Enum(ResourceType), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    role_permissions = relationship("RolePermission", back_populates="permission")

class RolePermission(Base):
    __tablename__ = "role_permissions"

    role_name = Column(String, ForeignKey("user_roles.name"), primary_key=True) # Assuming a user_roles table or similar for role names
    permission_id = Column(Integer, ForeignKey("permissions.id"), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    permission = relationship("Permission", back_populates="role_permissions")

# Note: A 'user_roles' table or similar containing distinct role names (UserRole enum) would be needed
# to link RolePermission to the actual roles assigned to users. We can add this or link directly to the User model if roles are fixed. 