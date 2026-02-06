from typing import Dict, Any, Optional, List, Set
from enum import Enum
import structlog
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets
from functools import wraps
import asyncio
from .factory import AgentType

# Configure logging
logger = structlog.get_logger(__name__)

class AgentPermission(Enum):
    """Available permissions for agent operations."""
    READ = "read"  # Can read agent responses
    WRITE = "write"  # Can send inputs to agent
    CREATE = "create"  # Can create new agent instances
    DELETE = "delete"  # Can delete agent instances
    CONFIGURE = "configure"  # Can modify agent configuration
    ADMIN = "admin"  # Full access to all operations

@dataclass
class AgentSecurityContext:
    """Security context for agent operations."""
    user_id: str
    permissions: Set[AgentPermission]
    api_key: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    expires_at: Optional[datetime] = None

class AgentSecurityManager:
    """Manages security for agent operations."""
    
    def __init__(
        self,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        token_ttl: int = 3600,  # 1 hour
        max_retries: int = 3,
        rate_limit: int = 100  # requests per minute
    ):
        """
        Initialize security manager.
        
        Args:
            jwt_secret: Secret key for JWT tokens
            jwt_algorithm: Algorithm for JWT signing
            token_ttl: Token time-to-live in seconds
            max_retries: Maximum retry attempts
            rate_limit: Rate limit per minute
        """
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.token_ttl = token_ttl
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self._rate_limiters: Dict[str, List[datetime]] = {}
        self._lock = asyncio.Lock()
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def create_token(self, security_context: AgentSecurityContext) -> str:
        """
        Create a JWT token for the security context.
        
        Args:
            security_context: Security context to encode
            
        Returns:
            JWT token string
        """
        payload = {
            "user_id": security_context.user_id,
            "permissions": [p.value for p in security_context.permissions],
            "api_key": security_context.api_key,
            "session_id": security_context.session_id,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_ttl)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[AgentSecurityContext]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Security context if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return AgentSecurityContext(
                user_id=payload["user_id"],
                permissions={AgentPermission(p) for p in payload["permissions"]},
                api_key=payload.get("api_key"),
                session_id=payload.get("session_id"),
                created_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"])
            )
        except jwt.InvalidTokenError as e:
            logger.error("token_verification_failed", error=str(e))
            return None
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if within rate limit, False otherwise
        """
        async with self._lock:
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=1)
            
            # Get user's request timestamps
            timestamps = self._rate_limiters.get(user_id, [])
            
            # Remove old timestamps
            timestamps = [ts for ts in timestamps if ts > window_start]
            
            # Check rate limit
            if len(timestamps) >= self.rate_limit:
                return False
            
            # Add new timestamp
            timestamps.append(now)
            self._rate_limiters[user_id] = timestamps
            return True
    
    def has_permission(
        self,
        security_context: AgentSecurityContext,
        required_permission: AgentPermission
    ) -> bool:
        """
        Check if security context has required permission.
        
        Args:
            security_context: Security context to check
            required_permission: Permission required
            
        Returns:
            True if has permission, False otherwise
        """
        if AgentPermission.ADMIN in security_context.permissions:
            return True
        return required_permission in security_context.permissions
    
    def require_permission(permission: AgentPermission):
        """
        Decorator to require specific permission for agent operations.
        
        Args:
            permission: Required permission
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, security_context: AgentSecurityContext, **kwargs):
                if not self.has_permission(security_context, permission):
                    raise PermissionError(f"Missing required permission: {permission.value}")
                return await func(self, *args, security_context=security_context, **kwargs)
            return wrapper
        return decorator
    
    def require_agent_type(agent_type: AgentType):
        """
        Decorator to require specific agent type for operations.
        
        Args:
            agent_type: Required agent type
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, agent_type: AgentType, **kwargs):
                if agent_type != agent_type:
                    raise ValueError(f"Invalid agent type: {agent_type.value}")
                return await func(self, *args, agent_type=agent_type, **kwargs)
            return wrapper
        return decorator

# Example usage:
"""
# Initialize security manager
security_manager = AgentSecurityManager(
    jwt_secret="your-secret-key",
    token_ttl=3600,
    rate_limit=100
)

# Create security context
context = AgentSecurityContext(
    user_id="user123",
    permissions={AgentPermission.READ, AgentPermission.WRITE},
    api_key=security_manager.generate_api_key()
)

# Create token
token = security_manager.create_token(context)

# Verify token
verified_context = security_manager.verify_token(token)

# Use decorators
class SecureAgent:
    def __init__(self, security_manager: AgentSecurityManager):
        self.security_manager = security_manager
    
    @AgentSecurityManager.require_permission(AgentPermission.READ)
    async def get_agent_response(
        self,
        agent_id: str,
        security_context: AgentSecurityContext
    ):
        # Check rate limit
        if not await self.security_manager.check_rate_limit(security_context.user_id):
            raise Exception("Rate limit exceeded")
        
        # Get agent response
        return {"response": "secure data"}
    
    @AgentSecurityManager.require_permission(AgentPermission.WRITE)
    @AgentSecurityManager.require_agent_type(AgentType.DOCUMENT)
    async def process_document(
        self,
        agent_type: AgentType,
        input_data: Dict[str, Any],
        security_context: AgentSecurityContext
    ):
        # Process document
        return {"status": "processed"}
"""
