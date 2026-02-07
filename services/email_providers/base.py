"""
Abstract base class for email providers.

This module defines the interface that all email providers must implement,
providing a consistent API for email delivery across different services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger()


class EmailStatus(Enum):
    """Status of an email send operation."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    REJECTED = "rejected"


@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    content: bytes
    content_type: str = "application/octet-stream"
    disposition: str = "attachment"  # or "inline"
    content_id: Optional[str] = None  # For inline images


@dataclass
class EmailMessage:
    """Represents an email message to be sent."""
    to: List[str]
    subject: str
    html_content: str
    from_email: str
    from_name: Optional[str] = None
    plain_content: Optional[str] = None
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    reply_to: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    attachments: Optional[List[EmailAttachment]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    template_id: Optional[str] = None  # For provider-managed templates
    template_data: Optional[Dict[str, Any]] = None


@dataclass
class EmailResult:
    """Result of an email send operation."""
    success: bool
    provider: str
    message_id: Optional[str] = None
    status: EmailStatus = EmailStatus.PENDING
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "success": self.success,
            "provider": self.provider,
            "message_id": self.message_id,
            "status": self.status.value,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat(),
        }


class EmailProvider(ABC):
    """
    Abstract base class for email providers.
    
    All concrete email providers (SendGrid, Resend, etc.) must inherit
    from this class and implement the abstract methods.
    """
    
    def __init__(self, name: str, api_key: str):
        """
        Initialize the email provider.
        
        Args:
            name: Provider name for logging and identification
            api_key: API key for authentication
        """
        self.name = name
        self.api_key = api_key
        self._initialized = False
        self._healthy = False
    
    @abstractmethod
    async def send(self, message: EmailMessage) -> EmailResult:
        """
        Send an email message.
        
        Args:
            message: The email message to send
            
        Returns:
            EmailResult containing the outcome of the send operation
        """
        pass
    
    @abstractmethod
    async def send_batch(self, messages: List[EmailMessage]) -> List[EmailResult]:
        """
        Send multiple email messages in batch.
        
        Args:
            messages: List of email messages to send
            
        Returns:
            List of EmailResult objects, one per message
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and operational.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider (verify API key, set up client).
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        pass
    
    async def validate_message(self, message: EmailMessage) -> tuple[bool, Optional[str]]:
        """
        Validate an email message before sending.
        
        Args:
            message: The message to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not message.to:
            return False, "No recipients specified"
        
        if not message.subject:
            return False, "No subject specified"
        
        if not message.html_content and not message.plain_content:
            return False, "No content specified (html or plain text required)"
        
        if not message.from_email:
            return False, "No from email specified"
        
        # Basic email format validation
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        for email in message.to:
            if not re.match(email_pattern, email):
                return False, f"Invalid email format: {email}"
        
        if not re.match(email_pattern, message.from_email):
            return False, f"Invalid from email format: {message.from_email}"
        
        return True, None
    
    def _log_send_attempt(self, message: EmailMessage) -> None:
        """Log an email send attempt."""
        logger.info(
            "email_send_attempt",
            provider=self.name,
            to=message.to,
            subject=message.subject[:50] if message.subject else None,
            has_attachments=bool(message.attachments),
        )
    
    def _log_send_result(self, message: EmailMessage, result: EmailResult) -> None:
        """Log an email send result."""
        log_method = logger.info if result.success else logger.error
        log_method(
            "email_send_result",
            provider=self.name,
            success=result.success,
            message_id=result.message_id,
            status=result.status.value,
            error=result.error_message,
            to=message.to,
        )
    
    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._initialized
    
    @property
    def is_healthy(self) -> bool:
        """Check if the provider is healthy."""
        return self._healthy


class EmailProviderError(Exception):
    """Base exception for email provider errors."""
    
    def __init__(self, message: str, provider: str, code: Optional[str] = None):
        self.message = message
        self.provider = provider
        self.code = code
        super().__init__(message)


class EmailRateLimitError(EmailProviderError):
    """Exception raised when rate limit is exceeded."""
    pass


class EmailAuthenticationError(EmailProviderError):
    """Exception raised for authentication failures."""
    pass


class EmailValidationError(EmailProviderError):
    """Exception raised for message validation failures."""
    pass
