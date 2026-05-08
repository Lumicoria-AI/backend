"""
Production Email Service with provider failover.

This module provides a robust email delivery service that:
- Supports multiple providers (SendGrid primary, Resend fallback)
- Implements automatic failover between providers
- Provides retry logic with exponential backoff
- Supports rate limiting to prevent provider abuse
- Renders email templates with Jinja2
- Logs all email operations for debugging and analytics
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from collections import defaultdict
import time

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import EmailStr
import structlog

from .email_providers.base import (
    EmailProvider,
    EmailMessage,
    EmailResult,
    EmailStatus,
    EmailProviderError,
    EmailRateLimitError,
    EmailAuthenticationError,
)
from .email_providers.sendgrid_provider import SendGridProvider
from .email_providers.resend_provider import ResendProvider
from ..core.config import settings

logger = structlog.get_logger()


class EmailProviderType(str, Enum):
    """Supported email providers."""
    SENDGRID = "sendgrid"
    RESEND = "resend"


class EmailService:
    """
    Production-grade email service with provider failover.
    
    Features:
    - Primary provider: SendGrid
    - Fallback provider: Resend
    - Automatic failover on provider failure
    - Retry logic with exponential backoff (3 attempts per provider)
    - Rate limiting (configurable per second)
    - Template rendering with Jinja2 caching
    - Comprehensive logging
    """
    
    def __init__(
        self,
        sendgrid_api_key: Optional[str] = None,
        resend_api_key: Optional[str] = None,
        primary_provider: EmailProviderType = EmailProviderType.SENDGRID,
        from_email: str = "noreply@lumicoria.ai",
        from_name: str = "Lumicoria.ai",
        sandbox_mode: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_per_second: float = 10.0,
    ):
        """
        Initialize the email service.
        
        Args:
            sendgrid_api_key: SendGrid API key (optional, falls back to settings)
            resend_api_key: Resend API key (optional, falls back to settings)
            primary_provider: Which provider to use first
            from_email: Default sender email address
            from_name: Default sender name
            sandbox_mode: If True, emails won't actually be sent (SendGrid only)
            max_retries: Maximum retry attempts per provider
            retry_delay: Initial delay between retries (exponential backoff)
            rate_limit_per_second: Maximum emails per second
        """
        # Configuration
        self.primary_provider = primary_provider
        self.from_email = from_email
        self.from_name = from_name
        self.sandbox_mode = sandbox_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_per_second = rate_limit_per_second
        
        # Get API keys from settings if not provided
        self._sendgrid_api_key = sendgrid_api_key or getattr(settings, 'SENDGRID_API_KEY', None)
        self._resend_api_key = resend_api_key or getattr(settings, 'RESEND_API_KEY', None)
        
        # Providers (lazy initialization)
        self._providers: Dict[EmailProviderType, Optional[EmailProvider]] = {
            EmailProviderType.SENDGRID: None,
            EmailProviderType.RESEND: None,
        }
        
        # Rate limiting
        self._last_send_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Template engine
        self.templates_dir = Path(__file__).parent / "templates" / "email"
        self._jinja_env: Optional[Environment] = None
        
        # Statistics
        self._stats = defaultdict(int)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the email service and providers.
        
        Returns:
            True if at least one provider was initialized successfully
        """
        if self._initialized:
            return True
        
        # Initialize Jinja2 environment
        if self.templates_dir.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=select_autoescape(['html', 'xml']),
                enable_async=True,
            )
        
        # Initialize providers
        initialized_any = False
        
        # SendGrid
        if self._sendgrid_api_key:
            try:
                provider = SendGridProvider(
                    api_key=self._sendgrid_api_key,
                    sandbox_mode=self.sandbox_mode
                )
                if await provider.initialize():
                    self._providers[EmailProviderType.SENDGRID] = provider
                    initialized_any = True
                    logger.info("email_provider_initialized", provider="sendgrid")
            except Exception as e:
                logger.warning("sendgrid_init_failed", error=str(e))
        
        # Resend
        if self._resend_api_key:
            try:
                provider = ResendProvider(api_key=self._resend_api_key)
                if await provider.initialize():
                    self._providers[EmailProviderType.RESEND] = provider
                    initialized_any = True
                    logger.info("email_provider_initialized", provider="resend")
            except Exception as e:
                logger.warning("resend_init_failed", error=str(e))
        
        self._initialized = initialized_any
        
        if not initialized_any:
            logger.error("no_email_providers_available")
        
        return initialized_any
    
    async def send(
        self,
        to: str | List[str],
        subject: str,
        template_name: str,
        template_data: Optional[Dict[str, Any]] = None,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmailResult:
        """
        Send an email using the configured providers.
        
        Args:
            to: Recipient email address(es)
            subject: Email subject line
            template_name: Name of the template file (without .html extension)
            template_data: Data to render into the template
            from_email: Override sender email
            from_name: Override sender name
            cc: CC recipients
            bcc: BCC recipients
            reply_to: Reply-to address
            attachments: List of attachment dicts with keys: filename, content, content_type
            tags: Tags for categorizing the email
            metadata: Additional metadata to attach
            
        Returns:
            EmailResult with the outcome
        """
        if not self._initialized:
            await self.initialize()
        
        # Apply rate limiting
        await self._apply_rate_limit()
        
        # Normalize recipients
        recipients = [to] if isinstance(to, str) else to
        
        # Render template
        try:
            html_content = await self._render_template(
                template_name,
                template_data or {}
            )
        except Exception as e:
            logger.error("template_render_error", template=template_name, error=str(e))
            return EmailResult(
                success=False,
                provider="none",
                status=EmailStatus.FAILED,
                error_message=f"Template rendering failed: {str(e)}",
                error_code="TEMPLATE_ERROR"
            )
        
        # Build message
        from .email_providers.base import EmailAttachment
        
        email_attachments = None
        if attachments:
            email_attachments = [
                EmailAttachment(
                    filename=att["filename"],
                    content=att["content"],
                    content_type=att.get("content_type", "application/octet-stream")
                )
                for att in attachments
            ]
        
        message = EmailMessage(
            to=recipients,
            subject=subject,
            html_content=html_content,
            from_email=from_email or self.from_email,
            from_name=from_name or self.from_name,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            attachments=email_attachments,
            tags=tags,
            metadata=metadata,
        )
        
        # Try sending with failover
        return await self._send_with_failover(message)
    
    async def send_raw(
        self,
        to: str | List[str],
        subject: str,
        html_content: str,
        plain_content: Optional[str] = None,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        **kwargs,
    ) -> EmailResult:
        """
        Send an email with raw HTML content (no template).
        
        Args:
            to: Recipient email address(es)
            subject: Email subject line
            html_content: Raw HTML content
            plain_content: Optional plain text version
            from_email: Override sender email
            from_name: Override sender name
            **kwargs: Additional message options
            
        Returns:
            EmailResult with the outcome
        """
        if not self._initialized:
            await self.initialize()
        
        await self._apply_rate_limit()
        
        recipients = [to] if isinstance(to, str) else to
        
        message = EmailMessage(
            to=recipients,
            subject=subject,
            html_content=html_content,
            plain_content=plain_content,
            from_email=from_email or self.from_email,
            from_name=from_name or self.from_name,
            **kwargs,
        )
        
        return await self._send_with_failover(message)
    
    async def send_batch(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[EmailResult]:
        """
        Send multiple emails in batch.
        
        Args:
            messages: List of message dicts with keys matching send() parameters
            
        Returns:
            List of EmailResult objects
        """
        results = []
        for msg_data in messages:
            result = await self.send(**msg_data)
            results.append(result)
        return results
    
    async def _send_with_failover(self, message: EmailMessage) -> EmailResult:
        """
        Send an email with automatic provider failover.
        
        Tries primary provider first, then falls back to secondary.
        Each provider gets max_retries attempts with exponential backoff.
        """
        # Determine provider order
        providers_to_try = self._get_provider_order()
        
        if not providers_to_try:
            return EmailResult(
                success=False,
                provider="none",
                status=EmailStatus.FAILED,
                error_message="No email providers configured",
                error_code="NO_PROVIDERS"
            )
        
        last_error = None
        
        for provider_type in providers_to_try:
            provider = self._providers.get(provider_type)
            if not provider:
                continue
            
            # Try this provider with retries
            result = await self._send_with_retries(provider, message)
            
            if result.success:
                self._stats[f"success_{provider_type.value}"] += 1
                logger.info(
                    "email_sent_successfully",
                    provider=provider_type.value,
                    subject=message.subject,
                    to=message.to
                )
                return result
            
            # Check if we should failover
            if self._should_failover(result):
                last_error = result.error_message
                self._stats[f"failover_from_{provider_type.value}"] += 1
                logger.warning(
                    "email_provider_failover",
                    from_provider=provider_type.value,
                    error=result.error_message
                )
                continue
            
            # Non-retriable error, don't failover
            return result
        
        # All providers failed
        self._stats["total_failures"] += 1
        return EmailResult(
            success=False,
            provider="all",
            status=EmailStatus.FAILED,
            error_message=f"All providers failed. Last error: {last_error}",
            error_code="ALL_PROVIDERS_FAILED"
        )
    
    async def _send_with_retries(
        self,
        provider: EmailProvider,
        message: EmailMessage
    ) -> EmailResult:
        """
        Send an email with retry logic.
        
        Uses exponential backoff: delay * (2 ^ attempt)
        """
        last_result = None
        
        for attempt in range(self.max_retries):
            try:
                result = await provider.send(message)
                
                if result.success:
                    return result
                
                last_result = result
                
                # Check if error is retriable
                if not self._is_retriable_error(result):
                    return result
                
                # Calculate backoff delay
                delay = self.retry_delay * (2 ** attempt)
                logger.info(
                    "email_send_retry",
                    provider=provider.name,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay=delay,
                    error=result.error_message
                )
                
                await asyncio.sleep(delay)
                
            except EmailAuthenticationError:
                # Don't retry auth errors
                return EmailResult(
                    success=False,
                    provider=provider.name,
                    status=EmailStatus.FAILED,
                    error_message="Authentication failed",
                    error_code="AUTH_ERROR"
                )
            
            except EmailRateLimitError:
                # Wait and retry for rate limits
                delay = self.retry_delay * (2 ** (attempt + 1))
                logger.warning(
                    "email_rate_limited",
                    provider=provider.name,
                    delay=delay
                )
                await asyncio.sleep(delay)
                
            except Exception as e:
                last_result = EmailResult(
                    success=False,
                    provider=provider.name,
                    status=EmailStatus.FAILED,
                    error_message=str(e),
                    error_code="UNEXPECTED_ERROR"
                )
        
        return last_result or EmailResult(
            success=False,
            provider=provider.name,
            status=EmailStatus.FAILED,
            error_message="Max retries exceeded",
            error_code="MAX_RETRIES"
        )
    
    def _get_provider_order(self) -> List[EmailProviderType]:
        """Get the order of providers to try based on primary setting."""
        if self.primary_provider == EmailProviderType.SENDGRID:
            return [EmailProviderType.SENDGRID, EmailProviderType.RESEND]
        else:
            return [EmailProviderType.RESEND, EmailProviderType.SENDGRID]
    
    def _should_failover(self, result: EmailResult) -> bool:
        """Determine if we should failover to another provider."""
        # Failover on provider errors, not on validation errors
        failover_codes = {
            "CLIENT_NOT_INITIALIZED",
            "AUTHENTICATION_ERROR",
            "AUTH_ERROR",
            "RATE_LIMIT_ERROR",
            "UNEXPECTED_ERROR",
            "MAX_RETRIES",
        }
        return result.error_code in failover_codes
    
    def _is_retriable_error(self, result: EmailResult) -> bool:
        """Determine if an error is retriable."""
        retriable_codes = {
            "RATE_LIMIT_ERROR",
            "UNEXPECTED_ERROR",
            "TIMEOUT",
            "CONNECTION_ERROR",
        }
        return result.error_code in retriable_codes
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to prevent provider abuse."""
        async with self._rate_limit_lock:
            current_time = time.monotonic()
            
            # Remove timestamps older than 1 second
            cutoff = current_time - 1.0
            self._last_send_times = [
                t for t in self._last_send_times if t > cutoff
            ]
            
            # Check if we're at the limit
            if len(self._last_send_times) >= self.rate_limit_per_second:
                # Wait until we can send again
                oldest = min(self._last_send_times)
                wait_time = 1.0 - (current_time - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Record this send
            self._last_send_times.append(time.monotonic())
    
    async def _render_template(
        self,
        template_name: str,
        data: Dict[str, Any]
    ) -> str:
        """Render an email template with the given data."""
        if not self._jinja_env:
            raise ValueError("Template engine not initialized")
        
        # Add default template variables
        data.setdefault("year", datetime.utcnow().year)
        data.setdefault("company_name", "Lumicoria.ai")
        data.setdefault("support_email", "support@lumicoria.ai")
        
        template = self._jinja_env.get_template(f"{template_name}.html")
        return await template.render_async(**data)
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check the health of all configured providers.
        
        Returns:
            Dict mapping provider names to their health status
        """
        if not self._initialized:
            await self.initialize()
        
        health = {}
        for provider_type, provider in self._providers.items():
            if provider:
                try:
                    health[provider_type.value] = await provider.health_check()
                except Exception:
                    health[provider_type.value] = False
            else:
                health[provider_type.value] = False
        
        return health
    
    def get_stats(self) -> Dict[str, int]:
        """Get email sending statistics."""
        return dict(self._stats)
    
    async def test_send(
        self,
        to: str,
        provider: Optional[EmailProviderType] = None
    ) -> EmailResult:
        """
        Send a test email to verify configuration.
        
        Args:
            to: Email address to send test to
            provider: Optional specific provider to test
            
        Returns:
            EmailResult with the outcome
        """
        return await self.send(
            to=to,
            subject="Lumicoria.ai - Test Email",
            template_name="test_notification",
            template_data={
                "subject": "Test Email",
                "user_name": "Test User",
                "message": "This is a test email to verify your email configuration is working correctly.",
            },
            tags=["test"],
            metadata={"test": True},
        )


# Singleton instance
_email_service: Optional[EmailService] = None


async def get_email_service() -> EmailService:
    """Get or create the email service singleton.

    Resend is the primary provider — SendGrid stays available as a
    fallback only when its API key is configured.  Resend's deliverability,
    plain-API surface, and lack of the dict/CustomArg footgun in the
    Python SDK make it the better default for transactional mail.
    """
    global _email_service

    if _email_service is None:
        _email_service = EmailService(
            sendgrid_api_key=getattr(settings, 'SENDGRID_API_KEY', None),
            resend_api_key=getattr(settings, 'RESEND_API_KEY', None),
            primary_provider=EmailProviderType.RESEND,
            from_email=getattr(settings, 'EMAIL_FROM_ADDRESS', 'noreply@lumicoria.ai'),
            from_name=getattr(settings, 'EMAIL_FROM_NAME', 'Lumicoria.ai'),
            sandbox_mode=getattr(settings, 'EMAIL_SANDBOX_MODE', False),
        )
        await _email_service.initialize()

    return _email_service
