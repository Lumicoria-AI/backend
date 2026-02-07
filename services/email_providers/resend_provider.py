"""
Resend email provider implementation.

This module provides integration with the Resend email delivery service,
offering a simple API for transactional emails with excellent deliverability.
"""

import asyncio
from typing import Dict, List, Optional, Any
import structlog

from .base import (
    EmailProvider,
    EmailMessage,
    EmailResult,
    EmailStatus,
    EmailProviderError,
    EmailRateLimitError,
    EmailAuthenticationError,
)

logger = structlog.get_logger()


class ResendProvider(EmailProvider):
    """
    Resend email provider implementation.
    
    Uses the official Resend Python SDK for email delivery.
    Provides a modern, developer-friendly API with excellent defaults.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Resend provider.
        
        Args:
            api_key: Resend API key (starts with re_)
        """
        super().__init__(name="resend", api_key=api_key)
        self._resend = None
    
    async def initialize(self) -> bool:
        """Initialize the Resend client."""
        try:
            import resend
            
            resend.api_key = self.api_key
            self._resend = resend
            self._initialized = True
            self._healthy = True
            
            logger.info("resend_initialized")
            return True
            
        except ImportError:
            logger.error("resend_import_error", message="resend package not installed")
            return False
        except Exception as e:
            logger.error("resend_init_error", error=str(e))
            return False
    
    async def send(self, message: EmailMessage) -> EmailResult:
        """Send an email via Resend."""
        if not self._initialized:
            await self.initialize()
        
        if not self._resend:
            return EmailResult(
                success=False,
                provider=self.name,
                status=EmailStatus.FAILED,
                error_message="Resend client not initialized",
                error_code="CLIENT_NOT_INITIALIZED"
            )
        
        # Validate message
        is_valid, error = await self.validate_message(message)
        if not is_valid:
            return EmailResult(
                success=False,
                provider=self.name,
                status=EmailStatus.REJECTED,
                error_message=error,
                error_code="VALIDATION_ERROR"
            )
        
        self._log_send_attempt(message)
        
        try:
            import base64
            
            # Build the email payload
            payload: Dict[str, Any] = {
                "from": f"{message.from_name} <{message.from_email}>" if message.from_name else message.from_email,
                "to": message.to,
                "subject": message.subject,
                "html": message.html_content,
            }
            
            # Optional plain text
            if message.plain_content:
                payload["text"] = message.plain_content
            
            # CC and BCC
            if message.cc:
                payload["cc"] = message.cc
            
            if message.bcc:
                payload["bcc"] = message.bcc
            
            # Reply-To
            if message.reply_to:
                payload["reply_to"] = message.reply_to
            
            # Custom headers
            if message.headers:
                payload["headers"] = message.headers
            
            # Tags (Resend supports up to 5 tags)
            if message.tags:
                payload["tags"] = [{"name": tag, "value": "true"} for tag in message.tags[:5]]
            
            # Attachments
            if message.attachments:
                payload["attachments"] = []
                for att in message.attachments:
                    attachment_data = {
                        "filename": att.filename,
                        "content": base64.b64encode(att.content).decode(),
                    }
                    if att.content_type:
                        attachment_data["content_type"] = att.content_type
                    payload["attachments"].append(attachment_data)
            
            # Send via thread pool (resend SDK is synchronous)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._resend.Emails.send(payload)
            )
            
            # Parse response
            if response and hasattr(response, "id"):
                result = EmailResult(
                    success=True,
                    provider=self.name,
                    message_id=response.id,
                    status=EmailStatus.SENT,
                    raw_response={"id": response.id}
                )
            elif isinstance(response, dict) and response.get("id"):
                result = EmailResult(
                    success=True,
                    provider=self.name,
                    message_id=response["id"],
                    status=EmailStatus.SENT,
                    raw_response=response
                )
            else:
                result = EmailResult(
                    success=False,
                    provider=self.name,
                    status=EmailStatus.FAILED,
                    error_message="Unexpected response from Resend",
                    error_code="UNEXPECTED_RESPONSE",
                    raw_response={"response": str(response)}
                )
            
            self._log_send_result(message, result)
            return result
            
        except Exception as e:
            error_str = str(e)
            error_code = "UNKNOWN_ERROR"
            
            # Handle specific Resend errors
            if "401" in error_str or "Invalid API Key" in error_str:
                error_code = "AUTHENTICATION_ERROR"
                raise EmailAuthenticationError(
                    message="Invalid Resend API key",
                    provider=self.name,
                    code=error_code
                )
            elif "429" in error_str or "rate limit" in error_str.lower():
                error_code = "RATE_LIMIT_ERROR"
                raise EmailRateLimitError(
                    message="Resend rate limit exceeded",
                    provider=self.name,
                    code=error_code
                )
            elif "validation" in error_str.lower():
                error_code = "VALIDATION_ERROR"
            
            result = EmailResult(
                success=False,
                provider=self.name,
                status=EmailStatus.FAILED,
                error_message=error_str,
                error_code=error_code
            )
            
            self._log_send_result(message, result)
            return result
    
    async def send_batch(self, messages: List[EmailMessage]) -> List[EmailResult]:
        """
        Send multiple emails in batch.
        
        Resend has a batch endpoint, but for reliability we send
        individually with concurrency control.
        """
        # Limit concurrency to respect rate limits
        semaphore = asyncio.Semaphore(5)  # Resend is stricter on rate limits
        
        async def send_with_semaphore(msg: EmailMessage) -> EmailResult:
            async with semaphore:
                # Small delay between sends to respect rate limits
                await asyncio.sleep(0.1)
                return await self.send(msg)
        
        results = await asyncio.gather(
            *[send_with_semaphore(msg) for msg in messages],
            return_exceptions=True
        )
        
        # Convert exceptions to EmailResult
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(EmailResult(
                    success=False,
                    provider=self.name,
                    status=EmailStatus.FAILED,
                    error_message=str(result),
                    error_code="BATCH_SEND_ERROR"
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def health_check(self) -> bool:
        """
        Check if Resend is operational.
        
        We verify by checking the API key is valid via the domains endpoint.
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._resend:
            self._healthy = False
            return False
        
        try:
            # List domains to verify API key
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._resend.Domains.list()
            )
            
            # If we get here without exception, the API key is valid
            self._healthy = True
            return True
            
        except Exception as e:
            logger.warning("resend_health_check_failed", error=str(e))
            self._healthy = False
            return False
    
    async def get_email_status(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a sent email.
        
        Args:
            email_id: The ID returned when sending the email
            
        Returns:
            Email details including status, or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._resend:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._resend.Emails.get(email_id)
            )
            return response if response else None
            
        except Exception as e:
            logger.warning("resend_get_email_failed", email_id=email_id, error=str(e))
            return None
