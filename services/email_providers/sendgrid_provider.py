"""
SendGrid email provider implementation.

This module provides integration with the SendGrid email delivery service,
supporting async sending, batch operations, and delivery tracking.
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


class SendGridProvider(EmailProvider):
    """
    SendGrid email provider implementation.
    
    Uses the official SendGrid Python SDK for email delivery.
    Supports async operations via thread pool execution.
    """
    
    def __init__(self, api_key: str, sandbox_mode: bool = False):
        """
        Initialize SendGrid provider.
        
        Args:
            api_key: SendGrid API key (starts with SG.)
            sandbox_mode: If True, emails won't actually be sent (for testing)
        """
        super().__init__(name="sendgrid", api_key=api_key)
        self.sandbox_mode = sandbox_mode
        self._client = None
        self._executor = None
    
    async def initialize(self) -> bool:
        """Initialize the SendGrid client."""
        try:
            from sendgrid import SendGridAPIClient
            
            self._client = SendGridAPIClient(self.api_key)
            self._initialized = True
            self._healthy = True
            
            logger.info("sendgrid_initialized", sandbox_mode=self.sandbox_mode)
            return True
            
        except ImportError:
            logger.error("sendgrid_import_error", message="sendgrid package not installed")
            return False
        except Exception as e:
            logger.error("sendgrid_init_error", error=str(e))
            return False
    
    async def send(self, message: EmailMessage) -> EmailResult:
        """Send an email via SendGrid."""
        if not self._initialized:
            await self.initialize()
        
        if not self._client:
            return EmailResult(
                success=False,
                provider=self.name,
                status=EmailStatus.FAILED,
                error_message="SendGrid client not initialized",
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
            from sendgrid.helpers.mail import (
                Mail,
                Email,
                To,
                Content,
                Attachment,
                FileContent,
                FileName,
                FileType,
                Disposition,
                ContentId,
                MailSettings,
                SandBoxMode,
            )
            import base64
            
            # Build the mail object
            mail = Mail()
            
            # From
            from_email = Email(message.from_email, message.from_name)
            mail.from_email = from_email
            
            # To recipients
            for recipient in message.to:
                mail.add_to(To(recipient))
            
            # CC and BCC
            if message.cc:
                for cc_email in message.cc:
                    mail.add_cc(cc_email)
            
            if message.bcc:
                for bcc_email in message.bcc:
                    mail.add_bcc(bcc_email)
            
            # Subject
            mail.subject = message.subject
            
            # Content
            if message.plain_content:
                mail.add_content(Content("text/plain", message.plain_content))
            mail.add_content(Content("text/html", message.html_content))
            
            # Reply-To
            if message.reply_to:
                mail.reply_to = Email(message.reply_to)
            
            # Custom headers
            if message.headers:
                for key, value in message.headers.items():
                    mail.add_header({key: value})
            
            # Attachments
            if message.attachments:
                for att in message.attachments:
                    attachment = Attachment()
                    attachment.file_content = FileContent(base64.b64encode(att.content).decode())
                    attachment.file_name = FileName(att.filename)
                    attachment.file_type = FileType(att.content_type)
                    attachment.disposition = Disposition(att.disposition)
                    if att.content_id:
                        attachment.content_id = ContentId(att.content_id)
                    mail.add_attachment(attachment)
            
            # Categories/Tags
            if message.tags:
                for tag in message.tags:
                    mail.add_category(tag)
            
            # Custom args (metadata)
            if message.metadata:
                for key, value in message.metadata.items():
                    mail.add_custom_arg({key: str(value)})
            
            # Sandbox mode for testing
            if self.sandbox_mode:
                mail_settings = MailSettings()
                mail_settings.sandbox_mode = SandBoxMode(True)
                mail.mail_settings = mail_settings
            
            # Send via thread pool (sendgrid SDK is synchronous)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.send(mail)
            )
            
            # Parse response
            if response.status_code in (200, 201, 202):
                # Extract message ID from headers
                message_id = None
                if response.headers:
                    message_id = response.headers.get("X-Message-Id")
                
                result = EmailResult(
                    success=True,
                    provider=self.name,
                    message_id=message_id,
                    status=EmailStatus.SENT,
                    raw_response={
                        "status_code": response.status_code,
                        "headers": dict(response.headers) if response.headers else {}
                    }
                )
            else:
                result = EmailResult(
                    success=False,
                    provider=self.name,
                    status=EmailStatus.FAILED,
                    error_message=f"SendGrid returned status {response.status_code}",
                    error_code=str(response.status_code),
                    raw_response={
                        "status_code": response.status_code,
                        "body": response.body.decode() if response.body else None
                    }
                )
            
            self._log_send_result(message, result)
            return result
            
        except Exception as e:
            error_str = str(e)
            error_code = "UNKNOWN_ERROR"
            
            # Handle specific SendGrid errors
            if "401" in error_str or "Unauthorized" in error_str:
                error_code = "AUTHENTICATION_ERROR"
                raise EmailAuthenticationError(
                    message="Invalid SendGrid API key",
                    provider=self.name,
                    code=error_code
                )
            elif "429" in error_str or "rate limit" in error_str.lower():
                error_code = "RATE_LIMIT_ERROR"
                raise EmailRateLimitError(
                    message="SendGrid rate limit exceeded",
                    provider=self.name,
                    code=error_code
                )
            
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
        
        SendGrid supports batch sending but for simplicity we'll
        send them individually with concurrency control.
        """
        # Limit concurrency to avoid rate limits
        semaphore = asyncio.Semaphore(10)
        
        async def send_with_semaphore(msg: EmailMessage) -> EmailResult:
            async with semaphore:
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
        Check if SendGrid is operational.
        
        We verify by checking the API key is valid.
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._client:
            self._healthy = False
            return False
        
        try:
            # Use scopes endpoint to verify API key is valid
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.client.scopes.get()
            )
            
            self._healthy = response.status_code == 200
            return self._healthy
            
        except Exception as e:
            logger.warning("sendgrid_health_check_failed", error=str(e))
            self._healthy = False
            return False
    
    async def get_delivery_status(self, message_id: str) -> Optional[EmailStatus]:
        """
        Get the delivery status of a sent email.
        
        Note: Requires SendGrid Event Webhook or Activity API access.
        """
        # This would require the Activity API which is a paid feature
        # For now, we return None indicating status is unknown
        return None
