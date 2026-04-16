"""
Lumicoria AI — S3-Compatible Dual-Write Storage Service

Writes to MinIO (primary) and Cloudflare R2 (backup) simultaneously.
All operations use the standard S3 protocol via boto3.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, Dict, List, Optional

import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)


class S3Client:
    """Thin wrapper around a boto3 S3 client."""

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        use_ssl: bool = False,
        region: str = "us-east-1",
        label: str = "s3",
    ):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.use_ssl = use_ssl
        self.region = region
        self.label = label
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            scheme = "https" if self.use_ssl else "http"
            endpoint = self.endpoint_url
            if not endpoint.startswith(("http://", "https://")):
                endpoint = f"{scheme}://{endpoint}"

            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=30,
                ),
            )
        return self._client

    # -- Bucket operations ---------------------------------------------------

    def ensure_bucket(self) -> None:
        client = self._get_client()
        try:
            client.head_bucket(Bucket=self.bucket)
        except client.exceptions.ClientError:
            client.create_bucket(Bucket=self.bucket)
            logger.info("Created S3 bucket", bucket=self.bucket, label=self.label)

    # -- Object operations ---------------------------------------------------

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        client = self._get_client()
        client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return {"bucket": self.bucket, "key": key, "size": len(data)}

    def download(self, key: str) -> bytes:
        client = self._get_client()
        response = client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def delete(self, key: str) -> None:
        client = self._get_client()
        client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.head_object(Bucket=self.bucket, Key=key)
            return True
        except client.exceptions.ClientError:
            return False

    def presigned_url(self, key: str, expiry: int = 3600) -> str:
        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiry,
        )

    def list_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        client = self._get_client()
        response = client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        objects = []
        for obj in response.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
        return objects

    def set_public_read_policy(self, prefix: str) -> None:
        """Set a bucket policy that allows anonymous read on a given prefix."""
        import json
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": f"PublicRead-{prefix.replace('/', '-')}",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self.bucket}/{prefix}*"],
                }
            ],
        }
        client = self._get_client()
        # Merge with existing policy if present
        try:
            existing = json.loads(client.get_bucket_policy(Bucket=self.bucket)["Policy"])
            # Avoid duplicates
            existing_sids = {s.get("Sid") for s in existing.get("Statement", [])}
            new_sid = policy["Statement"][0]["Sid"]
            if new_sid not in existing_sids:
                existing["Statement"].extend(policy["Statement"])
            policy = existing
        except client.exceptions.ClientError:
            pass  # No existing policy
        client.put_bucket_policy(Bucket=self.bucket, Policy=json.dumps(policy))
        logger.info("Set public-read policy", bucket=self.bucket, prefix=prefix)

    def public_url(self, key: str) -> str:
        """Return a direct (non-presigned) URL for a publicly readable object."""
        endpoint = self.endpoint_url
        scheme = "https" if self.use_ssl else "http"
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"{scheme}://{endpoint}"
        return f"{endpoint}/{self.bucket}/{key}"

    def health_check(self) -> bool:
        try:
            self._get_client().head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


class DualWriteStorageService:
    """
    S3 storage with simultaneous dual-write to MinIO (primary) and R2 (backup).

    - Reads always go to MinIO first; falls back to R2 on failure.
    - Writes go to both concurrently via asyncio.gather.
    - R2 write failures are logged but do not fail the request.
    """

    def __init__(self):
        self._primary: Optional[S3Client] = None
        self._secondary: Optional[S3Client] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Create clients and ensure buckets exist."""
        cfg = settings.s3

        self._primary = S3Client(
            endpoint_url=cfg.MINIO_ENDPOINT,
            access_key=cfg.MINIO_ACCESS_KEY,
            secret_key=cfg.MINIO_SECRET_KEY,
            bucket=cfg.MINIO_BUCKET,
            use_ssl=cfg.MINIO_USE_SSL,
            label="minio",
        )
        await asyncio.to_thread(self._primary.ensure_bucket)
        logger.info("MinIO primary storage ready", bucket=cfg.MINIO_BUCKET)

        # Set up R2 secondary if configured
        if cfg.DUAL_WRITE_ENABLED and cfg.R2_ENDPOINT and cfg.R2_ACCESS_KEY:
            self._secondary = S3Client(
                endpoint_url=cfg.R2_ENDPOINT,
                access_key=cfg.R2_ACCESS_KEY,
                secret_key=cfg.R2_SECRET_KEY,
                bucket=cfg.R2_BUCKET,
                use_ssl=True,  # R2 always uses HTTPS
                label="r2",
            )
            try:
                await asyncio.to_thread(self._secondary.ensure_bucket)
                logger.info("Cloudflare R2 secondary storage ready", bucket=cfg.R2_BUCKET)
            except Exception as e:
                logger.warning("R2 secondary storage unavailable — continuing with MinIO only", error=str(e))
                self._secondary = None
        else:
            logger.info("Dual-write disabled or R2 not configured — MinIO only")

        self._initialized = True

        # Set public-read policy on blog/ prefix so images are permanently accessible
        try:
            await asyncio.to_thread(self._primary.set_public_read_policy, "blog/")
        except Exception as e:
            logger.warning("Failed to set blog/ public-read policy", error=str(e))

    # -- Upload --------------------------------------------------------------

    async def upload_file(
        self,
        file_content: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """Upload to MinIO (required) and R2 (best-effort) simultaneously."""
        self._ensure_initialized()

        async def _upload_primary():
            return await asyncio.to_thread(self._primary.upload, key, file_content, content_type)

        async def _upload_secondary():
            if self._secondary is None:
                return None
            try:
                return await asyncio.to_thread(self._secondary.upload, key, file_content, content_type)
            except Exception as e:
                logger.error("R2 dual-write failed — file saved to MinIO only", key=key, error=str(e))
                return None

        primary_result, secondary_result = await asyncio.gather(
            _upload_primary(),
            _upload_secondary(),
        )

        return {
            "key": key,
            "size": len(file_content),
            "content_type": content_type,
            "minio": primary_result,
            "r2": secondary_result,
        }

    # -- Download ------------------------------------------------------------

    async def download_file(self, key: str) -> bytes:
        """Download from MinIO; fall back to R2 on failure."""
        self._ensure_initialized()
        try:
            return await asyncio.to_thread(self._primary.download, key)
        except Exception as primary_err:
            if self._secondary:
                logger.warning("MinIO download failed — trying R2", key=key, error=str(primary_err))
                return await asyncio.to_thread(self._secondary.download, key)
            raise

    # -- Delete --------------------------------------------------------------

    async def delete_file(self, key: str) -> bool:
        """Delete from both stores concurrently."""
        self._ensure_initialized()

        async def _del_primary():
            await asyncio.to_thread(self._primary.delete, key)

        async def _del_secondary():
            if self._secondary:
                try:
                    await asyncio.to_thread(self._secondary.delete, key)
                except Exception as e:
                    logger.error("R2 delete failed", key=key, error=str(e))

        await asyncio.gather(_del_primary(), _del_secondary())
        return True

    # -- Presigned URL -------------------------------------------------------

    async def get_presigned_url(self, key: str, expiry: Optional[int] = None) -> str:
        """Generate a presigned URL from the primary store (MinIO)."""
        self._ensure_initialized()
        if expiry is None:
            expiry = settings.s3.PRESIGNED_URL_EXPIRY
        return await asyncio.to_thread(self._primary.presigned_url, key, expiry)

    # -- List ----------------------------------------------------------------

    async def list_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        self._ensure_initialized()
        return await asyncio.to_thread(self._primary.list_objects, prefix)

    # -- Exists --------------------------------------------------------------

    async def file_exists(self, key: str) -> bool:
        self._ensure_initialized()
        return await asyncio.to_thread(self._primary.exists, key)

    # -- Health --------------------------------------------------------------

    async def health_check(self) -> bool:
        self._ensure_initialized()
        return await asyncio.to_thread(self._primary.health_check)

    # -- Public URL ----------------------------------------------------------

    async def set_public_read_policy(self, prefix: str) -> None:
        """Set public-read policy on a prefix (e.g. 'blog/') in the primary store."""
        self._ensure_initialized()
        await asyncio.to_thread(self._primary.set_public_read_policy, prefix)

    def get_public_url(self, key: str) -> str:
        """Return a permanent, non-presigned URL for a publicly readable object."""
        self._ensure_initialized()
        return self._primary.public_url(key)

    # -- Internal ------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("StorageService not initialized — call initialize() first")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
storage_service = DualWriteStorageService()
