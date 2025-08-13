# utils/storage_client.py

"""
Object Storage Client for S3-compatible storage systems.
Supports MinIO, DigitalOcean Spaces, AWS S3, and other S3-compatible storage.
"""

import asyncio
import functools
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
from io import BytesIO

from common.logger import LoggerFactory, LoggerType, LogLevel


class StorageClientError(Exception):
    """Custom exception for storage client operations."""

    pass


class StorageClient:
    """
    S3-compatible object storage client with async support.

    Supports MinIO, AWS S3, DigitalOcean Spaces, and other S3-compatible providers.
    Provides methods for upload, download, delete, and management operations.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:9000",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region_name: str = "us-east-1",
        bucket_name: str = "market-data",
        use_ssl: bool = False,
        verify_ssl: bool = True,
        signature_version: str = "s3v4",
        max_pool_connections: int = 50,
    ):
        """
        Initialize storage client.

        Args:
            endpoint_url: S3 endpoint URL (e.g., MinIO server URL)
            access_key: S3 access key (falls back to environment variables)
            secret_key: S3 secret key (falls back to environment variables)
            region_name: S3 region name
            bucket_name: Default bucket name for operations
            use_ssl: Whether to use SSL/TLS
            verify_ssl: Whether to verify SSL certificates
            signature_version: S3 signature version
            max_pool_connections: Maximum connection pool size
        """
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.use_ssl = use_ssl
        self.verify_ssl = verify_ssl

        # Initialize logger
        self.logger = LoggerFactory.get_logger(
            name="storage-client",
            logger_type=LoggerType.STANDARD,
            level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            log_file="logs/storage_client.log",
        )

        # Configure boto3 client
        config = Config(
            region_name=region_name,
            signature_version=signature_version,
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        )

        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=config,
                use_ssl=use_ssl,
                verify=verify_ssl,
            )

            self.s3_resource = boto3.resource(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=config,
                use_ssl=use_ssl,
                verify=verify_ssl,
            )

            self.logger.info(
                f"Storage client initialized with endpoint: {endpoint_url}"
            )

        except (NoCredentialsError, ClientError) as e:
            self.logger.error(f"Failed to initialize storage client: {e}")
            raise StorageClientError(f"Storage client initialization failed: {e}")

    async def _call(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    async def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create a bucket if it doesn't exist.

        Args:
            bucket_name: Bucket name (uses default if not provided)

        Returns:
            True if bucket created or already exists

        Raises:
            StorageClientError: If bucket creation fails
        """
        bucket = bucket_name or self.bucket_name
        try:
            # check tồn tại
            await self._call(self.s3_client.head_bucket, Bucket=bucket)
            self.logger.info(f"Bucket '{bucket}' already exists")
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchBucket"):
                try:
                    if self.region_name == "us-east-1":
                        await self._call(self.s3_client.create_bucket, Bucket=bucket)
                    else:
                        await self._call(
                            self.s3_client.create_bucket,
                            Bucket=bucket,
                            CreateBucketConfiguration={
                                "LocationConstraint": self.region_name
                            },
                        )
                    self.logger.info(f"Created bucket '{bucket}'")
                    return True
                except ClientError as ce:
                    self.logger.error(f"Failed to create bucket '{bucket}': {ce}")
                    raise StorageClientError(f"Failed to create bucket: {ce}")
            else:
                self.logger.error(f"Error checking bucket '{bucket}': {e}")
                raise StorageClientError(f"Error checking bucket: {e}")

    async def upload_file(
        self,
        local_file_path: Union[str, Path],
        object_key: str,
        bucket_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> bool:
        """
        Upload a file to object storage.

        Args:
            local_file_path: Local file path to upload
            object_key: Object key in storage
            bucket_name: Target bucket (uses default if not provided)
            metadata: Optional metadata to attach
            content_type: Content type for the object

        Returns:
            True if upload successful

        Raises:
            StorageClientError: If upload fails
        """
        bucket = bucket_name or self.bucket_name
        local_path = Path(local_file_path)

        if not local_path.exists():
            raise StorageClientError(f"Local file not found: {local_path}")

        try:
            # Ensure bucket exists
            await self.create_bucket(bucket)

            # Prepare upload parameters
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            if content_type:
                extra_args["ContentType"] = content_type

            # Upload file
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.upload_file,
                str(local_path),
                bucket,
                object_key,
                extra_args,
            )

            file_size = local_path.stat().st_size
            self.logger.info(
                f"Uploaded file '{local_path}' to '{bucket}/{object_key}' ({file_size} bytes)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to upload file '{local_path}': {e}")
            raise StorageClientError(f"Upload failed: {e}")

    async def download_file(
        self,
        object_key: str,
        local_file_path: Union[str, Path],
        bucket_name: Optional[str] = None,
    ) -> bool:
        """
        Download a file from object storage.

        Args:
            object_key: Object key in storage
            local_file_path: Local destination path
            bucket_name: Source bucket (uses default if not provided)

        Returns:
            True if download successful

        Raises:
            StorageClientError: If download fails
        """
        bucket = bucket_name or self.bucket_name
        local_path = Path(local_file_path)

        try:
            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.download_file,
                bucket,
                object_key,
                str(local_path),
            )

            file_size = local_path.stat().st_size
            self.logger.info(
                f"Downloaded '{bucket}/{object_key}' to '{local_path}' ({file_size} bytes)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to download '{bucket}/{object_key}': {e}")
            raise StorageClientError(f"Download failed: {e}")

    async def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        bucket_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> bool:
        """
        Upload bytes data to object storage.

        Args:
            data: Bytes data to upload
            object_key: Object key in storage
            bucket_name: Target bucket (uses default if not provided)
            metadata: Optional metadata to attach
            content_type: Content type for the object

        Returns:
            True if upload successful

        Raises:
            StorageClientError: If upload fails
        """
        bucket = bucket_name or self.bucket_name

        try:
            # Ensure bucket exists
            await self.create_bucket(bucket)

            # Prepare upload parameters
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            if content_type:
                extra_args["ContentType"] = content_type

            # Upload bytes
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.upload_fileobj,
                BytesIO(data),
                bucket,
                object_key,
                extra_args,
            )

            self.logger.info(f"Uploaded {len(data)} bytes to '{bucket}/{object_key}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to upload bytes to '{bucket}/{object_key}': {e}")
            raise StorageClientError(f"Upload failed: {e}")

    async def download_bytes(
        self,
        object_key: str,
        bucket_name: Optional[str] = None,
    ) -> bytes:
        """
        Download object as bytes.

        Args:
            object_key: Object key in storage
            bucket_name: Source bucket (uses default if not provided)

        Returns:
            Object data as bytes

        Raises:
            StorageClientError: If download fails
        """
        bucket = bucket_name or self.bucket_name

        try:
            buffer = BytesIO()
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.download_fileobj,
                bucket,
                object_key,
                buffer,
            )

            data = buffer.getvalue()
            self.logger.info(
                f"Downloaded {len(data)} bytes from '{bucket}/{object_key}'"
            )
            return data

        except Exception as e:
            self.logger.error(f"Failed to download '{bucket}/{object_key}': {e}")
            raise StorageClientError(f"Download failed: {e}")

    async def delete_object(
        self,
        object_key: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """
        Delete an object from storage.

        Args:
            object_key: Object key to delete
            bucket_name: Source bucket (uses default if not provided)

        Returns:
            True if deletion successful

        Raises:
            StorageClientError: If deletion fails
        """
        bucket = bucket_name or self.bucket_name
        try:
            await self._call(
                self.s3_client.delete_object, Bucket=bucket, Key=object_key
            )
            self.logger.info(f"Deleted object '{bucket}/{object_key}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete '{bucket}/{object_key}': {e}")
            raise StorageClientError(f"Delete failed: {e}")

    async def list_objects(
        self,
        prefix: str = "",
        bucket_name: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        List objects in bucket with optional prefix filter.

        Args:
            prefix: Object key prefix filter
            bucket_name: Source bucket (uses default if not provided)
            max_keys: Maximum number of keys to return

        Returns:
            List of object information dictionaries

        Raises:
            StorageClientError: If listing fails
        """
        bucket = bucket_name or self.bucket_name

        try:
            # Prepare parameters for list_objects_v2
            params = {"Bucket": bucket, "MaxKeys": max_keys}
            if prefix and prefix.strip():  # Only add Prefix if it's not empty
                params["Prefix"] = prefix

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(**params),
            )

            objects = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    objects.append(
                        {
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "etag": obj["ETag"].strip('"'),
                        }
                    )

            self.logger.info(
                f"Listed {len(objects)} objects in '{bucket}' with prefix '{prefix}'"
            )
            return objects

        except Exception as e:
            self.logger.error(f"Failed to list objects in '{bucket}': {e}")
            raise StorageClientError(f"StorageClient: List failed: {e}")

    async def object_exists(
        self,
        object_key: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """
        Check if an object exists in storage.

        Args:
            object_key: Object key to check
            bucket_name: Source bucket (uses default if not provided)

        Returns:
            True if object exists
        """
        bucket = bucket_name or self.bucket_name

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(Bucket=bucket, Key=object_key),
            )
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                self.logger.error(f"Error checking object '{bucket}/{object_key}': {e}")
                raise StorageClientError(f"Error checking object: {e}")

    async def get_object_info(
        self,
        object_key: str,
        bucket_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get object metadata and information.

        Args:
            object_key: Object key to check
            bucket_name: Source bucket (uses default if not provided)

        Returns:
            Object information dictionary or None if not found
        """
        bucket = bucket_name or self.bucket_name

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(Bucket=bucket, Key=object_key),
            )

            return {
                "key": object_key,
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"].strip('"'),
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            else:
                self.logger.error(
                    f"Error getting object info '{bucket}/{object_key}': {e}"
                )
                raise StorageClientError(f"Error getting object info: {e}")

    async def generate_presigned_url(
        self,
        object_key: str,
        bucket_name: Optional[str] = None,
        expiration: int = 3600,
        http_method: str = "GET",
    ) -> str:
        """
        Generate a presigned URL for temporary access to an object.

        Args:
            object_key: Object key
            bucket_name: Source bucket (uses default if not provided)
            expiration: URL expiration time in seconds
            http_method: HTTP method (GET, PUT, etc.)

        Returns:
            Presigned URL string

        Raises:
            StorageClientError: If URL generation fails
        """
        bucket = bucket_name or self.bucket_name

        try:
            url = await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.generate_presigned_url,
                f"{http_method.lower()}_object",
                {"Bucket": bucket, "Key": object_key},
                expiration,
            )

            self.logger.info(
                f"Generated presigned URL for '{bucket}/{object_key}' (expires in {expiration}s)"
            )
            return url

        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL: {e}")
            raise StorageClientError(f"Presigned URL generation failed: {e}")

    async def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage client and bucket information.

        Returns:
            Storage information dictionary
        """
        try:
            bucket_info = {}
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.s3_client.head_bucket(Bucket=self.bucket_name),
                )
                bucket_info["exists"] = True
                bucket_info["region"] = (
                    response.get("ResponseMetadata", {})
                    .get("HTTPHeaders", {})
                    .get("x-amz-bucket-region")
                )
            except ClientError:
                bucket_info["exists"] = False

            # List some objects to get usage stats
            objects = await self.list_objects(max_keys=1000)
            total_size = sum(obj["size"] for obj in objects)

            return {
                "storage_type": "s3_compatible",
                "endpoint_url": self.endpoint_url,
                "bucket_name": self.bucket_name,
                "region": self.region_name,
                "bucket_info": bucket_info,
                "total_objects": len(objects),
                "total_size_bytes": total_size,
                "use_ssl": self.use_ssl,
            }

        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {
                "storage_type": "s3_compatible",
                "endpoint_url": self.endpoint_url,
                "bucket_name": self.bucket_name,
                "error": str(e),
            }
