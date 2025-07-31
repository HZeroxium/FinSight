# utils/storage_client.py

"""
Object Storage Client for S3-compatible storage systems.
Supports MinIO, DigitalOcean Spaces, AWS S3, and other S3-compatible storage.
Adapted for AI Prediction service artifact management.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
from io import BytesIO

from ..core.config import get_settings
from common.logger.logger_factory import LoggerFactory


class StorageClientError(Exception):
    """Custom exception for storage client operations."""

    pass


class StorageClient:
    """
    S3-compatible object storage client with async support for AI predictions.

    Supports MinIO, AWS S3, DigitalOcean Spaces, and other S3-compatible providers.
    Provides methods for upload, download, delete, and management operations.
    Optimized for model artifacts, experiment data, and ML pipeline outputs.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region_name: Optional[str] = None,
        bucket_name: Optional[str] = None,
        use_ssl: Optional[bool] = None,
        verify_ssl: bool = True,
        signature_version: str = "s3v4",
        max_pool_connections: int = 50,
    ):
        """
        Initialize storage client with configuration from settings.

        Args:
            endpoint_url: S3 endpoint URL (falls back to config)
            access_key: S3 access key (falls back to config/environment)
            secret_key: S3 secret key (falls back to config/environment)
            region_name: S3 region name (falls back to config)
            bucket_name: Default bucket name (falls back to config)
            use_ssl: Whether to use SSL/TLS (falls back to config)
            verify_ssl: Whether to verify SSL certificates
            signature_version: S3 signature version
            max_pool_connections: Maximum connection pool size
        """
        self.settings = get_settings()

        # Use provided values or fallback to configuration/environment
        self.endpoint_url = endpoint_url or self.settings.cloud_storage_endpoint
        self.access_key = (
            access_key
            or self.settings.cloud_storage_access_key
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        self.secret_key = (
            secret_key
            or self.settings.cloud_storage_secret_key
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        self.region_name = region_name or self.settings.cloud_storage_region
        self.bucket_name = bucket_name or self.settings.cloud_storage_bucket
        self.use_ssl = (
            use_ssl
            if use_ssl is not None
            else (self.endpoint_url and self.endpoint_url.startswith("https://"))
        )
        self.verify_ssl = verify_ssl

        # Initialize logger
        self.logger = LoggerFactory.get_logger("StorageClient")

        # Configure boto3 client
        config = Config(
            region_name=self.region_name,
            signature_version=signature_version,
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        )

        try:
            # For AWS S3, endpoint_url should be None
            endpoint = (
                self.endpoint_url
                if self.endpoint_url and not self.endpoint_url.endswith("amazonaws.com")
                else None
            )

            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name,
                config=config,
                use_ssl=self.use_ssl,
                verify=verify_ssl,
            )

            self.s3_resource = boto3.resource(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name,
                config=config,
                use_ssl=self.use_ssl,
                verify=verify_ssl,
            )

            self.logger.info(
                f"Storage client initialized with endpoint: {endpoint or 'AWS S3'}, bucket: {self.bucket_name}"
            )

        except (NoCredentialsError, ClientError) as e:
            self.logger.error(f"Failed to initialize storage client: {e}")
            raise StorageClientError(f"Storage client initialization failed: {e}")

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
            # Check if bucket exists
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.head_bucket, {"Bucket": bucket}
            )
            self.logger.debug(f"Bucket '{bucket}' already exists")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    if self.region_name == "us-east-1":
                        # us-east-1 doesn't need location constraint
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.s3_client.create_bucket, {"Bucket": bucket}
                        )
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.s3_client.create_bucket,
                            {
                                "Bucket": bucket,
                                "CreateBucketConfiguration": {
                                    "LocationConstraint": self.region_name
                                },
                            },
                        )

                    self.logger.info(f"Created bucket '{bucket}'")
                    return True

                except ClientError as create_error:
                    self.logger.error(
                        f"Failed to create bucket '{bucket}': {create_error}"
                    )
                    raise StorageClientError(f"Failed to create bucket: {create_error}")
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
            )

            file_size = local_path.stat().st_size
            self.logger.info(
                f"Uploaded file '{local_path.name}' to '{bucket}/{object_key}' ({file_size} bytes)"
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
                f"Downloaded '{bucket}/{object_key}' to '{local_path.name}' ({file_size} bytes)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to download '{bucket}/{object_key}': {e}")
            raise StorageClientError(f"Download failed: {e}")

    async def upload_directory(
        self,
        local_dir_path: Union[str, Path],
        object_key_prefix: str,
        bucket_name: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a directory and its contents to object storage.

        Args:
            local_dir_path: Local directory path to upload
            object_key_prefix: Object key prefix for uploaded files
            bucket_name: Target bucket (uses default if not provided)
            include_patterns: List of file patterns to include
            exclude_patterns: List of file patterns to exclude

        Returns:
            Upload result dictionary with file list and summary

        Raises:
            StorageClientError: If upload fails
        """
        bucket = bucket_name or self.bucket_name
        local_path = Path(local_dir_path)

        if not local_path.exists() or not local_path.is_dir():
            raise StorageClientError(f"Local directory not found: {local_path}")

        try:
            # Ensure bucket exists
            await self.create_bucket(bucket)

            uploaded_files = []
            total_size = 0

            # Walk through directory and upload files
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    # Apply include/exclude patterns if specified
                    if include_patterns and not any(
                        file_path.match(pattern) for pattern in include_patterns
                    ):
                        continue
                    if exclude_patterns and any(
                        file_path.match(pattern) for pattern in exclude_patterns
                    ):
                        continue

                    # Calculate relative path for object key
                    relative_path = file_path.relative_to(local_path)
                    object_key = f"{object_key_prefix.rstrip('/')}/{relative_path}"

                    # Upload file
                    success = await self.upload_file(file_path, object_key, bucket)
                    if success:
                        file_size = file_path.stat().st_size
                        uploaded_files.append(
                            {
                                "local_path": str(file_path),
                                "object_key": object_key,
                                "size": file_size,
                            }
                        )
                        total_size += file_size

            self.logger.info(
                f"Uploaded directory '{local_path}' to '{bucket}/{object_key_prefix}' "
                f"({len(uploaded_files)} files, {total_size} bytes)"
            )

            return {
                "success": True,
                "uploaded_files": uploaded_files,
                "total_files": len(uploaded_files),
                "total_size": total_size,
                "bucket": bucket,
                "object_key_prefix": object_key_prefix,
            }

        except Exception as e:
            self.logger.error(f"Failed to upload directory '{local_path}': {e}")
            raise StorageClientError(f"Directory upload failed: {e}")

    async def download_directory(
        self,
        object_key_prefix: str,
        local_dir_path: Union[str, Path],
        bucket_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download all objects with a prefix to a local directory.

        Args:
            object_key_prefix: Object key prefix to download
            local_dir_path: Local destination directory
            bucket_name: Source bucket (uses default if not provided)

        Returns:
            Download result dictionary

        Raises:
            StorageClientError: If download fails
        """
        bucket = bucket_name or self.bucket_name
        local_path = Path(local_dir_path)

        try:
            # Create local directory
            local_path.mkdir(parents=True, exist_ok=True)

            # List objects with prefix
            objects = await self.list_objects(
                prefix=object_key_prefix, bucket_name=bucket
            )

            downloaded_files = []
            total_size = 0

            for obj in objects:
                object_key = obj["key"]

                # Calculate local file path
                relative_key = object_key[len(object_key_prefix.rstrip("/")) + 1 :]
                local_file_path = local_path / relative_key

                # Download file
                success = await self.download_file(object_key, local_file_path, bucket)
                if success:
                    downloaded_files.append(
                        {
                            "object_key": object_key,
                            "local_path": str(local_file_path),
                            "size": obj["size"],
                        }
                    )
                    total_size += obj["size"]

            self.logger.info(
                f"Downloaded directory '{bucket}/{object_key_prefix}' to '{local_path}' "
                f"({len(downloaded_files)} files, {total_size} bytes)"
            )

            return {
                "success": True,
                "downloaded_files": downloaded_files,
                "total_files": len(downloaded_files),
                "total_size": total_size,
                "local_path": str(local_path),
            }

        except Exception as e:
            self.logger.error(
                f"Failed to download directory '{bucket}/{object_key_prefix}': {e}"
            )
            raise StorageClientError(f"Directory download failed: {e}")

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
                self.s3_client.head_object,
                {"Bucket": bucket, "Key": object_key},
            )
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                self.logger.error(f"Error checking object '{bucket}/{object_key}': {e}")
                raise StorageClientError(f"Error checking object: {e}")

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
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.list_objects_v2,
                {"Bucket": bucket, "Prefix": prefix, "MaxKeys": max_keys},
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

            self.logger.debug(
                f"Listed {len(objects)} objects in '{bucket}' with prefix '{prefix}'"
            )
            return objects

        except Exception as e:
            self.logger.error(f"Failed to list objects in '{bucket}': {e}")
            raise StorageClientError(f"List failed: {e}")

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
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.delete_object,
                {"Bucket": bucket, "Key": object_key},
            )

            self.logger.info(f"Deleted object '{bucket}/{object_key}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete '{bucket}/{object_key}': {e}")
            raise StorageClientError(f"Delete failed: {e}")

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
                self.s3_client.head_object,
                {"Bucket": bucket, "Key": object_key},
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
