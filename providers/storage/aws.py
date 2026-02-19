"""AWS S3 implementation of StorageProvider."""
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from providers.storage.base import StorageProvider, ObjectInfo


class S3StorageProvider(StorageProvider):
    """Stores and retrieves objects using Amazon S3."""

    def __init__(self) -> None:
        self._bucket_default = os.getenv("AWS_S3_BUCKET", "documents")
        region = os.getenv("AWS_REGION", "us-east-1")
        self._s3 = boto3.client("s3", region_name=region)
        self._region = region

    def _object_url(self, bucket: str, key: str) -> str:
        return f"https://{bucket}.s3.{self._region}.amazonaws.com/{key}"

    def _ensure_bucket(self, bucket: str) -> None:
        try:
            self._s3.head_bucket(Bucket=bucket)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                kwargs: dict = {"Bucket": bucket}
                if self._region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": self._region
                    }
                self._s3.create_bucket(**kwargs)
            else:
                raise

    def upload(
        self,
        file_path: Path,
        key: str,
        bucket: str = "documents",
        overwrite: bool = True,
    ) -> ObjectInfo:
        bucket = bucket or self._bucket_default
        self._ensure_bucket(bucket)

        if not overwrite:
            try:
                self._s3.head_object(Bucket=bucket, Key=key)
                # Object exists and overwrite is False — skip
                head = self._s3.head_object(Bucket=bucket, Key=key)
                return ObjectInfo(
                    bucket=bucket,
                    key=key,
                    url=self._object_url(bucket, key),
                    size_bytes=head["ContentLength"],
                )
            except ClientError:
                pass  # Object doesn't exist, proceed with upload

        self._s3.upload_file(str(file_path), bucket, key)
        head = self._s3.head_object(Bucket=bucket, Key=key)
        return ObjectInfo(
            bucket=bucket,
            key=key,
            url=self._object_url(bucket, key),
            size_bytes=head["ContentLength"],
        )

    def upload_directory(
        self,
        directory: Path,
        bucket: str = "documents",
        pattern: str = "*.pdf",
    ) -> list[ObjectInfo]:
        results = []
        for file_path in directory.glob(pattern):
            info = self.upload(file_path, key=file_path.name, bucket=bucket)
            print(f"Uploaded: {file_path.name} → {info.url}")
            results.append(info)
        return results

    def download(
        self,
        key: str,
        bucket: str = "documents",
        local_path: Path | None = None,
    ) -> Path:
        bucket = bucket or self._bucket_default
        if local_path is None:
            local_path = Path(f"data/downloads/{key}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket, key, str(local_path))
        return local_path

    def list_objects(self, bucket: str = "documents") -> list[str]:
        bucket = bucket or self._bucket_default
        paginator = self._s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
