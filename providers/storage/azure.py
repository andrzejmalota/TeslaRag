"""Azure Blob Storage implementation of StorageProvider."""
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from providers.storage.base import StorageProvider, ObjectInfo


class AzureBlobStorageProvider(StorageProvider):
    """Stores and retrieves objects using Azure Blob Storage."""

    def _client(self) -> BlobServiceClient:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            return BlobServiceClient.from_connection_string(connection_string)
        account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        if not account_url:
            raise EnvironmentError(
                "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL."
            )
        return BlobServiceClient(account_url, credential=DefaultAzureCredential())

    def upload(
        self,
        file_path: Path,
        key: str,
        bucket: str = "documents",
        overwrite: bool = True,
    ) -> ObjectInfo:
        client = self._client()
        container_client = client.get_container_client(bucket)
        if not container_client.exists():
            container_client.create_container()

        blob_client = container_client.get_blob_client(key)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

        props = blob_client.get_blob_properties()
        return ObjectInfo(
            bucket=bucket,
            key=key,
            url=blob_client.url,
            size_bytes=props.size,
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
        client = self._client()
        blob_client = client.get_blob_client(bucket, key)

        if local_path is None:
            local_path = Path(f"data/downloads/{key}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        return local_path

    def list_objects(self, bucket: str = "documents") -> list[str]:
        client = self._client()
        container_client = client.get_container_client(bucket)
        return [blob.name for blob in container_client.list_blobs()]
