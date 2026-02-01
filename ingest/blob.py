"""Azure Blob Storage operations for RAG ingestion."""
import os
from pathlib import Path
from dataclasses import dataclass

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BlobInfo:
    """Information about an uploaded blob."""
    container: str
    blob_name: str
    url: str
    size_bytes: int


def get_blob_service_client() -> BlobServiceClient:
    """Get authenticated blob service client."""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)
    
    # Fallback to DefaultAzureCredential (managed identity, CLI, etc.)
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    return BlobServiceClient(account_url, credential=DefaultAzureCredential())


def upload_pdf(
    file_path: Path,
    container_name: str = "documents",
    overwrite: bool = True
) -> BlobInfo:
    """
    Upload a PDF file to Azure Blob Storage.
    
    Args:
        file_path: Local path to PDF file
        container_name: Target container name
        overwrite: Whether to overwrite existing blob
        
    Returns:
        BlobInfo with upload details
    """
    client = get_blob_service_client()
    container_client = client.get_container_client(container_name)
    
    # Create container if it doesn't exist
    if not container_client.exists():
        container_client.create_container()
    
    blob_name = file_path.name
    blob_client = container_client.get_blob_client(blob_name)
    
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=overwrite)
    
    properties = blob_client.get_blob_properties()
    
    return BlobInfo(
        container=container_name,
        blob_name=blob_name,
        url=blob_client.url,
        size_bytes=properties.size
    )


def upload_directory(
    directory: Path,
    container_name: str = "documents",
    pattern: str = "*.pdf"
) -> list[BlobInfo]:
    """Upload all matching files from a directory."""
    results = []
    for file_path in directory.glob(pattern):
        info = upload_pdf(file_path, container_name, overwrite=True)
        print(f"Uploaded: {file_path.name} → {info.url}")
        results.append(info)
    return results


def download_blob(
    blob_name: str,
    container_name: str = "documents",
    download_path: Path = None
) -> Path:
    """Download a blob to local storage."""
    client = get_blob_service_client()
    blob_client = client.get_blob_client(container_name, blob_name)
    
    if download_path is None:
        download_path = Path(f"data/downloads/{blob_name}")
    
    download_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(download_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    
    return download_path


def list_blobs(container_name: str = "documents") -> list[str]:
    """List all blobs in a container."""
    client = get_blob_service_client()
    container_client = client.get_container_client(container_name)
    return [blob.name for blob in container_client.list_blobs()]