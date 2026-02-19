"""
Backward-compatible blob storage helpers.

These functions delegate to whichever StorageProvider is selected in
config/settings.yaml (or the STORAGE_PROVIDER env var).  New code should
import from `providers` directly.
"""
from pathlib import Path

from dotenv import load_dotenv

from providers import get_storage_provider
from providers.storage.base import ObjectInfo

# Keep BlobInfo as an alias so existing imports keep working.
BlobInfo = ObjectInfo

load_dotenv()


def upload_pdf(
    file_path: Path,
    container_name: str = "documents",
    overwrite: bool = True,
) -> ObjectInfo:
    return get_storage_provider().upload(
        file_path, key=file_path.name, bucket=container_name, overwrite=overwrite
    )


def upload_directory(
    directory: Path,
    container_name: str = "documents",
    pattern: str = "*.pdf",
) -> list[ObjectInfo]:
    return get_storage_provider().upload_directory(directory, bucket=container_name, pattern=pattern)


def download_blob(
    blob_name: str,
    container_name: str = "documents",
    download_path: Path | None = None,
) -> Path:
    return get_storage_provider().download(blob_name, bucket=container_name, local_path=download_path)


def list_blobs(container_name: str = "documents") -> list[str]:
    return get_storage_provider().list_objects(bucket=container_name)
