"""Abstract base class for object/blob storage providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ObjectInfo:
    """Provider-agnostic information about an uploaded object."""
    bucket: str       # container (Azure) or bucket (AWS)
    key: str          # blob name (Azure) or S3 key (AWS)
    url: str
    size_bytes: int


class StorageProvider(ABC):
    """Common interface for cloud object storage (S3, Azure Blob, etc.)."""

    @abstractmethod
    def upload(
        self,
        file_path: Path,
        key: str,
        bucket: str = "documents",
        overwrite: bool = True,
    ) -> ObjectInfo:
        """Upload a file and return metadata about the stored object."""

    @abstractmethod
    def upload_directory(
        self,
        directory: Path,
        bucket: str = "documents",
        pattern: str = "*.pdf",
    ) -> list[ObjectInfo]:
        """Upload all files matching *pattern* from *directory*."""

    @abstractmethod
    def download(
        self,
        key: str,
        bucket: str = "documents",
        local_path: Path | None = None,
    ) -> Path:
        """Download an object to local disk and return the local path."""

    @abstractmethod
    def list_objects(self, bucket: str = "documents") -> list[str]:
        """Return the keys of every object in *bucket*."""
