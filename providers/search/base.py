"""Abstract base class for vector search / index providers."""
from abc import ABC, abstractmethod


class SearchProvider(ABC):
    """Common interface for cloud vector search services (Azure AI Search, OpenSearch, etc.)."""

    @abstractmethod
    def create_or_update_index(self, index_name: str) -> None:
        """Create the index if it does not exist, or update its schema."""

    @abstractmethod
    def upsert_documents(self, documents: list[dict]) -> None:
        """Insert or update documents (with pre-computed embeddings) in the index."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        index_name: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Run a vector similarity search and return the top-k results."""
