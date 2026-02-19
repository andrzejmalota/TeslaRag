"""Azure AI Search implementation of SearchProvider."""
import os

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from providers.search.base import SearchProvider
from index.schema import build_tesla_manual_index


class AzureSearchProvider(SearchProvider):
    """Creates and queries indexes using Azure AI Search."""

    def _credential(self) -> AzureKeyCredential:
        api_key = os.environ["AZURE_SEARCH_API_KEY"]
        return AzureKeyCredential(api_key)

    def _endpoint(self) -> str:
        endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        if not endpoint:
            raise EnvironmentError("Set the AZURE_SEARCH_ENDPOINT environment variable.")
        return endpoint

    def create_or_update_index(self, index_name: str) -> None:
        client = SearchIndexClient(
            endpoint=self._endpoint(),
            credential=self._credential(),
        )
        index = build_tesla_manual_index(index_name)
        client.create_or_update_index(index)
        print(f"Index '{index_name}' created or updated in Azure AI Search.")

    def upsert_documents(self, documents: list[dict]) -> None:
        index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "tesla-manual-v1")
        client = SearchClient(
            endpoint=self._endpoint(),
            index_name=index_name,
            credential=self._credential(),
        )
        client.upload_documents(documents=documents)

    def search(
        self,
        query_vector: list[float],
        index_name: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        from azure.search.documents.models import VectorizedQuery

        client = SearchClient(
            endpoint=self._endpoint(),
            index_name=index_name,
            credential=self._credential(),
        )
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="contentVector",
        )
        results = client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
        )
        return [dict(r) for r in results]
