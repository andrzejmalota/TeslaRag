"""
Search index module — delegates to the configured search provider.
"""
import os

from providers import get_search_provider

INDEX_NAME = os.environ.get("SEARCH_INDEX_NAME", "tesla-manual-v1")


def upsert_embeddings(embedded_chunks: list[dict]) -> None:
    """Upsert embedded chunks into the configured search index."""
    if not embedded_chunks:
        return
    provider = get_search_provider()
    provider.upsert_documents(embedded_chunks)
