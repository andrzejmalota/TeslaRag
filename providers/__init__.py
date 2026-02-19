"""Cloud provider abstractions for storage and search."""
from providers.factory import get_storage_provider, get_search_provider

__all__ = ["get_storage_provider", "get_search_provider"]
