"""
Provider factory — reads config/settings.yaml (with env var overrides)
and returns the correct cloud provider implementation.

Priority: environment variable > settings.yaml > built-in default (aws)
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml

from providers.storage.base import StorageProvider
from providers.search.base import SearchProvider

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _setting(key: str, default: str) -> str:
    """Return env var if set, otherwise fall back to config file, then *default*."""
    env_val = os.getenv(key)
    if env_val:
        return env_val
    return _load_config().get(key, default)


def get_storage_provider() -> StorageProvider:
    """Return the configured storage provider (default: aws)."""
    provider = _setting("STORAGE_PROVIDER", "aws")

    if provider == "aws":
        from providers.storage.aws import S3StorageProvider
        return S3StorageProvider()

    if provider == "azure":
        from providers.storage.azure import AzureBlobStorageProvider
        return AzureBlobStorageProvider()

    raise ValueError(
        f"Unknown STORAGE_PROVIDER: {provider!r}. Supported values: 'aws', 'azure'."
    )


def get_search_provider() -> SearchProvider:
    """Return the configured search provider (default: aws)."""
    provider = _setting("SEARCH_PROVIDER", "aws")

    if provider == "aws":
        from providers.search.aws import OpenSearchProvider
        return OpenSearchProvider()

    if provider == "azure":
        from providers.search.azure import AzureSearchProvider
        return AzureSearchProvider()

    raise ValueError(
        f"Unknown SEARCH_PROVIDER: {provider!r}. Supported values: 'aws', 'azure'."
    )
