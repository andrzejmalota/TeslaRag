"""Create or update the vector search index via the configured search provider."""
import os

from providers import get_search_provider


INDEX_NAME = os.environ.get("SEARCH_INDEX_NAME", "tesla-manual-v1")


def create_or_update_index(index_name: str = INDEX_NAME) -> None:
    provider = get_search_provider()
    provider.create_or_update_index(index_name)


if __name__ == "__main__":
    create_or_update_index()
