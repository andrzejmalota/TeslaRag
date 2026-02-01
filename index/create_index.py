import os
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

from rag.index.schema import build_tesla_manual_index

def index_chunks(docs: list[dict]) -> None:
    """Upserts into Azure AI Search"""



def create_or_update_index():
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    api_key = os.environ["AZURE_SEARCH_API_KEY"]

    index_name = "tesla-manual-v1"

    client = SearchIndexClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )

    index = build_tesla_manual_index(index_name)

    # Idempotent: create or update
    client.create_or_update_index(index)

    print(f"✅ Index '{index_name}' created or updated.")


if __name__ == "__main__":
    create_or_update_index()
