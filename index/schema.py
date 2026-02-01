from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large


def build_tesla_manual_index(index_name: str) -> SearchIndex:
    """
    Build Azure AI Search index schema for Tesla Manual RAG (v1).
    This function is PURE: no network calls, no side effects.
    """

    fields = [
        # --- Primary key ---
        SearchField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            retrievable=True,
        ),

        # --- Content ---
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),

        # --- Vector ---
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),

        # --- Metadata ---
        SearchField(
            name="doc_id",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name="section",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name="page_start",
            type=SearchFieldDataType.Int32,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name="page_end",
            type=SearchFieldDataType.Int32,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name="car_model",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name="model_year",
            type=SearchFieldDataType.Int32,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SearchField(
            name="embedding_version",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                metric="cosine",
            )
        ],
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )
