"""AWS OpenSearch implementation of SearchProvider."""
import os

from providers.search.base import SearchProvider

EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large


class OpenSearchProvider(SearchProvider):
    """Creates and queries indexes using Amazon OpenSearch Service."""

    def __init__(self) -> None:
        self._endpoint = os.environ.get("AWS_OPENSEARCH_ENDPOINT")
        self._index_name = os.environ.get("AWS_OPENSEARCH_INDEX_NAME", "tesla-manual-v1")

    def _client(self):
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "es",
            session_token=credentials.token,
        )
        if not self._endpoint:
            raise EnvironmentError("Set the AWS_OPENSEARCH_ENDPOINT environment variable.")
        host = self._endpoint.replace("https://", "").replace("http://", "")
        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

    def create_or_update_index(self, index_name: str) -> None:
        client = self._client()
        body = {
            "settings": {
                "index": {
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "contentVector": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSIONS,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "doc_id": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                    "car_model": {"type": "keyword"},
                    "model_year": {"type": "integer"},
                    "source": {"type": "keyword"},
                    "embedding_version": {"type": "keyword"},
                }
            },
        }
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=body)
            print(f"Index '{index_name}' created in OpenSearch.")
        else:
            print(f"Index '{index_name}' already exists in OpenSearch.")

    def upsert_documents(self, documents: list[dict]) -> None:
        from opensearchpy.helpers import bulk

        client = self._client()
        actions = [
            {
                "_index": self._index_name,
                "_id": doc["id"],
                "_source": doc,
            }
            for doc in documents
        ]
        bulk(client, actions)

    def search(
        self,
        query_vector: list[float],
        index_name: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        client = self._client()
        knn_query: dict = {
            "knn": {
                "contentVector": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        }
        if filters:
            query_body = {
                "query": {
                    "bool": {
                        "must": [knn_query],
                        "filter": [{"term": {k: v}} for k, v in filters.items()],
                    }
                }
            }
        else:
            query_body = {"query": knn_query}

        response = client.search(index=index_name, body=query_body, size=top_k)
        return [hit["_source"] for hit in response["hits"]["hits"]]
