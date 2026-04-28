import os
import json
import boto3
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger               = logging.getLogger(__name__)
AWS_REGION           = os.environ.get("AWS_REGION", "us-east-1")

# ─────────────────────────────────────────
# Factor 3: Config from environment
# Factor 4: OpenSearch is a backing service
# ─────────────────────────────────────────
OPENSEARCH_ENDPOINT  = os.environ.get("OPENSEARCH_ENDPOINT", "")
OPENSEARCH_INDEX     = "job-descriptions"

# Use IAM role credentials (Factor 15 — no hardcoded keys)
credentials = boto3.Session().get_credentials()
aws_auth    = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    AWS_REGION,
    "es",
    session_token=credentials.token
)

# OpenSearch client
opensearch_client = OpenSearch(
    hosts               = [OPENSEARCH_ENDPOINT],
    http_auth           = aws_auth,
    use_ssl             = True,
    verify_certs        = True,
    connection_class    = RequestsHttpConnection
)


def search_jobs(cv_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    RAG Step — search OpenSearch for the most relevant job descriptions
    using the CV's vector embedding.

    Args:
        cv_vector: the embedding vector from embedder.py
        top_k: number of matching jobs to return

    Returns:
        list of matching job descriptions
    """
    try:
        # KNN (K-Nearest Neighbors) vector search
        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": cv_vector,
                        "k":      top_k
                    }
                }
            }
        }

        response = opensearch_client.search(
            index = OPENSEARCH_INDEX,
            body  = query
        )

        hits = response["hits"]["hits"]
        jobs = [hit["_source"] for hit in hits]

        logger.info(f"Found {len(jobs)} matching jobs from OpenSearch")
        return jobs

    except Exception as e:
        logger.error(f"OpenSearch search failed: {e}")
        # Return empty list — analyzer will still work without RAG results
        return []
