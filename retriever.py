import logging

logger = logging.getLogger(__name__)


def search_jobs(cv_vector: list, top_k: int = 5) -> list:
    """
    RAG Step — returns empty list since OpenSearch is not used.
    The LLM will analyze the CV directly without vector search context.

    Args:
        cv_vector: the embedding vector (not used)
        top_k: number of results (not used)

    Returns:
        empty list
    """
    logger.info("Skipping vector search — using direct LLM analysis")
    return []
