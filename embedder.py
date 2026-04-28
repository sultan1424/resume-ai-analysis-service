import os
import json
import boto3
import logging

logger     = logging.getLogger(__name__)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ─────────────────────────────────────────
# Factor 3: Config from environment
# Factor 4: Bedrock is a backing service
# ─────────────────────────────────────────
BEDROCK_EMBED_MODEL_ID = os.environ.get(
    "BEDROCK_EMBED_MODEL_ID",
    "amazon.titan-embed-text-v1"
)

bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def embed_text(text: str) -> list[float]:
    """
    Convert text into a vector using AWS Bedrock Titan Embeddings.
    This vector is then used to search OpenSearch for matching jobs (RAG).

    Args:
        text: the CV text extracted by extractor.py

    Returns:
        a list of floats representing the text as a vector
    """
    try:
        # Truncate text to 8000 chars — Titan has a token limit
        truncated = text[:8000]

        response = bedrock_client.invoke_model(
            modelId     = BEDROCK_EMBED_MODEL_ID,
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps({"inputText": truncated})
        )

        response_body = json.loads(response["body"].read())
        embedding     = response_body["embedding"]

        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        return embedding

    except Exception as e:
        logger.error(f"Bedrock embedding failed: {e}")
        raise
