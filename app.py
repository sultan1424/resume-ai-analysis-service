import os
import logging
from flask import Flask, request, jsonify
from extractor import extract_text
from embedder import embed_text
from retriever import search_jobs
from analyzer import analyze_cv

# ─────────────────────────────────────────
# Factor 11: Logs — write to stdout only
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────
# Factor 3: Config — read from environment variables
# ─────────────────────────────────────────
S3_BUCKET_NAME      = os.environ.get("S3_BUCKET_NAME")
DYNAMODB_TABLE      = os.environ.get("DYNAMODB_TABLE")
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
PORT                = int(os.environ.get("PORT", 8081))

# ─────────────────────────────────────────
# Factor 4: Backing Services
# ─────────────────────────────────────────
import boto3
dynamodb      = boto3.resource("dynamodb", region_name=AWS_REGION)
results_table = dynamodb.Table(DYNAMODB_TABLE) if DYNAMODB_TABLE else None
s3_client     = boto3.client("s3", region_name=AWS_REGION)


# ─────────────────────────────────────────
# Factor 14: Telemetry — health check
# ─────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "ai-analysis-service"}), 200


# ─────────────────────────────────────────
# Factor 13: API First
# POST /analyze — start CV analysis (called by upload service)
# ─────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    data   = request.get_json()
    cv_id  = data.get("cv_id")
    s3_key = data.get("s3_key")

    if not cv_id or not s3_key:
        return jsonify({"error": "cv_id and s3_key are required"}), 400

    logger.info(f"Starting analysis for cv_id={cv_id}")

    # Update DynamoDB status to "processing"
    try:
        results_table.update_item(
            Key={"cv_id": cv_id},
            UpdateExpression="SET #s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": "processing"}
        )
    except Exception as e:
        logger.error(f"DynamoDB update failed: {e}")

    # ── Step 1: Download CV from S3 ──
    try:
        response  = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        cv_bytes  = response["Body"].read()
        logger.info(f"Downloaded CV from S3: {s3_key}")
    except Exception as e:
        logger.error(f"S3 download failed: {e}")
        _update_status(cv_id, "failed")
        return jsonify({"error": "Failed to download CV"}), 500

    # ── Step 2: Extract text from CV ──
    try:
        cv_text = extract_text(cv_bytes)
        logger.info(f"Extracted text from CV, length={len(cv_text)}")
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        _update_status(cv_id, "failed")
        return jsonify({"error": "Failed to extract text"}), 500

    # ── Step 3: Embed CV text ──
    try:
        cv_vector = embed_text(cv_text)
        logger.info(f"Generated embedding vector for cv_id={cv_id}")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        _update_status(cv_id, "failed")
        return jsonify({"error": "Failed to embed text"}), 500

    # ── Step 4: RAG — search for matching jobs ──
    try:
        matching_jobs = search_jobs(cv_vector)
        logger.info(f"Found {len(matching_jobs)} matching jobs for cv_id={cv_id}")
    except Exception as e:
        logger.error(f"Job search failed: {e}")
        matching_jobs = []

    # ── Step 5: Analyze CV with Bedrock LLM ──
    try:
        results = analyze_cv(cv_text, matching_jobs)
        logger.info(f"Analysis complete for cv_id={cv_id}")
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        _update_status(cv_id, "failed")
        return jsonify({"error": "Failed to analyze CV"}), 500

    # ── Step 6: Save results to DynamoDB ──
    try:
        results_table.update_item(
            Key={"cv_id": cv_id},
            UpdateExpression="SET #s = :s, skills = :sk, job_recommendations = :jr, summary = :su",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s":  "completed",
                ":sk": results["skills"],
                ":jr": results["job_recommendations"],
                ":su": results["summary"]
            }
        )
        logger.info(f"Results saved to DynamoDB for cv_id={cv_id}")
    except Exception as e:
        logger.error(f"DynamoDB save failed: {e}")
        return jsonify({"error": "Failed to save results"}), 500

    return jsonify({
        "message": "Analysis complete",
        "cv_id":   cv_id,
        "status":  "completed"
    }), 200


def _update_status(cv_id, status):
    """Helper — update status in DynamoDB"""
    try:
        results_table.update_item(
            Key={"cv_id": cv_id},
            UpdateExpression="SET #s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": status}
        )
    except Exception as e:
        logger.error(f"Status update failed: {e}")


# ─────────────────────────────────────────
# Factor 7: Port Binding
# Factor 9: Disposability — gunicorn handles SIGTERM
# ─────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"AI Analysis Service starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
