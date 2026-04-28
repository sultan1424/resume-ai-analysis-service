import os
import json
import boto3
import logging

logger           = logging.getLogger(__name__)
AWS_REGION       = os.environ.get("AWS_REGION", "us-east-1")

# ─────────────────────────────────────────
# Factor 3: Config from environment
# Factor 4: Bedrock is a backing service
# ─────────────────────────────────────────
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-sonnet-20240229-v1:0"
)

bedrock_client   = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def analyze_cv(cv_text: str, matching_jobs: list[dict]) -> dict:
    """
    Send CV text + matching job descriptions to Bedrock LLM.
    Returns extracted skills and job recommendations.

    Args:
        cv_text:      raw text of the CV from extractor.py
        matching_jobs: relevant jobs retrieved by retriever.py (RAG)

    Returns:
        dict with skills, job_recommendations, and summary
    """
    # Format matching jobs for the prompt
    jobs_context = ""
    if matching_jobs:
        jobs_context = "\n\nRelevant job descriptions from our database:\n"
        for i, job in enumerate(matching_jobs, 1):
            jobs_context += f"\n{i}. {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}\n"
            jobs_context += f"   Required skills: {job.get('skills', '')}\n"

    # ─────────────────────────────────────────
    # The prompt — instructs the LLM what to do
    # ─────────────────────────────────────────
    prompt = f"""You are an expert career advisor and CV analyzer.

Analyze the following CV and provide:
1. A list of technical and soft skills found in the CV
2. Top 5 job role recommendations based on the CV
3. A brief professional summary of the candidate

{jobs_context}

CV Content:
{cv_text[:6000]}

Respond ONLY in this exact JSON format:
{{
    "skills": ["skill1", "skill2", "skill3"],
    "job_recommendations": [
        {{
            "title": "Job Title",
            "match_percentage": 90,
            "reason": "Why this job fits"
        }}
    ],
    "summary": "Brief professional summary of the candidate"
}}"""

    try:
        # Call Bedrock Claude model
        response = bedrock_client.invoke_model(
            modelId     = BEDROCK_MODEL_ID,
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":        1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )

        response_body = json.loads(response["body"].read())
        raw_text      = response_body["content"][0]["text"]

        # Parse the JSON response from the LLM
        results = json.loads(raw_text)
        logger.info("LLM analysis completed successfully")
        return results

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        # Return a safe default if parsing fails
        return {
            "skills":              ["Could not extract skills"],
            "job_recommendations": [],
            "summary":             "Analysis could not be completed"
        }
    except Exception as e:
        logger.error(f"Bedrock LLM call failed: {e}")
        raise
