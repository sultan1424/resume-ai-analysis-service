import os
import json
import boto3
import logging

logger           = logging.getLogger(__name__)
AWS_REGION       = os.environ.get("AWS_REGION", "us-east-1")

# ─────────────────────────────────────────
# Factor 3: Config from environment
# Factor 4: Bedrock is a backing service
# Using Amazon Titan Text — available on all AWS accounts
# ─────────────────────────────────────────
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "amazon.titan-text-express-v1"
)

bedrock_client   = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def analyze_cv(cv_text: str, matching_jobs: list[dict]) -> dict:
    """
    Send CV text + matching job descriptions to Bedrock LLM.
    Returns extracted skills and job recommendations.
    """
    jobs_context = ""
    if matching_jobs:
        jobs_context = "\n\nRelevant job descriptions from our database:\n"
        for i, job in enumerate(matching_jobs, 1):
            jobs_context += f"\n{i}. {job.get('title', 'Unknown')}\n"
            jobs_context += f"   Required skills: {job.get('skills', '')}\n"

    prompt = f"""You are an expert career advisor and CV analyzer.

Analyze the following CV and provide:
1. A list of technical and soft skills found in the CV
2. Top 5 job role recommendations based on the CV
3. A brief professional summary of the candidate

{jobs_context}

CV Content:
{cv_text[:6000]}

Respond ONLY in this exact JSON format with no extra text:
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
        response = bedrock_client.invoke_model(
            modelId     = BEDROCK_MODEL_ID,
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 1000,
                    "temperature":   0.3,
                    "topP":          0.9
                }
            })
        )

        response_body = json.loads(response["body"].read())
        raw_text      = response_body["results"][0]["outputText"]

        start    = raw_text.find("{")
        end      = raw_text.rfind("}") + 1
        json_str = raw_text[start:end]

        results = json.loads(json_str)
        logger.info("LLM analysis completed successfully")
        return results

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {
            "skills":              ["Could not extract skills"],
            "job_recommendations": [],
            "summary":             "Analysis could not be completed"
        }
    except Exception as e:
        logger.error(f"Bedrock LLM call failed: {e}")
        raise