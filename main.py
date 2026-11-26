
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai

app = FastAPI(
    title="AI Affiliate Agency",
    description="AI-powered affiliate research platform",
    version="4.0.0"
)

MODEL_NAME = "gemini-3.5-flash-lite"
class ResearchRequest(BaseModel):
    niche: str = "AI tools and SaaS software"
    number_of_programs: int = Field(default=10, ge=1, le=10)


class AIRequest(BaseModel):
    prompt: str


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing in Railway."
        )

    return genai.Client(api_key=api_key)


def generate_ai_response(prompt: str):
    client = get_gemini_client()

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if not response.text:
                raise Exception("Gemini returned an empty response.")

            return response.text

        except Exception as error:
            error_message = str(error)
            print(f"Gemini error: {error_message}")

            if "429" in error_message:
                raise HTTPException(
                    status_code=429,
                    detail="Gemini free-tier quota or rate limit reached. Try again later."
                )

            if "503" in error_message:
                if attempt == 0:
                    time.sleep(3)
                    continue

                raise HTTPException(
                    status_code=503,
                    detail="Gemini is temporarily unavailable. Try again later."
                )

            raise HTTPException(
                status_code=502,
                detail="Gemini request failed. Check Railway logs."
            )


@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AI Affiliate Agency",
        "version": "4.0.0",
        "ai_model": MODEL_NAME
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ai-test")
def ai_test():
    prompt = """
    Introduce yourself as the AI assistant
    of an AI and SaaS affiliate marketing agency.

    Explain your main responsibilities in two short sentences.
    """

    result = generate_ai_response(prompt)

    return {
        "status": "success",
        "model": MODEL_NAME,
        "response": result
    }


@app.post("/research")
def research(request: ResearchRequest):

    prompt = f"""
    You are the Affiliate Market Research Analyst
    of an AI and SaaS affiliate marketing agency.

    Your task is to analyze this niche:

    NICHE: {request.niche}

    Prepare a report covering up to
    {request.number_of_programs} potential affiliate opportunities.

    For each opportunity analyze:

    1. Company and product name
    2. Product category
    3. Official website, if known
    4. Affiliate program, if known
    5. Possible commission model
    6. Recurring commission potential
    7. Target customers
    8. Main advantages
    9. Main disadvantages
    10. Marketing potential

    IMPORTANT RULES:

    - You do not have live internet search in this version.
    - Do not claim that you visited or verified websites.
    - Never invent exact commission percentages.
    - Never invent cookie durations.
    - Never claim an affiliate program is currently open
      unless that information has been verified.
    - Mark unknown information as "Not verified".
    - Clearly distinguish facts from estimates.
    - Do not invent sources or URLs.

    At the end:

    A. Rank the strongest opportunities.
    B. Explain why they are promising.
    C. Recommend three products for further verification.
    D. Explain what information must be checked
       on each company's official affiliate page.

    Return a clear, professional research report.
    """

    result = generate_ai_response(prompt)

    return {
        "status": "success",
        "agent": "Affiliate Market Researcher",
        "model": MODEL_NAME,
        "niche": request.niche,
        "programs_requested": request.number_of_programs,
        "live_web_search": False,
        "report": result
    }


@app.post("/ask-ai")
def ask_ai(request: AIRequest):

    result = generate_ai_response(request.prompt)

    return {
        "status": "success",
        "model": MODEL_NAME,
        "response": result
    }

class AffiliateAnalysisRequest(BaseModel):
    company_name: str
    product_name: str
    product_description: str = "Not provided"
    affiliate_details: str = "Not verified"
    target_audience: str = "Not specified"


@app.post("/analyze-affiliate")
def analyze_affiliate(request: AffiliateAnalysisRequest):

    prompt = f"""
    You are the Affiliate Program Analyst
    of a professional AI and SaaS affiliate agency.

    Analyze the following affiliate opportunity:

    COMPANY: {request.company_name}
    PRODUCT: {request.product_name}
    DESCRIPTION: {request.product_description}
    AFFILIATE DETAILS: {request.affiliate_details}
    TARGET AUDIENCE: {request.target_audience}

    Analyze these areas:

    1. Product value and market demand.
    2. Target audience and ideal customers.
    3. Affiliate commission model.
    4. Recurring revenue potential.
    5. Cookie duration and program conditions.
    6. Competition and marketing difficulty.
    7. Advantages and disadvantages.
    8. Potential marketing channels.
    9. Main risks.
    10. Overall opportunity score from 1 to 10.

    IMPORTANT RULES:

    - Do not invent commission percentages.
    - Do not invent cookie durations.
    - Do not claim program details are verified.
    - Mark missing information as "Not verified".
    - Clearly distinguish facts from estimates.
    - Do not invent sources or URLs.
    - Do not promise earnings or guaranteed results.

    Finish with:
    A. Overall assessment.
    B. Recommendation: High, Medium or Low potential.
    C. Information that must be verified.
    D. Recommended next steps.

    Write a structured professional report.
    """

    result = generate_ai_response(prompt)

    return {
        "status": "success",
        "agent": "Affiliate Analyst",
        "model": MODEL_NAME,
        "company": request.company_name,
        "product": request.product_name,
        "analysis": result
    }

@app.post("/full-workflow")
def full_workflow(request: ResearchRequest):

    # STEP 1: Research Agent
    research_prompt = f"""
    You are the Research Agent of an AI and SaaS affiliate agency.

    Research potential affiliate opportunities in this niche:
    {request.niche}

    Identify up to {request.number_of_programs} potential products.

    For each product include:
    - Company name
    - Product name
    - Product category
    - Target audience
    - Possible affiliate program
    - Possible commission model
    - Marketing potential

    IMPORTANT:
    No live web search is available.
    Do not invent commission rates or cookie durations.
    Mark unverified information clearly.
    Do not invent sources or URLs.

    Return a structured report.
    """

    research_result = generate_ai_response(research_prompt)

    # STEP 2: Affiliate Analyst
    analysis_prompt = f"""
    You are the Affiliate Analyst of an AI and SaaS affiliate agency.

    Analyze the following research report:

    {research_result}

    Evaluate each identified opportunity based on:

    1. Product value
    2. Target audience
    3. Market demand
    4. Affiliate commission potential
    5. Recurring revenue potential
    6. Competition
    7. Marketing difficulty
    8. Advantages and disadvantages
    9. Business risks

    IMPORTANT:
    - Do not invent commission rates.
    - Do not invent cookie durations.
    - Mark unverified details clearly.
    - Distinguish facts from estimates.
    - Do not promise earnings.

    Finish with:
    - Ranked opportunities
    - Best three opportunities for further verification
    - Final recommendations
    - Next steps

    Return a professional analysis.
    """

    analysis_result = generate_ai_response(analysis_prompt)

    return {
        "status": "success",
        "workflow": "Research to Recommendation",
        "model": MODEL_NAME,
        "niche": request.niche,
        "research_report": research_result,
        "final_analysis": analysis_result
    }
