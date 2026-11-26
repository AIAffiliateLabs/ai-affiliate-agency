
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(
    title="AI Affiliate Agency",
    description="AI-powered affiliate research platform",
    version="3.0.0"
)

class ResearchRequest(BaseModel):
    niche: str = "AI and SaaS software"
    number_of_programs: int = 10

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AI Affiliate Agency",
        "version": "3.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing."
        )

    return genai.Client(api_key=api_key)

@app.get("/ai-test")
def ai_test():
    client = get_gemini_client()

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents="Introduce yourself in two short sentences."
        )
        return {
            "status": "success",
            "response": response.text
        }
    except Exception as error:
        print(f"Gemini error: {error}")
        raise HTTPException(
            status_code=502,
            detail="Gemini connection failed."
        )

@app.post("/research")
def research(request: ResearchRequest):
    client = get_gemini_client()

    prompt = f"""
    You are the Affiliate Market Research Agent
    for an AI and SaaS affiliate marketing agency.

    Research the niche: {request.niche}

    Find up to {request.number_of_programs}
    relevant affiliate programs.

    For each program, investigate:
    1. Company and product name
    2. Product category
    3. Official website
    4. Affiliate program website
    5. Commission structure
    6. Whether recurring commissions are offered
    7. Cookie duration
    8. Main affiliate requirements
    9. Potential advantages and disadvantages

    Use current Google Search information.
    Include source URLs for factual claims.

    Never invent commission rates, cookie durations,
    program availability, or affiliate terms.

    If information cannot be verified, write
    "Not verified".

    Finish with a ranked list of the strongest
    opportunities and explain your ranking.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        return {
            "status": "success",
            "agent": "Affiliate Market Researcher",
            "niche": request.niche,
            "report": response.text
        }

    except Exception as error:
        print(f"Research error: {error}")
        raise HTTPException(
            status_code=502,
            detail="Research failed. Check Railway logs."
        )
