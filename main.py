
import os
from fastapi import FastAPI, HTTPException
from google import genai

app = FastAPI(
    title="AI Affiliate Agency",
    description="AI-powered affiliate marketing automation platform",
    version="1.1.0"
)

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AI Affiliate Agency",
        "message": "Our AI platform is starting!"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ai-test")
def ai_test():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing."
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=(
                "You are the AI assistant of an automated "
                "affiliate marketing agency. "
                "Introduce yourself in two short sentences."
            )
        )

        return {
            "status": "success",
            "ai_model": "Gemini",
            "response": response.text
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Gemini connection failed. Check Railway logs."
        )
